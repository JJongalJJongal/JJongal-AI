"""
아이들과 대화하며 동화 줄거리를 수집하는 AI 챗봇 모듈
"""
from typing import List, Dict, Optional, Any
from pathlib import Path
import random

# 자체 개발 모듈 임포트
from ..rag_system import RAGSystem
from .message_formatter import MessageFormatter
from .story_collector import StoryCollector
from .conversation_manager import ConversationManager
from .story_analyzer import StoryAnalyzer

# 공통 유틸리티 모듈 임포트
from shared.utils.logging_utils import get_module_logger
from shared.utils.openai_utils import initialize_client
from shared.configs.prompts_config import load_chatbot_a_prompts

# 로거 설정
logger = get_module_logger(__name__)

class StoryCollectionChatBot:
    """
    아이들과 대화하며 동화 줄거리를 수집하는 AI 챗봇 클래스
    
    Attributes:
        conversation_history (List[Dict]): 대화 내역을 저장하는 리스트 (role, content)
        age_group (int): 아이의 연령대 (4-9세)
        child_name (str): 아이의 이름
        interests (List[str]): 아이의 관심사 목록
        chatbot_name (str): 챗봇의 이름
        prompts (Dict): JSON 파일에서 로드한 프롬프트
        story_outline (Dict): 수집된 이야기 줄거리
    """
    
    def __init__(self, token_limit: int = 10000):
        """
        챗봇 초기화 및 기본 속성 설정
        
        Args:
            token_limit (int, optional): 전체 대화에서 사용 가능한 최대 토큰 수. 기본값은 10000.
        """
        self.age_group = None          # 아이의 연령대 (ex: 4-9세)
        self.child_name = None         # 아이의 이름
        self.interests = []            # 아이의 관심사
        self.chatbot_name = "부기"
        
        # 프롬프트 로드
        self.prompts = load_chatbot_a_prompts()
        
        # 모듈화된 컴포넌트 초기화
        self.conversation = ConversationManager(token_limit)
        self.formatter = MessageFormatter(self.prompts)
        self.collector = StoryCollector()
        
        # OpenAI 클라이언트 초기화
        try:
            self.openai_client = initialize_client()
        except Exception as e:
            logger.error(f"OpenAI 클라이언트 초기화 실패: {e}")
            self.openai_client = None
        
        # RAG 시스템 초기화
        try:
            self.rag_system = RAGSystem()
            # 샘플 스토리 추가 (필요 시)
            # self.rag_system.import_sample_stories()
        except Exception as e:
            logger.error(f"RAG 시스템 초기화 중 오류 발생: {e}")
            self.rag_system = None
            
        # 스토리 분석기 초기화
        self.analyzer = StoryAnalyzer(self.openai_client, self.rag_system)
        
        # 시스템 메시지 설정
        self.system_message = None
    
    # 아이 정보 업데이트하는 Function
    def update_child_info(self, child_name: str = None, age: int = None, interests: List[str] = None):
        """
        아이 정보를 업데이트하는 메서드
        
        Args:
            child_name (str, optional): 아이의 이름
            age (int, optional): 아이의 나이
            interests (List[str], optional): 아이의 관심사 목록
        """
        if child_name:
            self.child_name = child_name
        if age:
            self.age_group = age
        if interests:
            self.interests = interests
        
        # 메시지 포맷터 업데이트
        self.formatter = MessageFormatter(
            self.prompts, 
            self.child_name, 
            self.age_group, 
            self.interests, 
            self.chatbot_name
        )
            
        logger.info(f"아이 정보 업데이트: 이름={self.child_name}, 나이={self.age_group}, 관심사={self.interests}")
    
    def add_to_conversation(self, role: str, content: str):
        """대화 내역에 메시지 추가"""
        self.conversation.add_message(role, content)
    
    def get_conversation_history(self) -> List[Dict]:
        """대화 내역 반환"""
        return self.conversation.get_conversation_history()
    
    def initialize_chat(self, child_name: str, age: int, interests: List[str] = None, chatbot_name: str = "부기") -> str:
        """
        챗봇과의 대화를 초기화하는 함수
        
        Args:
            child_name (str): 아이의 이름
            age (int): 아이의 나이
            interests (List[str], optional): 아이의 관심사 목록
            chatbot_name (str, optional): 챗봇의 이름
            
        Returns:
            str: 초기 인사 메시지
        """
        # 입력 검증 및 기본값 설정
        if not child_name or not isinstance(child_name, str):
            raise ValueError("아이의 이름은 필수고, 문자열이어야 합니다.")
        if not isinstance(age, int) or age < 4 or age > 9:
            raise ValueError("아이의 나이는 4-9세 사이 정수이어야 합니다.")
        if interests and not isinstance(interests, list):
            raise ValueError("관심사는 리스트 형태여야 합니다.")
        if chatbot_name and not isinstance(chatbot_name, str):
            raise ValueError("챗봇의 이름은 문자열 형태여야 합니다.")
        
        # 챗봇 정보 업데이트
        self.update_child_info(child_name, age, interests)
        self.chatbot_name = chatbot_name
        
        # 메시지 포맷터 업데이트
        self.formatter = MessageFormatter(
            self.prompts, 
            self.child_name, 
            self.age_group, 
            self.interests, 
            self.chatbot_name
        )
        
        # 시스템 메시지 설정
        self.system_message = self.formatter.get_system_message()
        
        # 인사말 생성
        greeting = self.formatter.get_greeting()
        
        # 대화 시작
        self.add_to_conversation("assistant", greeting)
        
        return greeting
    
    def suggest_story_element(self, user_input: str) -> str:
        """
        사용자 입력을 분석하여 이야기 요소 제안
        
        Args:
            user_input (str): 사용자 입력
            
        Returns:
            str: 다음 대화 제안
        """
        # 사용자 입력 분석
        self.collector.analyze_user_response(user_input, self.openai_client)
        
        # 대화 길이 확인
        conversation_length = len(self.get_conversation_history())
        
        # 단계 전환 여부 확인
        if self.collector.should_transition_to_next_stage(conversation_length):
            self.collector.transition_to_next_stage(conversation_length)
            return self.formatter.get_stage_transition_message(self.collector.get_current_stage())
            
        # 현재 수집 단계에 따라 다른 질문 반환
        if random.random() < 0.7:  # 70% 확률로 단계별 질문
            return self.formatter.get_story_prompting_question(self.collector.get_current_stage())
        else:  # 30% 확률로 후속 질문
            if random.random() < 0.5:  # 격려
                return self.formatter.get_encouragement()
            else:  # 후속 질문
                return self.formatter.get_follow_up_question()
    
    def get_response(self, user_input: str) -> str:
        """
        사용자의 입력에 대한 AI 응답을 생성하는 함수
        
        Args:
            user_input (str): 사용자의 입력 메시지
            
        Returns:
            str: AI가 생성한 응답 메시지
        """
        try:
            # 토큰 제한 확인
            if self.conversation.is_token_limit_reached():
                return self.conversation.token_limit_reached_message
            
            # 사용자 메시지 추가
            self.add_to_conversation("user", user_input)
            
            # 연령대에 맞는 언어 설정 가져오기
            age_language = self.formatter.get_age_appropriate_language()
            
            # 시스템 메시지에 연령대별 언어 정보 추가
            enhanced_system_message = f"{self.system_message}\n\n추가 정보: 이 아이({self.age_group}세)에게 적합한 어휘와 개념:\n"
            enhanced_system_message += f"어휘: {', '.join(age_language.get('vocabulary', []))}\n"
            enhanced_system_message += f"문장 길이: {age_language.get('sentence_length', '')}\n"
            enhanced_system_message += f"개념: {', '.join(age_language.get('concepts', []))}\n"
            
            # 현재 이야기 단계 추가
            enhanced_system_message += f"\n현재 이야기 수집 단계: {self.collector.get_current_stage()}"
            
            # RAG 시스템에서 관련 정보 검색
            if self.rag_system and self.interests:
                try:
                    # 아이의 관심사 또는 최근 대화 내용 기반 검색
                    search_query = self.interests[0] if self.interests else user_input
                    similar_stories = self.rag_system.get_similar_stories(search_query, self.age_group, n_results=1)
                    
                    if similar_stories:
                        story = similar_stories[0]
                        enhanced_system_message += f"\n\n참고 스토리:\n제목: {story['title']}\n요약: {story['summary']}"
                except Exception as e:
                    logger.error(f"RAG 시스템 검색 중 오류 발생: {e}")
            
            # 이야기 제안 힌트 추가
            next_suggestion = self.suggest_story_element(user_input)
            enhanced_system_message += f"\n\n다음 대화 제안: {next_suggestion}"
            
            # GPT-4o-mini를 사용하여 응답 생성
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": enhanced_system_message},
                    *self.conversation.get_recent_messages(5)  # 최근 5개의 메시지만 사용
                ],
                temperature=0.9,  # 더 창의적이고 다양한 응답을 위해 temperature 증가
                max_tokens=500
            )
            
            # 토큰 사용량 업데이트
            if hasattr(response, 'usage'):
                prompt_tokens = response.usage.prompt_tokens
                completion_tokens = response.usage.completion_tokens
                
                self.conversation.update_token_usage(prompt_tokens, completion_tokens)
            
            # 응답 추출 및 저장
            assistant_response = response.choices[0].message.content
            self.add_to_conversation("assistant", assistant_response)
            
            return assistant_response
            
        except Exception as e:
            logger.error(f"응답 생성 중 오류 발생: {str(e)}")
            return f"미안해 {self.child_name}야. 조금만 후에 다시 이야기할까?"
    
    def suggest_story_theme(self) -> Dict:
        """
        수집된 대화 내용을 바탕으로 이야기 주제 제안
        
        Returns:
            Dict: 이야기 주제 및 줄거리 포맷
        """
        # 아이 정보 및 대화 내역 준비
        story_collection_prompt = self.formatter.format_story_collection_prompt()
        
        # 스토리 분석기를 통해 주제 제안
        story_theme = self.analyzer.suggest_story_theme(
            self.get_conversation_history(),
            self.child_name,
            self.age_group,
            self.interests,
            story_collection_prompt
        )
        
        return story_theme
    
    def get_conversation_summary(self) -> str:
        """
        대화 내용을 요약하는 함수
        
        Returns:
            str: 대화 내용 요약
        """
        # 토큰 제한 확인
        if self.conversation.is_token_limit_reached():
            return self.conversation.token_limit_reached_message
        
        # 스토리 분석기를 통해 대화 요약
        return self.analyzer.get_conversation_summary(
            self.get_conversation_history(),
            self.child_name,
            self.age_group
        )
    
    def save_conversation(self, file_path: str) -> bool:
        """
        대화 내역을 JSON 파일로 저장
        
        Args:
            file_path (str): 저장할 파일 경로
            
        Returns:
            bool: 성공 여부
        """
        try:
            # 추가 데이터 구성
            additional_data = {
                "child_info": {
                    "name": self.child_name,
                    "age": self.age_group,
                    "interests": self.interests
                },
                "story_stage": self.collector.get_current_stage(),
                "story_elements": self.collector.get_story_elements()
            }
            
            # 스토리 개요가 있으면 추가
            story_outline = self.analyzer.get_story_outline()
            if story_outline:
                additional_data["story_outline"] = story_outline
            
            # 대화 관리자를 통해 저장
            return self.conversation.save_conversation(file_path, additional_data)
            
        except Exception as e:
            logger.error(f"대화 내역 저장 실패: {e}")
            return False
    
    def load_conversation(self, file_path: str) -> bool:
        """
        대화 내역 JSON 파일 로드
        
        Args:
            file_path (str): 로드할 파일 경로
            
        Returns:
            bool: 성공 여부
        """
        try:
            # 대화 관리자를 통해 로드
            data = self.conversation.load_conversation(file_path)
            
            if not data:
                return False
            
            # 아이 정보 복원
            child_info = data.get("child_info", {})
            self.update_child_info(
                child_info.get("name"),
                child_info.get("age"),
                child_info.get("interests", [])
            )
            
            # 스토리 수집 상태 복원
            self.collector.update_from_saved_data(data)
            
            # 스토리 개요 복원
            if "story_outline" in data:
                self.analyzer.story_outline = data["story_outline"]
            
            return True
            
        except Exception as e:
            logger.error(f"대화 내역 로드 실패: {e}")
            return False
    
    def get_token_usage(self) -> Dict[str, int]:
        """
        현재 토큰 사용량 반환
        
        Returns:
            Dict[str, int]: 토큰 사용량 정보
        """
        return self.conversation.get_token_usage() 