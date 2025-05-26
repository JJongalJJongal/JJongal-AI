"""
부기 (ChatBot A) 메시지 처리기

메시지 포맷팅, 한국어 조사 처리, 연령대별 언어 적응을 담당하는 프로세서
기존 MessageFormatter의 기능을 BaseProcessor를 상속받아 재구성
"""
from typing import Dict, List, Any, Optional
import random

from .base_processor import BaseProcessor
from shared.utils.logging_utils import get_module_logger
from shared.utils.korean_utils import has_final_consonant, format_with_josa

logger = get_module_logger(__name__)

class MessageProcessor(BaseProcessor):
    """
    메시지 포맷팅 및 한국어 처리를 담당하는 프로세서
    
    BaseProcessor를 상속받아 메시지 포맷팅 기능을 제공
    """
    
    def __init__(self, prompts: Dict, child_name: Optional[str] = None, 
                 age_group: Optional[int] = None, interests: List[str] = None,
                 chatbot_name: str = "부기"):
        """
        메시지 프로세서 초기화
        
        Args:
            prompts: 프롬프트 템플릿 모음
            child_name: 아이의 이름
            age_group: 아이의 연령대
            interests: 아이의 관심사 목록
            chatbot_name: 챗봇의 이름
        """
        super().__init__()
        self.prompts = prompts
        self.child_name = child_name
        self.age_group = age_group
        self.interests = interests or []
        self.chatbot_name = chatbot_name
        
    def initialize(self) -> bool:
        """
        프로세서 초기화
        
        Returns:
            bool: 초기화 성공 여부
        """
        try:
            # 필수 프롬프트 템플릿 확인
            required_templates = ['system_message_template', 'greeting_templates']
            for template in required_templates:
                if template not in self.prompts:
                    logger.warning(f"필수 템플릿 누락: {template}")
            
            self.is_initialized = True
            logger.info("MessageProcessor 초기화 완료")
            return True
            
        except Exception as e:
            logger.error(f"MessageProcessor 초기화 실패: {e}")
            return False
    
    def process(self, input_data: Any) -> Any:
        """
        메시지 처리 (포맷팅)
        
        Args:
            input_data: 처리할 메시지 데이터
            
        Returns:
            Any: 포맷팅된 메시지
        """
        if not self.validate_input(input_data):
            return None
            
        if isinstance(input_data, dict):
            message_type = input_data.get('type')
            
            if message_type == 'greeting':
                return self.get_greeting()
            elif message_type == 'system_message':
                return self.get_system_message()
            elif message_type == 'story_question':
                stage = input_data.get('stage', 'character')
                return self.get_story_prompting_question(stage)
            elif message_type == 'follow_up':
                return self.get_follow_up_question()
            elif message_type == 'encouragement':
                return self.get_encouragement()
            elif message_type == 'stage_transition':
                stage = input_data.get('stage', 'character')
                return self.get_stage_transition_message(stage)
        
        return str(input_data)
    
    def update_child_info(self, child_name: str = None, age_group: int = None, 
                         interests: List[str] = None, chatbot_name: str = None):
        """
        아이 정보 업데이트
        
        Args:
            child_name: 아이의 이름
            age_group: 아이의 연령대
            interests: 아이의 관심사 목록
            chatbot_name: 챗봇의 이름
        """
        if child_name is not None:
            self.child_name = child_name
        if age_group is not None:
            self.age_group = age_group
        if interests is not None:
            self.interests = interests
        if chatbot_name is not None:
            self.chatbot_name = chatbot_name
    
    def get_system_message(self) -> str:
        """
        시스템 메시지 반환 및 포맷팅
        
        Returns:
            str: 포맷팅된 시스템 메시지
        """
        system_template = self.prompts.get('system_message_template', '')
        
        # 포맷팅을 위한 변수 준비
        interests_str = ", ".join(self.interests) if self.interests else "다양한 주제"
        
        # system_template이 리스트인 경우 처리
        if isinstance(system_template, list):
            formatted_items = []
            for item in system_template:
                try:
                    # 각 항목에 포맷팅 시도
                    formatted_item = item.format(
                        child_name=self.child_name or "아이",
                        name=self.child_name or "아이",
                        age_group=self.age_group or "5",
                        age=self.age_group or "5",
                        interests=interests_str,
                        chatbot_name=self.chatbot_name
                    )
                    formatted_items.append(formatted_item)
                except KeyError as e:
                    logger.warning(f"시스템 메시지 포맷팅 중 키 오류: {e}")
                    formatted_items.append(item)
            
            return "\n".join(formatted_items)
        else:
            try:
                return system_template.format(
                    child_name=self.child_name or "아이",
                    name=self.child_name or "아이",
                    age_group=self.age_group or "5",
                    age=self.age_group or "5",
                    interests=interests_str,
                    chatbot_name=self.chatbot_name
                )
            except KeyError as e:
                logger.warning(f"시스템 메시지 포맷팅 중 키 오류: {e}")
                return system_template
    
    def get_greeting(self) -> str:
        """
        인사말 반환
        
        Returns:
            str: 포맷팅된 인사말
        """
        greeting_templates = self.prompts.get('greeting_templates', [])
        
        if not greeting_templates:
            greeting_templates = [
                "안녕 {child_name}아/야! 난 부기야. 오늘은 우리 재미있는 이야기를 만들어볼까?",
                "반가워 {child_name}아/야! 부기라고 해. 함께 신나는 동화를 만들어보자!",
                "{child_name}아/야, 안녕! 나는 부기야. 너랑 같이 멋진 이야기를 만들고 싶어!"
            ]
        
        greeting = random.choice(greeting_templates)
        
        try:
            # 아/야 패턴 처리
            if "아/야" in greeting:
                greeting = greeting.replace("{child_name}아/야", format_with_josa(self.child_name or "친구", "아/야"))
            
            greeting = greeting.format(
                child_name=self.child_name or "친구",
                name=self.child_name or "친구",
                age=self.age_group or 5,
                chatbot_name=self.chatbot_name
            )
        except Exception as e:
            logger.warning(f"인사말 포맷팅 중 오류: {e}")
            greeting = f"안녕! 난 {self.chatbot_name}야. 재미있는 이야기를 함께 만들어보자!"
            
        return greeting
    
    def get_story_prompting_question(self, story_stage: str) -> str:
        """
        현재 이야기 수집 단계에 맞는 질문 반환
        
        Args:
            story_stage: 현재 이야기 수집 단계
            
        Returns:
            str: 이야기 수집 단계에 맞는 질문
        """
        story_questions = self.prompts.get('story_prompting_questions', {})
        current_stage_questions = story_questions.get(story_stage, [])
        
        if not current_stage_questions:
            return self.get_follow_up_question()
        
        question = random.choice(current_stage_questions)
        return self._format_korean_message(question)
    
    def get_follow_up_question(self) -> str:
        """
        랜덤한 후속 질문 반환 및 포맷팅
        
        Returns:
            str: 포맷팅된 후속 질문
        """
        questions = self.prompts.get('follow_up_questions', [])
        question = random.choice(questions) if questions else "더 자세히 이야기해 주세요."
        return self._format_korean_message(question)
    
    def get_encouragement(self) -> str:
        """
        격려 메시지 반환 및 포맷팅
        
        Returns:
            str: 포맷팅된 격려 메시지
        """
        encouragements = self.prompts.get('encouragement_messages', [])
        
        if not encouragements:
            encouragements = [
                "와! {name}아/야 정말 재미있는 생각이야!",
                "좋아! {name}이/가 말한 게 너무 멋져!",
                "우와! {name}은/는 정말 상상력이 풍부하구나!"
            ]
        
        encouragement = random.choice(encouragements)
        return self._format_korean_message(encouragement)
    
    def get_stage_transition_message(self, story_stage: str) -> str:
        """
        단계 전환 메시지 반환
        
        Args:
            story_stage: 전환할 이야기 단계
            
        Returns:
            str: 단계 전환 메시지
        """
        transition_messages = self.prompts.get('stage_transition_messages', {})
        stage_messages = transition_messages.get(story_stage, [])
        
        if not stage_messages:
            default_messages = {
                "setting": ["이제 이야기가 어디서 일어나는지 생각해보자!"],
                "problem": ["그럼 이제 어떤 문제가 생겼는지 이야기해볼까?"],
                "resolution": ["마지막으로 어떻게 문제를 해결할지 생각해보자!"]
            }
            stage_messages = default_messages.get(story_stage, ["다음 단계로 넘어가자!"])
        
        message = random.choice(stage_messages)
        return self._format_korean_message(message)
    
    def get_age_appropriate_language(self) -> Dict:
        """
        연령대에 맞는 언어 설정 반환
        
        Returns:
            Dict: 연령대별 언어 설정
        """
        age_language = self.prompts.get('age_appropriate_language', {})
        age_key = f"age_{self.age_group}" if self.age_group else "age_5"
        
        return age_language.get(age_key, {
            "vocabulary": ["간단한", "재미있는", "신나는"],
            "sentence_length": "짧고 간단한 문장",
            "concepts": ["기본적인 감정", "친숙한 동물", "일상적인 상황"]
        })
    
    def format_story_collection_prompt(self) -> str:
        """
        이야기 수집 프롬프트 포맷팅
        
        Returns:
            str: 포맷팅된 이야기 수집 프롬프트
        """
        prompt_template = self.prompts.get('story_collection_prompt', '')
        interests_str = ", ".join(self.interests) if self.interests else "다양한 주제"
        
        try:
            return prompt_template.format(
                child_name=self.child_name or "아이",
                name=self.child_name or "아이",
                age_group=self.age_group or 5,
                age=self.age_group or 5,
                interests=interests_str,
                chatbot_name=self.chatbot_name
            )
        except KeyError as e:
            logger.warning(f"이야기 수집 프롬프트 포맷팅 중 키 오류: {e}")
            return prompt_template
    
    def _format_korean_message(self, message: str) -> str:
        """
        한국어 조사 처리를 포함한 메시지 포맷팅
        
        Args:
            message: 포맷팅할 메시지
            
        Returns:
            str: 포맷팅된 메시지
        """
        if not self.child_name:
            return message.replace("{name}", "친구")
        
        has_final = has_final_consonant(self.child_name)
        
        # 각종 조사 처리
        replacements = {
            "{name}아/야": f"{self.child_name}아" if has_final else f"{self.child_name}야",
            "{name}이/가": f"{self.child_name}이" if has_final else f"{self.child_name}가",
            "{name}은/는": f"{self.child_name}은" if has_final else f"{self.child_name}는",
            "{name}을/를": f"{self.child_name}을" if has_final else f"{self.child_name}를",
            "{name}과/와": f"{self.child_name}과" if has_final else f"{self.child_name}와",
            "{name}": self.child_name
        }
        
        for pattern, replacement in replacements.items():
            message = message.replace(pattern, replacement)
        
        return message 