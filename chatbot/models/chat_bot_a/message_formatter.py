"""
메시지 포맷팅을 담당하는 모듈
"""
from typing import Dict, List, Any, Optional
import random

from shared.utils.logging_utils import get_module_logger
from shared.utils.korean_utils import has_final_consonant, format_with_josa

logger = get_module_logger(__name__)

class MessageFormatter:
    """
    챗봇 메시지의 포맷팅을 담당하는 클래스
    """
    
    def __init__(self, prompts: Dict, child_name: Optional[str] = None, 
                 age_group: Optional[int] = None, interests: List[str] = None,
                 chatbot_name: str = "부기"):
        """
        메시지 포맷터 초기화
        
        Args:
            prompts: 프롬프트 템플릿 모음
            child_name: 아이의 이름
            age_group: 아이의 연령대
            interests: 아이의 관심사 목록
            chatbot_name: 챗봇의 이름
        """
        self.prompts = prompts
        self.child_name = child_name
        self.age_group = age_group
        self.interests = interests or []
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
                        age_group=self.age_group or "5",
                        interests=interests_str,
                        chatbot_name=self.chatbot_name
                    )
                    formatted_items.append(formatted_item)
                except KeyError as e:
                    # 키 오류 시 원본 항목 사용
                    logger.warning(f"시스템 메시지 포맷팅 중 키 오류: {e}")
                    formatted_items.append(item)
            
            # 모든 항목을 결합
            return "\n".join(formatted_items)
        else:
            # 문자열인 경우
            try:
                return system_template.format(
                    child_name=self.child_name or "아이",
                    age_group=self.age_group or "5",
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
            # 기본 인사말
            greeting_templates = [
                "안녕 {child_name}아/야! 난 부기야. 오늘은 우리 재미있는 이야기를 만들어볼까?",
                "반가워 {child_name}아/야! 부기라고 해. 함께 신나는 동화를 만들어보자!",
                "{child_name}아/야, 안녕! 나는 부기야. 너랑 같이 멋진 이야기를 만들고 싶어!"
            ]
        
        # 랜덤 인사말 선택
        greeting = random.choice(greeting_templates)
        
        # 인사말에 조사 패턴이 있는 경우 처리
        try:
            # 아/야 패턴 처리
            if "아/야" in greeting:
                greeting = greeting.replace("{child_name}아/야", format_with_josa(self.child_name or "친구", "아/야"))
            
            # 그 외 포맷팅 처리
            greeting = greeting.format(
                child_name=self.child_name or "친구",
                chatbot_name=self.chatbot_name
            )
        except Exception as e:
            logger.warning(f"인사말 포맷팅 중 오류: {e}")
            # 기본 인사말 반환
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
        
        # 이름과 조사 처리
        if self.child_name:
            has_final = has_final_consonant(self.child_name)
            
            # 아/야 처리
            child_name_with_ya = f"{self.child_name}아" if has_final else f"{self.child_name}야"
            question = question.replace("{name}아/야", child_name_with_ya)
            
            # 이/가 처리
            child_name_with_ga = f"{self.child_name}이" if has_final else f"{self.child_name}가"
            question = question.replace("{name}이/가", child_name_with_ga)
            
            # 은/는 처리
            child_name_with_eun = f"{self.child_name}은" if has_final else f"{self.child_name}는"
            question = question.replace("{name}은/는", child_name_with_eun)
            
            # 을/를 처리
            child_name_with_eul = f"{self.child_name}을" if has_final else f"{self.child_name}를"
            question = question.replace("{name}을/를", child_name_with_eul)
            
            # 과/와 처리
            child_name_with_gwa = f"{self.child_name}과" if has_final else f"{self.child_name}와"
            question = question.replace("{name}과/와", child_name_with_gwa)
            
            # 기본 이름 대체
            question = question.replace("{name}", self.child_name)
        
        return question
    
    def get_follow_up_question(self) -> str:
        """
        랜덤한 후속 질문 반환 및 포맷팅
        
        Returns:
            str: 포맷팅된 후속 질문
        """
        questions = self.prompts.get('follow_up_questions', [])
        question = random.choice(questions) if questions else "더 자세히 이야기해 주세요."
        
        if self.child_name:
            has_final = has_final_consonant(self.child_name)
            # 아/야 처리
            child_name_with_ya = f"{self.child_name}아" if has_final else f"{self.child_name}야"
            question = question.replace("{name}아/야", child_name_with_ya)
            
            # 이/가 처리
            child_name_with_ga = f"{self.child_name}이" if has_final else f"{self.child_name}가"
            question = question.replace("{name}이/가", child_name_with_ga)
            
            # 은/는 처리
            child_name_with_eun = f"{self.child_name}은" if has_final else f"{self.child_name}는"
            question = question.replace("{name}은/는", child_name_with_eun)
            
            # 을/를 처리
            child_name_with_eul = f"{self.child_name}을" if has_final else f"{self.child_name}를"
            question = question.replace("{name}을/를", child_name_with_eul)
            
            # 과/와 처리
            child_name_with_gwa = f"{self.child_name}과" if has_final else f"{self.child_name}와"
            question = question.replace("{name}과/와", child_name_with_gwa)
            
            # 기본 이름 대체
            question = question.replace("{name}", self.child_name)
            
        return question
    
    def get_encouragement(self) -> str:
        """
        랜덤한 격려 문구 반환 및 포맷팅
        
        Returns:
            str: 포맷팅된 격려 문구
        """
        encouragements = self.prompts.get('encouragement_phrases', [])
        encouragement = random.choice(encouragements) if encouragements else "좋아요!"
        
        if self.child_name:
            has_final = has_final_consonant(self.child_name)
            # 아/야 처리
            child_name_with_ya = f"{self.child_name}아" if has_final else f"{self.child_name}야"
            encouragement = encouragement.replace("{name}아/야", child_name_with_ya)
            
            # 이/가 처리
            child_name_with_ga = f"{self.child_name}이" if has_final else f"{self.child_name}가"
            encouragement = encouragement.replace("{name}이/가", child_name_with_ga)
            
            # 은/는 처리
            child_name_with_eun = f"{self.child_name}은" if has_final else f"{self.child_name}는"
            encouragement = encouragement.replace("{name}은/는", child_name_with_eun)
            
            # 을/를 처리
            child_name_with_eul = f"{self.child_name}을" if has_final else f"{self.child_name}를"
            encouragement = encouragement.replace("{name}을/를", child_name_with_eul)
            
            # 과/와 처리
            child_name_with_gwa = f"{self.child_name}과" if has_final else f"{self.child_name}와"
            encouragement = encouragement.replace("{name}과/와", child_name_with_gwa)
            
            # 기본 이름 대체
            encouragement = encouragement.replace("{name}", self.child_name)
            
        return encouragement
    
    def get_age_appropriate_language(self) -> Dict:
        """
        아이의 연령대에 맞는 언어 설정 반환
        
        Returns:
            Dict: 연령대에 맞는 언어 설정
        """
        age_ranges = self.prompts.get('age_appropriate_language', {})
        
        if not self.age_group:
            return {}
        
        if 4 <= self.age_group <= 5:
            return age_ranges.get('4-5', {})
        elif 6 <= self.age_group <= 7:
            return age_ranges.get('6-7', {})
        elif 8 <= self.age_group <= 9:
            return age_ranges.get('8-9', {})
        else:
            return {}
    
    def format_story_collection_prompt(self) -> str:
        """
        스토리 수집 프롬프트 포맷팅
        
        Returns:
            str: 포맷팅된 스토리 수집 프롬프트
        """
        template = self.prompts.get('story_collection_prompt_template', '')
        
        # 관심사 문자열 준비
        interests_str = ", ".join(self.interests) if self.interests else "다양한 주제"
        
        # 이름에 맞는 조사 처리
        formatted_template = template
        if self.child_name:
            has_final = has_final_consonant(self.child_name)
            # 과/와 조사 처리
            child_name_with_gwa = f"{self.child_name}과" if has_final else f"{self.child_name}와"
            formatted_template = formatted_template.replace("{child_name}와/과", child_name_with_gwa)
        
        # 나머지 변수 포맷팅
        return formatted_template.format(
            child_name=self.child_name,
            age_group=self.age_group,
            interests=interests_str
        )
        
    def get_stage_transition_message(self, story_stage: str) -> str:
        """
        단계 전환 시 자연스러운 전환 메시지 생성
        
        Args:
            story_stage: 전환된 이야기 수집 단계
        
        Returns:
            str: 단계 전환 메시지
        """
        stage_messages = {
            "setting": [
                "이제 {name}이/가 얘기해준 친구들이 어디에서 살고 있을지 생각해볼까?",
                "재미있는 친구들이네! 이 친구들이 어떤 곳에서 모험을 하면 좋을까?",
                "{name}아/야, 그 친구들이 사는 세계는 어떤 곳인지 상상해볼래?",
                "멋진 캐릭터들이야! 이제 이 친구들이 어디에서 살고 있는지 이야기해줄래?"
            ],
            "problem": [
                "{name}아/야, 그런 멋진 곳에서 어떤 문제가 생길 수 있을까?",
                "그런 신기한 세계에서 우리 친구들에게 어떤 어려움이 찾아올까?",
                "{name}이/가 만든 세계에서 어떤 모험이 시작될 것 같아?",
                "그 곳에서 주인공이 해결해야 할 어떤 문제가 생길 수 있을까?"
            ],
            "resolution": [
                "{name}아/야, 그런 어려운 문제를 어떻게 해결하면 좋을까?",
                "우리 친구들이 그 문제를 해결하려면 어떻게 해야 할까?",
                "{name}이/가 생각하기에 주인공은 어떻게 그 위기를 극복할 수 있을까?",
                "그런 어려움을 어떻게 이겨낼 수 있을지 {name}의 생각이 궁금해!"
            ]
        }
        
        messages = stage_messages.get(story_stage, ["다음 이야기도 들려줘!"])
        message = random.choice(messages)
        
        # 이름과 조사 처리
        if self.child_name:
            has_final = has_final_consonant(self.child_name)
            # 아/야 처리
            child_name_with_ya = f"{self.child_name}아" if has_final else f"{self.child_name}야"
            message = message.replace("{name}아/야", child_name_with_ya)
            
            # 이/가 처리
            child_name_with_ga = f"{self.child_name}이" if has_final else f"{self.child_name}가"
            message = message.replace("{name}이/가", child_name_with_ga)
            
            # 은/는 처리
            child_name_with_eun = f"{self.child_name}은" if has_final else f"{self.child_name}는"
            message = message.replace("{name}은/는", child_name_with_eun)
            
            # 을/를 처리
            child_name_with_eul = f"{self.child_name}을" if has_final else f"{self.child_name}를"
            message = message.replace("{name}을/를", child_name_with_eul)
            
            # 과/와 처리
            child_name_with_gwa = f"{self.child_name}과" if has_final else f"{self.child_name}와"
            message = message.replace("{name}과/와", child_name_with_gwa)
            
            # 기본 이름 대체
            message = message.replace("{name}", self.child_name)
            
        return message 