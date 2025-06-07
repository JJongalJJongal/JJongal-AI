"""
ChatBot A 통합 메시지 처리기

메시지 포맷팅, 처리, 검증을 담당하는 통합 프로세서
기존 MessageFormatter와 MessageProcessor의 기능을 하나로 통합
"""
from typing import Dict, List, Any, Optional
import random
import sys
import os

from shared.utils.logging_utils import get_module_logger
from shared.utils.korean_utils import has_final_consonant, format_with_josa

# 중앙화된 유틸리티 임포트
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
try:
    from shared.utils.age_group_utils import AgeGroupManager
    from shared.configs.consolidated_prompts import ConsolidatedPrompts
    CENTRALIZED_UTILS_AVAILABLE = True
except ImportError:
    CENTRALIZED_UTILS_AVAILABLE = False

logger = get_module_logger(__name__)

class UnifiedMessageProcessor:
    """
    메시지 포맷팅, 처리, 검증을 담당하는 통합 프로세서
    
    기존의 분산된 메시지 관련 기능들을 하나의 프로세서로 통합하여
    더 효율적이고 일관된 메시지 처리를 제공
    
    주요 기능:
    - 메시지 포맷팅 및 조사 처리
    - 연령대별 언어 적응
    - 시스템 메시지 생성
    - 대화 단계별 질문 생성
    - 메시지 검증 및 필터링
    """
    
    def __init__(self, prompts: Dict = None, child_name: Optional[str] = None, 
                 age_group: Optional[int] = None, interests: List[str] = None,
                 chatbot_name: str = "부기", enhanced_mode: bool = False):
        """
        통합 메시지 프로세서 초기화
        
        Args:
            prompts: 프롬프트 템플릿 모음
            child_name: 아이의 이름
            age_group: 아이의 연령대
            interests: 아이의 관심사 목록
            chatbot_name: 챗봇의 이름
            enhanced_mode: Enhanced 모드 사용 여부
        """
        self.prompts = prompts or {}
        self.child_name = child_name
        self.age_group = age_group
        self.interests = interests or []
        self.chatbot_name = chatbot_name
        self.enhanced_mode = enhanced_mode
        
        # 메시지 처리 통계
        self.processed_messages = 0
        self.format_errors = 0
        self.validation_failures = 0
        
        # 캐시된 설정
        self._cached_language_settings = None
        self._cached_system_message = None
        
        logger.info(f"통합 메시지 프로세서 초기화 완료 - Enhanced: {enhanced_mode}")
    
    # ==========================================
    # 기본 정보 관리
    # ==========================================
    
    def update_child_info(self, child_name: str = None, age_group: int = None, 
                         interests: List[str] = None, chatbot_name: str = None):
        """
        아이 정보 업데이트 및 캐시 무효화
        
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
        
        # 캐시 무효화
        self._cached_language_settings = None
        self._cached_system_message = None
        
        logger.info(f"아이 정보 업데이트: {self.child_name}({self.age_group}세)")
    
    # ==========================================
    # 시스템 메시지 생성
    # ==========================================
    
    def get_system_message(self, message_type: str = "conversation") -> str:
        """
        시스템 메시지 반환 및 포맷팅
        
        Args:
            message_type: 메시지 타입 (conversation, story_collection 등)
            
        Returns:
            str: 포맷팅된 시스템 메시지
        """
        # 캐시된 메시지가 있고 기본 타입이면 반환
        if self._cached_system_message and message_type == "conversation":
            return self._cached_system_message
        
        # 중앙화된 프롬프트 사용 시도
        if CENTRALIZED_UTILS_AVAILABLE:
            try:
                system_message = ConsolidatedPrompts.get_system_message(
                    "chatbot_a", 
                    message_type,
                    child_name=self.child_name or "친구",
                    age_group=self.age_group or 5,
                    chatbot_name=self.chatbot_name,
                    interests=", ".join(self.interests) if self.interests else "다양한 주제"
                )
                
                if system_message:
                    if message_type == "conversation":
                        self._cached_system_message = system_message
                    return system_message
            except Exception as e:
                logger.warning(f"중앙화된 프롬프트 사용 실패: {e}")
        
        # 폴백: 기존 프롬프트 사용
        return self._get_legacy_system_message(message_type)
    
    def _get_legacy_system_message(self, message_type: str) -> str:
        """레거시 시스템 메시지 생성"""
        system_template = self.prompts.get('system_message_template', '')
        
        if not system_template:
            # 기본 시스템 메시지
            return f"""당신은 {self.chatbot_name}라는 이름의 친근한 AI 친구입니다.
{self.age_group or 5}세 아이 {self.child_name or '친구'}와 재미있는 대화를 나누며 
동화 만들기를 도와주는 역할을 합니다."""
        
        # 포맷팅 변수 준비
        interests_str = ", ".join(self.interests) if self.interests else "다양한 주제"
        
        try:
            if isinstance(system_template, list):
                formatted_items = []
                for item in system_template:
                    formatted_item = self._format_template_item(item, interests_str)
                    formatted_items.append(formatted_item)
                return "\n".join(formatted_items)
            else:
                return self._format_template_item(system_template, interests_str)
        except Exception as e:
            logger.error(f"시스템 메시지 포맷팅 오류: {e}")
            self.format_errors += 1
            return system_template
    
    def _format_template_item(self, template: str, interests_str: str) -> str:
        """템플릿 항목 포맷팅"""
        try:
            return template.format(
                child_name=self.child_name or "친구",
                name=self.child_name or "친구",
                age_group=self.age_group or 5,
                age=self.age_group or 5,
                interests=interests_str,
                chatbot_name=self.chatbot_name
            )
        except KeyError as e:
            logger.warning(f"템플릿 키 오류: {e}")
            return template
    
    # ==========================================
    # 인사말 및 기본 메시지
    # ==========================================
    
    def get_greeting(self) -> str:
        """
        연령대에 맞는 인사말 생성
        
        Returns:
            str: 포맷팅된 인사말
        """
        # 중앙화된 프롬프트 사용 시도
        if CENTRALIZED_UTILS_AVAILABLE and self.age_group:
            try:
                from shared.utils.age_group_utils import AgeGroup
                age_group_enum = AgeGroupManager.determine_age_group(self.age_group)
                greeting_templates = ConsolidatedPrompts.get_encouragement(age_group_enum)
                
                if greeting_templates:
                    greeting = random.choice(greeting_templates)
                    return self._apply_korean_formatting(greeting)
            except Exception as e:
                logger.warning(f"중앙화된 인사말 사용 실패: {e}")
        
        # 폴백: 기존 방식
        return self._get_legacy_greeting()
    
    def _get_legacy_greeting(self) -> str:
        """레거시 인사말 생성"""
        greeting_templates = self.prompts.get('greeting_templates', [])
        
        if not greeting_templates:
            greeting_templates = [
                "안녕 {child_name}아/야! 난 {chatbot_name}야. 오늘은 우리 재미있는 이야기를 만들어볼까?",
                "반가워 {child_name}아/야! {chatbot_name}라고 해. 함께 신나는 동화를 만들어보자!",
                "{child_name}아/야, 안녕! 나는 {chatbot_name}야. 너랑 같이 멋진 이야기를 만들고 싶어!"
            ]
        
        greeting = random.choice(greeting_templates)
        return self._apply_korean_formatting(greeting)
    
    # ==========================================
    # 이야기 수집 관련 메시지
    # ==========================================
    
    def get_story_prompting_question(self, story_stage: str) -> str:
        """
        현재 이야기 수집 단계에 맞는 질문 반환
        
        Args:
            story_stage: 현재 이야기 수집 단계
            
        Returns:
            str: 이야기 수집 단계에 맞는 질문
        """
        # 중앙화된 프롬프트 사용 시도
        if CENTRALIZED_UTILS_AVAILABLE and self.age_group:
            try:
                age_group_enum = AgeGroupManager.determine_age_group(self.age_group)
                questions = ConsolidatedPrompts.get_story_collection_prompt(story_stage, age_group_enum)
                
                if questions:
                    question = random.choice(questions)
                    return self._apply_korean_formatting(question)
            except Exception as e:
                logger.warning(f"중앙화된 질문 사용 실패: {e}")
        
        # 폴백: 기존 방식
        return self._get_legacy_story_question(story_stage)
    
    def _get_legacy_story_question(self, story_stage: str) -> str:
        """레거시 이야기 질문 생성"""
        story_questions = self.prompts.get('story_prompting_questions', {})
        current_stage_questions = story_questions.get(story_stage, [])
        
        if not current_stage_questions:
            return self.get_follow_up_question()
        
        question = random.choice(current_stage_questions)
        return self._apply_korean_formatting(question)
    
    def get_follow_up_question(self) -> str:
        """
        후속 질문 생성
        
        Returns:
            str: 포맷팅된 후속 질문
        """
        # 중앙화된 프롬프트 사용 시도
        if CENTRALIZED_UTILS_AVAILABLE and self.age_group:
            try:
                age_group_enum = AgeGroupManager.determine_age_group(self.age_group)
                questions = ConsolidatedPrompts.get_follow_up_questions(age_group_enum)
                
                if questions:
                    question = random.choice(questions)
                    return self._apply_korean_formatting(question)
            except Exception as e:
                logger.warning(f"중앙화된 후속 질문 사용 실패: {e}")
        
        # 폴백: 기존 방식
        questions = self.prompts.get('follow_up_questions', [
            "더 자세히 이야기해 주세요.",
            "그래서 어떻게 됐을까요?",
            "정말 재미있네요! 계속 들려주세요."
        ])
        
        question = random.choice(questions)
        return self._apply_korean_formatting(question)
    
    def get_encouragement(self) -> str:
        """
        격려 문구 생성
        
        Returns:
            str: 포맷팅된 격려 문구
        """
        # 중앙화된 프롬프트 사용 시도
        if CENTRALIZED_UTILS_AVAILABLE and self.age_group:
            try:
                age_group_enum = AgeGroupManager.determine_age_group(self.age_group)
                encouragements = ConsolidatedPrompts.get_encouragement(age_group_enum)
                
                if encouragements:
                    encouragement = random.choice(encouragements)
                    return self._apply_korean_formatting(encouragement)
            except Exception as e:
                logger.warning(f"중앙화된 격려 문구 사용 실패: {e}")
        
        # 폴백: 기존 방식
        encouragements = self.prompts.get('encouragement_phrases', [
            "와! 정말 좋은 생각이야!",
            "멋진 아이디어네!",
            "상상력이 정말 대단해!",
            "더 이야기해줘!"
        ])
        
        encouragement = random.choice(encouragements)
        return self._apply_korean_formatting(encouragement)
    
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
        return self._apply_korean_formatting(message)
    
    # ==========================================
    # 한국어 포맷팅 및 조사 처리
    # ==========================================
    
    def _apply_korean_formatting(self, text: str) -> str:
        """
        한국어 조사 처리 및 포맷팅 적용
        
        Args:
            text: 원본 텍스트
            
        Returns:
            str: 포맷팅된 텍스트
        """
        if not self.child_name:
            # 이름이 없으면 기본 포맷팅만
            return text.format(
                name="친구",
                child_name="친구",
                age=self.age_group or 5,
                age_group=self.age_group or 5,
                chatbot_name=self.chatbot_name
            )
        
        try:
            # 조사 패턴 처리
            formatted_text = text
            
            # 아/야 패턴
            if "아/야" in formatted_text:
                formatted_text = formatted_text.replace(
                    "{name}아/야", 
                    format_with_josa(self.child_name, "아/야")
                )
                formatted_text = formatted_text.replace(
                    "{child_name}아/야", 
                    format_with_josa(self.child_name, "아/야")
                )
            
            # 이/가 패턴
            if "이/가" in formatted_text:
                formatted_text = formatted_text.replace(
                    "{name}이/가", 
                    format_with_josa(self.child_name, "이/가")
                )
                formatted_text = formatted_text.replace(
                    "{child_name}이/가", 
                    format_with_josa(self.child_name, "이/가")
                )
            
            # 은/는 패턴
            if "은/는" in formatted_text:
                formatted_text = formatted_text.replace(
                    "{name}은/는", 
                    format_with_josa(self.child_name, "은/는")
                )
                formatted_text = formatted_text.replace(
                    "{child_name}은/는", 
                    format_with_josa(self.child_name, "은/는")
                )
            
            # 을/를 패턴
            if "을/를" in formatted_text:
                formatted_text = formatted_text.replace(
                    "{name}을/를", 
                    format_with_josa(self.child_name, "을/를")
                )
                formatted_text = formatted_text.replace(
                    "{child_name}을/를", 
                    format_with_josa(self.child_name, "을/를")
                )
            
            # 과/와 패턴
            if "과/와" in formatted_text:
                formatted_text = formatted_text.replace(
                    "{name}과/와", 
                    format_with_josa(self.child_name, "과/와")
                )
                formatted_text = formatted_text.replace(
                    "{child_name}과/와", 
                    format_with_josa(self.child_name, "과/와")
                )
            
            # 기본 변수 포맷팅
            formatted_text = formatted_text.format(
                name=self.child_name,
                child_name=self.child_name,
                age=self.age_group or 5,
                age_group=self.age_group or 5,
                chatbot_name=self.chatbot_name
            )
            
            self.processed_messages += 1
            return formatted_text
            
        except Exception as e:
            logger.error(f"한국어 포맷팅 오류: {e}")
            self.format_errors += 1
            # 오류 시 기본 포맷팅
            return text.format(
                name=self.child_name,
                child_name=self.child_name,
                age=self.age_group or 5,
                age_group=self.age_group or 5,
                chatbot_name=self.chatbot_name
            )
    
    # ==========================================
    # 연령대별 언어 설정
    # ==========================================
    
    def get_age_appropriate_language(self) -> Dict[str, Any]:
        """
        아이의 연령대에 맞는 언어 설정 반환
        
        Returns:
            Dict: 연령대에 맞는 언어 설정
        """
        if not self.age_group:
            return {}
        
        # 캐시된 설정이 있으면 반환
        if self._cached_language_settings:
            return self._cached_language_settings
        
        # 중앙화된 유틸리티 사용 시도
        if CENTRALIZED_UTILS_AVAILABLE:
            try:
                settings = AgeGroupManager.get_language_settings(self.age_group)
                self._cached_language_settings = settings
                return settings
            except Exception as e:
                logger.warning(f"중앙화된 언어 설정 사용 실패: {e}")
        
        # 폴백: 기존 로직
        return self._get_legacy_language_settings()
    
    def _get_legacy_language_settings(self) -> Dict[str, Any]:
        """레거시 언어 설정"""
        age_ranges = self.prompts.get('age_appropriate_language', {})
        
        if 4 <= self.age_group <= 7:
            settings = age_ranges.get('4-7', age_ranges.get('4-5', {}))
        elif 8 <= self.age_group <= 9:
            settings = age_ranges.get('8-9', {})
        else:
            settings = {}
        
        # 기본값 설정
        default_settings = {
            "vocabulary": "basic",
            "sentence_length": "short",
            "use_repetition": True,
            "use_sound_words": True,
            "emotional_complexity": "basic"
        }
        
        # 기본값과 병합
        result = {**default_settings, **settings}
        self._cached_language_settings = result
        return result
    
    # ==========================================
    # 메시지 검증 및 필터링
    # ==========================================
    
    def validate_message(self, message: str) -> Dict[str, Any]:
        """
        메시지 검증 및 품질 평가
        
        Args:
            message: 검증할 메시지
            
        Returns:
            Dict: 검증 결과
        """
        validation_result = {
            "is_valid": True,
            "issues": [],
            "suggestions": [],
            "age_appropriate": True,
            "length_appropriate": True,
            "complexity_score": 0.5
        }
        
        try:
            # 길이 검증
            if len(message) > 500:
                validation_result["length_appropriate"] = False
                validation_result["issues"].append("메시지가 너무 깁니다")
                validation_result["suggestions"].append("더 짧게 나누어 전달하세요")
            
            # 연령 적합성 검증
            if self.age_group and CENTRALIZED_UTILS_AVAILABLE:
                try:
                    appropriateness = AgeGroupManager.validate_age_appropriateness(
                        self.age_group, 
                        {"content": message, "vocabulary_level": "auto_detect"}
                    )
                    
                    validation_result["age_appropriate"] = appropriateness["is_appropriate"]
                    validation_result["issues"].extend(appropriateness.get("issues", []))
                    validation_result["suggestions"].extend(appropriateness.get("suggestions", []))
                    
                except Exception as e:
                    logger.warning(f"연령 적합성 검증 실패: {e}")
            
            # 복잡도 점수 계산 (간단한 휴리스틱)
            word_count = len(message.split())
            sentence_count = message.count('.') + message.count('!') + message.count('?') + 1
            avg_sentence_length = word_count / sentence_count if sentence_count > 0 else word_count
            
            if avg_sentence_length <= 5:
                validation_result["complexity_score"] = 0.2
            elif avg_sentence_length <= 10:
                validation_result["complexity_score"] = 0.5
            else:
                validation_result["complexity_score"] = 0.8
            
            # 전체 유효성 결정
            validation_result["is_valid"] = (
                validation_result["age_appropriate"] and 
                validation_result["length_appropriate"] and
                len(validation_result["issues"]) == 0
            )
            
            if not validation_result["is_valid"]:
                self.validation_failures += 1
            
        except Exception as e:
            logger.error(f"메시지 검증 중 오류: {e}")
            validation_result["is_valid"] = False
            validation_result["issues"].append(f"검증 오류: {str(e)}")
        
        return validation_result
    
    # ==========================================
    # 통계 및 상태 관리
    # ==========================================
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """처리 통계 반환"""
        total_processed = self.processed_messages
        error_rate = (self.format_errors / total_processed) if total_processed > 0 else 0
        validation_failure_rate = (self.validation_failures / total_processed) if total_processed > 0 else 0
        
        return {
            "total_processed": total_processed,
            "format_errors": self.format_errors,
            "validation_failures": self.validation_failures,
            "error_rate": round(error_rate, 3),
            "validation_failure_rate": round(validation_failure_rate, 3),
            "child_info": {
                "name": self.child_name,
                "age": self.age_group,
                "interests_count": len(self.interests)
            },
            "cache_status": {
                "language_settings_cached": self._cached_language_settings is not None,
                "system_message_cached": self._cached_system_message is not None
            }
        }
    
    def reset_cache(self):
        """캐시 초기화"""
        self._cached_language_settings = None
        self._cached_system_message = None
        logger.info("메시지 프로세서 캐시 초기화")
    
    def reset_stats(self):
        """통계 초기화"""
        self.processed_messages = 0
        self.format_errors = 0
        self.validation_failures = 0
        logger.info("메시지 프로세서 통계 초기화")
    
    # ==========================================
    # Enhanced 모드 지원 메서드들
    # ==========================================
    
    def get_enhanced_greeting(self, age: int) -> str:
        """Enhanced 연령별 특화 인사말"""
        age_specific_greetings = {
            4: f"안녕 {self.child_name or '친구'}야! 나는 {self.chatbot_name}야! 오늘 뭐 하고 놀까?",
            5: f"안녕 {self.child_name or '친구'}야! 나는 {self.chatbot_name}야. 재미있는 얘기 같이 만들어볼까?",
            6: f"안녕 {self.child_name or '친구'}야! {self.chatbot_name}라고 해! 신나는 모험 이야기 만들어보자!",
            7: f"안녕 {self.child_name or '친구'}야! 나는 {self.chatbot_name}야. 멋진 이야기를 함께 만들어보자!",
            8: f"안녕 {self.child_name or '친구'}야! {self.chatbot_name}야. 너만의 특별한 이야기를 만들어보자!",
            9: f"안녕 {self.child_name or '친구'}야! 나는 {self.chatbot_name}야. 창의적인 이야기를 함께 만들어볼까?"
        }
        
        greeting = age_specific_greetings.get(age, age_specific_greetings[5])
        
        # 관심사가 있으면 추가
        if self.interests:
            interest_str = ", ".join(self.interests[:2])
            greeting += f" {interest_str} 좋아한다고 들었어!"
        
        return greeting
    
    def get_token_limit_message(self) -> str:
        """토큰 제한 메시지"""
        return f"안녕, {self.child_name or '친구'}! 우리 대화가 너무 길어져서 여기서 마무리해야겠어. 오늘 정말 재미있는 이야기를 들려줘서 고마워!"
    
    def validate_response(self, response: str) -> Dict[str, Any]:
        """응답 검증"""
        validation_result = {
            "is_valid": True,
            "age_appropriate": True,
            "encouraging": True,
            "clear": True,
            "issues": []
        }
        
        if not response or len(response.strip()) < 5:
            validation_result["is_valid"] = False
            validation_result["issues"].append("응답이 너무 짧음")
        
        # 연령 적절성 검사
        if self.age_group:
            inappropriate_words = ["어려운", "복잡한", "무서운", "슬픈"]
            if any(word in response for word in inappropriate_words):
                validation_result["age_appropriate"] = False
                validation_result["issues"].append("연령에 부적절한 표현")
        
        # 격려성 검사
        encouraging_words = ["좋아", "멋져", "훌륭", "재미있", "대단", "와"]
        if not any(word in response for word in encouraging_words):
            validation_result["encouraging"] = False
            validation_result["issues"].append("격려적 표현 부족")
        
        # 명확성 검사
        if len(response) > 200:
            validation_result["clear"] = False
            validation_result["issues"].append("응답이 너무 김")
        
        self.validation_failures += len(validation_result["issues"])
        return validation_result
    
    def get_error_message(self) -> str:
        """에러 메시지"""
        error_messages = [
            f"앗, 잠깐만! {self.chatbot_name}가 생각을 정리하고 있어요.",
            "조금 기다려줄래요? 더 재미있는 이야기를 준비하고 있거든요!",
            "어라? 뭔가 꼬였네요. 다시 말해줄 수 있어요?"
        ]
        return random.choice(error_messages)
    
    def adjust_for_age_group(self, response: str, age_group: int) -> str:
        """연령대별 응답 조정"""
        if age_group <= 5:
            # 더 단순하고 친근하게
            response = response.replace("합니다", "해요")
            response = response.replace("입니다", "예요")
            response = response.replace("그렇습니다", "그래요")
        elif age_group >= 8:
            # 조금 더 정중하게
            response = response.replace("해", "해요")
            response = response.replace("야", "예요")
        
        return response
    
    def make_age_appropriate(self, response: str) -> str:
        """연령 적절하게 만들기"""
        # 어려운 단어 대체
        replacements = {
            "복잡한": "재미있는",
            "어려운": "신나는",
            "무서운": "스릴 넘치는",
            "슬픈": "조금 아쉬운",
            "문제": "도전",
            "실패": "다른 시도"
        }
        
        for old, new in replacements.items():
            response = response.replace(old, new)
        
        return response
    
    def add_encouragement(self, response: str) -> str:
        """격려 표현 추가"""
        encouragements = ["정말 좋은 아이디어네요!", "와, 멋져요!", "그거 재미있겠다!", "대단해요!"]
        
        if not any(enc in response for enc in encouragements):
            encouragement = random.choice(encouragements)
            response = f"{encouragement} {response}"
        
        return response
    
    def clarify_message(self, response: str) -> str:
        """메시지 명확화"""
        # 너무 긴 문장 나누기
        if len(response) > 150:
            sentences = response.split('.')
            if len(sentences) > 2:
                response = '. '.join(sentences[:2]) + '.'
        
        # 명확한 표현으로 변경
        clarifications = {
            "그것": "그 아이디어",
            "이것": "이 생각",
            "저것": "저 이야기"
        }
        
        for old, new in clarifications.items():
            response = response.replace(old, new)
        
        return response

    # ==========================================
    # 기존 유틸리티 및 상태 관리 메서드들
    # ========================================== 