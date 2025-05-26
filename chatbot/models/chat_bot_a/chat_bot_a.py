"""
아이들과 대화하며 동화 줄거리를 수집하는 AI 챗봇 (부기)
"""
from typing import List, Dict, Any
import warnings

# 새로운 통합 엔진들
from .core.story_engine import StoryEngine
from .processors.unified_message_processor import UnifiedMessageProcessor
from .conversation_manager import ConversationManager

# 공통 유틸리티
from shared.utils.logging_utils import get_module_logger
from shared.utils.openai_utils import initialize_client
from shared.configs.prompts_config import load_chatbot_a_prompts

# 레거시 호환성 
try:
    from ..rag_system import RAGSystem
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False

logger = get_module_logger(__name__)

class ChatBotA:
    """
    ChatBot A - 통합 엔진
    
    주요 기능:
    - 통합된 이야기 엔진 (수집 + 분석)
    - 통합된 메시지 프로세서 (포맷팅 + 처리)
    - 향상된 성능 및 유지보수성
    - legacy 호환성 유지 (별칭 제공)
    """
    
    def __init__(self, token_limit: int = 10000, enable_rag: bool = True, 
                 legacy_compatibility: bool = True):
        """
        ChatBot A 초기화
        
        Args:
            token_limit: 대화 토큰 제한
            enable_rag: RAG 시스템 사용 여부
            legacy_compatibility: 레거시 호환성 유지 여부
        """
        # 기본 설정
        self.token_limit = token_limit
        self.enable_rag = enable_rag
        self.legacy_compatibility = legacy_compatibility
        
        # 아이 정보
        self.child_name = None
        self.age_group = None
        self.interests = []
        self.chatbot_name = "부기"
        
        # 프롬프트 로드
        self.prompts = load_chatbot_a_prompts()
        
        # OpenAI 클라이언트 초기화
        try:
            self.openai_client = initialize_client()
        except Exception as e:
            logger.error(f"OpenAI 클라이언트 초기화 실패: {e}")
            self.openai_client = None
        
        # RAG 시스템 초기화
        self.rag_system = None
        if enable_rag and RAG_AVAILABLE:
            try:
                self.rag_system = RAGSystem()
                logger.info("RAG 시스템 초기화 완료")
            except Exception as e:
                logger.warning(f"RAG 시스템 초기화 실패: {e}")
        
        # 핵심 엔진들 초기화
        self._initialize_engines()
        
        # 레거시 호환성 설정
        if legacy_compatibility:
            self._setup_legacy_compatibility()
        
        logger.info("ChatBot A (리팩토링 버전) 초기화 완료")
    
    def _initialize_engines(self):
        """핵심 엔진들 초기화"""
        # 대화 관리자
        self.conversation = ConversationManager(self.token_limit)
        
        # 통합 이야기 엔진
        self.story_engine = StoryEngine(
            openai_client=self.openai_client,
            rag_system=self.rag_system,
            conversation_manager=self.conversation
        )
        
        # 통합 메시지 프로세서
        self.message_processor = UnifiedMessageProcessor(
            prompts=self.prompts,
            child_name=self.child_name,
            age_group=self.age_group,
            interests=self.interests,
            chatbot_name=self.chatbot_name
        )
        
        logger.info("통합 엔진들 초기화 완료")
    
    def _setup_legacy_compatibility(self):
        """레거시 호환성을 위한 별칭 설정"""
        # 기존 API와의 호환성을 위한 별칭들
        self.collector = self.story_engine  # StoryCollector 호환
        self.analyzer = self.story_engine   # StoryAnalyzer 호환
        self.formatter = self.message_processor  # MessageFormatter 호환
        
        logger.info("레거시 호환성 설정 완료")
    
    # ==========================================
    # 기본 정보 관리
    # ==========================================
    
    def update_child_info(self, child_name: str = None, age: int = None, 
                         interests: List[str] = None, chatbot_name: str = None):
        """
        아이 정보 업데이트
        
        Args:
            child_name: 아이의 이름
            age: 아이의 나이
            interests: 아이의 관심사 목록
            chatbot_name: 챗봇의 이름
        """
        # 정보 업데이트
        if child_name is not None:
            self.child_name = child_name
        if age is not None:
            self.age_group = age
        if interests is not None:
            self.interests = interests
        if chatbot_name is not None:
            self.chatbot_name = chatbot_name
        
        # 엔진들에 정보 전파
        self.message_processor.update_child_info(
            child_name=self.child_name,
            age_group=self.age_group,
            interests=self.interests,
            chatbot_name=self.chatbot_name
        )
        
        logger.info(f"아이 정보 업데이트: {self.child_name}({self.age_group}세)")
    
    # ==========================================
    # 대화 관리
    # ==========================================
    
    def initialize_chat(self, child_name: str, age: int, interests: List[str] = None, 
                       chatbot_name: str = "부기") -> str:
        """
        챗봇과의 대화 초기화
        
        Args:
            child_name: 아이의 이름
            age: 아이의 나이
            interests: 아이의 관심사 목록
            chatbot_name: 챗봇의 이름
            
        Returns:
            str: 초기 인사 메시지
        """
        # 입력 검증
        if not child_name or not isinstance(child_name, str):
            raise ValueError("아이의 이름은 필수고, 문자열이어야 합니다.")
        if not isinstance(age, int) or age < 4 or age > 9:
            raise ValueError("아이의 나이는 4-9세 사이 정수이어야 합니다.")
        
        # 아이 정보 업데이트
        self.update_child_info(child_name, age, interests, chatbot_name)
        
        # 인사말 생성
        greeting = self.message_processor.get_greeting()
        
        # 대화 시작
        self.add_to_conversation("assistant", greeting)
        
        return greeting
    
    def get_response(self, user_input: str) -> str:
        """
        사용자 입력에 대한 응답 생성
        
        Args:
            user_input: 사용자 입력
            
        Returns:
            str: 챗봇 응답
        """
        try:
            # 토큰 제한 확인
            if self.conversation.is_token_limit_reached():
                return self.conversation.token_limit_reached_message
            
            # 사용자 메시지 추가
            self.add_to_conversation("user", user_input)
            
            # 사용자 응답 분석
            analysis_result = self.story_engine.analyze_user_response(
                user_input, self.openai_client
            )
            
            # 응답 생성 전략 결정
            response = self._generate_contextual_response(user_input, analysis_result)
            
            # 단계 전환 확인
            conversation_length = len(self.get_conversation_history())
            if self.story_engine.should_transition_to_next_stage(conversation_length):
                self.story_engine.transition_to_next_stage(conversation_length)
                stage_message = self.message_processor.get_stage_transition_message(
                    self.story_engine.get_current_stage()
                )
                response += f"\n\n{stage_message}"
            
            # 응답 검증
            validation = self.message_processor.validate_message(response)
            if not validation["is_valid"]:
                logger.warning(f"응답 검증 실패: {validation['issues']}")
                # 필요시 응답 수정
                response = self._fix_response_issues(response, validation)
            
            # 응답 추가
            self.add_to_conversation("assistant", response)
            
            return response
            
        except Exception as e:
            logger.error(f"응답 생성 중 오류: {e}")
            return f"미안해 {self.child_name or '친구'}야. 조금만 후에 다시 이야기할까?"
    
    def _generate_contextual_response(self, user_input: str, analysis_result: Dict) -> str:
        """
        컨텍스트를 고려한 응답 생성
        
        Args:
            user_input: 사용자 입력
            analysis_result: 분석 결과
            
        Returns:
            str: 생성된 응답
        """
        # 품질 점수에 따른 응답 전략
        quality_score = analysis_result.get("quality_score", 0.5)
        
        if quality_score > 0.7:
            # 고품질 응답 - 격려 + 후속 질문
            encouragement = self.message_processor.get_encouragement()
            follow_up = self.message_processor.get_follow_up_question()
            return f"{encouragement} {follow_up}"
        
        elif quality_score > 0.3:
            # 중간 품질 - 단계별 질문
            return self.message_processor.get_story_prompting_question(
                self.story_engine.get_current_stage()
            )
        
        else:
            # 저품질 - 격려 위주
            return self.message_processor.get_encouragement()
    
    def _fix_response_issues(self, response: str, validation: Dict) -> str:
        """응답 문제 수정"""
        fixed_response = response
        
        # 길이 문제 해결
        if not validation["length_appropriate"]:
            # 응답을 절반으로 줄임
            sentences = fixed_response.split('.')
            if len(sentences) > 1:
                fixed_response = '.'.join(sentences[:len(sentences)//2]) + '.'
        
        return fixed_response
    
    # ==========================================
    # 이야기 관련 기능
    # ==========================================
    
    def suggest_story_theme(self) -> Dict:
        """
        수집된 대화를 바탕으로 이야기 주제 제안
        
        Returns:
            Dict: 이야기 주제 및 구조
        """
        conversation_history = self.get_conversation_history()
        
        return self.story_engine.suggest_story_theme(
            conversation_history=conversation_history,
            child_name=self.child_name or "친구",
            age_group=self.age_group or 5,
            interests=self.interests,
            story_collection_prompt=self._get_story_collection_prompt()
        )
    
    def get_conversation_summary(self) -> str:
        """
        대화 내용 요약
        
        Returns:
            str: 대화 요약
        """
        conversation_history = self.get_conversation_history()
        
        return self.story_engine.get_conversation_summary(
            conversation_history=conversation_history,
            child_name=self.child_name or "친구",
            age_group=self.age_group or 5
        )
    
    def _get_story_collection_prompt(self) -> str:
        """이야기 수집 프롬프트 생성"""
        # 기본 프롬프트 템플릿
        template = self.prompts.get('story_collection_prompt_template', '')
        
        if not template:
            return f"""
            {self.child_name or '친구'}와의 대화를 바탕으로 재미있는 동화를 만들어주세요.
            연령대: {self.age_group or 5}세
            관심사: {', '.join(self.interests) if self.interests else '다양한 주제'}
            """
        
        # 포맷팅
        interests_str = ", ".join(self.interests) if self.interests else "다양한 주제"
        
        return template.format(
            child_name=self.child_name or "친구",
            age_group=self.age_group or 5,
            interests=interests_str
        )
    
    # ==========================================
    # 대화 히스토리 관리
    # ==========================================
    
    def add_to_conversation(self, role: str, content: str):
        """대화 내역에 메시지 추가"""
        self.conversation.add_message(role, content)
    
    def get_conversation_history(self) -> List[Dict]:
        """대화 내역 반환"""
        return self.conversation.get_conversation_history()
    
    def save_conversation(self, file_path: str) -> bool:
        """
        대화 내역을 파일로 저장
        
        Args:
            file_path: 저장할 파일 경로
            
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
                "story_stage": self.story_engine.get_current_stage(),
                "story_elements": self.story_engine.get_story_elements(),
                "collection_stats": self.story_engine.get_collection_stats(),
                "processing_stats": self.message_processor.get_processing_stats()
            }
            
            # 이야기 개요 추가
            story_outline = self.story_engine.get_story_outline()
            if story_outline:
                additional_data["story_outline"] = story_outline
            
            return self.conversation.save_conversation(file_path, additional_data)
            
        except Exception as e:
            logger.error(f"대화 저장 실패: {e}")
            return False
    
    def load_conversation(self, file_path: str) -> bool:
        """
        대화 내역을 파일에서 로드
        
        Args:
            file_path: 로드할 파일 경로
            
        Returns:
            bool: 성공 여부
        """
        try:
            data = self.conversation.load_conversation(file_path)
            
            if not data:
                return False
            
            # 아이 정보 복원
            child_info = data.get("child_info", {})
            self.update_child_info(
                child_name=child_info.get("name"),
                age=child_info.get("age"),
                interests=child_info.get("interests", [])
            )
            
            # 엔진 상태 복원
            self.story_engine.update_from_saved_data(data)
            
            return True
            
        except Exception as e:
            logger.error(f"대화 로드 실패: {e}")
            return False
    
    # ==========================================
    # 상태 및 통계
    # ==========================================
    
    def get_system_status(self) -> Dict[str, Any]:
        """시스템 상태 반환"""
        return {
            "version": "3.0.0-refactored",
            "child_info": {
                "name": self.child_name,
                "age": self.age_group,
                "interests": self.interests,
                "chatbot_name": self.chatbot_name
            },
            "conversation_stats": self.conversation.get_token_usage(),
            "story_stats": self.story_engine.get_collection_stats(),
            "message_stats": self.message_processor.get_processing_stats(),
            "engines": {
                "openai_available": self.openai_client is not None,
                "rag_available": self.rag_system is not None,
                "legacy_compatibility": self.legacy_compatibility
            }
        }
    
    def get_capabilities(self) -> List[str]:
        """사용 가능한 기능 목록 반환"""
        capabilities = [
            "대화 관리",
            "이야기 수집",
            "이야기 분석",
            "메시지 포맷팅",
            "연령대별 언어 적응",
            "대화 저장/로드",
            "통계 제공"
        ]
        
        if self.openai_client:
            capabilities.extend([
                "AI 응답 생성",
                "고급 분석",
                "이야기 주제 제안"
            ])
        
        if self.rag_system:
            capabilities.extend([
                "RAG 기반 풍부화",
                "벡터 데이터베이스 활용"
            ])
        
        return capabilities
    
    # ==========================================
    # 레거시 호환성 메서드들
    # ==========================================
    
    def suggest_story_element(self, user_input: str) -> str:
        """레거시 호환: 이야기 요소 제안"""
        warnings.warn(
            "suggest_story_element는 deprecated되었습니다. get_response를 사용하세요.",
            DeprecationWarning,
            stacklevel=2
        )
        return self.get_response(user_input)
    
    def get_token_usage(self) -> Dict[str, int]:
        """레거시 호환: 토큰 사용량"""
        return self.conversation.get_token_usage()


# 기존 클래스명과의 호환성을 위한 별칭
StoryCollectionChatBot = ChatBotA 