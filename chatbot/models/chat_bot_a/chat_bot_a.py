"""
아이들과 대화하며 동화 줄거리를 수집하는 AI 챗봇 (부기) - Enhanced v2.0

개선된 프롬프트 시스템과 호환:
- 연령별 특화 수집 (4-7세, 8-9세)
- 성능 추적 및 최적화
- Enhanced 모드 지원
- Chat Bot B와의 향상된 협업
"""
from typing import List, Dict, Any, Optional
import time

# 새로운 통합 엔진들
from .core.story_engine import StoryEngine
from .processors.unified_message_processor import UnifiedMessageProcessor
from .conversation_manager import ConversationManager
from chatbot.data.vector_db.core import VectorDB

# 공통 유틸리티
from shared.utils.logging_utils import get_module_logger
from shared.utils.openai_utils import initialize_client
from shared.configs.prompts_config import load_chatbot_a_prompts


logger = get_module_logger(__name__)

class ChatBotA:
    """
    ChatBot A
    
    주요 기능:
    - 개선된 이야기 수집 엔진 (연령별 특화)
    - 향상된 메시지 프로세싱 
    - Chat Bot B와의 원활한 협업
    - 성능 추적 및 최적화
    - Enhanced 프롬프트 시스템 호환
    """
    
    def __init__(self, 
                 token_limit: int = 10000, 
                 enable_rag: bool = True, 
                 legacy_compatibility: bool = True,
                 enhanced_mode: bool = True,
                 enable_performance_tracking: bool = True):
        """
        ChatBot A 초기화
        
        Args:
            token_limit: 대화 토큰 제한
            enable_rag: RAG 시스템 사용 여부
            legacy_compatibility: 레거시 호환성 유지 여부
            enhanced_mode: Enhanced 모드 사용 여부
            enable_performance_tracking: 성능 추적 활성화
        """
        # 기본 설정
        self.token_limit = token_limit
        self.enable_rag = enable_rag
        self.legacy_compatibility = legacy_compatibility
        self.enhanced_mode = enhanced_mode
        self.enable_performance_tracking = enable_performance_tracking
        
        # 프롬프트 버전 설정
        self.prompt_version = "Enhanced v2.0" if enhanced_mode else "Standard v1.0"
        
        # 아이 정보
        self.child_name = None
        self.age_group = None
        self.interests = []
        self.chatbot_name = "부기"
        
        # 성능 메트릭
        self.performance_metrics = {
            "total_conversations": 0,
            "successful_story_collections": 0,
            "average_collection_time": 0,
            "enhanced_mode_usage": 0,
            "age_group_statistics": {},
            "collaboration_success_rate": 0
        }
        
        # 프롬프트 로드
        self.prompts = load_chatbot_a_prompts()
        
        # OpenAI 클라이언트 초기화
        try:
            self.openai_client = initialize_client()
        except Exception as e:
            logger.error(f"OpenAI 클라이언트 초기화 실패: {e}")
            self.openai_client = None
        
        # RAG 시스템 초기화
        try:
            self.rag_system = VectorDB()
            logger.info("RAG 시스템 초기화 완료")
        except Exception as e:
            logger.error(f"RAG 시스템 초기화 실패: {e}")
            self.rag_system = None
        
        # 핵심 엔진들 초기화
        self._initialize_engines()
        
        # 레거시 호환성 설정
        if legacy_compatibility:
            self._setup_legacy_compatibility()
        
        logger.info(f"ChatBot A (Enhanced v2.0) 초기화 완료 - Mode: {self.prompt_version}")
    
    def _initialize_engines(self):
        """핵심 엔진들 초기화 (Enhanced)"""
        # 대화 관리자
        self.conversation = ConversationManager(self.token_limit)
        
        # Enhanced 이야기 엔진
        self.story_engine = StoryEngine(
            openai_client=self.openai_client,
            rag_system=self.rag_system,
            conversation_manager=self.conversation,
            enhanced_mode=self.enhanced_mode,
            performance_tracking=self.enable_performance_tracking
        )
        
        # Enhanced 메시지 프로세서
        self.message_processor = UnifiedMessageProcessor(
            prompts=self.prompts,
            child_name=self.child_name,
            age_group=self.age_group,
            interests=self.interests,
            chatbot_name=self.chatbot_name,
            enhanced_mode=self.enhanced_mode
        )
        
        logger.info("Enhanced 통합 엔진들 초기화 완료")
    
    def _setup_legacy_compatibility(self):
        """레거시 호환성을 위한 별칭 설정"""
        # 기존 API와의 호환성을 위한 별칭들
        self.collector = self.story_engine  # StoryCollector 호환
        self.analyzer = self.story_engine   # StoryAnalyzer 호환
        self.formatter = self.message_processor  # MessageFormatter 호환
        
        logger.info("레거시 호환성 설정 완료")
    
    # ==========================================
    # Enhanced 기본 정보 관리
    # ==========================================
    
    def update_child_info(self, child_name: str = None, age: int = None, 
                         interests: List[str] = None, chatbot_name: str = None):
        """
        아이 정보 업데이트 (Enhanced)
        
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
            # 연령대별 통계 업데이트
            if self.enable_performance_tracking:
                age_group_key = self._get_age_group_key(age)
                if age_group_key not in self.performance_metrics["age_group_statistics"]:
                    self.performance_metrics["age_group_statistics"][age_group_key] = 0
                self.performance_metrics["age_group_statistics"][age_group_key] += 1
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
        
        # 스토리 엔진에 연령별 설정 전파
        if hasattr(self.story_engine, 'set_age_specific_mode'):
            self.story_engine.set_age_specific_mode(self.age_group)
        
        logger.info(f"Enhanced 아이 정보 업데이트: {self.child_name}({self.age_group}세) - {self.prompt_version}")
    
    def _get_age_group_key(self, age: int) -> str:
        """연령대 키 반환"""
        if 4 <= age <= 7:
            return "age_4_7"
        elif 8 <= age <= 9:
            return "age_8_9"
        else:
            return "age_other"
    
    # ==========================================
    # Enhanced 대화 관리
    # ==========================================
    
    def initialize_chat(self, child_name: str, age: int, interests: List[str] = None, 
                       chatbot_name: str = "부기") -> str:
        """
        Enhanced 챗봇과의 대화 초기화
        
        Args:
            child_name: 아이의 이름
            age: 아이의 나이
            interests: 아이의 관심사 목록
            chatbot_name: 챗봇의 이름
            
        Returns:
            str: 연령별 특화된 초기 인사 메시지
        """
        # 입력 검증
        if not child_name or not isinstance(child_name, str):
            raise ValueError("아이의 이름은 필수고, 문자열이어야 합니다.")
        if not isinstance(age, int) or age < 4 or age > 9:
            raise ValueError("아이의 나이는 4-9세 사이 정수이어야 합니다.")
        
        # 성능 추적 시작
        start_time = time.time()
        if self.enable_performance_tracking:
            self.performance_metrics["total_conversations"] += 1
            if self.enhanced_mode:
                self.performance_metrics["enhanced_mode_usage"] += 1
        
        # 아이 정보 업데이트
        self.update_child_info(child_name, age, interests, chatbot_name)
        
        # Enhanced 연령별 인사말 생성
        if self.enhanced_mode:
            greeting = self.message_processor.get_enhanced_greeting(age)
        else:
            greeting = self.message_processor.get_greeting()
        
        # 대화 시작
        self.add_to_conversation("assistant", greeting)
        
        # 성능 메트릭 업데이트
        if self.enable_performance_tracking:
            initialization_time = time.time() - start_time
            logger.info(f"Enhanced 대화 초기화 완료: {initialization_time:.2f}초")
        
        return greeting
    
    def get_response(self, user_input: str) -> str:
        """
        Enhanced 사용자 입력에 대한 응답 생성
        
        Args:
            user_input: 사용자 입력
            
        Returns:
            str: 연령별 최적화된 챗봇 응답
        """
        try:
            # 토큰 제한 확인
            if self.conversation.is_token_limit_reached():
                return self.message_processor.get_token_limit_message()
            
            # 사용자 입력 추가
            self.add_to_conversation("user", user_input)
            
            # Enhanced 분석 수행
            analysis_result = self.story_engine.analyze_input(
                user_input, 
                enhanced_mode=self.enhanced_mode,
                age_group=self.age_group
            )
            
            # Enhanced 응답 생성
            response = self._generate_enhanced_response(user_input, analysis_result)
            
            # 응답 검증 및 개선
            validation = self.message_processor.validate_response(response)
            if not validation['is_valid']:
                response = self._fix_response_issues(response, validation)
            
            # 응답 추가
            self.add_to_conversation("assistant", response)
            
            return response
            
        except Exception as e:
            logger.error(f"Enhanced 응답 생성 중 오류: {e}")
            return self.message_processor.get_error_message()
    
    def _generate_enhanced_response(self, user_input: str, analysis_result: Dict) -> str:
        """Enhanced 맥락적 응답 생성"""
        conversation_history = self.conversation.get_recent_messages(5)
        
        if self.enhanced_mode:
            # Enhanced 프롬프트 시스템 사용
            response_context = {
                "user_input": user_input,
                "analysis": analysis_result,
                "conversation_history": conversation_history,
                "child_age": self.age_group,
                "child_interests": self.interests,
                "enhanced_mode": True,
                "prompt_version": self.prompt_version
            }
            
            return self.story_engine.generate_enhanced_response(response_context)
        else:
            # 기본 모드
            return self.story_engine.generate_contextual_response(
                user_input, analysis_result, conversation_history
            )
    
    def _fix_response_issues(self, response: str, validation: Dict) -> str:
        """Enhanced 응답 문제 수정"""
        if not validation.get('age_appropriate', True):
            # 연령별 적절성 수정
            if self.enhanced_mode:
                return self.message_processor.adjust_for_age_group(response, self.age_group)
            else:
                return self.message_processor.make_age_appropriate(response)
        
        if not validation.get('encouraging', True):
            return self.message_processor.add_encouragement(response)
        
        if not validation.get('clear', True):
            return self.message_processor.clarify_message(response)
        
        return response
    
    # ==========================================
    # Enhanced 스토리 수집 기능
    # ==========================================
    
    def suggest_story_theme(self) -> Dict:
        """Enhanced 연령별 스토리 주제 제안"""
        if self.enhanced_mode and self.age_group:
            return self.story_engine.suggest_enhanced_theme(
                age_group=self.age_group,
                interests=self.interests
            )
        else:
            # StoryEngine.suggest_story_theme은 매개변수가 필요함
            conversation_history = self.conversation.get_all_messages() if self.conversation else []
            return self.story_engine.suggest_story_theme(
                conversation_history=conversation_history,
                child_name=self.child_name or "친구",
                age_group=self.age_group or 5,
                interests=self.interests or [],
                story_collection_prompt="대화에서 수집된 정보를 바탕으로 이야기 주제를 제안해주세요."
            )
    
    def get_story_outline_for_chatbot_b(self) -> Dict[str, Any]:
        """
        Chat Bot B를 위한 Enhanced 스토리 개요 생성
        
        Returns:
            Dict: Enhanced 스토리 개요 (v2.0 호환)
        """
        try:
            collection_start_time = time.time()
            
            # Enhanced 스토리 수집
            if self.enhanced_mode:
                story_outline = self.story_engine.create_enhanced_story_outline(
                    conversation_history=self.conversation.get_all_messages(),
                    child_age=self.age_group,
                    child_interests=self.interests,
                    child_name=self.child_name
                )
            else:
                story_outline = self.story_engine.create_story_outline()
            
            # Chat Bot B 호환성 메타데이터 추가
            story_outline.update({
                "chatbot_a_version": self.prompt_version,
                "enhanced_mode": self.enhanced_mode,
                "collection_method": "enhanced_conversation" if self.enhanced_mode else "basic_conversation",
                "child_profile": {
                    "name": self.child_name,
                    "age": self.age_group,
                    "interests": self.interests
                },
                "collaboration_metadata": {
                    "source": "chatbot_a",
                    "timestamp": time.time(),
                    "conversation_length": len(self.conversation.get_all_messages())
                }
            })
            
            # 성능 메트릭 업데이트
            if self.enable_performance_tracking:
                collection_time = time.time() - collection_start_time
                self.performance_metrics["successful_story_collections"] += 1
                self._update_average_collection_time(collection_time)
                self.performance_metrics["collaboration_success_rate"] = (
                    self.performance_metrics["successful_story_collections"] / 
                    max(1, self.performance_metrics["total_conversations"])
                )
            
            logger.info(f"Enhanced 스토리 개요 생성 완료 for Chat Bot B")
            return story_outline
            
        except Exception as e:
            logger.error(f"Enhanced 스토리 개요 생성 실패: {e}")
            raise
    
    def _update_average_collection_time(self, new_time: float):
        """평균 수집 시간 업데이트"""
        current_avg = self.performance_metrics["average_collection_time"]
        successful_count = self.performance_metrics["successful_story_collections"]
        
        if successful_count == 1:
            self.performance_metrics["average_collection_time"] = new_time
        else:
            self.performance_metrics["average_collection_time"] = (
                (current_avg * (successful_count - 1) + new_time) / successful_count
        )
    
    def get_conversation_summary(self) -> str:
        """Enhanced 대화 요약"""
        if self.enhanced_mode:
            return self.story_engine.create_enhanced_summary(
                self.conversation.get_all_messages(),
                age_group=self.age_group
            )
        else:
            return self.story_engine.create_conversation_summary()
    
    # ==========================================
    # Enhanced 시스템 상태 및 메트릭
    # ==========================================
    
    def get_system_status(self) -> Dict[str, Any]:
        """Enhanced 시스템 상태 조회"""
        status = {
            "openai_client": self.openai_client is not None,
            "rag_system": self.rag_system is not None,
            "story_engine": self.story_engine is not None,
            "message_processor": self.message_processor is not None,
            "conversation_manager": self.conversation is not None,
            "enhanced_mode": self.enhanced_mode,
            "prompt_version": self.prompt_version,
            "child_info_set": all([self.child_name, self.age_group]),
            "conversation_active": len(self.conversation.get_all_messages()) > 0,
            "performance_tracking": self.enable_performance_tracking
        }
        
        # 성능 메트릭 추가
        if self.enable_performance_tracking:
            status["performance_metrics"] = self.performance_metrics
        
        return status
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """성능 메트릭 조회"""
        return self.performance_metrics if self.enable_performance_tracking else {}
    
    def get_capabilities(self) -> List[str]:
        """Enhanced 기능 목록"""
        base_capabilities = [
            "자연스러운 대화",
            "스토리 요소 수집",
            "아이 친화적 응답",
            "대화 맥락 유지",
            "안전한 콘텐츠 필터링"
        ]
        
        if self.enhanced_mode:
            enhanced_capabilities = [
                "연령별 특화 대화 (4-7세, 8-9세)",
                "향상된 프롬프트 엔지니어링",
                "체인 오브 소트 추론",
                "성능 추적 및 최적화",
                "Chat Bot B와 향상된 협업"
            ]
            base_capabilities.extend(enhanced_capabilities)
        
        if self.rag_system:
            base_capabilities.append("RAG 기반 지식 검색")
        
        return base_capabilities
    
    # ==========================================
    # 레거시 호환성 메서드들
    # ==========================================
    
    def add_to_conversation(self, role: str, content: str):
        """대화 기록에 메시지 추가"""
        self.conversation.add_message(role, content)
    
    def get_conversation_history(self) -> List[Dict]:
        """대화 기록 반환"""
        return self.conversation.get_all_messages()
    
    def save_conversation(self, file_path: str) -> bool:
        """대화 저장 (Enhanced 메타데이터 포함)"""
        try:
            conversation_data = {
                "messages": self.conversation.get_all_messages(),
                "child_info": {
                    "name": self.child_name,
                    "age": self.age_group,
                    "interests": self.interests
                },
                "metadata": {
                    "chatbot_version": self.prompt_version,
                    "enhanced_mode": self.enhanced_mode,
                    "timestamp": time.time()
            }
            }
            
            if self.enable_performance_tracking:
                conversation_data["performance_metrics"] = self.performance_metrics
            
            return self.conversation.save_conversation(file_path, conversation_data)
        except Exception as e:
            logger.error(f"Enhanced 대화 저장 실패: {e}")
            return False
    
    def load_conversation(self, file_path: str) -> bool:
        """대화 로드 (Enhanced 메타데이터 지원)"""
        try:
            conversation_data = self.conversation.load_conversation(file_path)
            
            if conversation_data:
                # 메타데이터 복원
                metadata = conversation_data.get("metadata", {})
                child_info = conversation_data.get("child_info", {})
            
            # 아이 정보 복원
            self.update_child_info(
                child_name=child_info.get("name"),
                age=child_info.get("age"),
                interests=child_info.get("interests", [])
            )
            
                # 성능 메트릭 복원
            if self.enable_performance_tracking and "performance_metrics" in conversation_data:
                    self.performance_metrics.update(conversation_data["performance_metrics"])
                    logger.info(f"Enhanced 대화 로드 완료: {metadata.get('chatbot_version', 'unknown')}")
            return True
            
            return False
        except Exception as e:
            logger.error(f"Enhanced 대화 로드 실패: {e}")
            return False
    
    def suggest_story_element(self, user_input: str) -> str:
        """Enhanced 스토리 요소 제안"""
        if self.enhanced_mode:
            return self.story_engine.suggest_enhanced_element(
                user_input, 
                age_group=self.age_group,
                interests=self.interests
            )
        else:
            return self.story_engine.suggest_story_element(user_input)
    
    def get_token_usage(self) -> Dict[str, int]:
        """토큰 사용량 조회"""
        return self.conversation.get_token_usage()


# 기존 클래스명과의 호환성을 위한 별칭
StoryCollectionChatBot = ChatBotA 