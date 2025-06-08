"""
아이들과 대화하며 동화 줄거리를 수집하는 AI 챗봇 (부기) - Enhanced v3.0

개선된 영구 메모리 시스템과 LangChain 통합:
- SQLite 기반 영구 대화 저장
- LangChain RunnableWithMessageHistory 활용
- 세션 기반 대화 연속성
- 자동 토큰 관리 및 메시지 트리밍
- 연령별 특화 수집 (4-7세, 8-9세)
"""
from typing import List, Dict, Any, Optional
import time
import uuid
from datetime import datetime

# 새로운 영구 메모리 시스템
from .core.persistent_memory import ConversationMemoryManager, create_session_id, restore_session_from_file

# 기존 통합 엔진들
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
    ChatBot A - Enhanced v3.0 (영구 메모리 시스템)
    
    주요 기능:
    - SQLite 기반 영구 대화 저장 및 관리
    - LangChain RunnableWithMessageHistory 통합
    - 세션 기반 대화 연속성 보장
    - 자동 토큰 관리 및 메시지 트리밍
    - 개선된 이야기 수집 엔진 (연령별 특화)
    - Chat Bot B와의 원활한 협업
    """
    
    def __init__(self, 
                 vector_db_instance: VectorDB,
                 token_limit: int = 10000, 
                 use_langchain: bool = True,
                 legacy_compatibility: bool = True,
                 enhanced_mode: bool = True,
                 enable_performance_tracking: bool = True,
                 session_id: str = None,
                 db_path: str = "chatbot/data/conversations.db"):
        """
        ChatBot A 초기화
        
        Args:
            vector_db_instance: 미리 초기화된 VectorDB 인스턴스
            token_limit: 대화 토큰 제한 
            use_langchain: LangChain 사용 여부
            legacy_compatibility: 레거시 호환성 유지 여부
            enhanced_mode: Enhanced 모드 사용 여부
            enable_performance_tracking: 성능 추적 활성화
            session_id: 세션 ID (없으면 자동 생성)
            db_path: SQLite 데이터베이스 경로
        """
        # 기본 설정
        self.token_limit = token_limit
        self.legacy_compatibility = legacy_compatibility
        self.enhanced_mode = enhanced_mode
        self.user_data = {}
        self.story_data = {}
        self.enable_performance_tracking = enable_performance_tracking
        self.use_langchain = use_langchain
        self.db_path = db_path
        
        # 세션 관리
        self.session_id = session_id or create_session_id("default")
        self.session_active = False
        
        # 프롬프트 버전 설정
        self.prompt_version = "Enhanced v3.0" if enhanced_mode else "Standard v1.0"
        
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
            logger.info("OpenAI 클라이언트 초기화 성공")
        except Exception as e:
            logger.warning(f"OpenAI 클라이언트 초기화 실패: {e}")
            logger.warning("fallback 모드로 동작합니다. 실제 GPT 응답을 원한다면 'pip install openai'를 실행하세요.")
            self.openai_client = None
        
        # RAG 시스템 설정
        self.rag_system = vector_db_instance
        if self.rag_system:
            logger.info("RAG 시스템 (VectorDB) 주입 완료")
            self.enable_rag = True
        else:
            logger.warning("RAG 시스템 (VectorDB)이 주입되지 않았거나 None입니다.")
            self.enable_rag = False
        
        # 영구 메모리 시스템 초기화
        self._initialize_memory_system()
        
        # 핵심 엔진들 초기화
        self._initialize_engines()
        
        # 레거시 호환성 설정
        if legacy_compatibility:
            self._setup_legacy_compatibility()
        
        logger.info(f"ChatBot A (Enhanced v3.0) 초기화 완료 - Session: {self.session_id}, Mode: {self.prompt_version}, RAG: {self.enable_rag}")
    
    def _initialize_memory_system(self):
        """영구 메모리 시스템 초기화"""
        try:
            if self.use_langchain and self.openai_client:
                # LangChain 기반 영구 메모리 관리자 초기화
                self.memory_manager = ConversationMemoryManager(
                    openai_client=self.openai_client,
                    model_name="gpt-4o-mini",
                    max_tokens=self.token_limit,
                    db_path=self.db_path
                )
                
                # 현재 세션의 히스토리 가져오기
                self.conversation_history = self.memory_manager.get_session_history(self.session_id)
                
                logger.info(f"영구 메모리 시스템 초기화 완료 - Session: {self.session_id}")
            else:
                # LangChain 없이는 기존 방식 사용
                self.memory_manager = None
                self.conversation_history = None
                logger.warning("LangChain 미사용 또는 OpenAI 클라이언트 없음 - 기본 메모리 사용")
                
        except Exception as e:
            logger.error(f"영구 메모리 시스템 초기화 실패: {e}")
            self.memory_manager = None
            self.conversation_history = None
    
    def _initialize_engines(self):
        """핵심 엔진들 초기화 (Enhanced)"""
        # LangChain 기반 대화 관리자 (use_langchain가 True이고 OpenAI 클라이언트가 있을 때)
        if self.use_langchain and self.openai_client:
            try:
                from .core.legacy_integration import LegacyIntegrationManager
                self.conversation_manager = LegacyIntegrationManager(
                    token_limit=self.token_limit,
                    use_langchain=True,
                    openai_client=self.openai_client,
                    rag_system=self.rag_system,
                    prompts=self.prompts
                )
                self.conversation = self.conversation_manager.conversation
                logger.info("LangChain 기반 대화 엔진 초기화 완료")
            except Exception as e:
                logger.warning(f"LangChain 초기화 실패, 기본 모드로 전환: {e}")
                self.conversation = ConversationManager(self.token_limit)
        else:
            # 기본 대화 관리자
            self.conversation = ConversationManager(self.token_limit)
        
        # Enhanced 이야기 엔진
        self.story_engine = StoryEngine(
            openai_client=self.openai_client,
            rag_system=self.rag_system,
            user_data=self.user_data,
            story_data=self.story_data,
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
        Enhanced v3.0 - 영구 메모리 기반 응답 생성
        
        Args:
            user_input: 사용자 입력
            
        Returns:
            str: LangChain 기반 연속성있는 챗봇 응답
        """
        try:
            # 영구 메모리 시스템 사용 (최우선)
            if self.memory_manager and self.conversation_history:
                # 비동기 응답 생성을 동기로 래핑
                import asyncio
                try:
                    # 현재 이벤트 루프에서 실행 중인 경우
                    loop = asyncio.get_running_loop()
                    # 새 스레드에서 비동기 함수 실행
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(
                            asyncio.run, 
                            self._generate_persistent_response(user_input)
                        )
                        response = future.result(timeout=30)  # 30초 타임아웃
                    
                    logger.info(f"영구 메모리 응답 생성 완료: '{response[:50]}...'")
                    return response
                    
                except RuntimeError:
                    # 이벤트 루프가 없는 경우 직접 실행
                    response = asyncio.run(self._generate_persistent_response(user_input))
                    logger.info(f"영구 메모리 응답 생성 완료: '{response[:50]}...'")
                    return response
                    
                except Exception as e:
                    logger.error(f"영구 메모리 응답 생성 실패: {e}")
                    # 폴백으로 기존 방식 사용
            
            # 폴백: 기존 방식 사용
            logger.warning("영구 메모리 시스템을 사용할 수 없음, 기존 방식으로 폴백")
            return self._generate_fallback_response(user_input)
            
        except Exception as e:
            logger.error(f"응답 생성 중 치명적 오류: {e}")
            return f"죄송해요 {self.child_name or '친구'}아! 잠시 문제가 생겼어요. 다시 말해주실 수 있나요?"
    
    async def _generate_persistent_response(self, user_input: str) -> str:
        """영구 메모리 기반 응답 생성 (비동기)"""
        try:
            # LangChain 메모리 관리자로 응답 생성
            response = await self.memory_manager.generate_response(
                session_id=self.session_id,
                user_input=user_input,
                child_name=self.child_name or "친구",
                age_group=self.age_group or 5,
                interests=self.interests or []
            )
            
            # 세션 활성화 표시
            self.session_active = True
            
            # 성능 메트릭 업데이트
            if self.enable_performance_tracking:
                self.performance_metrics["successful_story_collections"] += 1
            
            return response
            
        except Exception as e:
            logger.error(f"영구 메모리 응답 생성 중 오류: {e}")
            # 에러 시 폴백 응답
            return f"안녕 {self.child_name or '친구'}아! 재미있는 이야기를 함께 만들어보자!"
    
    def _generate_fallback_response(self, user_input: str) -> str:
        """기존 방식의 폴백 응답 생성"""
        try:
            # 토큰 제한 확인
            if hasattr(self.conversation, 'is_token_limit_reached') and self.conversation.is_token_limit_reached():
                return self.message_processor.get_token_limit_message()
            
            # 사용자 입력 추가
            self.add_to_conversation("user", user_input)
            
            # Enhanced 분석 수행
            analysis_result = self.story_engine.analyze_input(
                user_input, 
                enhanced_mode=self.enhanced_mode,
                age_group=self.age_group
            )
            
            # 기본 응답 생성
            response = self._generate_enhanced_response(user_input, analysis_result)
            
            # 응답 추가
            self.add_to_conversation("assistant", response)
            
            return response
            
        except Exception as e:
            logger.error(f"폴백 응답 생성 중 오류: {e}")
            return f"안녕 {self.child_name or '친구'}아! 오늘은 어떤 재미있는 이야기를 만들어볼까?"
    
    def _generate_enhanced_response(self, user_input: str, analysis_result: Dict) -> str:
        """Enhanced 맥락적 응답 생성"""
        conversation_history = self.conversation.get_recent_messages(5)
        
        # LangChain을 사용할 수 있는 경우 우선 시도
        if self.use_langchain and self.openai_client and hasattr(self.conversation_manager, 'get_enhanced_response'):
            try:
                response = self.conversation_manager.get_enhanced_response(
                    user_input=user_input,
                    child_name=self.child_name,
                    age_group=self.age_group,
                    interests=self.interests,
                    chatbot_name=self.chatbot_name
                )
                logger.info("LangChain 기반 응답 생성 완료")
                return response
            except Exception as e:
                logger.warning(f"LangChain 응답 생성 실패, fallback 모드로 전환: {e}")
        
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
    
    def get_story_data(self) -> Dict:
        """현재 생성된 스토리 데이터를 반환합니다."""
        return self.story_engine.get_story_data()

    def suggest_story_idea(self) -> Dict:
        """
        대화 기록을 바탕으로 동화 아이디어를 제안합니다.
        """
        # StoryEngine의 suggest_story_idea를 호출 (비동기 처리 필요)
        # 여기서는 간단한 동기적 호출 예시를 보여주나, 실제로는 비동기 처리가 필요함
        # conversation_history = self.conversation_manager.get_history() # 예시
        conversation_history = [] # 임시
        
        # asyncio.run()은 이미 실행 중인 이벤트 루프에서 호출할 수 없으므로
        # 실제 FastAPI 환경에서는 바로 await를 사용해야 합니다.
        # loop = asyncio.get_event_loop()
        # return loop.run_until_complete(self.story_engine.suggest_story_idea(conversation_history))
        
        # 임시 동기 반환 (실제 사용 시 주의)
        return {
            "title": "임시 아이디어",
            "summary": "RAG를 통해 생성된 재미있는 모험 이야기"
        }

    def suggest_story_theme(self) -> Dict:
        """
        대화 기록을 바탕으로 이야기 주제를 제안합니다.
        """
        try:
            conversation_history = self.conversation.get_conversation_history()
            
            # StoryEngine의 suggest_story_theme 메서드 호출
            story_theme = self.story_engine.suggest_story_theme(
                conversation_history=conversation_history,
                child_name=self.child_name,
                age_group=self.age_group,
                interests=self.interests,
                story_collection_prompt="수집된 대화를 바탕으로 이야기 주제를 생성해주세요"
            )
            
            return story_theme
            
        except Exception as e:
            logger.error(f"이야기 주제 제안 중 오류: {e}")
            return {
                "title": f"{self.child_name}의 모험",
                "theme": "우정과 모험",
                "plot_summary": "더 많은 대화가 필요해!",
                "educational_value": "상상력과 창의성",
                "target_age": self.age_group,
                "estimated_length": "짧음",
                "key_scenes": []
            }

    def reset_story(self):
        """스토리 데이터를 초기화합니다."""
        self.story_engine.reset_story()
        logger.info("ChatBot A 스토리 데이터 리셋 완료.")
    
    def get_story_outline_for_chatbot_b(self) -> Dict[str, Any]:
        """
        Chat Bot B (꼬기) 를 위한 강화된 스토리 개요 생성 (영구 메모리 기반)
        
        Returns:
            Dict: 실제 대화 내용을 분석한 스토리 개요
        """
        try:
            collection_start_time = time.time()
            
            # 영구 메모리 시스템에서 대화 기록 가져오기
            conversation_history = self._get_conversation_history_for_story()
            
            logger.info(f"스토리 개요 생성: 대화 기록 {len(conversation_history)}개 메시지 분석 중...")
            
            # 실제 대화 내용이 있는지 확인
            if not conversation_history or len(conversation_history) < 2:
                logger.warning("충분한 대화 기록이 없어 기본 스토리 개요 생성")
                return self._create_default_story_outline()
            
            # Enhanced 스토리 수집 (영구 메모리 기반)
            if self.enhanced_mode and self.memory_manager:
                # 비동기 처리를 동기로 래핑
                import asyncio
                try:
                    story_outline = asyncio.run(self._generate_enhanced_story_outline(conversation_history))
                except RuntimeError:
                    # 이미 이벤트 루프가 실행 중인 경우 동기 방식으로 폴백
                    story_outline = self._generate_sync_story_outline(conversation_history)
            else:
                # 기본 모드에서도 실제 대화 내용 분석
                story_outline = self.story_engine.suggest_story_theme(
                    conversation_history=conversation_history,
                    child_name=self.child_name or "친구",
                    age_group=self.age_group or 5,
                    interests=self.interests or [],
                    story_collection_prompt="실제 대화 내용을 바탕으로 창의적인 이야기 구조를 생성해줘."
                )
            
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
                    "conversation_length": len(self.conversation.get_conversation_history())
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
            # 에러 발생 시 기본 스토리 개요 반환
            return self._create_default_story_outline()
    
    def _get_conversation_history_for_story(self) -> List[Dict]:
        """스토리 생성을 위한 대화 기록 가져오기"""
        try:
            if self.memory_manager and self.conversation_history:
                # 영구 메모리에서 대화 기록 가져오기
                messages = self.conversation_history.messages
                conversation_history = []
                
                for msg in messages:
                    if hasattr(msg, 'content') and msg.content.strip():
                        if "Human" in str(type(msg)):
                            conversation_history.append({
                                "role": "user",
                                "content": msg.content
                            })
                        elif "AI" in str(type(msg)):
                            conversation_history.append({
                                "role": "assistant", 
                                "content": msg.content
                            })
                
                logger.info(f"영구 메모리에서 {len(conversation_history)}개 메시지 추출")
                return conversation_history
            else:
                # 폴백: 기존 방식
                return self.conversation.get_conversation_history() if hasattr(self.conversation, 'get_conversation_history') else []
                
        except Exception as e:
            logger.error(f"대화 기록 추출 실패: {e}")
            return []
    
    async def _generate_enhanced_story_outline(self, conversation_history: List[Dict]) -> Dict[str, Any]:
        """영구 메모리 기반 Enhanced 스토리 개요 생성 (비동기)"""
        try:
            # 대화 요약 생성
            conversation_summary = await self.memory_manager.generate_response(
                session_id=self.session_id,
                user_input="지금까지의 대화를 바탕으로 아이가 좋아할 만한 창의적인 동화의 주요 요소들(등장인물, 배경, 갈등, 해결방법)을 추출해서 간단히 요약해줘.",
                child_name=self.child_name or "친구",
                age_group=self.age_group or 5,
                interests=self.interests or []
            )
            
            # StoryEngine으로 상세 분석
            detailed_analysis = self.story_engine.suggest_story_theme(
                conversation_history=conversation_history,
                child_name=self.child_name or "친구",
                age_group=self.age_group or 5,
                interests=self.interests or [],
                story_collection_prompt="실제 대화 내용과 AI 요약을 바탕으로 구체적이고 창의적인 동화 구조를 생성해줘."
            )
            
            # 대화 기반 키워드 추출
            user_messages = [msg["content"] for msg in conversation_history if msg["role"] == "user"]
            extracted_keywords = []
            for msg in user_messages:
                # 간단한 키워드 추출 (실제로는 더 정교한 NLP 사용 가능)
                words = msg.split()
                keywords = [w for w in words if len(w) > 2 and w.isalpha()]
                extracted_keywords.extend(keywords[:3])  # 메시지당 최대 3개 키워드
            
            # Enhanced 메타데이터 추가
            enhanced_outline = {
                **detailed_analysis,
                "conversation_summary": conversation_summary,
                "extracted_keywords": list(set(extracted_keywords)),
                "actual_conversation_length": len(conversation_history),
                "story_generation_method": "enhanced_memory_based",
                "conversation_analysis": {
                    "user_messages": len([msg for msg in conversation_history if msg["role"] == "user"]),
                    "ai_responses": len([msg for msg in conversation_history if msg["role"] == "assistant"]),
                    "total_words": sum(len(msg["content"].split()) for msg in conversation_history),
                    "conversation_topics": extracted_keywords[:10]  # 상위 10개 토픽
                }
            }
            
            logger.info(f"Enhanced 스토리 개요 생성 완료: {len(extracted_keywords)}개 키워드 추출")
            return enhanced_outline
            
        except Exception as e:
            logger.error(f"Enhanced 스토리 개요 생성 중 오류: {e}")
            return self._generate_sync_story_outline(conversation_history)
    
    def _generate_sync_story_outline(self, conversation_history: List[Dict]) -> Dict[str, Any]:
        """동기 방식 스토리 개요 생성"""
        try:
            # 기존 StoryEngine 사용
            story_outline = self.story_engine.suggest_story_theme(
                conversation_history=conversation_history,
                child_name=self.child_name or "친구",
                age_group=self.age_group or 5,
                interests=self.interests or [],
                story_collection_prompt="실제 대화 내용을 바탕으로 창의적인 동화 구조를 생성해줘."
            )
            
            # 기본 메타데이터 추가
            story_outline.update({
                "actual_conversation_length": len(conversation_history),
                "story_generation_method": "sync_conversation_based"
            })
            
            return story_outline
            
        except Exception as e:
            logger.error(f"동기 스토리 개요 생성 중 오류: {e}")
            return self._create_default_story_outline()
    
    def _create_default_story_outline(self) -> Dict[str, Any]:
        """기본 스토리 개요 생성 (대화 기록이 부족할 때)"""
        return {
            "title": f"{self.child_name or '친구'}의 특별한 모험",
            "theme": "우정과 모험",
            "plot_summary": "용감한 주인공이 친구들과 함께 신나는 모험을 떠나는 이야기",
            "educational_value": "협력과 문제해결",
            "target_age": self.age_group or 5,
            "estimated_length": "중간",
            "key_scenes": [
                "모험의 시작",
                "친구들과의 만남", 
                "어려움 극복",
                "행복한 결말"
            ],
            "characters": [
                {"name": self.child_name or "친구", "role": "주인공"},
                {"name": "동물 친구", "role": "조력자"}
            ],
            "setting": "마법의 숲",
            "story_generation_method": "default_template",
            "actual_conversation_length": 0,
            "child_profile": {
                "name": self.child_name,
                "age": self.age_group,
                "interests": self.interests
            },
            "collaboration_metadata": {
                "source": "chatbot_a_default",
                "timestamp": time.time(),
                "conversation_length": 0
            }
        }
    
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
                self.conversation.get_conversation_history(),
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
            "conversation_active": len(self.conversation.get_conversation_history()) > 0,
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
    
    def add_user_message(self, content: str):
        """사용자 메시지를 대화 기록에 추가 (편의성 메서드)"""
        if self.memory_manager and self.conversation_history:
            # 영구 메모리 시스템 사용
            from langchain_core.messages import HumanMessage
            self.conversation_history.add_message(HumanMessage(content=content))
            logger.debug(f"[CHATBOT_A] 영구 메모리에 사용자 메시지 추가: '{content[:30]}...'")
        else:
            # 기존 방식 폴백
            self.conversation.add_message("user", content)
            logger.debug(f"[CHATBOT_A] 사용자 메시지 추가: '{content[:30]}...'")
    
    def add_ai_message(self, content: str):
        """AI 메시지를 대화 기록에 추가 (편의성 메서드)"""
        if self.memory_manager and self.conversation_history:
            # 영구 메모리 시스템 사용
            from langchain_core.messages import AIMessage
            self.conversation_history.add_message(AIMessage(content=content))
            logger.debug(f"[CHATBOT_A] 영구 메모리에 AI 메시지 추가: '{content[:30]}...'")
        else:
            # 기존 방식 폴백
            self.conversation.add_message("assistant", content)
            logger.debug(f"[CHATBOT_A] AI 메시지 추가: '{content[:30]}...'")
    
    # ==========================================
    # Enhanced v3.0 세션 관리 메서드
    # ==========================================
    
    def create_new_session(self, child_name: str = None) -> str:
        """새 세션 생성"""
        try:
            # 새 세션 ID 생성
            self.session_id = create_session_id(child_name or self.child_name or "unknown")
            
            # 영구 메모리 시스템 재초기화
            if self.memory_manager:
                self.conversation_history = self.memory_manager.get_session_history(self.session_id)
                
                # 아이 정보가 있으면 세션에 저장
                if self.child_name:
                    self.conversation_history.update_session_info(
                        self.child_name, 
                        self.age_group or 5, 
                        self.interests or []
                    )
            
            self.session_active = False
            logger.info(f"새 세션 생성 완료: {self.session_id}")
            return self.session_id
            
        except Exception as e:
            logger.error(f"새 세션 생성 실패: {e}")
            return None
    
    def restore_session(self, session_id: str) -> bool:
        """기존 세션 복원"""
        try:
            if not self.memory_manager:
                logger.warning("영구 메모리 시스템이 없어 세션 복원 불가")
                return False
            
            # 세션 히스토리 가져오기
            history = self.memory_manager.get_session_history(session_id)
            stats = history.get_session_stats()
            
            if stats:
                # 세션 정보 복원
                self.session_id = session_id
                self.conversation_history = history
                self.child_name = stats.get("child_name")
                self.age_group = stats.get("age_group")
                self.interests = stats.get("interests", [])
                self.session_active = True
                
                logger.info(f"세션 복원 완료: {session_id} ({stats.get('message_count', 0)}개 메시지)")
                return True
            else:
                logger.warning(f"세션을 찾을 수 없음: {session_id}")
                return False
                
        except Exception as e:
            logger.error(f"세션 복원 실패: {e}")
            return False
    
    def get_session_info(self) -> Dict[str, Any]:
        """현재 세션 정보 반환"""
        info = {
            "session_id": self.session_id,
            "active": self.session_active,
            "child_name": self.child_name,
            "age_group": self.age_group,
            "interests": self.interests,
            "memory_type": "persistent" if self.memory_manager else "temporary"
        }
        
        # 메모리 관리자가 있으면 상세 정보 추가
        if self.memory_manager and self.conversation_history:
            stats = self.conversation_history.get_session_stats()
            info.update(stats)
        
        return info
    
    def export_conversation(self) -> Dict[str, Any]:
        """현재 대화를 JSON 형태로 내보내기"""
        try:
            if self.memory_manager and self.conversation_history:
                messages = []
                for msg in self.conversation_history.messages:
                    if hasattr(msg, 'content'):
                        role = "user" if "Human" in str(type(msg)) else "assistant"
                        messages.append({
                            "role": role,
                            "content": msg.content,
                            "timestamp": datetime.now().isoformat()
                        })
                
                return {
                    "session_info": self.get_session_info(),
                    "messages": messages,
                    "export_time": datetime.now().isoformat()
                }
            else:
                # 폴백: 기존 방식
                return {
                    "session_info": self.get_session_info(),
                    "messages": self.get_conversation_history(),
                    "export_time": datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"대화 내보내기 실패: {e}")
            return {}
    
    def cleanup_old_conversations(self, days: int = 30) -> int:
        """오래된 대화 정리"""
        try:
            if self.memory_manager:
                return self.memory_manager.cleanup_old_sessions(days)
            return 0
        except Exception as e:
            logger.error(f"대화 정리 실패: {e}")
            return 0
        logger.debug(f"[CHATBOT_A] AI 메시지 추가: '{content[:30]}...'")
    
    def get_conversation_history(self) -> List[Dict]:
        """대화 기록 반환"""
        return self.conversation.get_conversation_history()
    
    def save_conversation(self, file_path: str) -> bool:
        """대화 저장 (Enhanced 메타데이터 포함)"""
        try:
            conversation_data = {
                "messages": self.conversation.get_conversation_history(),
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

