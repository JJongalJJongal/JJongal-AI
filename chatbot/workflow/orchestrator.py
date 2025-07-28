"""
CCB_AI Workflow Orchestrator

부기(ChatBot A)와 꼬기(ChatBot B) 간의 완전한 통합 워크플로우를 관리하는
중앙 오케스트레이터.

이 시스템은 이야기 수집부터 최종 멀티미디어 동화 생성까지의
전체 파이프라인을 자동화.
"""

import os
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable

from shared.utils.logging_utils import get_module_logger

# Story Schema Import
from .story_schema import (
    StoryDataSchema, StoryStage, StoryElement, ElementType,
    ChildProfile, ConversationSummary, GeneratedStory, MultimediaAssets
)
from .state_manager import StateManager # 상태 관리자
from .pipeline_manager import PipelineManager # 파이프라인 관리자
from .multimedia_coordinator import MultimediaCoordinator # 멀티미디어 조정자 (Image, Audio 생성)

# 로깅 설정
logger = get_module_logger(__name__)

# 챗봇 모듈 임포트
try:
    from chatbot.models.chat_bot_a import ChatBotA
    from chatbot.models.chat_bot_b import ChatBotB
except ImportError as e:
    logger.warning(f"Chatbot module Import 실패: {e}")

class WorkflowOrchestrator:
    """
    CCB_AI workflow orchestrator
    
    전체 이야기 생성 파이프라인을 관리하고 조정:
    1. 부기(ChatBot A)를 통한 이야기 요소 수집
    2. 데이터 검증 및 변환
    3. 꼬기(ChatBot B)를 통한 완전한 이야기 생성
    4. 멀티미디어 생성 (이미지, 오디오)
    5. 최종 동화 완성 및 배포
    """
    
    def __init__(
        self,
        output_dir: str = "output",
        enable_multimedia: bool = True,
        enable_voice: bool = False,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Workflow orchestrator initialization
        
        Args:
            output_dir: 출력 디렉토리 경로
            enable_multimedia: 멀티미디어 생성 활성화
            enable_voice: 음성 처리 활성화
            config: 추가 설정
        """
        self.output_dir = output_dir
        self.enable_multimedia = enable_multimedia
        self.enable_voice = enable_voice
        self.config = config or {}
        
        # 로깅 설정
        self.logger = get_module_logger(__name__)
        
        # 필수 디렉토리 확인/생성
        self._ensure_output_directories()
        
        # 상태 관리자
        self.state_manager = StateManager(output_dir)
        
        # 파이프라인 관리자
        self.pipeline_manager = PipelineManager()
        
        # 멀티미디어 조정자 (Image, Audio 생성)
        if enable_multimedia:
            self.multimedia_coordinator = MultimediaCoordinator(output_dir)
        else:
            self.multimedia_coordinator = None
        
        # 챗봇 인스턴스
        self.chat_bot_a = None
        self.chat_bot_b = None
        
        # 이벤트 핸들러
        self.event_handlers: Dict[str, List[Callable]] = {
            "stage_changed": [], # 파이프라인 단계 변경 이벤트
            "error_occurred": [], # 오류 발생 이벤트
            "story_completed": [], # 이야기 완료 이벤트
            "progress_updated": [] # 진행 상태 업데이트 이벤트
        }
        
        # 활성 스토리 추적
        self.active_stories: Dict[str, StoryDataSchema] = {} # 활성 스토리 목록 (story_id -> StoryDataSchema)
        
        # 챗봇 초기화
        self.initialize_chatbots()
        
        self.logger.info(f"WorkflowOrchestrator 초기화 완료 (output_dir: {output_dir})")
    
    def _ensure_output_directories(self):
        """출력 디렉토리 구조 확인/생성"""
        import os
        
        directories_to_create = [
            self.output_dir,                                          # output
            os.path.join(self.output_dir, "workflow_states"),         # workflow_states
            os.path.join(self.output_dir, "metadata"),                # metadata  
            os.path.join(self.output_dir, "stories"),                 # stories
            os.path.join(self.output_dir, "temp"),                    # temp
            os.path.join(self.output_dir, "temp", "images"),          # temp/images
            os.path.join(self.output_dir, "temp", "audio"),           # temp/audio
            os.path.join(self.output_dir, "temp", "voice_samples"),   # temp/voice_samples
            os.path.join(self.output_dir, "conversations"),           # conversations
        ]
        
        created_count = 0
        for directory in directories_to_create:
            try:
                if not os.path.exists(directory):
                    os.makedirs(directory, exist_ok=True)
                    self.logger.debug(f"디렉토리 생성: {directory}")
                    created_count += 1
            except Exception as e:
                self.logger.error(f"디렉토리 생성 실패: {directory} - {e}")
                
        if created_count > 0:
            self.logger.info(f"WorkflowOrchestrator: {created_count}개 디렉토리 생성됨")
        else:
            self.logger.debug("WorkflowOrchestrator: 모든 디렉토리 확인됨")
    
    def initialize_chatbots(self):
        """챗봇 인스턴스 초기화"""
        try:
            # VectorDB 초기화 (ChatBotA 필요)
            try:
                from chatbot.data.vector_db.core import VectorDB
                import os
                
                # .env에서 VectorDB 경로 읽기 (통일된 환경변수 사용)
                chroma_base = os.getenv("CHROMA_DB_PATH", "/app/chatbot/data/vector_db")
                vector_db_path = os.path.join(chroma_base, "main")  # main DB 사용
                self.logger.info(f"VectorDB 경로 환경변수: {vector_db_path}")
                
                # VectorDB 초기화 (ChromaDB 경로 포함)
                vector_db = VectorDB(
                    persist_directory=vector_db_path,      # ChromaDB 저장 경로
                    embedding_model="nlpai-lab/KURE-v1",   # 한국어 임베딩 모델 (KURE-v1 : 2025년 기준 가장 성능이 높은 한국어 임베딩 모델)
                    use_hybrid_mode=True,                  # Hybrid mode(Memory+Disk)
                    memory_cache_size=1000,               # 메모리 캐시 크기
                    enable_lfu_cache=True                 # LFU 캐시 정책
                )
                self.logger.info(f"VectorDB 초기화 완료: {vector_db_path}")
            except Exception as e:
                self.logger.warning(f"VectorDB 초기화 실패: {e}, None으로 진행")
                vector_db = None
            
            # ChatBot A (부기) 초기화
            self.chat_bot_a = ChatBotA(
                vector_db_instance=vector_db,      # VectorDB 인스턴스
                token_limit=10000,                 # 토큰 제한
                use_langchain=True,                # LangChain 사용 유무
                legacy_compatibility=True,         # 레거시 호환성
                enhanced_mode=True,                # Enhanced 모드 (RAG)
                enable_performance_tracking=True   # 성능 추적
            )
            
            # ChatBot B (꼬기) 초기화
            self.chat_bot_b = ChatBotB(
                vector_db_path=vector_db_path,  # VectorDB 경로 전달
                collection_name="fairy_tales",
                use_enhanced_generators=True,
                enable_performance_tracking=True
            )
            
            self.logger.info("Chatbot B Instance 초기화 완료")
            
        except Exception as e:
            self.logger.error(f"Chatbot 초기화 실패: {e}")
            raise
    
    def add_event_handler(self, event_type: str, handler: Callable):
        """이벤트 핸들러 추가"""
        if event_type in self.event_handlers:
            self.event_handlers[event_type].append(handler)
    
    def _emit_event(self, event_type: str, data: Dict[str, Any]):
        """이벤트 발생"""
        for handler in self.event_handlers.get(event_type, []):
            try:
                handler(data)
            except Exception as e:
                self.logger.error(f"이벤트 핸들러 실행 실패 ({event_type}): {e}")
    
    async def create_story(
        self,
        child_profile: ChildProfile,
        conversation_data: Optional[Dict[str, Any]] = None,
        story_preferences: Optional[Dict[str, Any]] = None
    ) -> StoryDataSchema:
        """
        완전한 이야기 생성 Workflow 실행
        
        Args:
            child_profile: 아이 프로필 정보
            conversation_data: 기존 대화 데이터 (선택사항)
            story_preferences: 이야기 선호도 설정
            
        Returns:
            완성된 이야기 데이터
        """
        # 새 스토리 스키마 생성
        story_schema = StoryDataSchema()  
        story_schema.child_profile = child_profile # 아이 프로필 정보 설정
        
        story_id = story_schema.metadata.story_id
        self.active_stories[story_id] = story_schema
        
        try:
            self.logger.info(f"이야기 생성 시작: {story_id}")
            
            # 1단계: 이야기 요소 수집 (ChatBot A)
            await self._stage_collect_story_elements(story_schema, conversation_data)
            
            # 2단계: 데이터 검증
            await self._stage_validate_data(story_schema)
            
            # 3단계: 이야기 생성 (ChatBot B)
            await self._stage_generate_story(story_schema, story_preferences)
            
            # 4단계: 멀티미디어 생성
            if self.enable_multimedia:
                await self._stage_generate_multimedia(story_schema)
            
            # 5단계: 최종 완성
            await self._stage_finalize_story(story_schema)
            
            self.logger.info(f"이야기 생성 완료: {story_id}")
            self._emit_event("story_completed", {"story_id": story_id, "story": story_schema})
            
            return story_schema
            
        except Exception as e:
            self.logger.error(f"이야기 생성 실패 ({story_id}): {e}")
            story_schema.add_error("workflow_error", str(e))
            self._emit_event("error_occurred", {"story_id": story_id, "error": str(e)})
            raise
        
        finally:
            # 상태 저장
            await self.state_manager.save_story_state(story_schema)
    
    async def _stage_collect_story_elements(
        self,
        story_schema: StoryDataSchema,
        conversation_data: Optional[Dict[str, Any]] = None
    ):
        """1단계: 이야기 요소 수집"""
        story_schema.update_stage(StoryStage.COLLECTION, "이야기 요소 수집 시작")
        
        try:
            if not self.chat_bot_a:
                self.initialize_chatbots()
            
            self.logger.info(f"대화 데이터 상태: {conversation_data is not None}")
            
            # 기존 대화 데이터가 있으면 분석
            if conversation_data and conversation_data.get("messages"):
                self.logger.info("대화 데이터를 분석하여 이야기 요소 추출 중...")
                elements = await self._analyze_conversation_data(conversation_data)
                for element in elements:
                    story_schema.add_story_element(element)
                
                # 대화 요약 생성
                summary = self._create_conversation_summary(conversation_data)
                story_schema.conversation_summary = summary
                
                self.logger.info(f"대화 분석으로 추출된 이야기 요소: {len(elements)}개")
            
            # 대화 데이터가 없거나 요소가 부족한 경우: 기본 이야기 요소 생성
            current_elements = story_schema.get_all_elements()
            self.logger.info(f"현재 이야기 요소 수: {len(current_elements)}")
            
            if len(current_elements) == 0 or not story_schema.is_ready_for_generation():
                self.logger.info("대화 데이터가 없거나 필수 요소가 부족하여 기본 이야기 요소 생성 중...")
                await self._generate_default_story_elements(story_schema)
                
                # 생성 후 다시 확인
                final_elements = story_schema.get_all_elements()
                self.logger.info(f"기본 요소 생성 후 총 이야기 요소: {len(final_elements)}개")
                
                # 요소별 개수 확인
                for element_type in ElementType:
                    count = len(story_schema.get_elements_by_type(element_type))
                    self.logger.info(f"  - {element_type.value}: {count}개")
            
            self.logger.info(f"이야기 요소 수집 완료: {len(story_schema.get_all_elements())}개 요소")
            
        except Exception as e:
            story_schema.add_error("collection_error", f"이야기 요소 수집 실패: {e}")
            self.logger.error(f"이야기 요소 수집 중 오류: {e}", exc_info=True)
            raise
    
    async def _stage_validate_data(self, story_schema: StoryDataSchema):
        """2단계: 데이터 검증"""
        story_schema.update_stage(StoryStage.VALIDATION, "데이터 검증 시작")
        
        try:
            # 필수 요소 확인
            if not story_schema.is_ready_for_generation():
                missing_elements = []
                required_types = [ElementType.CHARACTER, ElementType.SETTING, ElementType.PROBLEM]
                
                for element_type in required_types:
                    if not story_schema.get_elements_by_type(element_type):
                        missing_elements.append(element_type.value)
                
                self.logger.warning(f"필수 이야기 요소 부족: {missing_elements}")
                self.logger.info("부족한 요소들을 자동으로 생성합니다...")
                
                # 부족한 요소들을 자동 생성
                await self._generate_default_story_elements(story_schema)
                
                # 재검증
                if not story_schema.is_ready_for_generation():
                    # 재생성 후에도 부족하면 오류
                    final_missing = []
                    for element_type in required_types:
                        if not story_schema.get_elements_by_type(element_type):
                            final_missing.append(element_type.value)
                    raise ValueError(f"필수 이야기 요소 생성 실패: {final_missing}")
                
                self.logger.info("부족한 요소들이 성공적으로 생성되었습니다")
            
            # 아이 프로필 검증
            if not story_schema.child_profile:
                raise ValueError("아이 프로필 정보가 필요합니다")
            
            # 연령대별 적절성 검증
            await self._validate_age_appropriateness(story_schema)
            
            self.logger.info("데이터 검증 완료")
            
        except Exception as e:
            story_schema.add_error("validation_error", f"데이터 검증 실패: {e}")
            raise
    
    async def _stage_generate_story(
        self,
        story_schema: StoryDataSchema,
        story_preferences: Optional[Dict[str, Any]] = None
    ):
        """3단계: 이야기 생성"""
        story_schema.update_stage(StoryStage.GENERATION, "이야기 생성 시작")
        
        try:
            if not self.chat_bot_b:
                self.initialize_chatbots()
            
            # 이야기 생성 요청 데이터 준비
            generation_request = self._prepare_generation_request(story_schema, story_preferences)
            
            # ChatBot B를 통한 이야기 생성
            generated_content = await self._generate_story_content(generation_request)
            
            # 생성된 이야기 저장
            generated_story = GeneratedStory(
                content=generated_content["content"],
                chapters=generated_content.get("chapters", []),
                word_count=len(generated_content["content"].split()),
                generation_model="gpt-4o-mini",
                quality_score=generated_content.get("quality_score", 0.8)
            )
            
            story_schema.generated_story = generated_story
            
            self.logger.info(f"이야기 생성 완료: {generated_story.word_count}단어")
            
        except Exception as e:
            story_schema.add_error("generation_error", f"이야기 생성 실패: {e}")
            raise
    
    async def _stage_generate_multimedia(self, story_schema: StoryDataSchema):
        """4단계: 멀티미디어 생성"""
        if not self.multimedia_coordinator:
            return
        
        story_schema.update_stage(StoryStage.MULTIMEDIA, "멀티미디어 생성 시작")
        
        try:
            # 이미지 생성
            images = await self.multimedia_coordinator.generate_images(story_schema)
            
            # 오디오 생성
            audio_files = await self.multimedia_coordinator.generate_audio(story_schema)
            
            # 멀티미디어 자산 저장
            multimedia_assets = MultimediaAssets(
                images=images,
                audio_files=audio_files
            )
            
            story_schema.multimedia_assets = multimedia_assets
            
            self.logger.info(f"멀티미디어 생성 완료: {len(images)}개 이미지, {len(audio_files)}개 오디오")
            
        except Exception as e:
            story_schema.add_error("multimedia_error", f"멀티미디어 생성 실패: {e}")
            # 멀티미디어 생성 실패는 치명적이지 않음
            self.logger.warning(f"멀티미디어 생성 실패, 계속 진행: {e}")
    
    async def _stage_finalize_story(self, story_schema: StoryDataSchema):
        """5단계: 최종 완성"""
        story_schema.update_stage(StoryStage.COMPLETION, "이야기 완성")
        
        try:
            # 최종 파일 저장
            story_dir = os.path.join(self.output_dir, "stories", story_schema.metadata.story_id)
            os.makedirs(story_dir, exist_ok=True)
            
            # 스토리 데이터 저장
            story_file = os.path.join(story_dir, "story_data.json")
            story_schema.save_to_file(story_file)
            
            # 텍스트 파일 저장
            if story_schema.generated_story:
                text_file = os.path.join(story_dir, "story.txt")
                with open(text_file, 'w', encoding='utf-8') as f:
                    f.write(story_schema.generated_story.content)
            
            # 메타데이터 업데이트
            story_schema.metadata.updated_at = datetime.now()
            
            self.logger.info(f"이야기 완성: {story_dir}")
            
        except Exception as e:
            story_schema.add_error("finalization_error", f"이야기 완성 실패: {e}")
            raise
    
    async def _generate_default_story_elements(self, story_schema: StoryDataSchema):
        """아이 프로필을 기반으로 기본 이야기 요소 생성"""
        child_profile = story_schema.child_profile
        
        try:
            self.logger.info(f"기본 이야기 요소 생성 시작 - 아이: {child_profile.name}, 나이: {child_profile.age}")
            
            # 기본 캐릭터 생성 (아이의 이름과 관심사 기반)
            from .story_schema import StoryElement, ElementType
            
            character_content = f"{child_profile.name}이라는 {child_profile.age}살 아이"
            character_element = StoryElement(
                element_type=ElementType.CHARACTER, # 이야기 요소 타입 (캐릭터, 설정, 문제)
                content=character_content, # 이야기 요소 내용
                confidence_score=0.8, # 신뢰도 점수 (0.0 ~ 1.0)
                source_conversation="child_profile_generation" # Source 대화 데이터 추적
            )
            story_schema.add_story_element(character_element) # 이야기 요소 추가
            self.logger.info(f"캐릭터 요소 생성: {character_content}")
            
            # 기본 설정 생성 (관심사 기반)
            if child_profile.interests:
                primary_interest = child_profile.interests[0] if child_profile.interests else "모험" # 관심사 중 첫 번째 항목
                setting_content = f"{primary_interest}와 관련된 신비로운 장소" # 설정 내용
            else:
                setting_content = "마법의 숲" # 기본 설정 내용
                
            setting_element = StoryElement(   # 설정 요소 생성
                element_type=ElementType.SETTING, # 이야기 요소 타입 (캐릭터, 설정, 문제)
                content=setting_content, # 이야기 요소 내용
                confidence_score=0.7, # 신뢰도 점수 (0.0 ~ 1.0)
                source_conversation="child_profile_generation" # Source 대화 데이터 추적
            )
            story_schema.add_story_element(setting_element) # 이야기 요소 추가
            self.logger.info(f"설정 요소 생성: {setting_content}")
            
            # 기본 문제/갈등 생성 (연령대별)
            if child_profile.age <= 6:
                problem_content = "도움이 필요한 친구를 만나게 되는 문제"
            else:
                problem_content = "해결해야 할 수수께끼나 미션"
                
            problem_element = StoryElement(
                element_type=ElementType.PROBLEM, # 이야기 요소 타입 (캐릭터, 설정, 문제)
                content=problem_content, # 이야기 요소 내용
                confidence_score=0.7, # 신뢰도 점수 (0.0 ~ 1.0)
                source_conversation="child_profile_generation" # Source 대화 데이터 추적
            )
            story_schema.add_story_element(problem_element)
            self.logger.info(f"문제 요소 생성: {problem_content}") # 문제 요소 생성 로그 출력
            
            # 생성 후 검증
            total_elements = len(story_schema.get_all_elements()) # 총 이야기 요소 개수
            character_count = len(story_schema.get_elements_by_type(ElementType.CHARACTER)) # 캐릭터 요소 개수
            setting_count = len(story_schema.get_elements_by_type(ElementType.SETTING)) # 설정 요소 개수
            problem_count = len(story_schema.get_elements_by_type(ElementType.PROBLEM)) # 문제 요소 개수
            
            self.logger.info(f"기본 이야기 요소 생성 완료: 총 {total_elements}개") # 기본 이야기 요소 생성 완료 로그 출력
            self.logger.info(f"  - 캐릭터: {character_count}개, 설정: {setting_count}개, 문제: {problem_count}개") # 캐릭터, 설정, 문제 요소 개수 로그 출력
            
            # 준비 상태 재확인
            is_ready = story_schema.is_ready_for_generation() # 이야기 생성 준비 상태 확인
            self.logger.info(f"이야기 생성 준비 상태: {is_ready}") # 이야기 생성 준비 상태 로그 출력
            
        except Exception as e:
            self.logger.error(f"기본 이야기 요소 생성 실패: {e}", exc_info=True) # 기본 이야기 요소 생성 실패 로그 출력
            raise 

    async def _analyze_conversation_data(self, conversation_data: Dict[str, Any]) -> List[StoryElement]:
        """대화 데이터 분석하여 이야기 요소 추출"""
        elements = [] # 이야기 요소 목록
        
        # ChatBot A의 분석 기능 사용
        if self.chat_bot_a:
            try:
                # 대화 내용에서 이야기 요소 추출
                messages = conversation_data.get("messages", []) # 대화 데이터에서 메시지 추출
                
                # 모든 메시지를 하나의 문자열로 합치기
                user_messages = []
                for message in messages:
                    if message.get("role") == "user": # 사용자 메시지만 추출
                        user_messages.append(message.get("content", ""))
                
                if user_messages:
                    # ChatBot A의 개선된 analyze_user_response 메서드 사용
                    combined_input = " ".join(user_messages)
                    
                    # 개선된 분석 - analyze_user_response 사용
                    if hasattr(self.chat_bot_a.story_engine, 'analyze_user_response'):
                        analysis_result = self.chat_bot_a.story_engine.analyze_user_response(
                            user_input=combined_input,
                            openai_client=getattr(self.chat_bot_a, 'openai_client', None)
                        )
                    else:
                        # 폴백: 기존 analyze_input 메서드
                        analysis_result = self.chat_bot_a.story_engine.analyze_input(
                            combined_input, 
                            enhanced_mode=True,
                            age_group=getattr(self.chat_bot_a, 'age_group', 5)
                        )
                    
                    # 분석 결과를 StoryElement로 변환
                    from .story_schema import StoryElement, ElementType
                    
                    # 개선된 분석 결과 처리
                    keywords = analysis_result.get("keywords", [])
                    story_elements_data = analysis_result.get("story_elements", {})
                    interests = analysis_result.get("interests", [])
                    
                    # 구조화된 이야기 요소 우선 처리
                    if story_elements_data:
                        # 캐릭터 추출
                        for character in story_elements_data.get("characters", []):
                            elements.append(StoryElement(
                                element_type=ElementType.CHARACTER,
                                content=character,
                                keywords=[character],
                                confidence_score=0.9,
                                source_conversation="enhanced_analysis"
                            ))
                        
                        # 설정 추출
                        for setting in story_elements_data.get("settings", []):
                            elements.append(StoryElement(
                                element_type=ElementType.SETTING,
                                content=setting,
                                keywords=[setting],
                                confidence_score=0.8,
                                source_conversation="enhanced_analysis"
                            ))
                        
                        # 감정/문제 요소 추출
                        for emotion in story_elements_data.get("emotions", []):
                            elements.append(StoryElement(
                                element_type=ElementType.PROBLEM,
                                content=f"{emotion}과 관련된 이야기",
                                keywords=[emotion],
                                confidence_score=0.7,
                                source_conversation="enhanced_analysis"
                            ))
                        
                        # 물건/도구 요소
                        for obj in story_elements_data.get("objects", []):
                            elements.append(StoryElement(
                                element_type=ElementType.CHARACTER,  # 또는 새로운 타입
                                content=f"특별한 {obj}",
                                keywords=[obj],
                                confidence_score=0.6,
                                source_conversation="enhanced_analysis"
                            ))
                    
                    # 키워드 기반 보완적 추출
                    if keywords:
                        # 개선된 패턴 매칭
                        character_patterns = ["공주", "왕자", "토끼", "강아지", "고양이", "곰", "사자", "친구", "아이", "엄마", "아빠", "요정", "마법사", "용", "유니콘"]
                        setting_patterns = ["숲", "바다", "산", "집", "학교", "공원", "마을", "성", "하늘", "동굴", "정원", "마법나라", "우주"]
                        problem_patterns = ["모험", "여행", "탐험", "문제", "도움", "구하기", "찾기", "걱정", "무서운", "신나는"]
                        
                        # 캐릭터 요소 추출
                        character_keywords = [k for k in keywords if any(char_word in k for char_word in character_patterns)]
                        for keyword in character_keywords:
                            # 중복 체크
                            if not any(elem.content == keyword for elem in elements):
                                elements.append(StoryElement(
                                    element_type=ElementType.CHARACTER,
                                    content=keyword,
                                    keywords=[keyword],
                                    confidence_score=0.75,
                                    source_conversation="keyword_analysis"
                                ))
                        
                        # 설정 요소 추출
                        setting_keywords = [k for k in keywords if any(setting_word in k for setting_word in setting_patterns)]
                        for keyword in setting_keywords:
                            if not any(elem.content == keyword for elem in elements):
                                elements.append(StoryElement(
                                    element_type=ElementType.SETTING,
                                    content=keyword,
                                    keywords=[keyword],
                                    confidence_score=0.7,
                                    source_conversation="keyword_analysis"
                                ))
                        
                        # 문제/모험 요소 추출
                        problem_keywords = [k for k in keywords if any(problem_word in k for problem_word in problem_patterns)]
                        for keyword in problem_keywords:
                            content = f"{keyword}과 관련된 상황"
                            if not any(content in elem.content for elem in elements):
                                elements.append(StoryElement(
                                    element_type=ElementType.PROBLEM,
                                    content=content,
                                    keywords=[keyword],
                                    confidence_score=0.65,
                                    source_conversation="keyword_analysis"
                                ))
                    
                    # 발견된 관심사를 로그에 기록
                    if interests:
                        self.logger.info(f"대화에서 발견된 관심사: {interests}")
                        
                    self.logger.info(f"개선된 분석으로 {len(elements)}개 요소 추출 (키워드: {len(keywords)}개, 구조화 요소: {len(story_elements_data)}개)")
                        
            except Exception as e:
                self.logger.warning(f"대화 데이터 분석 실패: {e}")
        
        return elements # 이야기 요소 목록 반환
    
    def _create_conversation_summary(self, conversation_data: Dict[str, Any]) -> ConversationSummary:
        """대화 요약 생성"""
        messages = conversation_data.get("messages", []) # 대화 데이터에서 메시지 추출
        
        return ConversationSummary(
            total_messages=len(messages), # 메시지 개수
            conversation_duration=conversation_data.get("duration", 0), # 대화 지속 시간
            key_topics=conversation_data.get("topics", []), # 주요 주제
            emotional_tone=conversation_data.get("tone", "neutral"), # 감정 톤
            engagement_level=conversation_data.get("engagement", 0.5), # 참여 수준
            summary_text=conversation_data.get("summary", "") # 요약 텍스트
        )
    
    async def _validate_age_appropriateness(self, story_schema: StoryDataSchema):
        """연령대별 적절성 검증"""
        if not story_schema.child_profile:
            return
        
        age = story_schema.child_profile.age # 아이 나이 정보 가져오기
        
        # 연령대별 검증 로직
        for element in story_schema.get_all_elements():
            if age < 5 and any(word in element.content.lower() for word in ["무서운", "위험한", "슬픈"]): # 연령대별 검증 로직
                element.confidence_score *= 0.5  # 신뢰도 감소
    
    def _prepare_generation_request(
        self,
        story_schema: StoryDataSchema, # 이야기 스키마
        story_preferences: Optional[Dict[str, Any]] = None # 이야기 선호도 정보
    ) -> Dict[str, Any]:
        """이야기 생성 요청 데이터 준비"""
        return {
            "child_profile": story_schema.child_profile.to_dict(), # 아이 프로필 정보
            "story_elements": {
                element_type.value: [element.to_dict() for element in elements] # 이야기 요소 정보
                for element_type, elements in story_schema.story_elements.items() # 이야기 요소 타입과 요소 목록 추출
            },
            "preferences": story_preferences or {}, # 이야기 선호도 정보
            "conversation_summary": story_schema.conversation_summary.to_dict() if story_schema.conversation_summary else None # 대화 요약 정보
        }
    
    async def _generate_story_content(self, generation_request: Dict[str, Any]) -> Dict[str, Any]:
        """ChatBot B를 통한 이야기 내용 생성"""
        self.logger.info(f"_generate_story_content 시작")
        
        if not self.chat_bot_b:
            self.logger.error(f"ChatBot B가 None입니다.")
            raise ValueError("ChatBot B가 초기화되지 않았습니다.")
        
        self.logger.info(f"ChatBot B 상태 정상")
        
        try:
            # ChatBot B 설정: 아이 프로필에서 나이 정보 가져오기
            child_profile = generation_request.get("child_profile", {}) # 아이 프로필 정보 가져오기
            age = child_profile.get("age", 5) # 아이 나이 정보 가져오기
            self.logger.info(f"아이 연령: {age}")
            
            self.chat_bot_b.set_target_age(age) # 아이 나이 정보 설정
            self.logger.info(f"ChatBot B 연령 설정 완료")
            
            # 스토리 개요 설정 (generation_request를 스토리 개요로 변환)
            story_outline = {
                "child_profile": child_profile, # 아이 프로필 정보  
                "story_elements": generation_request.get("story_elements", {}), # 이야기 요소 정보
                "preferences": generation_request.get("preferences", {}), # 이야기 선호도 정보
                "conversation_summary": generation_request.get("conversation_summary") # 대화 요약 정보
            }
            self.chat_bot_b.set_story_outline(story_outline) # 스토리 개요 설정
            self.logger.info(f"ChatBot B 스토리 개요 설정 완료")
            
            # ChatBot B의 기존 메서드 호출 - 텍스트 중심 생성
            self.logger.info(f"ChatBot B generate_text_only 호출 시작...")
            result = await self.chat_bot_b.generate_text_only(
                use_enhanced=True,  # Enhanced 모드 사용
                progress_callback=None # 진행 상태 콜백 함수
            )
            self.logger.info(f"ChatBot B generate_text_only 호출 완료!")
            
            # 결과 포맷을 오케스트레이터 형식에 맞게 변환
            story_data = result.get("story_data", {})
            chapters = story_data.get("chapters", [])
            
            # chapters에서 전체 스토리 텍스트 추출
            full_content = ""
            if chapters:
                for chapter in chapters:
                    # Enhanced TextGenerator의 새로운 구조: narration + dialogues
                    chapter_content = ""
                    
                    # 1. content 또는 chapter_content (legacy)
                    if chapter.get("content"):
                        chapter_content = chapter.get("content")
                    elif chapter.get("chapter_content"):
                        chapter_content = chapter.get("chapter_content")
                    # 2. Enhanced 구조: narration + dialogues
                    else:
                        narration = chapter.get("narration", "")
                        dialogues = chapter.get("dialogues", [])
                        
                        # 내레이션 추가
                        if narration:
                            chapter_content += narration
                        
                        # 대화 추가
                        for dialogue in dialogues:
                            speaker = dialogue.get("speaker", "")
                            text = dialogue.get("text", "")
                            if speaker and text:
                                chapter_content += f"\n{speaker}: \"{text}\""
                    
                    if chapter_content:
                        # 챕터 제목 추가 (있는 경우)
                        chapter_title = chapter.get("chapter_title", "")
                        if chapter_title:
                            full_content += f"**{chapter_title}**\n\n"
                        full_content += chapter_content + "\n\n"
            
            formatted_result = {
                "content": full_content.strip(), # chapters에서 추출한 전체 이야기 내용
                "chapters": chapters, # 이야기 장 정보
                "quality_score": result.get("quality_score", 0.8), # 이야기 품질 점수
                "generation_metadata": result.get("metadata", {}) # 생성 메타데이터
            }
            
            return formatted_result # 결과 반환
            
        except Exception as e:
            self.logger.error(f"ChatBot B 이야기 생성 실패: {e}")
            self.logger.error(f"Exception 타입: {type(e)}")
            import traceback
            self.logger.error(f"Full traceback: {traceback.format_exc()}")
            # 폴백: 기본 이야기 생성
            self.logger.info(f"폴백 스토리 생성으로 진행...")
            return await self._generate_fallback_story(generation_request)
    
    async def _generate_fallback_story(self, generation_request: Dict[str, Any]) -> Dict[str, Any]:
        """폴백 이야기 생성"""
        # 기본적인 이야기 템플릿 사용
        elements = generation_request.get("story_elements", {})
        characters = elements.get("character", []) # 캐릭터 정보 가져오기
        settings = elements.get("setting", []) # 설정 정보 가져오기
        problems = elements.get("problem", []) # 문제 정보 가져오기
        
        character_name = characters[0]["content"] if characters else "주인공" # 캐릭터 이름
        setting_desc = settings[0]["content"] if settings else "마법의 숲" # 설정 설명
        problem_desc = problems[0]["content"] if problems else "모험을 떠나야 했어요" # 문제 설명
        
        # 이야기 내용 생성
        story_content = f"""
        옛날 옛적에 {character_name}이(가) {setting_desc}에 살고 있었어요.
        어느 날 {character_name}은(는) {problem_desc}.
        용기를 내어 문제를 해결한 {character_name}은(는) 행복하게 살았답니다.
        """
        
        return {
            "content": story_content.strip(), # 이야기 내용
            "chapters": [{"title": "시작", "content": story_content.strip()}], # 이야기 장 정보
            "quality_score": 0.6 # 이야기 품질 점수
        }
    
    async def get_story_status(self, story_id: str) -> Optional[Dict[str, Any]]: # 이야기 상태 조회
        """이야기 상태 조회"""
        if story_id in self.active_stories: # 활성 이야기 목록에 있는지 확인
            story = self.active_stories[story_id] # 활성 이야기 목록에서 이야기 찾기
            return {
                "story_id": story_id, # 이야기 ID
                "current_stage": story.current_stage.value, # 현재 단계
                "completion_percentage": story.get_completion_percentage(), # 완료 퍼센트
                "errors": story.errors, # 에러 목록
                "created_at": story.metadata.created_at.isoformat(), # 생성 시간
                "updated_at": story.metadata.updated_at.isoformat() # 업데이트 시간
            }
        
        # 저장된 상태에서 조회
        return await self.state_manager.get_story_status(story_id) 
    
    async def resume_story(self, story_id: str) -> StoryDataSchema: # 중단된 이야기 재개
        """중단된 이야기 재개"""
        story_schema = await self.state_manager.load_story_state(story_id) # 이야기 상태 로드
        if not story_schema: # 이야기 상태가 없으면 에러 발생
            raise ValueError(f"이야기를 찾을 수 없습니다: {story_id}")
        
        self.active_stories[story_id] = story_schema # 활성 이야기 목록에 추가
        
        # 현재 단계에 따라 재개
        if story_schema.current_stage == StoryStage.COLLECTION: # 수집 단계
            await self._stage_collect_story_elements(story_schema)
        elif story_schema.current_stage == StoryStage.VALIDATION: # 검증 단계
            await self._stage_validate_data(story_schema)
        elif story_schema.current_stage == StoryStage.GENERATION: # 생성 단계
            await self._stage_generate_story(story_schema)
        elif story_schema.current_stage == StoryStage.MULTIMEDIA: # 멀티미디어 단계
            await self._stage_generate_multimedia(story_schema)
        
        return story_schema # 이야기 스키마 반환
    
    def get_active_stories(self) -> List[str]: # 활성 이야기 목록 반환
        """활성 이야기 목록 반환"""
        return list(self.active_stories.keys()) # 활성 이야기 목록 반환
    
    async def cancel_story(self, story_id: str): # 이야기 생성 취소
        """이야기 생성 취소"""
        if story_id in self.active_stories: # 활성 이야기 목록에 있는지 확인
            story = self.active_stories[story_id] # 활성 이야기 목록에서 이야기 찾기
            story.add_error("cancelled", "사용자에 의해 취소됨") # 에러 추가
            await self.state_manager.save_story_state(story) # 이야기 상태 저장
            del self.active_stories[story_id] # 활성 이야기 목록에서 이야기 제거
            
            self.logger.info(f"이야기 생성 취소: {story_id}") # 로그 출력