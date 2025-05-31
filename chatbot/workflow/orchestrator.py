"""
CCB_AI Workflow Orchestrator

부기(ChatBot A)와 꼬기(ChatBot B) 간의 완전한 통합 워크플로우를 관리하는
중앙 오케스트레이터입니다.

이 시스템은 이야기 수집부터 최종 멀티미디어 동화 생성까지의
전체 파이프라인을 자동화합니다.
"""

import logging
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
import os

from .story_schema import (
    StoryDataSchema, StoryStage, StoryElement, ElementType,
    ChildProfile, ConversationSummary, GeneratedStory, MultimediaAssets
)
from .state_manager import StateManager
from .pipeline_manager import PipelineManager
from .integration_api import IntegrationAPI
from .multimedia_coordinator import MultimediaCoordinator

# 챗봇 모듈 임포트
try:
    from chatbot.models.chat_bot_a import ChatBotA
    from chatbot.models.chat_bot_b import ChatBotB
except ImportError as e:
    logging.warning(f"챗봇 모듈 임포트 실패: {e}")

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
        workflow orchestrator initialization
        
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
        self.logger = logging.getLogger(__name__)
        
        # 상태 관리자
        self.state_manager = StateManager(output_dir)
        
        # 파이프라인 관리자
        self.pipeline_manager = PipelineManager()
        
        # 통합 API
        self.integration_api = IntegrationAPI()
        
        # 멀티미디어 조정자
        if enable_multimedia:
            self.multimedia_coordinator = MultimediaCoordinator(output_dir)
        else:
            self.multimedia_coordinator = None
        
        # 챗봇 인스턴스
        self.chat_bot_a = None
        self.chat_bot_b = None
        
        # 이벤트 핸들러
        self.event_handlers: Dict[str, List[Callable]] = {
            "stage_changed": [],
            "error_occurred": [],
            "story_completed": [],
            "progress_updated": []
        }
        
        # 현재 활성 스토리
        self.active_stories: Dict[str, StoryDataSchema] = {}
        
        self.logger.info("워크플로우 오케스트레이터 초기화 완료")
    
    def initialize_chatbots(self):
        """챗봇 인스턴스 초기화"""
        try:
            # ChatBot A (부기) 초기화
            self.chat_bot_a = ChatBotA(
                use_langchain=True,
                use_legacy_mode=False
            )
            
            # ChatBot B (꼬기) 초기화
            self.chat_bot_b = ChatBotB()
            
            self.logger.info("챗봇 인스턴스 초기화 완료")
            
        except Exception as e:
            self.logger.error(f"챗봇 초기화 실패: {e}")
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
        완전한 이야기 생성 워크플로우 실행
        
        Args:
            child_profile: 아이 프로필 정보
            conversation_data: 기존 대화 데이터 (선택사항)
            story_preferences: 이야기 선호도 설정
            
        Returns:
            완성된 이야기 데이터
        """
        # 새 스토리 스키마 생성
        story_schema = StoryDataSchema()
        story_schema.child_profile = child_profile
        
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
            
            # 기존 대화 데이터가 있으면 분석
            if conversation_data:
                elements = await self._analyze_conversation_data(conversation_data)
                for element in elements:
                    story_schema.add_story_element(element)
            
            # 대화 요약 생성
            if conversation_data:
                summary = self._create_conversation_summary(conversation_data)
                story_schema.conversation_summary = summary
            
            self.logger.info(f"이야기 요소 수집 완료: {len(story_schema.get_all_elements())}개 요소")
            
        except Exception as e:
            story_schema.add_error("collection_error", f"이야기 요소 수집 실패: {e}")
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
                
                raise ValueError(f"필수 이야기 요소 부족: {missing_elements}")
            
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
                reading_level=story_schema.child_profile.language_level,
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
    
    async def _analyze_conversation_data(self, conversation_data: Dict[str, Any]) -> List[StoryElement]:
        """대화 데이터 분석하여 이야기 요소 추출"""
        elements = []
        
        # ChatBot A의 분석 기능 사용
        if self.chat_bot_a:
            try:
                # 대화 내용에서 이야기 요소 추출
                messages = conversation_data.get("messages", [])
                for message in messages:
                    if message.get("role") == "user":
                        # 사용자 메시지에서 이야기 요소 추출
                        extracted = await self.chat_bot_a.extract_story_elements(message.get("content", ""))
                        elements.extend(extracted)
                        
            except Exception as e:
                self.logger.warning(f"대화 데이터 분석 실패: {e}")
        
        return elements
    
    def _create_conversation_summary(self, conversation_data: Dict[str, Any]) -> ConversationSummary:
        """대화 요약 생성"""
        messages = conversation_data.get("messages", [])
        
        return ConversationSummary(
            total_messages=len(messages),
            conversation_duration=conversation_data.get("duration", 0),
            key_topics=conversation_data.get("topics", []),
            emotional_tone=conversation_data.get("tone", "neutral"),
            engagement_level=conversation_data.get("engagement", 0.5),
            summary_text=conversation_data.get("summary", "")
        )
    
    async def _validate_age_appropriateness(self, story_schema: StoryDataSchema):
        """연령대별 적절성 검증"""
        if not story_schema.child_profile:
            return
        
        age = story_schema.child_profile.age
        
        # 연령대별 검증 로직
        for element in story_schema.get_all_elements():
            if age < 5 and any(word in element.content.lower() for word in ["무서운", "위험한", "슬픈"]):
                element.confidence_score *= 0.5  # 신뢰도 감소
    
    def _prepare_generation_request(
        self,
        story_schema: StoryDataSchema,
        story_preferences: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """이야기 생성 요청 데이터 준비"""
        return {
            "child_profile": story_schema.child_profile.to_dict(),
            "story_elements": {
                element_type.value: [element.to_dict() for element in elements]
                for element_type, elements in story_schema.story_elements.items()
            },
            "preferences": story_preferences or {},
            "conversation_summary": story_schema.conversation_summary.to_dict() if story_schema.conversation_summary else None
        }
    
    async def _generate_story_content(self, generation_request: Dict[str, Any]) -> Dict[str, Any]:
        """ChatBot B를 통한 이야기 내용 생성"""
        if not self.chat_bot_b:
            raise ValueError("ChatBot B가 초기화되지 않았습니다")
        
        try:
            # ChatBot B의 생성 메서드 호출
            result = await self.chat_bot_b.generate_story(generation_request)
            return result
            
        except Exception as e:
            self.logger.error(f"ChatBot B 이야기 생성 실패: {e}")
            # 폴백: 기본 이야기 생성
            return await self._generate_fallback_story(generation_request)
    
    async def _generate_fallback_story(self, generation_request: Dict[str, Any]) -> Dict[str, Any]:
        """폴백 이야기 생성"""
        # 기본적인 이야기 템플릿 사용
        elements = generation_request.get("story_elements", {})
        characters = elements.get("character", [])
        settings = elements.get("setting", [])
        problems = elements.get("problem", [])
        
        character_name = characters[0]["content"] if characters else "주인공"
        setting_desc = settings[0]["content"] if settings else "마법의 숲"
        problem_desc = problems[0]["content"] if problems else "모험을 떠나야 했어요"
        
        story_content = f"""
        옛날 옛적에 {character_name}이(가) {setting_desc}에 살고 있었어요.
        어느 날 {character_name}은(는) {problem_desc}.
        용기를 내어 문제를 해결한 {character_name}은(는) 행복하게 살았답니다.
        """
        
        return {
            "content": story_content.strip(),
            "chapters": [{"title": "시작", "content": story_content.strip()}],
            "quality_score": 0.6
        }
    
    async def get_story_status(self, story_id: str) -> Optional[Dict[str, Any]]:
        """이야기 상태 조회"""
        if story_id in self.active_stories:
            story = self.active_stories[story_id]
            return {
                "story_id": story_id,
                "current_stage": story.current_stage.value,
                "completion_percentage": story.get_completion_percentage(),
                "errors": story.errors,
                "created_at": story.metadata.created_at.isoformat(),
                "updated_at": story.metadata.updated_at.isoformat()
            }
        
        # 저장된 상태에서 조회
        return await self.state_manager.get_story_status(story_id)
    
    async def resume_story(self, story_id: str) -> StoryDataSchema:
        """중단된 이야기 재개"""
        story_schema = await self.state_manager.load_story_state(story_id)
        if not story_schema:
            raise ValueError(f"이야기를 찾을 수 없습니다: {story_id}")
        
        self.active_stories[story_id] = story_schema
        
        # 현재 단계에 따라 재개
        if story_schema.current_stage == StoryStage.COLLECTION:
            await self._stage_collect_story_elements(story_schema)
        elif story_schema.current_stage == StoryStage.VALIDATION:
            await self._stage_validate_data(story_schema)
        elif story_schema.current_stage == StoryStage.GENERATION:
            await self._stage_generate_story(story_schema)
        elif story_schema.current_stage == StoryStage.MULTIMEDIA:
            await self._stage_generate_multimedia(story_schema)
        
        return story_schema
    
    def get_active_stories(self) -> List[str]:
        """활성 이야기 목록 반환"""
        return list(self.active_stories.keys())
    
    async def cancel_story(self, story_id: str):
        """이야기 생성 취소"""
        if story_id in self.active_stories:
            story = self.active_stories[story_id]
            story.add_error("cancelled", "사용자에 의해 취소됨")
            await self.state_manager.save_story_state(story)
            del self.active_stories[story_id]
            
            self.logger.info(f"이야기 생성 취소: {story_id}") 