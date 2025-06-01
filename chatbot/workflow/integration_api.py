"""
CCB_AI Integration API

부기(ChatBot A)와 꼬기(ChatBot B) 간의 통신 및 외부 시스템과의
통합을 위한 RESTful API를 제공.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, TYPE_CHECKING
from datetime import datetime
from enum import Enum
from dataclasses import dataclass
import json

try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    logging.warning("FastAPI not available. API endpoints will not be functional.")

from .story_schema import StoryDataSchema, ChildProfile, AgeGroup, ElementType

# TYPE_CHECKING을 사용하여 순환 임포트 해결
if TYPE_CHECKING:
    from .orchestrator import WorkflowOrchestrator

class APIEndpoints(Enum):
    """API 엔드포인트"""
    CREATE_STORY = "/api/v1/stories"
    GET_STORY = "/api/v1/stories/{story_id}"
    GET_STORY_STATUS = "/api/v1/stories/{story_id}/status"
    LIST_STORIES = "/api/v1/stories"
    CANCEL_STORY = "/api/v1/stories/{story_id}/cancel"
    RESUME_STORY = "/api/v1/stories/{story_id}/resume"
    HEALTH_CHECK = "/api/v1/health"
    STATISTICS = "/api/v1/statistics"

# Pydantic 모델들 (FastAPI가 사용 가능한 경우에만)
if FASTAPI_AVAILABLE:
    class ChildProfileRequest(BaseModel):
        """아이 프로필 요청 모델"""
        name: str = Field(..., description="아이 이름")
        age: int = Field(..., ge=3, le=12, description="아이 나이 (3-12세)")
        interests: List[str] = Field(default=[], description="관심사 목록")
        language_level: str = Field(default="basic", description="언어 수준")
        special_needs: List[str] = Field(default=[], description="특별한 요구사항")
    
    class StoryCreationRequest(BaseModel):
        """이야기 생성 요청 모델"""
        child_profile: ChildProfileRequest
        conversation_data: Optional[Dict[str, Any]] = Field(None, description="기존 대화 데이터")
        story_preferences: Optional[Dict[str, Any]] = Field(None, description="이야기 선호도")
        enable_multimedia: bool = Field(True, description="멀티미디어 생성 활성화")
    
    class StoryResponse(BaseModel):
        """이야기 응답 모델"""
        story_id: str
        status: str
        message: str
        data: Optional[Dict[str, Any]] = None
    
    class StatusResponse(BaseModel):
        """상태 응답 모델"""
        story_id: str
        current_stage: str
        workflow_state: str
        progress_percentage: float
        error_count: int
        created_at: str
        updated_at: str
    
    class HealthResponse(BaseModel):
        """헬스체크 응답 모델"""
        status: str
        timestamp: str
        version: str
        active_stories: int
        total_stories: int

class IntegrationAPI:
    """
    CCB_AI 통합 API
    
    워크플로우 오케스트레이터와 외부 시스템 간의 통신을 담당합니다.
    """
    
    def __init__(self, orchestrator: Optional["WorkflowOrchestrator"] = None):
        """
        통합 API 초기화
        
        Args:
            orchestrator: 워크플로우 오케스트레이터 인스턴스
        """
        self.orchestrator = orchestrator
        self.logger = logging.getLogger(__name__)
        
        # FastAPI 앱 (사용 가능한 경우)
        if FASTAPI_AVAILABLE:
            self.app = FastAPI(
                title="CCB_AI Integration API",
                description="부기와 꼬기 통합 이야기 생성 API",
                version="1.0.0"
            )
            self._setup_routes()
        else:
            self.app = None
            self.logger.warning("FastAPI를 사용할 수 없어 API 엔드포인트가 비활성화됩니다.")
    
    def set_orchestrator(self, orchestrator: "WorkflowOrchestrator"):
        """오케스트레이터 설정"""
        self.orchestrator = orchestrator
    
    def _setup_routes(self):
        """API 라우트 설정"""
        if not FASTAPI_AVAILABLE or not self.app:
            return
        
        @self.app.post(APIEndpoints.CREATE_STORY.value, response_model=StoryResponse)
        async def create_story(request: StoryCreationRequest, background_tasks: BackgroundTasks):
            """새 이야기 생성"""
            try:
                if not self.orchestrator:
                    raise HTTPException(status_code=500, detail="오케스트레이터가 초기화되지 않았습니다")
                
                # 아이 프로필 변환
                age_group = self._determine_age_group(request.child_profile.age)
                child_profile = ChildProfile(
                    name=request.child_profile.name,
                    age=request.child_profile.age,
                    age_group=age_group,
                    interests=request.child_profile.interests,
                    language_level=request.child_profile.language_level,
                    special_needs=request.child_profile.special_needs
                )
                
                # 백그라운드에서 이야기 생성 시작
                background_tasks.add_task(
                    self._create_story_background,
                    child_profile,
                    request.conversation_data,
                    request.story_preferences
                )
                
                # 임시 스토리 ID 생성 (실제로는 오케스트레이터에서 생성됨)
                import uuid
                story_id = str(uuid.uuid4())
                
                return StoryResponse(
                    story_id=story_id,
                    status="started",
                    message="이야기 생성이 시작되었습니다",
                    data={"child_name": child_profile.name}
                )
                
            except Exception as e:
                self.logger.error(f"이야기 생성 요청 실패: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get(APIEndpoints.GET_STORY.value.replace("{story_id}", "{story_id}"), response_model=StoryResponse)
        async def get_story(story_id: str):
            """이야기 조회"""
            try:
                if not self.orchestrator:
                    raise HTTPException(status_code=500, detail="오케스트레이터가 초기화되지 않았습니다")
                
                # 이야기 상태 로드
                story_schema = await self.orchestrator.state_manager.load_story_state(story_id)
                if not story_schema:
                    raise HTTPException(status_code=404, detail="이야기를 찾을 수 없습니다")
                
                return StoryResponse(
                    story_id=story_id,
                    status="found",
                    message="이야기 조회 성공",
                    data=story_schema.to_dict()
                )
                
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"이야기 조회 실패: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get(APIEndpoints.GET_STORY_STATUS.value.replace("{story_id}", "{story_id}"), response_model=StatusResponse)
        async def get_story_status(story_id: str):
            """이야기 상태 조회"""
            try:
                if not self.orchestrator:
                    raise HTTPException(status_code=500, detail="오케스트레이터가 초기화되지 않았습니다")
                
                status = await self.orchestrator.get_story_status(story_id)
                if not status:
                    raise HTTPException(status_code=404, detail="이야기를 찾을 수 없습니다")
                
                return StatusResponse(**status)
                
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"이야기 상태 조회 실패: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get(APIEndpoints.LIST_STORIES.value)
        async def list_stories(active_only: bool = False):
            """이야기 목록 조회"""
            try:
                if not self.orchestrator:
                    raise HTTPException(status_code=500, detail="오케스트레이터가 초기화되지 않았습니다")
                
                if active_only:
                    stories = await self.orchestrator.state_manager.list_active_stories()
                else:
                    stories = await self.orchestrator.state_manager.list_all_stories()
                
                return {
                    "stories": stories,
                    "count": len(stories),
                    "active_only": active_only
                }
                
            except Exception as e:
                self.logger.error(f"이야기 목록 조회 실패: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post(APIEndpoints.CANCEL_STORY.value.replace("{story_id}", "{story_id}"))
        async def cancel_story(story_id: str):
            """이야기 생성 취소"""
            try:
                if not self.orchestrator:
                    raise HTTPException(status_code=500, detail="오케스트레이터가 초기화되지 않았습니다")
                
                await self.orchestrator.cancel_story(story_id)
                
                return {"message": f"이야기 생성이 취소되었습니다: {story_id}"}
                
            except Exception as e:
                self.logger.error(f"이야기 취소 실패: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post(APIEndpoints.RESUME_STORY.value.replace("{story_id}", "{story_id}"))
        async def resume_story(story_id: str, background_tasks: BackgroundTasks):
            """중단된 이야기 재개"""
            try:
                if not self.orchestrator:
                    raise HTTPException(status_code=500, detail="오케스트레이터가 초기화되지 않았습니다")
                
                # 백그라운드에서 이야기 재개
                background_tasks.add_task(self._resume_story_background, story_id)
                
                return {"message": f"이야기 재개가 시작되었습니다: {story_id}"}
                
            except Exception as e:
                self.logger.error(f"이야기 재개 실패: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get(APIEndpoints.HEALTH_CHECK.value, response_model=HealthResponse)
        async def health_check():
            """헬스체크"""
            try:
                active_stories = len(self.orchestrator.get_active_stories()) if self.orchestrator else 0
                
                if self.orchestrator:
                    all_stories = await self.orchestrator.state_manager.list_all_stories()
                    total_stories = len(all_stories)
                else:
                    total_stories = 0
                
                return HealthResponse(
                    status="healthy",
                    timestamp=datetime.now().isoformat(),
                    version="1.0.0",
                    active_stories=active_stories,
                    total_stories=total_stories
                )
                
            except Exception as e:
                self.logger.error(f"헬스체크 실패: {e}")
                return HealthResponse(
                    status="unhealthy",
                    timestamp=datetime.now().isoformat(),
                    version="1.0.0",
                    active_stories=0,
                    total_stories=0
                )
        
        @self.app.get(APIEndpoints.STATISTICS.value)
        async def get_statistics():
            """통계 정보 조회"""
            try:
                if not self.orchestrator:
                    raise HTTPException(status_code=500, detail="오케스트레이터가 초기화되지 않았습니다")
                
                stats = await self.orchestrator.state_manager.get_workflow_statistics()
                return stats
                
            except Exception as e:
                self.logger.error(f"통계 조회 실패: {e}")
                raise HTTPException(status_code=500, detail=str(e))
    
    async def _create_story_background(
        self,
        child_profile: ChildProfile,
        conversation_data: Optional[Dict[str, Any]],
        story_preferences: Optional[Dict[str, Any]]
    ):
        """백그라운드에서 이야기 생성"""
        try:
            if self.orchestrator:
                await self.orchestrator.create_story(
                    child_profile=child_profile,
                    conversation_data=conversation_data,
                    story_preferences=story_preferences
                )
        except Exception as e:
            self.logger.error(f"백그라운드 이야기 생성 실패: {e}")
    
    async def _resume_story_background(self, story_id: str):
        """백그라운드에서 이야기 재개"""
        try:
            if self.orchestrator:
                await self.orchestrator.resume_story(story_id)
        except Exception as e:
            self.logger.error(f"백그라운드 이야기 재개 실패: {e}")
    
    def _determine_age_group(self, age: int) -> AgeGroup:
        """나이에 따른 연령대 결정"""
        if age <= 7:
            return AgeGroup.YOUNG_CHILDREN
        else:
            return AgeGroup.ELEMENTARY
    
    # 비-FastAPI 메서드들 (직접 호출용)
    async def create_story_direct(
        self,
        child_profile: ChildProfile,
        conversation_data: Optional[Dict[str, Any]] = None,
        story_preferences: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """직접 이야기 생성 (FastAPI 없이)"""
        try:
            if not self.orchestrator:
                raise ValueError("오케스트레이터가 초기화되지 않았습니다")
            
            story_schema = await self.orchestrator.create_story(
                child_profile=child_profile,
                conversation_data=conversation_data,
                story_preferences=story_preferences
            )
            
            return {
                "success": True,
                "story_id": story_schema.metadata.story_id,
                "data": story_schema.to_dict()
            }
            
        except Exception as e:
            self.logger.error(f"직접 이야기 생성 실패: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_story_direct(self, story_id: str) -> Dict[str, Any]:
        """직접 이야기 조회 (FastAPI 없이)"""
        try:
            if not self.orchestrator:
                raise ValueError("오케스트레이터가 초기화되지 않았습니다")
            
            story_schema = await self.orchestrator.state_manager.load_story_state(story_id)
            if not story_schema:
                return {
                    "success": False,
                    "error": "이야기를 찾을 수 없습니다"
                }
            
            return {
                "success": True,
                "story_id": story_id,
                "data": story_schema.to_dict()
            }
            
        except Exception as e:
            self.logger.error(f"직접 이야기 조회 실패: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_status_direct(self, story_id: str) -> Dict[str, Any]:
        """직접 상태 조회 (FastAPI 없이)"""
        try:
            if not self.orchestrator:
                raise ValueError("오케스트레이터가 초기화되지 않았습니다")
            
            status = await self.orchestrator.get_story_status(story_id)
            if not status:
                return {
                    "success": False,
                    "error": "이야기를 찾을 수 없습니다"
                }
            
            return {
                "success": True,
                "status": status
            }
            
        except Exception as e:
            self.logger.error(f"직접 상태 조회 실패: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_app(self):
        """FastAPI 앱 인스턴스 반환"""
        return self.app
    
    def is_api_available(self) -> bool:
        """API 사용 가능 여부 확인"""
        return FASTAPI_AVAILABLE and self.app is not None 