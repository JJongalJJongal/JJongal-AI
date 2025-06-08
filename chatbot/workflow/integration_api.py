"""
CCB_AI Integration API

부기(ChatBot A)와 꼬기(ChatBot B) 간의 통신 및 외부 시스템과의
통합을 위한 RESTful API를 제공.
"""

import os
# HuggingFace Tokenizers 병렬 처리 경고 해결
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum
import uuid
from contextlib import asynccontextmanager

try:
    from fastapi import FastAPI, HTTPException, Depends, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
    from pydantic import BaseModel, Field, field_validator
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    logging.warning("FastAPI 라이브러리가 없습니다. API 엔드포인트가 비활성화됩니다.")

from .story_schema import ChildProfile, AgeGroup

class APIEndpoints(Enum):
    """API Endpoints"""
    CREATE_STORY = "/api/v1/stories" # 이야기 생성
    GET_STORY = "/api/v1/stories/{story_id}" # 이야기 조회
    GET_STORY_STATUS = "/api/v1/stories/{story_id}/status" # 이야기 상태 조회
    LIST_STORIES = "/api/v1/stories" # 이야기 목록 조회
    CANCEL_STORY = "/api/v1/stories/{story_id}/cancel" # 이야기 생성 취소
    RESUME_STORY = "/api/v1/stories/{story_id}/resume" # 이야기 재개
    HEALTH_CHECK = "/api/v1/health" # 헬스체크
    STATISTICS = "/api/v1/statistics" # 통계

# Rate limiting 설정
if FASTAPI_AVAILABLE:
    limiter = Limiter(key_func=get_remote_address)
else:
    limiter = None

# 보안 설정
security = HTTPBearer(auto_error=True) if FASTAPI_AVAILABLE else None

# Pydantic 모델들 
if FASTAPI_AVAILABLE:
    class ChildProfileRequest(BaseModel):
        """아이 프로필 요청 모델"""
        name: str = Field(..., min_length=1, max_length=50, description="아이 이름")
        age: int = Field(..., ge=3, le=12, description="아이 나이 (3-12세)")
        interests: List[str] = Field(default=[], max_items=10, description="관심사 목록")
        language_level: str = Field(default="basic", pattern="^(basic|intermediate|advanced)$", description="언어 수준")
        special_needs: List[str] = Field(default=[], max_items=5, description="특별한 요구사항")
        
        @field_validator('interests')
        def validate_interests(cls, v):
            for interest in v:
                if len(interest) > 30:
                    raise ValueError('각 관심사는 30자를 초과할 수 없습니다')
            return v
    
    class StoryCreationRequest(BaseModel):
        """이야기 생성 요청 모델"""
        child_profile: ChildProfileRequest
        conversation_data: Optional[Dict[str, Any]] = Field(None, description="기존 대화 데이터")
        story_preferences: Optional[Dict[str, Any]] = Field(None, description="이야기 선호도")
        enable_multimedia: bool = Field(True, description="멀티미디어 생성 활성화")
        
        class Config:
            str_max_length = 10000  # 전체 요청 크기 제한
    
    class StandardResponse(BaseModel):
        """표준 응답 모델"""
        success: bool = Field(..., description="성공 여부")
        message: str = Field(..., description="응답 메시지")
        data: Optional[Dict[str, Any]] = Field(None, description="응답 데이터")
        error_code: Optional[str] = Field(None, description="에러 코드")
    
    class StoryResponse(StandardResponse):
        """이야기 응답 모델"""
        story_id: Optional[str] = Field(None, description="이야기 ID")
    
    class StatusResponse(BaseModel):
        """상태 응답 모델"""
        story_id: str
        current_stage: str
        workflow_state: str
        progress_percentage: float = Field(..., ge=0, le=100)
        error_count: int = Field(..., ge=0)
        created_at: str
        updated_at: str
        errors: Optional[List[str]] = Field(default=[], description="발생한 오류들")
    
    class HealthResponse(BaseModel):
        """헬스체크 응답 모델"""
        # 헬스체크란 서비스의 상태 확인 의미.
        status: str
        timestamp: str
        version: str
        active_stories: int
        total_stories: int

# AuthProcessor 인스턴스 생성
from chatbot.models.voice_ws.processors.auth_processor import AuthProcessor
_auth_processor = AuthProcessor()

# 인증 검증 함수
async def verify_auth(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    """API 인증 검증 - AuthProcessor를 활용한 실제 토큰 검증"""
    if not credentials:
        # 토큰이 없는 경우 401 반환 (실제 서비스에서는 인증 필수)
        raise HTTPException(
            status_code=401, 
            detail="인증 토큰이 필요합니다. Authorization 헤더에 'Bearer <token>'을 포함해주세요."
        )
    
    # AuthProcessor를 통한 토큰 검증
    token = credentials.credentials
    is_valid = _auth_processor.validate_token(token)
    
    if not is_valid:
        raise HTTPException(
            status_code=401, 
            detail="유효하지 않거나 만료된 토큰입니다."
        )
    
    # 토큰이 유효한 경우 사용자 정보 반환
    if token == "development_token":
        return {"user_id": "development_user", "token_type": "development"}
    else:
        # JWT 토큰인 경우 페이로드 디코딩하여 사용자 정보 추출
        try:
            import jwt
            payload = jwt.decode(token, _auth_processor.JWT_SECRET_KEY, algorithms=[_auth_processor.JWT_ALGORITHM])
            return {"user_id": payload.get("sub", "unknown"), "token_type": "jwt", "payload": payload}
        except:
            # JWT 디코딩 실패해도 검증은 통과했으므로 기본 정보 반환
            return {"user_id": "authenticated_user", "token_type": "unknown"}

class IntegrationAPI:
    """
    CCB_AI 통합 API
    
    워크플로우 오케스트레이터와 외부 시스템 간의 통신을 담당합니다.
    """
    
    def __init__(self, orchestrator: Optional[Any] = None):
        """
        통합 API 초기화
        
        Args:
            orchestrator: 워크플로우 오케스트레이터 인스턴스
        """
        self.orchestrator = orchestrator
        self.logger = logging.getLogger(__name__)
        self.story_states: Dict[str, Dict[str, Any]] = {}  # 스토리 상태 캐시
        
        # FastAPI 앱 (사용 가능한 경우)
        if FASTAPI_AVAILABLE:
            @asynccontextmanager
            async def lifespan(app: FastAPI):
                # 시작 시 초기화
                self.logger.info("Integration API 시작")
                yield
                # 종료 시 정리
                self.logger.info("Integration API 종료")
                
            self.app = FastAPI(
                title="CCB_AI Integration API",
                description="부기와 꼬기 통합 이야기 생성 API",
                version="1.0.0",
                lifespan=lifespan
            )
            
            # CORS 설정
            self.app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],  # 실제 운영에서는 특정 도메인으로 제한
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )
            
            # Rate limiting 설정
            if limiter:
                self.app.state.limiter = limiter
                self.app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
            
            self._setup_routes()
        else:
            self.app = None
            self.logger.warning("FastAPI를 사용할 수 없어 API 엔드포인트가 비활성화됩니다.")
    
    def set_orchestrator(self, orchestrator: Any):
        """오케스트레이터 설정"""
        self.orchestrator = orchestrator
    
    def _setup_routes(self):
        """API 라우트 설정"""
        if not FASTAPI_AVAILABLE or not self.app:
            return
        
        @self.app.post(APIEndpoints.CREATE_STORY.value, response_model=StoryResponse)
        @limiter.limit("5/minute") if limiter else lambda x: x
        async def create_story(
            request: Request,
            story_request: StoryCreationRequest, 
            auth: dict = Depends(verify_auth)
        ):
            """새 이야기 생성"""
            try:
                # Try to initialize orchestrator if not already done
                global orchestrator_initialized
                if not self.orchestrator and not orchestrator_initialized:
                    orchestrator_initialized = init_orchestrator()
                
                if not self.orchestrator:
                    return StoryResponse(
                        success=False,
                        message="오케스트레이터가 초기화되지 않았습니다",
                        error_code="ORCHESTRATOR_NOT_INITIALIZED"
                    )
                
                # 아이 프로필 변환
                age_group = self._determine_age_group(story_request.child_profile.age)
                child_profile = ChildProfile(
                    name=story_request.child_profile.name,
                    age=story_request.child_profile.age,
                    age_group=age_group,
                    interests=story_request.child_profile.interests,
                    language_level=story_request.child_profile.language_level,
                    special_needs=story_request.child_profile.special_needs
                )
                
                # 오케스트레이터에서 story_id 먼저 생성
                story_id = await self._create_story_with_id(
                    child_profile,
                    story_request.conversation_data,
                    story_request.story_preferences
                )
                
                return StoryResponse(
                    success=True,
                    story_id=story_id,
                    message="이야기 생성이 시작되었습니다",
                    data={
                        "child_name": child_profile.name,
                        "estimated_completion_time": "3-5분"
                    }
                )
                
            except Exception as e:
                self.logger.error(f"이야기 생성 요청 실패: {e}", exc_info=True)
                return StoryResponse(
                    success=False,
                    message=f"이야기 생성 중 오류가 발생했습니다: {str(e)}",
                    error_code="STORY_CREATION_FAILED"
                )
        
        @self.app.get(APIEndpoints.GET_STORY.value.replace("{story_id}", "{story_id}"), response_model=StandardResponse)
        async def get_story(story_id: str, auth: dict = Depends(verify_auth)):
            """이야기 조회"""
            try:
                if not self.orchestrator:
                    return StandardResponse(
                        success=False,
                        message="오케스트레이터가 초기화되지 않았습니다",
                        error_code="ORCHESTRATOR_NOT_INITIALIZED"
                    )
                
                # 이야기 상태 로드
                story_schema = await self.orchestrator.state_manager.load_story_state(story_id)
                if not story_schema:
                    return StandardResponse(
                        success=False,
                        message="이야기를 찾을 수 없습니다",
                        error_code="STORY_NOT_FOUND"
                    )
                
                return StandardResponse(
                    success=True,
                    message="이야기 조회 성공",
                    data=story_schema.to_dict()
                )
                
            except Exception as e:
                self.logger.error(f"이야기 조회 실패: {e}", exc_info=True)
                return StandardResponse(
                    success=False,
                    message=f"이야기 조회 중 오류가 발생했습니다: {str(e)}",
                    error_code="STORY_RETRIEVAL_FAILED"
                )
        
        @self.app.get(APIEndpoints.GET_STORY_STATUS.value.replace("{story_id}", "{story_id}"), response_model=StatusResponse)
        async def get_story_status(story_id: str, auth: dict = Depends(verify_auth)):
            """이야기 상태 조회"""
            try:
                if not self.orchestrator:
                    raise HTTPException(status_code=500, detail="오케스트레이터가 초기화되지 않았습니다")
                
                status = await self.orchestrator.get_story_status(story_id)
                if not status:
                    raise HTTPException(status_code=404, detail="이야기를 찾을 수 없습니다")
                
                # 타입 안전성을 위한 검증
                required_fields = ['story_id', 'current_stage', 'workflow_state', 'progress_percentage', 'error_count', 'created_at', 'updated_at']
                for field in required_fields:
                    if field not in status:
                        status[field] = "unknown" if field in ['current_stage', 'workflow_state'] else 0
                
                return StatusResponse(**status)
                
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"이야기 상태 조회 실패: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=f"상태 조회 중 오류가 발생했습니다: {str(e)}")
        
        @self.app.get(APIEndpoints.LIST_STORIES.value, response_model=StandardResponse)
        async def list_stories(active_only: bool = False, auth: dict = Depends(verify_auth)):
            """이야기 목록 조회"""
            try:
                if not self.orchestrator:
                    return StandardResponse(
                        success=False,
                        message="오케스트레이터가 초기화되지 않았습니다",
                        error_code="ORCHESTRATOR_NOT_INITIALIZED"
                    )
                
                if active_only:
                    stories = await self.orchestrator.state_manager.list_active_stories()
                else:
                    stories = await self.orchestrator.state_manager.list_all_stories()
                
                return StandardResponse(
                    success=True,
                    message="이야기 목록 조회 성공",
                    data={
                        "stories": stories,
                        "count": len(stories),
                        "active_only": active_only
                    }
                )
                
            except Exception as e:
                self.logger.error(f"이야기 목록 조회 실패: {e}", exc_info=True)
                return StandardResponse(
                    success=False,
                    message=f"목록 조회 중 오류가 발생했습니다: {str(e)}",
                    error_code="LIST_RETRIEVAL_FAILED"
                )
        
        @self.app.post(APIEndpoints.CANCEL_STORY.value.replace("{story_id}", "{story_id}"), response_model=StandardResponse)
        async def cancel_story(story_id: str, auth: dict = Depends(verify_auth)):
            """이야기 생성 취소"""
            try:
                if not self.orchestrator:
                    return StandardResponse(
                        success=False,
                        message="오케스트레이터가 초기화되지 않았습니다",
                        error_code="ORCHESTRATOR_NOT_INITIALIZED"
                    )
                
                await self.orchestrator.cancel_story(story_id)
                
                return StandardResponse(
                    success=True,
                    message=f"이야기 생성이 취소되었습니다",
                    data={"story_id": story_id}
                )
                
            except Exception as e:
                self.logger.error(f"이야기 취소 실패: {e}", exc_info=True)
                return StandardResponse(
                    success=False,
                    message=f"취소 중 오류가 발생했습니다: {str(e)}",
                    error_code="CANCELLATION_FAILED"
                )
        
        @self.app.post(APIEndpoints.RESUME_STORY.value.replace("{story_id}", "{story_id}"), response_model=StandardResponse)
        async def resume_story(story_id: str, auth: dict = Depends(verify_auth)):
            """중단된 이야기 재개"""
            try:
                if not self.orchestrator:
                    return StandardResponse(
                        success=False,
                        message="오케스트레이터가 초기화되지 않았습니다",
                        error_code="ORCHESTRATOR_NOT_INITIALIZED"
                    )
                
                # 동기적으로 재개 상태만 확인하고 비동기로 실행
                resume_task = asyncio.create_task(self.orchestrator.resume_story(story_id))
                
                return StandardResponse(
                    success=True,
                    message=f"이야기 재개가 시작되었습니다",
                    data={"story_id": story_id}
                )
                
            except Exception as e:
                self.logger.error(f"이야기 재개 실패: {e}", exc_info=True)
                return StandardResponse(
                    success=False,
                    message=f"재개 중 오류가 발생했습니다: {str(e)}",
                    error_code="RESUME_FAILED"
                )
        
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
                self.logger.error(f"헬스체크 실패: {e}", exc_info=True)
                return HealthResponse(
                    status="unhealthy",
                    timestamp=datetime.now().isoformat(),
                    version="1.0.0",
                    active_stories=0,
                    total_stories=0
                )
        
        @self.app.get(APIEndpoints.STATISTICS.value, response_model=StandardResponse)
        async def get_statistics(auth: dict = Depends(verify_auth)):
            """통계 정보 조회"""
            try:
                if not self.orchestrator:
                    return StandardResponse(
                        success=False,
                        message="오케스트레이터가 초기화되지 않았습니다",
                        error_code="ORCHESTRATOR_NOT_INITIALIZED"
                    )
                
                stats = await self.orchestrator.state_manager.get_workflow_statistics()
                return StandardResponse(
                    success=True,
                    message="통계 조회 성공",
                    data=stats
                )
                
            except Exception as e:
                self.logger.error(f"통계 조회 실패: {e}", exc_info=True)
                return StandardResponse(
                    success=False,
                    message=f"통계 조회 중 오류가 발생했습니다: {str(e)}",
                    error_code="STATISTICS_FAILED"
                )
    
    async def _create_story_with_id(
        self,
        child_profile: ChildProfile,
        conversation_data: Optional[Dict[str, Any]],
        story_preferences: Optional[Dict[str, Any]]
    ) -> str:
        """Story ID를 먼저 생성하고 백그라운드에서 실행"""
        try:
            # Story ID 먼저 생성
            story_id = str(uuid.uuid4())
            self.logger.info(f"새 스토리 ID 생성: {story_id}")
            
            # 상태 초기화
            self.story_states[story_id] = {
                "status": "initializing",
                "created_at": datetime.now().isoformat(),
                "child_profile": child_profile.to_dict() if hasattr(child_profile, 'to_dict') else child_profile.__dict__
            }
            self.logger.info(f"스토리 상태 초기화 완료: {story_id}")
            
            # 백그라운드에서 실제 생성 시작
            self.logger.info(f"백그라운드 태스크 생성 시작: {story_id}")
            task = asyncio.create_task(self._create_story_background(
                story_id,
                child_profile,
                conversation_data,
                story_preferences
            ))
            self.logger.info(f"백그라운드 태스크 생성 완료: {story_id}, task: {task}")
            
            return story_id
            
        except Exception as e:
            self.logger.error(f"Story ID 생성 실패: {e}", exc_info=True)
            raise
    
    async def _create_story_background(
        self,
        story_id: str,
        child_profile: ChildProfile,
        conversation_data: Optional[Dict[str, Any]],
        story_preferences: Optional[Dict[str, Any]]
    ):
        """Background에서 이야기 생성"""
        self.logger.info(f"Background 이야기 생성 시작 (ID: {story_id})")
        
        try:
            # 상태 업데이트
            if story_id in self.story_states:
                self.story_states[story_id]["status"] = "in_progress"
                self.logger.info(f"상태 업데이트 완료: {story_id} -> in_progress")
            else:
                self.logger.warning(f"Story state not found for {story_id}")
            
            self.logger.info(f"Orchestrator 상태: {self.orchestrator is not None}")
            
            if self.orchestrator:
                self.logger.info(f"이야기 생성 호출 시작 (ID: {story_id})")
                # 실제 이야기 생성
                story_schema = await self.orchestrator.create_story(
                    child_profile=child_profile,
                    conversation_data=conversation_data,
                    story_preferences=story_preferences
                )
                
                self.logger.info(f"이야기 생성 완료 (ID: {story_id})")
                
                # 성공 상태 업데이트
                self.story_states[story_id].update({
                    "status": "completed",
                    "completed_at": datetime.now().isoformat(),
                    "story_data": story_schema.to_dict() if story_schema else None
                })
                
                self.logger.info(f"Background 이야기 생성 성공 (ID: {story_id})")
            else:
                self.logger.error(f"Orchestrator가 초기화되지 않음 (ID: {story_id})")
                
        except Exception as e:
            self.logger.error(f"Background 이야기 생성 실패 (ID: {story_id}): {e}", exc_info=True)
            # 실패 상태 업데이트
            if story_id in self.story_states:
                self.story_states[story_id].update({
                    "status": "failed",
                    "error": str(e),
                    "failed_at": datetime.now().isoformat()
                })
            else:
                self.logger.error(f"스토리 상태 업데이트 실패 - story_id {story_id}")
    
    def _determine_age_group(self, age: int) -> AgeGroup:
        """나이에 따른 연령대 결정"""
        if age <= 7:
            return AgeGroup.YOUNG_CHILDREN
        else:
            return AgeGroup.ELEMENTARY

# 전역 변수 설정 -> 지연 Import (겹치는 Import 방지)
# uvicorn 시작 시 찾을 변수
# 오케스트레이터 초기화 여부 확인
if FASTAPI_AVAILABLE:
    # Import orchestrator
    try:
        from .orchestrator import WorkflowOrchestrator
        ORCHESTRATOR_AVAILABLE = True
    except ImportError as e:
        logging.warning(f"WorkflowOrchestrator를 가져올 수 없습니다: {e}")
        ORCHESTRATOR_AVAILABLE = False
    
    # 통합 API 인스턴스 생성
    integration_api = IntegrationAPI()
    
    # Orchestrator 초기화 함수
    def init_orchestrator():
        if ORCHESTRATOR_AVAILABLE:
            try:
                orchestrator = WorkflowOrchestrator(
                    output_dir="/app/output", # 출력 디렉토리 경로
                    enable_multimedia=True,
                    enable_voice=False
                )
                # 챗봇 초기화
                orchestrator.initialize_chatbots()
                
                # 통합 API에 오케스트레이터 설정
                integration_api.set_orchestrator(orchestrator)
                logging.info("WorkflowOrchestrator 초기화 및 연결 완료") # 로그 출력
                return True
            except Exception as e:
                logging.warning(f"WorkflowOrchestrator 초기화 실패: {e}")
                logging.info("API는 오케스트레이터 없이 실행됩니다") # 로그 출력
                return False
        return False
    
    # 첫 번째 API 호출 시 Orchestrator 초기화
    orchestrator_initialized = False
    
    # uvicorn 시작 시 찾을 변수
    app = integration_api.app
else:
    # FastAPI 사용 불가 시 대체해야 할 변수
    app = None
    integration_api = None

# 로그 레벨 설정 (애플리케이션 레벨에서 설정해야 함 - uvicorn은 자체 로거만 설정)
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True  # 기존 설정을 덮어쓰기
) 