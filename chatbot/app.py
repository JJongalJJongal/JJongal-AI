"""
CCB_AI 통합 FastAPI 애플리케이션

WebSocket 음성 인터페이스와 스토리 생성 API를 통합한 메인 서버입니다.
"""
import asyncio
import os
import sys
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI, WebSocket, HTTPException, Response, Query, status, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)
load_dotenv(os.path.join(project_root, '.env'))

# 로깅
from shared.utils.logging_utils import get_module_logger

# Voice WebSocket 컴포넌트
from chatbot.models.voice_ws.core.connection_engine import ConnectionEngine
from chatbot.models.voice_ws.processors.auth_processor import AuthProcessor
from chatbot.models.voice_ws.processors.audio_processor import AudioProcessor 
from chatbot.models.voice_ws.handlers.audio_handler import handle_audio_websocket
from chatbot.models.voice_ws.handlers.story_handler import handle_story_generation_websocket
from chatbot.data.vector_db.core import VectorDB

# Integration API 컴포넌트
from chatbot.workflow.orchestrator import WorkflowOrchestrator
from chatbot.workflow.story_schema import ChildProfile, AgeGroup

# Integration API 모델들
from chatbot.workflow.integration_api import (
    StoryCreationRequest, StoryResponse, StandardResponse, HealthResponse,
    verify_auth
)

logger = get_module_logger(__name__)
logger.info("=== 🚀 CHATBOT.APP.PY 모듈이 로드되었습니다! ===")

# 전역 컴포넌트
connection_engine = ConnectionEngine()
auth_processor = AuthProcessor()
audio_processor = AudioProcessor()
orchestrator = None

@asynccontextmanager
async def lifespan_manager(app: FastAPI):
    """서비스 생명주기 관리"""
    global orchestrator
    
    # 시작 시 초기화
    logger.info("꼬꼬북 AI 시스템 시작 중...")
    logger.info(f"작업 디렉토리: {os.getcwd()}")
    logger.info(f"Python 버전: {sys.version.split()[0]}")
    
    # VectorDB 사전 로드
    logger.info("설치된 패키지 확인...")
    try:
        app.state.vector_db = VectorDB()
        logger.info("VectorDB 사전 로드 완료")
    except Exception as e:
        logger.error(f"VectorDB 사전 로드 실패: {e}")
        app.state.vector_db = None
    
    # 워크플로우 시스템 초기화
    try:
        logger.info("워크플로우 시스템 초기화 중...")
        
        # 출력 디렉토리 설정
        output_dir = os.getenv("MULTIMEDIA_OUTPUT_DIR", "/app/output")
        
        # WorkflowOrchestrator 초기화 (내부에서 StateManager, PipelineManager, MultimediaCoordinator 생성)
        orchestrator = WorkflowOrchestrator(
            output_dir=output_dir,
            enable_multimedia=os.getenv("ENABLE_MULTIMEDIA", "true").lower() == "true"
        )
        
        logger.info("워크플로우 시스템 초기화 완료")
        
    except Exception as e:
        logger.error(f"워크플로우 시스템 초기화 실패: {e}")
        orchestrator = None
    
    logger.info("FastAPI 서버 시작 중... (포트: 8000)")
    
    # WebSocket 정리 태스크 시작
    asyncio.create_task(connection_engine.cleanup_inactive_clients())
    
    yield
    
    # 종료 시 정리
    logger.info("서비스 종료 중...")
    connection_engine.set_shutdown_event()
    await connection_engine.close_all_connections()
    logger.info("서비스 종료 완료")

# FastAPI 애플리케이션 생성
app = FastAPI(
    title="꼬꼬북 AI 통합 서버",
    description="아동 음성 인터페이스 및 동화 생성 통합 API",
    version="1.0.0",
    lifespan=lifespan_manager
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8080", "*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# 전역 오류 처리
@app.middleware("http")
async def catch_exceptions_middleware(request, call_next):
    """전역 오류 처리 미들웨어"""
    try:
        return await call_next(request)
    except Exception as e:
        logger.error(f"처리되지 않은 서버 오류: {e}")
        return Response(
            content='{"detail": "서버 내부 오류가 발생했습니다"}',
            status_code=500,
            media_type="application/json"
        )

# ===========================================
# WebSocket 엔드포인트
# ===========================================

@app.websocket("/ws/audio")
async def audio_endpoint(
    websocket: WebSocket,
    child_name: str = Query(None),
    age: int = Query(None),
    interests: Optional[str] = Query(None),
):
    """WebSocket 오디오 처리 엔드포인트"""
    if not await auth_processor.validate_connection(websocket):
        return
        
    await handle_audio_websocket(
        websocket,
        child_name,
        age,
        interests,
        connection_engine=connection_engine,
        audio_processor=audio_processor
    )

@app.websocket("/ws/story_generation")
async def story_generation_endpoint(
    websocket: WebSocket,
    child_name: str = Query(None),
    age: int = Query(None),
    interests: Optional[str] = Query(None),
    token: Optional[str] = Query(None)
):
    """WebSocket 스토리 생성 엔드포인트"""
    if not await auth_processor.validate_connection(websocket):
        return
        
    await handle_story_generation_websocket(
        websocket,
        child_name,
        age,
        interests,
        token
    )

# ===========================================
# HTTP API 엔드포인트
# ===========================================

@app.get("/health")
async def health_check():
    """서버 상태 확인 엔드포인트"""
    if audio_processor.whisper_model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Whisper 모델이 초기화되지 않았습니다"
        )
    return {
        "status": "online", 
        "whisper_model": "loaded",
        "orchestrator": orchestrator is not None
    }

# 헬퍼 함수들
async def _create_story_with_orchestrator(
    child_profile: ChildProfile,
    conversation_data: Optional[dict],
    story_preferences: Optional[dict]
) -> str:
    """Orchestrator를 통해 이야기 생성하고 실제 story_id 반환"""
    try:
        if not orchestrator:
            raise RuntimeError("오케스트레이터가 초기화되지 않았습니다")
        
        logger.info("이야기 생성 시작")
        logger.info(f"아이 프로필: {child_profile.name}, 나이: {child_profile.age}")
        logger.info(f"관심사: {child_profile.interests}")
        
        # Orchestrator가 story_id를 생성하고 실제 이야기 생성을 백그라운드에서 시작
        logger.info("오케스트레이터 create_story 호출 중...")
        story_schema = await orchestrator.create_story(
            child_profile=child_profile,
            conversation_data=conversation_data,
            story_preferences=story_preferences
        )
        
        logger.info(f"오케스트레이터 create_story 반환됨: {story_schema is not None}")
        if story_schema:
            logger.info(f"스토리 스키마 단계: {story_schema.current_stage}")
            logger.info(f"생성된 스토리 내용 길이: {len(story_schema.generated_story.content) if story_schema.generated_story else 'None'}")
        
        # Orchestrator가 생성한 실제 story_id 반환
        actual_story_id = story_schema.metadata.story_id
        logger.info(f"이야기 생성 완료: {actual_story_id}")
        
        return actual_story_id
        
    except Exception as e:
        logger.error(f"이야기 생성 실패: {e}", exc_info=True)
        raise

def _determine_age_group(age: int) -> AgeGroup:
    """나이에 따른 연령대 결정"""
    if age <= 7:
        return AgeGroup.YOUNG_CHILDREN
    else:
        return AgeGroup.ELEMENTARY

# ===========================================
# 스토리 생성 API 엔드포인트
# ===========================================

@app.post("/api/v1/stories", response_model=StoryResponse)
async def create_story(
    request: Request,
    story_request: StoryCreationRequest, 
    auth: dict = Depends(verify_auth)
):
    """새 이야기 생성"""
    print("🔥🔥🔥 PRINT: CREATE_STORY 함수 호출됨!!! 🔥🔥🔥")
    logger.info("🔥🔥🔥 LOGGER: CREATE_STORY 함수 호출됨!!! 🔥🔥🔥")
    logger.info("=== 스토리 생성 API 호출됨 ===")
    logger.info(f"요청 데이터: 아이 이름={story_request.child_profile.name}, 나이={story_request.child_profile.age}")
    
    try:
        logger.info("오케스트레이터 상태 확인 중...")
        if not orchestrator:
            logger.error("오케스트레이터가 None입니다!")
            return StoryResponse(
                success=False,
                message="오케스트레이터가 초기화되지 않았습니다",
                error_code="ORCHESTRATOR_NOT_INITIALIZED"
            )
        
        logger.info("오케스트레이터 정상 확인됨. 아이 프로필 변환 중...")
        
        # 아이 프로필 변환
        age_group = _determine_age_group(story_request.child_profile.age)
        child_profile = ChildProfile(
            name=story_request.child_profile.name,
            age=story_request.child_profile.age,
            age_group=age_group,
            interests=story_request.child_profile.interests,
            language_level=story_request.child_profile.language_level,
            special_needs=story_request.child_profile.special_needs
        )
        
        logger.info(f"아이 프로필 변환 완료: {child_profile.name}, 연령대: {age_group}")
        
        # 오케스트레이터에서 story_id 먼저 생성
        logger.info("_create_story_with_orchestrator 호출 중...")
        story_id = await _create_story_with_orchestrator(
            child_profile,
            story_request.conversation_data,
            story_request.story_preferences
        )
        
        logger.info(f"스토리 생성 완료! Story ID: {story_id}")
        
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
        logger.error(f"이야기 생성 요청 실패: {e}", exc_info=True)
        return StoryResponse(
            success=False,
            message=f"이야기 생성 중 오류가 발생했습니다: {str(e)}",
            error_code="STORY_CREATION_FAILED"
        )

@app.get("/api/v1/stories/{story_id}/status", response_model=StandardResponse)
async def get_story_status(story_id: str, auth: dict = Depends(verify_auth)):
    """이야기 상태 조회"""
    try:
        if not orchestrator:
            return StandardResponse(
                success=False,
                message="오케스트레이터가 초기화되지 않았습니다",
                error_code="ORCHESTRATOR_NOT_INITIALIZED"
            )
        
        # 이야기 상태 조회
        status = await orchestrator.get_story_status(story_id)
        if not status:
            return StandardResponse(
                success=False,
                message="이야기를 찾을 수 없습니다",
                error_code="STORY_NOT_FOUND"
            )
        
        return StandardResponse(
            success=True,
            message="이야기 상태 조회 성공",
            data=status
        )
        
    except Exception as e:
        logger.error(f"이야기 상태 조회 실패: {e}", exc_info=True)
        return StandardResponse(
            success=False,
            message=f"상태 조회 중 오류가 발생했습니다: {str(e)}",
            error_code="STATUS_RETRIEVAL_FAILED"
        )

@app.get("/api/v1/stories/{story_id}", response_model=StandardResponse)
async def get_story(story_id: str, auth: dict = Depends(verify_auth)):
    """이야기 조회"""
    try:
        if not orchestrator:
            return StandardResponse(
                success=False,
                message="오케스트레이터가 초기화되지 않았습니다",
                error_code="ORCHESTRATOR_NOT_INITIALIZED"
            )
        
        # 이야기 상태 로드
        story_schema = await orchestrator.state_manager.load_story_state(story_id)
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
        logger.error(f"이야기 조회 실패: {e}", exc_info=True)
        return StandardResponse(
            success=False,
            message=f"이야기 조회 중 오류가 발생했습니다: {str(e)}",
            error_code="STORY_RETRIEVAL_FAILED"
        )

@app.get("/api/v1/health", response_model=HealthResponse)
async def api_health_check():
    """API 헬스체크"""
    from datetime import datetime
    try:
        active_stories = len(orchestrator.get_active_stories()) if orchestrator else 0
        
        if orchestrator:
            all_stories = await orchestrator.state_manager.list_all_stories()
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
        logger.error(f"헬스체크 실패: {e}", exc_info=True)
        return HealthResponse(
            status="unhealthy",
            timestamp=datetime.now().isoformat(),
            version="1.0.0",
            active_stories=0,
            total_stories=0
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "chatbot.app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    ) 