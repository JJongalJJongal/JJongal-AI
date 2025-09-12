from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from contextlib import asynccontextmanager
import uvicorn
import os

# API v1 router import
from src.api.v1.endpoints.unified_audio import router as unified_audio_router
from src.api.v1.endpoints.chatbot_voice import router as chatbot_voice_router
from src.api.v1.endpoints.pipeline_monitoring import (
    router as pipeline_monitoring_router,
)
from src.api.v1.routers.api_routers import router as api_router
from src.api.v1.dependencies import ServiceFactory

# logging
from src.shared.utils.logging import get_module_logger

logger = get_module_logger(__name__)


# Lifecycle Event
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 애플리케이션 시작 전 초기화 작업
    logger.info("🚀 JJongal-AI API Server 시작")

    # Disable LangSmith tracing for development to avoid API errors
    os.environ["LANGCHAIN_TRACING_V2"] = "true"

    # API Key 검증 (개발 환경용 - 실제 키 없어도 서버 시작)
    elevenlabs_key = os.getenv("ELEVENLABS_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    if elevenlabs_key and not elevenlabs_key.startswith("sk_REPLACE"):
        logger.info("✅ ElevenLabs API Key 확인 완료")
    else:
        logger.info("🔧 개발 모드: ElevenLabs API Key 없이 실행 (음성 기능 제한)")

    if openai_key and not openai_key.startswith("sk-proj-REPLACE"):
        logger.info("✅ OpenAI API Key 확인 완료")
    else:
        logger.info("🔧 개발 모드: OpenAI API Key 없이 실행 (AI 기능 제한)")

    # Initialize AI models and services through ServiceFactory
    try:

        # Pre-initialize critical services to ensure they're loaded in background
        logger.info("🤖 AI 모델 초기화 시작...")

        chatbot_a = await ServiceFactory.get_instance("chatbot_a")
        logger.info("✅ ChatBot A (쫑이) 초기화 완료 - 백그라운드에서 상주 중")

        voice_processor = await ServiceFactory.get_instance("voice_processor")
        logger.info("✅ Voice Processing 서비스 초기화 완료")

        connection_manager = await ServiceFactory.get_instance("connection_manager")
        logger.info("✅ WebSocket 연결 관리자 초기화 완료")

        # Initialize model health monitoring
        from src.shared.monitoring.model_health_monitor import (
            initialize_model_monitoring,
        )

        await initialize_model_monitoring()
        logger.info("📊 AI 모델 건강 모니터링 시작")

        logger.info("🎯 모든 AI 서비스가 백그라운드에서 실행 중입니다")

    except Exception as e:
        logger.error(f"❌ AI 서비스 초기화 실패: {e}")

    # Test Vector DB connection
    try:
        from src.api.v1.dependencies import get_vector_db

        vector_db_info = await get_vector_db()
        if vector_db_info.get("status") == "connected":
            logger.info(
                f"✅ Vector DB 연결 완료 ({vector_db_info.get('collection_count', 0)} documents)"
            )
        else:
            logger.warning(
                f"⚠️ Vector DB 연결 문제: {vector_db_info.get('error', 'Unknown error')}"
            )
    except Exception as e:
        logger.error(f"❌ Vector DB 연결 테스트 실패: {e}")

    yield  # 애플리케이션 실행 중

    # 애플리케이션 종료 후 정리 작업
    logger.info("🔄 JJongal-AI API Server 종료 중...")

    try:
        # Cleanup model monitoring
        from src.shared.monitoring.model_health_monitor import cleanup_model_monitoring

        await cleanup_model_monitoring()
        logger.info("📊 AI 모델 모니터링 정리 완료")
    except Exception as e:
        logger.error(f"모델 모니터링 정리 실패: {e}")

    try:
        # Cleanup WebSocket connections
        connection_manager = await ServiceFactory.get_instance("connection_manager")
        await connection_manager.disconnect_all()
        logger.info("🔌 WebSocket 연결 정리 완료")
    except Exception as e:
        logger.error(f"WebSocket 정리 실패: {e}")

    try:
        # Cleanup temporary audio files
        from src.shared.utils.audio import cleanup_temp_audio_files

        cleanup_result = await cleanup_temp_audio_files(
            "output/temp/audio", max_age_hours=1
        )
        if cleanup_result["files_deleted"] > 0:
            logger.info(
                f"🧹 임시 오디오 파일 정리: {cleanup_result['files_deleted']}개 파일, "
                f"{cleanup_result['bytes_freed']} bytes 확보"
            )
    except Exception as e:
        logger.error(f"임시 파일 정리 실패: {e}")

    logger.info("✅ JJongal-AI API Server 종료 완료")


# FastAPI Instance
app = FastAPI(
    title="쫑알쫑알 동화 챗봇 API",
    description="AI 기반 개인화된 동화 생성 플랫폼",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS Middleware (Development / Operation)
app.add_middleware(
    CORSMiddleware,  # CORS 설정
    allow_origins=[
        "*"
    ],  # 모든 도메인 허용하는것이나, 실제 프로덕션 환경에서는 특정 도메인 (AWS EC2 인스턴스 주소) 허용
    allow_credentials=True,  # Client 측에서 쿠키 전송 허용
    allow_methods=[
        "*"
    ],  # 모든 HTTP Method 허용 (GET, POST, PUT, DELETE, OPTIONS, PATCH)
    allow_headers=["*"],  # 모든 헤더 허용
)

# 신뢰할 수 있는 Host Middleware
app.add_middleware(
    TrustedHostMiddleware,  # 신뢰할 수 있는 Host 설정
    allowed_hosts=[
        "*"
    ],  # 모든 도메인 허용하는것이나, 실제 프로덕션 환경에서는 특정 도메인 (AWS EC2 인스턴스 주소) 허용
)

# API Router 등록
# REST API 엔드포인트 (health, auth, stories 등)
app.include_router(api_router)

# ChatBot Voice Options API
app.include_router(
    chatbot_voice_router, prefix="/api/v1"  # /api/v1/chatbot-voice/* 엔드포인트들
)

# Pipeline Monitoring API
app.include_router(
    pipeline_monitoring_router,
    prefix="/api/v1",  # /api/v1/pipeline-monitoring/* 엔드포인트들
)

# WebSocket Router 등록
# 통합 쫑알쫑알 WebSocket API (API 문서 v1.0 기준)
app.include_router(
    unified_audio_router, prefix="/wss/v1"  # /wss/v1/audio 통합 엔드포인트
)


# 전역 예외 처리기
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": {"code": str(exc.status_code), "message": exc.detail},
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "error": {"code": "VALIDATION_ERROR", "message": str(exc)},
        },
    )


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
