from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from datetime import datetime
from contextlib import asynccontextmanager
import uvicorn

# API v1 router import
from chatbot.api.v1.routers import websocket_router
from chatbot.api.v1.ws.audio import router as audio_router

# logging 
from shared.utils.logging_utils import get_module_logger
logger = get_module_logger(__name__)

# Lifecycle Event
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 애플리케이션 시작 전 초기화 작업
    logger.info("쫑알쫑알 API Server 시작")
    # VectorDB 연결, ElevenLabs API Key 확인 등
    elevenlabs_key = os.getenv("ELEVENLABS_API_KEY")
    if elevenlabs_key:
        logger.info("ElevenLabs API Key 확인 완료")
    else:
        logger.error("ElevenLabs API Key 설정이 필요합니다.")
        
    try:
        logger.info("VectorDB 연결 준비")
    except Exception as e:
        logger.error(f"VectorDB 연결 실패 : {e}")
        
    logger.info("WebSocket 연결 관리자 초기화")
    
    yield # 애플리케이션 실행 중
    
    # 애플리케이션 종료 후 정리 작업
    logger.info("쫑알쫑알 API Server 종료")
    
    logger.info("WebSocket 연결 정리")
    
    logger.info("임시 파일 정리")
    
    logger.info("서버 종료 완료")

# FastAPI Instance
app = FastAPI(
    title="쫑알쫑알 동화 챗봇 API",
    description="AI 기반 개인화된 동화 생성 플랫폼",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Middleware (Development / Operation)
app.add_middleware(
    CORSMiddleware, # CORS 설정
    allow_origins=["*"], # 모든 도메인 허용하는것이나, 실제 프로덕션 환경에서는 특정 도메인 (AWS EC2 인스턴스 주소) 허용
    allow_credentials=True, # Client 측에서 쿠키 전송 허용
    allow_methods=["*"], # 모든 HTTP Method 허용 (GET, POST, PUT, DELETE, OPTIONS, PATCH)
    allow_headers=["*"], # 모든 헤더 허용
)

# 신뢰할 수 있는 Host Middleware
app.add_middleware(
    TrustedHostMiddleware, # 신뢰할 수 있는 Host 설정
    allowed_hosts=["*"] # 모든 도메인 허용하는것이나, 실제 프로덕션 환경에서는 특정 도메인 (AWS EC2 인스턴스 주소) 허용
)

# WebSocket Router 등록
app.include_router(
    websocket_router,
    prefix="/api/v1" # /api/v1/wss/...
)

app.include_router(
    audio_router,
    prefix="/wss/v1" # /wss/v1/audio/...
)

# 전역 예외 처리기
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"success": False, "error": {"code": str(exc.status_code), "message": exc.detail}}
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return JSONResponse(
        status_code=422,
        content={"success": False, "error": {"code": "VALIDATION_ERROR", "message": str(exc)}}
    )
    