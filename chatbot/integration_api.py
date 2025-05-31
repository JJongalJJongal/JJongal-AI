"""
꼬꼬북 통합 RESTful API 서버

이야기 생성 관리, RAG 시스템 연동, 사용자 피드백 등
다양한 비-실시간 상호작용을 위한 API를 제공합니다.
"""

import uvicorn
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os

from shared.utils.logging_utils import get_module_logger
from .routers import story_router
from chatbot.db import initialize_db as initialize_story_task_db

# 환경 변수 로드 (프로젝트 루트 .env 파일 기준)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..')) 
dotenv_path = os.path.join(project_root, '.env')
load_dotenv(dotenv_path=dotenv_path)

logger = get_module_logger("integration_api")

app = FastAPI(
    title="꼬꼬북 통합 API",
    description="이야기 생성 관리, RAG 시스템 연동, 사용자 피드백 등을 위한 RESTful API",
    version="0.1.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 실제 운영 시에는 특정 도메인으로 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- API Routers --- 
# (추후 라우터들을 추가하고 여기서 include_router 할 예정)
from .routers import story_router
# from .routers import rag_router, feedback_router
app.include_router(story_router.router, prefix="/stories", tags=["Story Management"])
# app.include_router(rag_router.router, prefix="/rag", tags=["RAG System"])
# app.include_router(feedback_router.router, prefix="/feedback", tags=["User Feedback"])

# --- Event Handlers ---
@app.on_event("startup")
async def startup_event():
    logger.info("통합 API 서버 시작됨")
    try:
        initialize_story_task_db()
        logger.info("이야기 작업 DB 초기화 성공.")
    except Exception as e:
        logger.error(f"이야기 작업 DB 초기화 실패: {e}", exc_info=True)
    # 필요한 초기화 작업 (예: DB 연결, 모델 로드 등)

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("통합 API 서버 종료됨")
    # 필요한 정리 작업

# --- Basic Endpoints ---
@app.get("/health", summary="API 상태 확인", tags=["General"])
async def health_check():
    """API 서버의 현재 상태를 반환합니다."""
    logger.info("Health check 요청 수신")
    return {"status": "healthy", "message": "Integration API is running."}

# --- Main execution (for direct run) ---
if __name__ == "__main__":
    port = int(os.getenv("INTEGRATION_API_PORT", 8001))
    reload = os.getenv("INTEGRATION_API_RELOAD", "false").lower() == "true"
    log_level = os.getenv("INTEGRATION_API_LOG_LEVEL", "info")

    logger.info(f"통합 API 서버 직접 실행 (포트: {port}, 자동 리로드: {reload})")
    uvicorn.run(
        "integration_api:app", 
        host="0.0.0.0", 
        port=port, 
        reload=reload,
        log_level=log_level,
        app_dir=os.path.dirname(__file__) # 현재 파일의 디렉토리를 app_dir로 설정
    ) 