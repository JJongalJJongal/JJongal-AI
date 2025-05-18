"""
FastAPI 앱 설정 모듈

이 모듈은 FastAPI 애플리케이션 설정과 라우팅을 제공합니다.
"""
import asyncio
import logging
from typing import Optional
from fastapi import FastAPI, WebSocket, HTTPException, Response, status, Depends, Query
from fastapi.middleware.cors import CORSMiddleware

from .auth import validate_token, get_test_token, validate_connection, extract_token_from_header
from .utils import setup_logging
from .connection import ConnectionManager
from .handlers import handle_audio_websocket, handle_story_generation_websocket

# FastAPI 애플리케이션 인스턴스 생성
app = FastAPI(
    title="꼬꼬북 음성 인터페이스 API", 
    description="아동 음성을 받아 동화 생성을 위한 API",
    version="1.0.0"
)

# CORS 설정 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 origin 허용 (프로덕션에서는 특정 도메인만 허용하도록 변경 필요)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 연결 관리자 인스턴스 초기화
connection_manager = ConnectionManager()

# 토큰 검증 의존성
async def verify_token(websocket: WebSocket):
    """
    WebSocket 헤더에서 토큰을 추출하고 검증하는 의존성 함수
    
    Args:
        websocket (WebSocket): WebSocket 연결 객체
        
    Returns:
        bool: 토큰이 유효하면 True
        
    Raises:
        HTTPException: 토큰이 없거나 유효하지 않은 경우
    """
    token = await extract_token_from_header(websocket)
    if not token:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        logging.warning("토큰 없음: WebSocket 연결 거부")
        return False
        
    if not validate_token(token):
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        logging.warning("토큰 검증 실패: WebSocket 연결 거부")
        return False
        
    return True

# 웹 서버 시작 이벤트 핸들러
@app.on_event("startup")
async def startup_event():
    """서버 시작 시 실행되는 함수"""
    # 로깅 설정
    setup_logging()
    logging.info("음성 WebSocket 서버 시작됨")
    
    # 비활성 클라이언트 정리 태스크 시작
    asyncio.create_task(ConnectionManager.cleanup_inactive_clients())
    
# 웹 서버 종료 이벤트 핸들러
@app.on_event("shutdown")
async def handle_shutdown():
    """서버 종료 시 실행되는 함수"""
    logging.info("음성 WebSocket 서버 종료됨")
    
    # 종료 이벤트 설정
    ConnectionManager.set_shutdown_event()
    
    # 활성 연결 정리
    await ConnectionManager.close_all_connections()

# 상태 확인 엔드포인트
@app.get("/health")
async def health_check():
    """서버 상태 확인 엔드포인트"""
    from .audio import whisper_model
    
    if whisper_model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Whisper 모델이 초기화되지 않았습니다"
        )
    return {"status": "online", "whisper_model": "loaded"}

# 활성 연결 정보 조회 엔드포인트
@app.get("/connections")
async def get_connections():
    """활성 연결 정보 조회 엔드포인트"""
    return {
        "connections": ConnectionManager.get_active_connections_info(),
        "count": ConnectionManager.get_client_count()
    }

# JWT 토큰 발급 테스트 엔드포인트
@app.get("/api/test-token")
async def api_test_token():
    """JWT 토큰 테스트 발급 엔드포인트"""
    token_info = get_test_token()
    if not token_info["token"]:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="토큰 생성 실패"
        )
    return token_info

# 오류 처리 미들웨어
@app.middleware("http")
async def catch_exceptions_middleware(request, call_next):
    """전역 오류 처리 미들웨어"""
    try:
        return await call_next(request)
    except Exception as e:
        logging.error(f"처리되지 않은 서버 오류: {e}")
        return Response(
            content=f'{{"detail": "서버 내부 오류가 발생했습니다"}}',
            status_code=500,
            media_type="application/json"
        )

# 오디오 WebSocket 엔드포인트
@app.websocket("/ws/audio")
async def audio_endpoint(
    websocket: WebSocket,
    child_name: str = Query(None),
    age: int = Query(None),
    interests: Optional[str] = Query(None)
):
    """
    WebSocket 오디오 처리 엔드포인트
    
    파라미터는 쿼리로 전달하되 토큰은 헤더로 전달 (Authorization: Bearer {token})
    
    Args:
        websocket (WebSocket): WebSocket 연결
        child_name (str): 아이 이름
        age (int): 아이 나이
        interests (str): 아이 관심사 (쉼표로 구분)
    """
    # 토큰 검증
    if not await validate_connection(websocket):
        return
        
    # WebSocket 핸들러로 처리 위임
    await handle_audio_websocket(websocket, child_name, age, interests)

# 동화 생성 WebSocket 엔드포인트
@app.websocket("/ws/story_generation")
async def story_generation_endpoint(
    websocket: WebSocket,
    child_name: str = Query(None),
    age: int = Query(None),
    interests: Optional[str] = Query(None)
):
    """
    동화 생성을 위한 WebSocket 엔드포인트
    
    파라미터는 쿼리로 전달하되 토큰은 헤더로 전달 (Authorization: Bearer {token})
    
    Args:
        websocket: WebSocket 연결
        child_name: 아이 이름
        age: 아이 나이
        interests: 관심사 (쉼표로 구분)
    """
    # 토큰 검증
    if not await validate_connection(websocket):
        return
        
    # WebSocket 핸들러로 처리 위임
    await handle_story_generation_websocket(websocket, child_name, age, interests) 