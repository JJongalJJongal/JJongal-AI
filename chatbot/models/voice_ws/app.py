"""
FastAPI 앱 설정 모듈 (모듈화된 구조)

이 모듈은 FastAPI 애플리케이션 설정과 라우팅을 제공하며,
모듈화된 core, processors, handlers를 사용합니다.
"""
import asyncio
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI, WebSocket, HTTPException, Response, status, Query
from fastapi.middleware.cors import CORSMiddleware

# 모듈화된 컴포넌트 임포트
from shared.utils.logging_utils import get_module_logger 
from .core.connection_engine import ConnectionEngine
from .processors.auth_processor import AuthProcessor
from .processors.audio_processor import AudioProcessor 
from .handlers.audio_handler import handle_audio_websocket
from .handlers.story_handler import handle_story_generation_websocket
from chatbot.data.vector_db.core import VectorDB # VectorDB 임포트 추가

logger = get_module_logger(__name__) # 로거 설정

# 서비스 시작/종료 관리
@asynccontextmanager # asynccontextmanager로 변경
async def lifespan_manager(app: FastAPI):
    # WebSocket 서버 시작
    logger.info("음성 WebSocket 서버 시작") 
    
    # VectorDB 인스턴스 사전 로드
    logger.info("VectorDB 사전 로드 중...")
    try:
        app.state.vector_db = VectorDB() # app.state에 저장
        logger.info("VectorDB 사전 로드 완료.")
    except Exception as e:
        logger.error(f"VectorDB 사전 로드 실패: {e}")
        app.state.vector_db = None
    
    print("서비스 시작")
    asyncio.create_task(connection_engine.cleanup_inactive_clients()) # 비활성 클라이언트 정리 task 시작
    
    yield
    
    # WebSocket 서버 종료
    logger.info("음성 WebSocket 서버 종료")
    connection_engine.set_shutdown_event() # 종료 이벤트 설정
    await connection_engine.close_all_connections() # 활성 연결 정리
    print("서비스 종료")
    
# FastAPI 애플리케이션 인스턴스 생성
app = FastAPI(
    title="꼬꼬북 음성 인터페이스 API", 
    description="아동 음성을 받아 동화 생성을 위한 API",
    lifespan=lifespan_manager
)

# CORS 설정 추가
app.add_middleware(
    CORSMiddleware, # CORS 설정
    allow_origins=["*"],  # 모든 출처 허용
    allow_credentials=True, # 자격 증명 허용
    allow_methods=["*"], # 모든 메서드 허용
    allow_headers=["*"], # 모든 헤더 허용
)

# 주요 컴포넌트 인스턴스화
connection_engine = ConnectionEngine() 
auth_processor = AuthProcessor()
audio_processor = AudioProcessor() 

# 토큰 검증 의존성 (AuthProcessor 사용)
async def verify_connection_token(websocket: WebSocket):
    """WebSocket 연결의 토큰을 검증하는 의존성 함수 (AuthProcessor 사용)"""
    if not await auth_processor.validate_connection(websocket):
        # validate_connection 내부에서 close 및 로깅 처리
        raise HTTPException(status_code=status.WS_1008_POLICY_VIOLATION, detail="토큰 검증 실패") # 토큰 검증 실패 예외 발생
    return True

# 상태 확인 엔드포인트
@app.get("/health")
async def health_check():
    """서버 상태 확인 엔드포인트"""
    if audio_processor.whisper_model is None: # AudioProcessor 인스턴스 사용
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Whisper 모델이 초기화되지 않았습니다"
        )
    return {"status": "online", "whisper_model": "loaded"}

# 활성 연결 정보 조회 엔드포인트
@app.get("/connections")
async def get_connections_info(): 
    """활성 연결 정보 조회 엔드포인트 (ConnectionEngine 사용)"""
    return {
        "connections": connection_engine.get_active_connections_info(),
        "count": connection_engine.get_client_count()
    }

# JWT 토큰 발급 테스트 엔드포인트
@app.get("/api/test-token")
async def api_test_token():
    """JWT 토큰 테스트 발급 엔드포인트 (AuthProcessor 사용)"""
    token_info = auth_processor.get_test_token()
    if not token_info["token"]:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="토큰 생성 실패"
        )
    return token_info

# 오류 처리 미들웨어 (유지)
@app.middleware("http")
async def catch_exceptions_middleware(request, call_next):
    """전역 오류 처리 미들웨어"""
    try:
        return await call_next(request)
    except Exception as e:
        logger.error(f"처리되지 않은 서버 오류: {e}")
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
    interests: Optional[str] = Query(None),
    # 의존성을 통해 토큰 검증
    # token_valid: bool = Depends(verify_connection_token) # FastAPI의 Depends는 HTTP에서만 제대로 동작, WS에서는 직접 호출
):
    """
    WebSocket 오디오 처리 엔드포인트
    """
    # 토큰 검증 (직접 호출)
    if not await auth_processor.validate_connection(websocket):
        return # validate_connection 내부에서 close 처리
        
    # WebSocket 핸들러로 처리 위임
    # 핸들러에 필요한 의존성(엔진, 프로세서) 전달
    await handle_audio_websocket(
        websocket, # WebSocket 객체
        child_name, # 아이 이름
        age, # 아이 연령대 (4-7세, 8-9세)
        interests, # 아이 관심사
        connection_engine=connection_engine, # ConnectionEngine 인스턴스
        audio_processor=audio_processor # AudioProcessor 인스턴스
        # chat_bot_a 등 필요한 다른 의존성도 전달 가능
    )

# 동화 생성 WebSocket 엔드포인트
@app.websocket("/ws/story_generation")
async def story_generation_endpoint(
    websocket: WebSocket,
    child_name: str = Query(None),
    age: int = Query(None),
    interests: Optional[str] = Query(None),
    # token_valid: bool = Depends(verify_connection_token) # 위와 동일한 이유로 직접 호출
):
    """
    동화 생성을 위한 WebSocket 엔드포인트
    """
    # 토큰 검증 (직접 호출)
    if not await auth_processor.validate_connection(websocket):
        return # validate_connection 내부에서 close 처리
        
    # WebSocket 핸들러로 처리 위임
    await handle_story_generation_websocket(
        websocket, 
        child_name, 
        age, 
        interests,
        connection_engine=connection_engine,
        audio_processor=audio_processor 
        # chat_bot_b 등 필요한 다른 의존성도 전달 가능
    ) 