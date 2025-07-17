"""
WebSocket JWT 인증 의존성

FastAPI 의존성 주입을 사용한 JWT 토큰 검증
"""
from fastapi import WebSocket, status, WebSocketException, Query, Depends, status
from typing import Dict, Any, Optional
from shared.utils.logging_utils import get_module_logger
import jwt
import asyncio
from datetime import datetime

from chatbot.models.voice_ws.processors.auth_processor import AuthProcessor

logger = get_module_logger(__name__)

# AuthProcessor Instance 생성
auth_processor = AuthProcessor() 

class WebSocketAuthError(WebSocketException):
    """WebSocket 인증 오류"""
    def __init__(self, reason: str = "Authentication Failed"):
        super().__init__(code=status.WS_1008_POLICY_VIOLATION, reason=reason)


async def verify_jwt_token(
    websocket: WebSocket,
    token: Optional[str] = Query(None, description="JWT 인증 토큰")
) -> Dict[str, Any]:
    
    """
    WebSocket JWT 토큰 검증 의존성
    
    Args:
        websocket: WebSocket 연결 객체
        token: Query parameter로 전달된 JWT 토큰
    
    Returns:
        Dict[str, Any]: 토큰에서 추출한 사용자 정보
        
    Raises:
        WebSocketAuthError: 인증 실패 시
    """
    if not token:
        logger.warning("WebSocket 연결 시도 : 토큰 없음")
        raise WebSocketAuthError("토큰이 필요합니다.") # 인증 실패 시 예외 발생
    
    try:
        user_info = auth_processor.verify_jwt_token(token)
        
        if not user_info:
            logger.warning("WebSocket 인증 실패 : 유효하지 않은 토큰")
            raise WebSocketAuthError("유효하지 않은 토큰입니다.") # 인증 실패 시 예외 발생
        
        logger.info(f"WebSocket 인증 성공 : {user_info.get('user_id', 'unknown')}") # 인증 성공 시
        return user_info
    
    except jwt.ExpiredSignatureError:
        logger.warning("WebSocket 인증 실패 : 토큰 만료")
        raise WebSocketAuthError("토큰이 만료되었습니다.") # 인증 실패 시 예외 발생
    
    except jwt.InvalidTokenError as e:
        logger.warning(f"WebSocket 인증 실패 : 잘못된 토큰 - {e}")
        raise WebSocketAuthError("잘못된 토큰입니다.") # 인증 실패 시 예외 발생
    
    except Exception as e:
        logger.error(f"WebSocket 인증 중 오류 : {e}")
        raise WebSocketAuthError("인증 처리 중 오류가 발생했습니다.") # 인증 실패 시 예외 발생
  
# 현재는 아이 접근 권한 검증 로직 없음. 추후 추가 예정
# async def verify_child_permission(
#     child_name: str, 
#     user_info: Dict[str, Any] = Depends(verify_jwt_token) # JWT 토큰 검증 의존성 주입
# ) -> Dict[str, Any]:
    
#     """
#     아이 접근 권한 검증 의존성
    
#     Args:
#         child_name: 접근하려는 아이 이름
#         user_info: JWT에서 추출한 사용자 정보
        
#     Returns:
#         Dict[str, Any]: 검증된 사용자 정보
    
#     Raises:
#         WebSocketAuthError: 권한 없음
#     """

       
async def get_connection_manager():
    """연결 관리자 의존성"""
    from shared.utils import ConnectionManager
    return ConnectionManager(connection_timeout=1800) # 30분 타임아웃

async def get_audio_processor():
    """오디오 processor 의존성"""
    from ...models.voice_ws.processors.audio_processor import AudioProcessor
    return AudioProcessor()

# 개발용 간단 인증
async def verify_development_token(
    websocket: WebSocket,
    token: Optional[str] = Query(None, description="개발용 토큰")
) -> Dict[str, Any]:
    """개발용 간단 인증"""
    if not token or token != "dev-token":
        raise WebSocketAuthError("개발 토큰이 필요합니다.")
    
    return {
        "user_id": "dev_user",
        "username": "개발용_사용자",
        # "role": "child",
    }