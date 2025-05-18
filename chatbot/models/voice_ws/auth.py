"""
JWT 인증 관련 기능

이 모듈은 JWT 토큰 생성, 검증 및 인증 처리 기능을 제공합니다.
"""
import os
import time
import logging
import jwt
from jwt.exceptions import InvalidTokenError
from fastapi import WebSocket, status

# JWT 설정
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "default_secret_key_for_development_only")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION = 3600  # 1시간

def generate_token(payload: dict) -> str:
    """
    JWT 토큰 생성 함수
    
    Args:
        payload (dict): 토큰에 포함할 페이로드
        
    Returns:
        str: 생성된 JWT 토큰
    """
    if not JWT_SECRET_KEY:
        logging.error("JWT_SECRET_KEY 환경 변수가 설정되지 않았습니다")
        return ""
        
    try:
        # JWT 토큰 생성
        token = jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
        return token
    except Exception as e:
        logging.error(f"토큰 생성 중 오류 발생: {e}")
        return ""

def validate_token(token: str) -> bool:
    """
    인증 토큰 검증 함수
    
    Args:
        token (str): 검증할 토큰 (JWT Token)
    
    Returns:
        bool: 유효한 토큰이면 True, 아니면 False
    """
    # 테스트 모드일 때는 test_token 허용
    if token == "test_token":
        logging.info("테스트 토큰 인증 성공")
        return True
        
    if not JWT_SECRET_KEY:
        logging.error("JWT_SECRET_KEY 환경 변수가 설정되지 않았습니다")
        return False
    
    try:
        # token decoding & verify
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        return True
    except InvalidTokenError:
        logging.warning("유효하지 않은 토큰입니다.")
        return False
    except Exception as e:
        logging.error(f"토큰 검증 중 오류 발생: {e}")
        return False

async def extract_token_from_header(websocket: WebSocket) -> str:
    """
    WebSocket 헤더에서 토큰 추출
    
    Args:
        websocket (WebSocket): WebSocket 연결 객체
    
    Returns:
        str: 추출된 토큰 또는 빈 문자열
    """
    headers = websocket.scope.get("headers", [])
    
    for header_name, header_value in headers:
        if header_name.decode().lower() == "authorization":
            auth_value = header_value.decode()
            # "Bearer {token}" 형식에서 토큰 추출
            if auth_value.startswith("Bearer "):
                return auth_value[7:]
    
    # 헤더에서 토큰을 찾지 못한 경우
    logging.warning("Authorization 헤더에서 토큰을 찾을 수 없습니다")
    return ""

async def validate_connection(websocket: WebSocket) -> bool:
    """
    WebSocket 연결 검증
    
    Args:
        websocket (WebSocket): WebSocket 연결 객체
    
    Returns:
        bool: 유효한 연결이면 True, 아니면 False
    """
    # 헤더에서 토큰 추출
    token = await extract_token_from_header(websocket)
    
    # 토큰 없으면 연결 거부
    if not token:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return False
    
    # 토큰 검증
    if not validate_token(token):
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return False
    
    return True

def get_test_token() -> dict:
    """
    테스트용 JWT 토큰 생성
    
    Returns:
        dict: 토큰 정보를 포함한 딕셔너리
    """
    payload = {
        "sub": "test_user",
        "role": "child",
        "iat": int(time.time()),
        "exp": int(time.time()) + JWT_EXPIRATION
    }
    token = generate_token(payload)
    
    return {
        "token": token,
        "expires_in": JWT_EXPIRATION,
        "token_type": "Bearer"
    } 