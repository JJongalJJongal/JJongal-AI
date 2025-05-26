"""
인증 처리 프로세서

JWT 토큰 생성, 검증 및 WebSocket 연결 인증 기능을 제공합니다.
"""
import os
import jwt
from datetime import datetime, timedelta, timezone 
from typing import Dict, Optional
from fastapi import WebSocket, status
from dotenv import load_dotenv

from shared.utils.logging_utils import get_module_logger 

logger = get_module_logger(__name__) 

# 환경 변수 로드
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..')) 
dotenv_path = os.path.join(project_root, '.env')
load_dotenv(dotenv_path=dotenv_path) 

JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 30))

class AuthProcessor:
    """
    인증 처리를 담당하는 프로세서
    """
    def __init__(self):
        logger.info("AuthProcessor 초기화 완료")

    def create_access_token(self, data: dict, expires_delta: Optional[timedelta] = None) -> Optional[str]:
        """JWT 액세스 토큰 생성"""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.now(timezone.utc) + expires_delta # 변경: timezone.utc 사용
        else:
            expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES) # 변경: timezone.utc 사용
        to_encode.update({"exp": expire})
        try:
            encoded_jwt = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
            return encoded_jwt
        except Exception as e:
            logger.error(f"토큰 생성 오류: {e}")
            return None

    def validate_token(self, token: str) -> bool:
        """JWT 토큰 검증"""
        if not token:
            return False
        try:
            payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
            # 추가 검증 로직 (예: payload 내용 확인)
            
            # 여기서는 토큰 만료 및 서명 검증만 수행
            return True
        except jwt.ExpiredSignatureError:
            logger.warning("토큰 만료")
            return False
        except jwt.InvalidTokenError as e:
            logger.warning(f"유효하지 않은 토큰: {e}")
            return False
        except Exception as e:
            logger.error(f"토큰 검증 중 예상치 못한 오류: {e}")
            return False

    async def extract_token_from_header(self, websocket: WebSocket) -> Optional[str]:
        """WebSocket 헤더에서 토큰 추출"""
        authorization = websocket.headers.get("Authorization")
        if authorization and authorization.startswith("Bearer "):
            return authorization.split("Bearer ")[1]
        # 쿼리 파라미터에서도 토큰 확인 (하위 호환성)
        token_query = websocket.query_params.get("token")
        if token_query:
            logger.warning("쿼리 파라미터로 토큰 전달됨 (헤더 사용 권장)")
            return token_query
        return None

    async def validate_connection(self, websocket: WebSocket) -> bool:
        """WebSocket 연결 유효성 검증 (토큰 포함)"""
        token = await self.extract_token_from_header(websocket)
        if not token:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="토큰 없음")
            logger.warning("토큰 없음: WebSocket 연결 거부")
            return False
        
        if not self.validate_token(token):
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="유효하지 않은 토큰")
            logger.warning("토큰 검증 실패: WebSocket 연결 거부")
            return False
        
        return True

    def get_test_token(self) -> Dict[str, Optional[str]]:
        """테스트용 JWT 토큰 생성"""
        # 실제 운영에서는 사용자 ID 등을 포함해야 함
        user_data = {"sub": "test_user@example.com", "user_id": 123}
        token = self.create_access_token(data=user_data)
        return {"token_type": "bearer", "token": token} 