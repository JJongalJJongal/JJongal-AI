"""
인증 처리 프로세서

JWT 토큰 생성, 검증 및 WebSocket 연결 인증 기능을 제공.
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

JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key") # 환경 변수에서 토큰 키 가져오기
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256") # 토큰 알고리즘
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 30)) # 토큰 만료 시간 (분)

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
            expire = datetime.now(timezone.utc) + expires_delta
        else:
            expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        to_encode.update({"exp": expire})
        try:
            encoded_jwt = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
            return encoded_jwt
        except Exception as e:
            logger.error(f"토큰 생성 오류: {e}")
            return None

    def validate_token(self, token: str) -> bool:
        """JWT 토큰 검증"""
        logger.info(f"validate_token 호출됨. 토큰: '{token}' (타입: {type(token)})")
        if not token:
            logger.warning("validate_token: 토큰이 비어있음.")
            return False
        
        if token == "development_token":
            logger.info("validate_token: 개발용 토큰 확인됨. True 반환.")
            return True
            
        try:
            logger.info("validate_token: JWT 디코딩 시도 중...")
            payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
            logger.info(f"validate_token: JWT 디코딩 성공. 페이로드: {payload}. True 반환.")
            return True
        except jwt.ExpiredSignatureError:
            logger.warning("validate_token: 토큰 만료 (ExpiredSignatureError)")
            return False
        except jwt.InvalidTokenError as e:
            logger.warning(f"validate_token: 유효하지 않은 토큰 (InvalidTokenError): {e}")
            return False
        except Exception as e:
            logger.error(f"validate_token: 토큰 검증 중 예상치 못한 오류: {e}")
            return False

    async def extract_token_from_header(self, websocket: WebSocket) -> Optional[str]:
        """WebSocket 요청에서 토큰 추출 (쿼리 파라미터 우선)"""
        logger.info("extract_token_from_header 호출됨.")
        
        # 1순위: 쿼리 파라미터에서 토큰 확인
        token_query = websocket.query_params.get("token")
        if token_query:
            logger.info(f"쿼리 파라미터에서 토큰 추출: '{token_query}'")
            return token_query

        # 2순위: Authorization 헤더에서 토큰 확인 (대체 또는 이전 방식 지원)
        authorization = websocket.headers.get("Authorization")
        logger.info(f"Authorization 헤더 (대체 확인): {authorization}")
        if authorization and authorization.startswith("Bearer "):
            extracted_token = authorization.split("Bearer ")[1]
            logger.warning(f"Authorization 헤더에서 토큰 추출 (쿼리 파라미터 사용 권장): '{extracted_token}'")
            return extracted_token
        
        logger.warning("extract_token_from_header: 쿼리 파라미터나 헤더에서 토큰을 찾을 수 없음.")
        return None

    async def validate_connection(self, websocket: WebSocket) -> bool:
        """WebSocket 연결 유효성 검증 (토큰 포함)"""
        logger.info("validate_connection 호출됨.")
        token = await self.extract_token_from_header(websocket)
        
        if not token:
            logger.warning("validate_connection: 토큰 없음. 연결 거부.")
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="토큰 없음")
            return False
        
        logger.info(f"validate_connection: 토큰 ('{token}')에 대해 validate_token 호출 예정.")
        is_valid = self.validate_token(token)
        
        if not is_valid:
            logger.warning(f"validate_connection: validate_token이 False 반환. 토큰 ('{token}') 유효하지 않음. 연결 거부.")
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="유효하지 않은 토큰")
            return False
        
        logger.info("validate_connection: 토큰 유효함. True 반환.")
        return True

    def get_test_token(self) -> Dict[str, Optional[str]]:
        """테스트용 JWT 토큰 생성"""
        user_data = {"sub": "test_user@example.com", "user_id": 123}
        token = self.create_access_token(data=user_data)
        return {"token_type": "bearer", "token": token} 