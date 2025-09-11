"""
JJongal Authentication Middleware
JWT 토큰 기반 인증 미들웨어
"""

import jwt
from typing import Optional, Dict, Any
from fastapi import Request, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from src.shared.utils.logging import get_module_logger
from src.shared.configs.app import get_env_vars

logger = get_module_logger(__name__)

class JWTAuthMiddleware(BaseHTTPMiddleware):
    """
    JWT 인증 미들웨어
    API 요청에 대한 토큰 검증
    """
    
    def __init__(self, app, excluded_paths: list = None):
        super().__init__(app)
        self.env_vars = get_env_vars()
        self.secret_key = self.env_vars.get("jwt_secret_key", "jjongal_default_secret")
        self.algorithm = "HS256"
        
        # 인증 제외 경로 목록
        self.excluded_paths = excluded_paths or [
            "/docs",
            "/redoc",
            "/openapi.json",
            "/health",
            "/api/v1/health",
            "/api/v1/auth/token",
            "/api/v1/auth/dev-token",
            "/favicon.ico"
        ]
    
    async def dispatch(self, request: Request, call_next):
        """
        요청 처리 전 인증 검증
        """
        # 제외 경로는 인증 건너뛰기
        if any(request.url.path.startswith(path) for path in self.excluded_paths):
            response = await call_next(request)
            return response
        
        # WebSocket 경로는 별도 처리 (query parameter로 토큰 전달)
        if request.url.path.startswith("/wss/"):
            response = await call_next(request)
            return response
        
        try:
            # Authorization 헤더 토큰 검증
            authorization = request.headers.get("Authorization")
            if not authorization:
                return self._unauthorized_response("Authorization header missing")
            
            if not authorization.startswith("Bearer "):
                return self._unauthorized_response("Invalid authorization format")
            
            token = authorization.replace("Bearer ", "")
            
            # 토큰 검증
            payload = self._verify_token(token)
            if not payload:
                return self._unauthorized_response("Invalid token")
            
            # 요청에 사용자 정보 추가
            request.state.user_id = payload.get("user_id")
            request.state.user_info = payload
            
            logger.debug(f"인증 성공: user_id={payload.get('user_id')}")
            
        except Exception as e:
            logger.warning(f"인증 실패: {e}")
            return self._unauthorized_response("Authentication failed")
        
        response = await call_next(request)
        return response
    
    def _verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        JWT 토큰 검증
        """
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("토큰 만료")
            return None
        except jwt.InvalidTokenError:
            logger.warning("유효하지 않은 토큰")
            return None
        except Exception as e:
            logger.error(f"토큰 검증 오류: {e}")
            return None
    
    def _unauthorized_response(self, message: str) -> JSONResponse:
        """
        인증 실패 응답 생성
        """
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={
                "success": False,
                "error": {
                    "code": "UNAUTHORIZED",
                    "message": message
                }
            },
            headers={"WWW-Authenticate": "Bearer"}
        )

class OptionalJWTAuthMiddleware(BaseHTTPMiddleware):
    """
    선택적 JWT 인증 미들웨어
    토큰이 있으면 검증, 없어도 계속 진행
    """
    
    def __init__(self, app):
        super().__init__(app)
        self.env_vars = get_env_vars()
        self.secret_key = self.env_vars.get("jwt_secret_key", "jjongal_default_secret")
        self.algorithm = "HS256"
    
    async def dispatch(self, request: Request, call_next):
        """
        요청 처리 전 선택적 인증 검증
        """
        try:
            authorization = request.headers.get("Authorization")
            if authorization and authorization.startswith("Bearer "):
                token = authorization.replace("Bearer ", "")
                payload = self._verify_token(token)
                
                if payload:
                    request.state.user_id = payload.get("user_id")
                    request.state.user_info = payload
                    request.state.authenticated = True
                    logger.debug(f"선택적 인증 성공: user_id={payload.get('user_id')}")
                else:
                    request.state.authenticated = False
            else:
                request.state.authenticated = False
                
        except Exception as e:
            logger.debug(f"선택적 인증 검증 오류 (계속함): {e}")
            request.state.authenticated = False
        
        response = await call_next(request)
        return response
    
    def _verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        JWT 토큰 검증
        """
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except Exception:
            return None

# FastAPI Security 객체
security = HTTPBearer(auto_error=False)

def get_current_user(credentials: Optional[HTTPAuthorizationCredentials] = None) -> Optional[Dict[str, Any]]:
    """
    현재 사용자 정보 반환 (종속성 함수)
    """
    if not credentials:
        return None
    
    try:
        env_vars = get_env_vars()
        secret_key = env_vars.get("jwt_secret_key", "jjongal_default_secret")
        
        payload = jwt.decode(credentials.credentials, secret_key, algorithms=["HS256"])
        return payload
        
    except Exception as e:
        logger.warning(f"사용자 정보 반환 실패: {e}")
        return None

def require_auth(credentials: HTTPAuthorizationCredentials = security) -> Dict[str, Any]:
    """
    필수 인증 종속성 (종속성 함수)
    """
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    user_info = get_current_user(credentials)
    if not user_info:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    return user_info