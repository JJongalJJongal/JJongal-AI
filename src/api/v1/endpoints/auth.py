"""
JJongal Authentication API Endpoints
JWT 토큰 기반 인증 시스템
"""

from typing import Dict, Any
from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import HTTPBearer
from pydantic import BaseModel
from datetime import datetime

from src.shared.utils.logging import get_module_logger
from ..dependencies import get_jwt_manager, JWTManager

logger = get_module_logger(__name__)

router = APIRouter(prefix="/auth", tags=["authentication"])
security = HTTPBearer()

# Request/Response Models
class TokenRequest(BaseModel):
    user_id: str
    password: str = "dev"  # 개발용

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int = 86400  # 24시간
    user_id: str

class RefreshTokenRequest(BaseModel):
    refresh_token: str

@router.post("/token", response_model=TokenResponse)
async def create_access_token(
    token_request: TokenRequest,
    jwt_manager: JWTManager = Depends(get_jwt_manager)
) -> TokenResponse:
    """
    JWT 토큰 생성
    개발용 간단 인증
    """
    try:
        logger.info(f"토큰 생성 요청: user_id={token_request.user_id}")
        
        # 간단한 인증 (실제로는 DB 연동 필요)
        if not token_request.user_id:
            raise HTTPException(
                status_code=400,
                detail="user_id is required"
            )
        
        # JWT 토큰 생성
        payload = {
            "user_id": token_request.user_id,
            "user_name": token_request.user_id,
            "role": "user",
            "created_at": datetime.now().isoformat()
        }
        
        access_token = jwt_manager.generate_token(payload)
        
        logger.info(f"토큰 생성 완료: user_id={token_request.user_id}")
        
        return TokenResponse(
            access_token=access_token,
            user_id=token_request.user_id
        )
        
    except Exception as e:
        logger.error(f"토큰 생성 실패: {e}")
        raise HTTPException(
            status_code=500,
            detail="Token creation failed"
        )

@router.post("/refresh", response_model=TokenResponse)
async def refresh_access_token(
    refresh_request: RefreshTokenRequest,
    jwt_manager: JWTManager = Depends(get_jwt_manager)
) -> TokenResponse:
    """
    JWT 토큰 갱신
    """
    try:
        # 기존 토큰 검증
        payload = jwt_manager.verify_token(refresh_request.refresh_token)
        user_id = payload.get("user_id")
        
        if not user_id:
            raise HTTPException(
                status_code=401,
                detail="Invalid refresh token"
            )
        
        # 새 토큰 생성
        new_payload = {
            "user_id": user_id,
            "user_name": payload.get("user_name", user_id),
            "role": payload.get("role", "user"),
            "created_at": datetime.now().isoformat()
        }
        
        new_token = jwt_manager.generate_token(new_payload)
        
        logger.info(f"토큰 갱신 완료: user_id={user_id}")
        
        return TokenResponse(
            access_token=new_token,
            user_id=user_id
        )
        
    except Exception as e:
        logger.error(f"토큰 갱신 실패: {e}")
        raise HTTPException(
            status_code=401,
            detail="Token refresh failed"
        )

@router.get("/verify")
async def verify_token(
    jwt_manager: JWTManager = Depends(get_jwt_manager),
    token: str = Depends(security)
) -> Dict[str, Any]:
    """
    JWT 토큰 검증
    """
    try:
        # Bearer 토큰에서 실제 토큰 추출
        if hasattr(token, 'credentials'):
            token_str = token.credentials
        else:
            token_str = str(token)
            
        payload = jwt_manager.verify_token(token_str)
        
        return {
            "valid": True,
            "user_id": payload.get("user_id"),
            "expires_at": payload.get("exp"),
            "issued_at": payload.get("iat")
        }
        
    except Exception as e:
        logger.error(f"토큰 검증 실패: {e}")
        raise HTTPException(
            status_code=401,
            detail="Invalid token"
        )

@router.get("/dev-token")
async def get_development_token(
    user_id: str = "dev_user_001",
    jwt_manager: JWTManager = Depends(get_jwt_manager)
) -> TokenResponse:
    """
    개발용 토큰 생성 (개발 환경용)
    """
    try:
        token = jwt_manager.create_development_token(user_id)
        
        return TokenResponse(
            access_token=token,
            user_id=user_id
        )
        
    except Exception as e:
        logger.error(f"개발 토큰 생성 실패: {e}")
        raise HTTPException(
            status_code=500,
            detail="Development token creation failed"
        )