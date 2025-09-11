"""
Health Check Router for JJongal-AI
상태 확인 라우터
"""

from fastapi import APIRouter
from ..endpoints.health import router as health_endpoint_router

# Health check router
router = APIRouter()

# Include health endpoints
router.include_router(health_endpoint_router, tags=["health"])