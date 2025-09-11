"""
API Router Module

FastAPI APIRouter 를 사용해서 API endpoints 를 관리

"""
from .api_routers import router as api_router
from .health_routers import router as health_router

__all__ = [
    "api_router",
    "health_router"
]