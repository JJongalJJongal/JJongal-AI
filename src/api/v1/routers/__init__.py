"""
Voice WebSocket Router Module

FastAPI APIRouter 를 사용해서 WebSocket endpoint 를 관리

"""
from .websocket_routers import router as websocket_router

__all__ = [
    "websocket_router"
]