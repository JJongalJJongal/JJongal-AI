from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from typing import Dict, Any
import json
import asyncio
from datetime import datetime

# API v1 Dependency 활용
from ..dependencies import verify_jwt_token, get_connection_manager, get_audio_processor
from ..models import AudioConfigMessage, TranscriptionMessage

# voice_ws
from chatbot.models.voice_ws.handlers.audio_handler import handler_audio_websocket
from chatbot.models.voice_ws.core.connection_engine import ConnectionEngine

# Logging
from shared.utils.logging_utils import get_module_logger
logger = get_module_logger(__name__)


router = APIRouter(prefix="/audio", tags=["WebSocket", "Audio"]) # WebSocket Router Setting

@router.websocket("/")
async def audio_websocket_endpoint(
    websocket: WebSocket,
    # JWT authentication
    user_info: Dict[str, Any] = Depends(verify_jwt_token),
    # Dependency Injection
    connection_manager: ConnectionEngine = Depends(get_connection_manager)
):
    """
    Real-time Audio Conversation WebSocket

    API 명세서
    1. Connection 수립
    2. Conversation 시작 (JSON)
    3. Audio Streaming (Binary)
    4. Server Response (JSON)
    5. Conversation 종료 (JSON)
    
    Query Parameters:
        token: JWT Authentication Token (Required)
        
    Example:
        wss://localhost:8000/wss/v1/audio?token=your_jwt_token
    """
    
    # 1. Connection 수립
    await websocket.accept()
    
    # Client 등록
    client_id = f"audio_{user_info['user_id']}_{datetime.now().timestamp()}"
    
    try:
        # 2. Conversation 시작 대기 (JSON)
        start_message = await websocket.receive_json()
        
        # 3. voice_ws handler 위임
        await handler_audio_websocket(
            websocket=websocket,
            user_info=user_info,
            client_id=client_id,
            start_message=start_message
        )
        
    except WebSocketDisconnect:
        logger.info(f"Audio WebSocket Connection Closed: {client_id}")
    except Exception as e:
        logger.error(f"Audio WebSocket Error: {client_id}")
    finally:
        # 5. Conversation 종료
        await connection_manager.unregister_client(client_id)
        

@router.websocket("/audio")  