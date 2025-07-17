""" WebSocket voice processing handlers """

from fastapi import WebSocket, WebSocketDisconnect
from typing import Dict, Any, Optional, List
import json
import asyncio
import os
from datetime import datetime
from pathlib import Path

from shared.utils import ConnectionManager
from shared.utils.logging_utils import get_module_logger
from shared.utils.audio_utils import generate_speech
from shared.utils.file_utils import ensure_directory

from chatbot.models.chat_bot_a import ChatBotA

logger = get_module_logger(__name__)

async def handle_voice_message(
    data: Dict[str, Any],
    client_id: str,
    websocket: WebSocket,
    connection_manager: ConnectionManager,
    user_info: Dict[str, Any]
) -> None:
    """
    Google STT Text Message process
    
    Args:
        data: JSON Message for client
        client_id: client's unique id
        websocket: WebSocket connection object
        connection_manager: WebSocket connection manager
        user_info: User information from JWT
    """
    try:
        message_type = data.get("type")
        
        if message_type == "start_conversation":
            await _handle_start_conversation(data, client_id, websocket, user_info)
            
        elif message_type == "user_message":
            await _handle_user_message(data, client_id, websocket, user_info)
        
        elif message_type == "end_conversation":
            await _handle_end_conversation(data, client_id, websocket, user_info)
        
        else:
            await websocket.send_json({
                "type": "error",
                "error_message": f"Invalid message type: {message_type}",
                "error_code": "UNKNWON_MESSAGE_TYPE"
            })
    
    except Exception as e:
        logger.error(f"Error voice message processing ({client_id}): {e}")
        await websocket.send_json({
            "type": "error",
            "error_message": "Internal Server Error",
            "error_code": "MESSAGE_PROCESSING_ERROR"
        })
    
async def handle_audio_stream(
    audio_data: bytes,
    client_id: str,
    websocket: WebSocket,
    connection_manager: ConnectionManager,
    user_info: Dict[str, Any]
) -> None:
    """
    Voice Clone Audio stream processing

    Args:
        audio_data: audio binary data for client
        client_id : client's unique id
        websocket: WebSocket connection object
        connection_manager: WebSocket connection manager
        user_info: User information from JWT
    """
    try:
        # child name from user_info
        child_name = user_info.get("child_name", "unknown")
        
        # sample directory for voice clone
        samples_dir = Path("output/temp/voice_samples") / child_name
        samples_dir.mkdir(parents=True, exist_ok=True)
        
        # audio file save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"sample_{timestamp}.mp3"
        file_path = samples_dir / file_name
        
        with open(file_path, "wb") as f:
            f.write(audio_data)
        
        logger.info(f"Voice clone sample saved: {file_path}")
        
        # sample count check
        sample_count = len(list(samples_dir.glob("sample_*.mp3")))
        
        # Process 
        await websocket.send_json({
            "type": "voice_clone_progress",
            "sample_count": sample_count,
            "ready_for_cloning": sample_count >= 5,
            "message": f"Saved... ({sample_count}/5)",
            "timestamp": datetime.now().isoformat()
        })
        
        
        # Voice clone create if 5 samples are saved
        if sample_count >= 5:
            await _create_voice_clone(child_name, samples_dir, websocket)
        
    except Exception as e:
        logger.error(f"Error audio stream processing ({client_id}): {e}")
        await websocket.send_json({
            "type": "error",
            "error_message": "Error processing audio stream",
            "error_code": "AUDIO_PROCESSING_ERROR"
        })

async def handler_audio_websocket(
    websocket: WebSocket,
    user_info: Dict[str, Any],
    client_id: str,
    start_message: Dict[str, Any]
) -> None:
    """
    Main WebSocket handler for JSON/Binary data routing

    Args:
        websocket: WebSocket connection object
        user_info: User information from JWT
        client_id: client's unique id
        start_message: Initial message from client
    """
    
    try:
        # ConnectionManager instance
        connection_manager = ConnectionManager()
        
        
        # Client connection 
        await connection_manager.connect(websocket, client_id, {
            "user_info": user_info,
            "connection_type": "audio",
            "start_message": start_message
        })
        
        logger.info(f"Audio WebSocket connected: {client_id}")
        
        # Send connection success
        await websocket.send_json({
            "type": "connection_success",
            "message": "Audio conversation connection successed",
            "client_id": client_id,
            "timestamp": datetime.now().isoformat()
        })
        
        # Message process lopp
        while True:
            try:
                # JSON message try
                message = await websocket.receive_json()
                await handle_voice_message(
                    message, client_id, websocket, connection_manager, user_info
                )
            except ValueError:
                # JSON -> Binary
                try:
                    audio_data = await websocket.receive_bytes()
                    await handle_audio_stream(
                        audio_data, client_id, websocket, connection_manager, user_info
                    )
    except WebSocketDisconnect:
        logger.error(f"Audio WebSocket disconnected: {client_id}")
        
    except Exception as e:
        logger.error(f"Audio handler error: {e}")
    
    finally:
        # Connection close
        connection_manager.disconnect(client_id)
        
            
async def _handle_start_conversation(data, client_id, websocket, user_info):
    """ Conversation start processing """
    child_name = data.get("payload", {}).get("child_name", "unknown")
    age = data.get("payload", {}).get("age", 7)
    
    # Chatbot A Instance 
               