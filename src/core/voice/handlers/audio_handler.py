"""
Modern Audio Handler for JJongal-AI WebSocket Communication

This module provides comprehensive audio handling capabilities for real-time
voice communication between children and AI chatbots.
"""

import asyncio
import base64
import json
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
from pathlib import Path

from fastapi import WebSocket, WebSocketDisconnect
from src.shared.utils.logging import get_module_logger
from src.shared.utils.audio import generate_speech, initialize_elevenlabs
from src.shared.configs.app import get_app_settings

logger = get_module_logger(__name__)


class AudioHandler:
    """
    Enhanced Audio Handler for WebSocket Communication
    
    Features:
    - Real-time audio message processing
    - TTS generation with cloned voices
    - Conversation state management
    - Audio configuration handling
    - Error recovery and validation
    """
    
    def __init__(self, websocket: WebSocket, client_id: str = None):
        self.websocket = websocket
        self.client_id = client_id or f"audio_{uuid.uuid4().hex[:8]}"
        
        # Conversation state
        self.conversation_history: List[Dict[str, Any]] = []
        self.conversation_id: Optional[str] = None
        self.is_active: bool = False
        self.session_metadata: Dict[str, Any] = {}
        
        # Audio configuration
        self.audio_config = {
            "tts_enabled": True,
            "voice_cloning_enabled": True,
            "audio_format": "mp3",
            "sample_rate": 44100,
            "bit_rate": 128
        }
        
        # Audio buffer for streaming
        self.audio_buffer: List[bytes] = []
        self.max_buffer_size = 10  # Maximum audio chunks in buffer
        
        # Services (will be injected via dependency injection)
        self.chatbot_a = None
        self.voice_processor = None
        self.elevenlabs_client = None
        
        logger.info(f"AudioHandler initialized for client: {self.client_id}")
        
    def inject_services(self, chatbot_a=None, voice_processor=None, **kwargs):
        """Inject required services via dependency injection"""
        self.chatbot_a = chatbot_a
        self.voice_processor = voice_processor
        
        # Initialize ElevenLabs client if not provided
        if not self.elevenlabs_client:
            self.elevenlabs_client = initialize_elevenlabs()
            
        logger.debug(f"Services injected for client: {self.client_id}")
    
    async def handle_text_message(self, message: Dict[str, Any]) -> None:
        """Enhanced text message processing with comprehensive routing"""
        try:
            message_type = message.get("type")
            
            if not message_type:
                await self._send_error("MESSAGE_TYPE_MISSING", "Message type is required")
                return
                
            logger.debug(f"Processing message type: {message_type} for client: {self.client_id}")
            
            # Route messages to appropriate handlers
            handler_map = {
                "start_conversation": self._handle_start_conversation,
                "user_message": self._handle_user_message,
                "end_conversation": self._handle_end_conversation,
                "audio_config": self._configure_audio,
                "voice_sample": self._handle_voice_sample,
                "get_conversation_status": self._handle_status_request
            }
            
            handler = handler_map.get(message_type)
            if handler:
                await handler(message)
            else:
                await self._send_error(
                    "UNKNOWN_MESSAGE_TYPE", 
                    f"Unknown message type: {message_type}",
                    {"supported_types": list(handler_map.keys())}
                )
                
        except Exception as e:
            logger.error(f"Error handling text message: {e}")
            await self._send_error("MESSAGE_PROCESSING_ERROR", str(e))
    
    async def _handle_user_message(self, message: Dict[str, Any]) -> None:
        """Process user message with AI response generation"""
        try:
            if not self.is_active:
                await self._send_error("CONVERSATION_NOT_ACTIVE", "No active conversation")
                return
                
            user_text = message.get("text", "").strip()
            audio_data = message.get("audio_data")  # Base64 encoded audio if available
            
            if not user_text and not audio_data:
                await self._send_error("EMPTY_MESSAGE", "Text or audio data required")
                return
            
            # Add to conversation history
            user_message_entry = {
                "role": "user",
                "content": user_text,
                "timestamp": datetime.now().isoformat(),
                "has_audio": bool(audio_data)
            }
            self.conversation_history.append(user_message_entry)
            
            # Process with ChatBot A if available
            if self.chatbot_a:
                try:
                    ai_response_data = await self.chatbot_a.get_response(
                        user_input=user_text,
                        session_id=self.conversation_id
                    )
                    
                    ai_response = ai_response_data.get("response", "미안해, 다시 말해줄래?")
                    
                    # Add AI response to history
                    ai_message_entry = {
                        "role": "assistant", 
                        "content": ai_response,
                        "timestamp": datetime.now().isoformat(),
                        "metadata": {
                            "story_elements_found": ai_response_data.get("story_elements_found", []),
                            "conversation_stage": ai_response_data.get("conversation_stage", "unknown"),
                            "completion_percentage": ai_response_data.get("completion_percentage", 0)
                        }
                    }
                    self.conversation_history.append(ai_message_entry)
                    
                except Exception as e:
                    logger.error(f"ChatBot A processing failed: {e}")
                    ai_response = "앗, 잠시 문제가 있었어. 다시 말해줄래?"
                    ai_response_data = {}
            else:
                ai_response = "안녕! 재미있는 이야기를 들려줘!"
                ai_response_data = {}
            
            # Generate TTS audio
            audio_url = None
            if self.audio_config.get("tts_enabled", True):
                audio_data = await self._generate_tts_audio(ai_response)
                if audio_data:
                    audio_base64 = base64.b64encode(audio_data).decode("utf-8")
                    audio_url = f"data:audio/{self.audio_config['audio_format']};base64,{audio_base64}"
            
            # Send response
            response = {
                "type": "ai_response",
                "status": "success",
                "data": {
                    "text": ai_response,
                    "audio_url": audio_url,
                    "user_text": user_text,
                    "conversation_id": self.conversation_id,
                    "timestamp": datetime.now().isoformat()
                },
                "metadata": ai_response_data.get("metadata", {})
            }
            
            await self.websocket.send_json(response)
            logger.debug(f"AI response sent to client: {self.client_id}")
            
        except Exception as e:
            logger.error(f"Error handling user message: {e}")
            await self._send_error("USER_MESSAGE_ERROR", str(e))
    
    async def _handle_start_conversation(self, message: Dict[str, Any]) -> None:
        """Initialize conversation with enhanced setup"""
        try:
            payload = message.get("payload", {})
            
            # Generate conversation ID if not provided
            self.conversation_id = (
                message.get("conversation_id") or 
                f"conv_{self.client_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            
            # Extract child information
            child_name = payload.get("child_name", "친구")
            child_age = payload.get("age", 7)
            child_interests = payload.get("interests", [])
            
            # Store session metadata
            self.session_metadata = {
                "child_name": child_name,
                "child_age": child_age, 
                "child_interests": child_interests,
                "start_time": datetime.now().isoformat(),
                "client_id": self.client_id
            }
            
            # Initialize ChatBot A session if available
            greeting_text = f"안녕 {child_name}아! 난 쫑이야!"
            if self.chatbot_a:
                try:
                    session_id = await self.chatbot_a.initialize_chat(
                        child_name=child_name,
                        child_age=child_age,
                        child_interests=child_interests
                    )
                    self.conversation_id = session_id
                    greeting_text = f"안녕 {child_name}아! 난 쫑이야! 오늘 재미있는 이야기 만들어보자!"
                except Exception as e:
                    logger.error(f"ChatBot A initialization failed: {e}")
            
            self.is_active = True
            
            # Generate welcome TTS
            audio_url = None
            if self.audio_config.get("tts_enabled", True):
                audio_data = await self._generate_tts_audio(greeting_text)
                if audio_data:
                    audio_base64 = base64.b64encode(audio_data).decode("utf-8")
                    audio_url = f"data:audio/{self.audio_config['audio_format']};base64,{audio_base64}"
            
            response = {
                "type": "conversation_started",
                "status": "success",
                "data": {
                    "conversation_id": self.conversation_id,
                    "greeting_text": greeting_text,
                    "audio_url": audio_url,
                    "child_name": child_name,
                    "session_metadata": self.session_metadata
                }
            }
            
            await self.websocket.send_json(response)
            logger.info(f"Conversation started for {child_name} (client: {self.client_id})")
            
        except Exception as e:
            logger.error(f"Error starting conversation: {e}")
            await self._send_error("CONVERSATION_START_ERROR", str(e))
    
    async def _handle_end_conversation(self, message: Dict[str, Any] = None) -> None:
        """End conversation with proper cleanup"""
        try:
            if not self.is_active:
                await self._send_error("CONVERSATION_NOT_ACTIVE", "No active conversation to end")
                return
            
            conversation_summary = {
                "conversation_id": self.conversation_id,
                "duration_minutes": 0,
                "message_count": len(self.conversation_history),
                "child_name": self.session_metadata.get("child_name", "unknown")
            }
            
            # Calculate duration if start time available
            if self.session_metadata.get("start_time"):
                try:
                    start_time = datetime.fromisoformat(self.session_metadata["start_time"])
                    duration = datetime.now() - start_time
                    conversation_summary["duration_minutes"] = round(duration.total_seconds() / 60, 1)
                except:
                    pass
            
            # Close ChatBot A session if available
            if self.chatbot_a and self.conversation_id:
                try:
                    session_summary = await self.chatbot_a.end_conversation(self.conversation_id)
                    conversation_summary.update(session_summary)
                except Exception as e:
                    logger.error(f"Error ending ChatBot A session: {e}")
            
            # Reset state
            self.is_active = False
            self.audio_buffer.clear()
            
            response = {
                "type": "conversation_ended",
                "status": "success", 
                "data": {
                    "conversation_id": self.conversation_id,
                    "summary": conversation_summary,
                    "timestamp": datetime.now().isoformat()
                }
            }
            
            await self.websocket.send_json(response)
            logger.info(f"Conversation ended for client: {self.client_id}")
            
            # Keep conversation_id for potential story generation
            
        except Exception as e:
            logger.error(f"Error ending conversation: {e}")
            await self._send_error("CONVERSATION_END_ERROR", str(e))
    
    async def _configure_audio(self, message: Dict[str, Any]) -> None:
        """Configure audio settings with validation"""
        try:
            config = message.get("config", {})
            
            # Validate and update audio configuration
            valid_configs = {
                "tts_enabled": bool,
                "voice_cloning_enabled": bool,
                "audio_format": lambda x: x in ["mp3", "wav", "ogg"],
                "sample_rate": lambda x: x in [22050, 44100, 48000],
                "bit_rate": lambda x: 64 <= x <= 320
            }
            
            updated_configs = {}
            for key, value in config.items():
                if key in valid_configs:
                    validator = valid_configs[key]
                    if callable(validator) and validator != bool:
                        if validator(value):
                            updated_configs[key] = value
                            self.audio_config[key] = value
                        else:
                            logger.warning(f"Invalid value for {key}: {value}")
                    elif validator == bool:
                        updated_configs[key] = bool(value)
                        self.audio_config[key] = bool(value)
                    else:
                        updated_configs[key] = value
                        self.audio_config[key] = value
            
            response = {
                "type": "audio_config_updated",
                "status": "success",
                "data": {
                    "updated_configs": updated_configs,
                    "current_config": self.audio_config
                }
            }
            
            await self.websocket.send_json(response)
            logger.debug(f"Audio config updated for client: {self.client_id}")
            
        except Exception as e:
            logger.error(f"Error configuring audio: {e}")
            await self._send_error("AUDIO_CONFIG_ERROR", str(e))
    
    async def _handle_voice_sample(self, message: Dict[str, Any]) -> None:
        """Handle voice sample collection for voice cloning"""
        try:
            if not self.voice_processor:
                await self._send_error("SERVICE_UNAVAILABLE", "Voice processing service not available")
                return
            
            audio_data = message.get("audio_data")  # Base64 encoded
            if not audio_data:
                await self._send_error("MISSING_AUDIO_DATA", "Audio data required for voice sample")
                return
            
            # Decode audio data
            try:
                audio_bytes = base64.b64decode(audio_data)
            except Exception as e:
                await self._send_error("INVALID_AUDIO_DATA", f"Failed to decode audio data: {e}")
                return
            
            # Collect voice sample
            user_id = self.session_metadata.get("child_name", self.client_id)
            success = await self.voice_processor.collect_user_audio_sample(
                user_id=user_id,
                audio_data=audio_bytes
            )
            
            if success:
                sample_count = self.voice_processor.get_sample_count(user_id)
                ready_for_cloning = self.voice_processor.is_ready_for_cloning(user_id)
                
                response_data = {
                    "sample_collected": True,
                    "samples_collected": sample_count,
                    "samples_needed": 5,
                    "ready_for_cloning": ready_for_cloning
                }
                
                # Auto-create voice clone when ready
                if ready_for_cloning:
                    voice_id, error = await self.voice_processor.create_instant_voice_clone(
                        user_id=user_id,
                        voice_name=f"{user_id}_voice_clone"
                    )
                    
                    if voice_id:
                        response_data.update({
                            "voice_cloned": True,
                            "voice_id": voice_id,
                            "message": f"{user_id}의 음성이 성공적으로 복제되었어요!"
                        })
                    else:
                        response_data.update({
                            "voice_cloned": False,
                            "error": error
                        })
                
                response = {
                    "type": "voice_sample_processed",
                    "status": "success",
                    "data": response_data
                }
            else:
                response = {
                    "type": "voice_sample_processed", 
                    "status": "error",
                    "error": "Failed to collect voice sample"
                }
            
            await self.websocket.send_json(response)
            
        except Exception as e:
            logger.error(f"Error handling voice sample: {e}")
            await self._send_error("VOICE_SAMPLE_ERROR", str(e))
    
    async def _handle_status_request(self, message: Dict[str, Any]) -> None:
        """Handle conversation status request"""
        try:
            status_data = {
                "client_id": self.client_id,
                "conversation_id": self.conversation_id,
                "is_active": self.is_active,
                "message_count": len(self.conversation_history),
                "audio_config": self.audio_config,
                "session_metadata": self.session_metadata,
                "timestamp": datetime.now().isoformat()
            }
            
            # Add ChatBot A status if available
            if self.chatbot_a and self.conversation_id:
                try:
                    # This would need to be implemented in ChatBot A
                    # chatbot_status = await self.chatbot_a.get_session_status(self.conversation_id)
                    # status_data["chatbot_status"] = chatbot_status
                    pass
                except:
                    pass
            
            response = {
                "type": "conversation_status",
                "status": "success",
                "data": status_data
            }
            
            await self.websocket.send_json(response)
            
        except Exception as e:
            logger.error(f"Error handling status request: {e}")
            await self._send_error("STATUS_REQUEST_ERROR", str(e))
    
    async def handle_binary_data(self, audio_data: bytes) -> None:
        """Handle binary audio data for real-time processing"""
        try:
            if not self.is_active:
                await self._send_error("CONVERSATION_NOT_ACTIVE", "No active conversation")
                return
            
            # Add to audio buffer
            self.audio_buffer.append(audio_data)
            
            # Maintain buffer size
            if len(self.audio_buffer) > self.max_buffer_size:
                self.audio_buffer.pop(0)
            
            # Process for voice cloning if enabled
            if (self.audio_config.get("voice_cloning_enabled", True) and 
                self.voice_processor):
                
                user_id = self.session_metadata.get("child_name", self.client_id)
                await self.voice_processor.collect_user_audio_sample(
                    user_id=user_id,
                    audio_data=audio_data
                )
            
            logger.debug(f"Binary audio data processed for client: {self.client_id}")
            
        except Exception as e:
            logger.error(f"Error handling binary data: {e}")
            await self._send_error("BINARY_DATA_ERROR", str(e))
    
    async def _generate_tts_audio(self, text: str) -> Optional[bytes]:
        """Generate TTS audio with cloned voice if available"""
        try:
            if not self.elevenlabs_client:
                logger.warning("ElevenLabs client not available for TTS")
                return None
            
            # Use cloned voice if available
            voice_id = None
            if self.voice_processor and self.session_metadata.get("child_name"):
                user_id = self.session_metadata["child_name"]
                voice_id = self.voice_processor.get_user_voice_id(user_id)
            
            audio_data, _ = await generate_speech(
                client=self.elevenlabs_client,
                text=text,
                voice_id=voice_id,  # Will use default voice if None
                output_path=None  # Return audio data directly
            )
            
            return audio_data
            
        except Exception as e:
            logger.error(f"TTS generation failed: {e}")
            return None
    
    async def _send_error(
        self, 
        error_code: str, 
        error_message: str, 
        details: Dict[str, Any] = None
    ) -> None:
        """Send error message to client"""
        try:
            error_response = {
                "type": "error",
                "status": "error",
                "error": {
                    "code": error_code,
                    "message": error_message,
                    "timestamp": datetime.now().isoformat(),
                    "client_id": self.client_id
                }
            }
            
            if details:
                error_response["error"]["details"] = details
            
            await self.websocket.send_json(error_response)
            logger.warning(f"Error sent to client {self.client_id}: {error_code} - {error_message}")
            
        except Exception as e:
            logger.error(f"Failed to send error message: {e}")
    
    def cleanup(self) -> None:
        """Cleanup resources and reset state"""
        try:
            self.is_active = False
            self.audio_buffer.clear()
            self.conversation_history.clear()
            self.session_metadata.clear()
            
            logger.info(f"AudioHandler cleanup completed for client: {self.client_id}")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


# Factory function for creating audio handlers
def create_audio_handler(websocket: WebSocket, client_id: str = None) -> AudioHandler:
    """Create configured AudioHandler instance"""
    return AudioHandler(websocket=websocket, client_id=client_id)


# WebSocket handler function for compatibility
async def handler_audio_websocket(
    websocket: WebSocket,
    user_info: Dict[str, Any],
    client_id: str,
    chatbot_a=None,
    voice_processor=None,
    **kwargs
) -> None:
    """
    Main WebSocket audio handler function
    
    This function maintains compatibility with existing routing while
    providing full functionality through the AudioHandler class.
    """
    audio_handler = create_audio_handler(websocket, client_id)
    audio_handler.inject_services(
        chatbot_a=chatbot_a,
        voice_processor=voice_processor,
        **kwargs
    )
    
    try:
        logger.info(f"WebSocket audio connection established: {client_id}")
        
        # Send connection success message
        await websocket.send_json({
            "type": "connection_established",
            "status": "success",
            "data": {
                "client_id": client_id,
                "user_info": user_info,
                "timestamp": datetime.now().isoformat()
            }
        })
        
        # Main message processing loop
        while True:
            try:
                # Try to receive JSON message first
                try:
                    message = await websocket.receive_json()
                    await audio_handler.handle_text_message(message)
                except:
                    # If JSON fails, try binary data
                    audio_data = await websocket.receive_bytes()
                    await audio_handler.handle_binary_data(audio_data)
                    
            except WebSocketDisconnect:
                logger.info(f"WebSocket disconnected normally: {client_id}")
                break
            except Exception as e:
                logger.error(f"Message processing error for {client_id}: {e}")
                await audio_handler._send_error("MESSAGE_PROCESSING_ERROR", str(e))
                
    except Exception as e:
        logger.error(f"WebSocket connection error for {client_id}: {e}")
    finally:
        # Cleanup
        audio_handler.cleanup()
        logger.info(f"WebSocket audio connection closed: {client_id}")