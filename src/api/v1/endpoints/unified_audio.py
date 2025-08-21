import asyncio
import base64
import json
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, Query

from ..dependencies import (
    verify_jwt_token_websocket,
    get_connection_manager,
    get_chatbot_a,
    get_voice_cloning_processor,
    get_jjong_ari_collaborate,
    websocket_session_context,
    Websocket_error_handler,
)

from src.shared.utils.logging import get_module_logger
from src.shared.utils.websocket import ConnectionManager

logger = get_module_logger(__name__)

router = APIRouter(prefix="/audio", tags=["WebSocket", "Audio", "JjongAl"])


class WebSocketMessageBuilder:

    @staticmethod
    def success(
        message_type: str, data: Dict[str, Any] = None, message: str = None
    ) -> Dict[str, Any]:
        # Create success response message
        response = {
            "type": message_type,
            "status": "success",
            "timestamp": datetime.now().isoformat(),
        }

        if data:
            response.update(data)
        if message:
            response["message"] = message
        return response

    @staticmethod
    def error(
        error_code: str, error_message: str, details: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        # Create error response message
        response = {
            "type": "error",
            "status": "error",
            "error_code": error_code,
            "error_message": error_message,
            "timestamp": datetime.now().isoformat(),
        }

        if details:
            response["details"] = details
        return response

    @staticmethod
    def progress(
        progress_type: str, current: int, total: int, message: str = None
    ) -> Dict[str, Any]:
        # Create progress response message
        response = {
            "type": progress_type,
            "status": "progress",
            "progress": {
                "current": current,
                "total": total,
                "percentage": round((current / total) * 100, 1) if total > 0 else 0,
            },
            "timestamp": datetime.now().isoformat(),
        }
        if message:
            response["message"] = message
        return response


class JjongAlAudioHandler:
    def __init__(self):
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.message_handlers = {
            "start_conversation": self.handle_start_conversation,
            "user_message": self.handle_user_message,
            "end_conversation": self.handle_end_versation,
            "audio_config": self.handle_audio_config,
        }

    async def handle_json_message(
        self,
        websocket: WebSocket,
        client_id: str,
        message: Dict[str, Any],
        services: Dict[str, Any],
    ) -> None:
        # Handle JSON message with dependency injection
        try:
            message_type = message.get("type")

            if message_type in self.message_handlers:
                handler = self.message_handlers[message_type]
                response = await handler(websocket, client_id, message, services)

                if response:
                    await websocket.send_json(response)
            else:
                await websocket.send_json(
                    WebSocketMessageBuilder.error(
                        "UNKNOWN_MESSAGE_TYPE", f"Unknown message type: {message_type}"
                    )
                )
        except Exception as e:
            logger.error(f"JSON message handling failed : {e}")
            await websocket.send_json(
                WebSocketMessageBuilder.error(
                    "MESSAGE_PROCESSING_ERROR", f"Message processing error: {str(e)}"
                )
            )

    async def handle_binary_data(
        self,
        websocket: WebSocket,
        client_id: str,
        audio_data: bytes,
        services: Dict[str, Any],
    ) -> None:
        # Handle binary audio data for voice cloning
        try:
            voice_processor = services.get("voice_processor")
            if not voice_processor:
                await websocket.send_json(
                    WebSocketMessageBuilder.error(
                        "VOICE_PROCESSOR_UNAVAILABLE", "Voice processor not available"
                    )
                )
                return

            session = self.get_session(client_id)
            if not session:
                await websocket.send_json(
                    WebSocketMessageBuilder.error(
                        "SESSION_NOT_FOUND", "Session not found"
                    )
                )
                return

            child_name = session.get("child_name", "unknown")

            # Collect voice sample
            success = await voice_processor.collect_user_audio_sample(
                user_id=child_name, audio_data=audio_data
            )

            if success:
                sample_count = voice_processor.get_sample_count(child_name)
                ready_for_cloning = voice_processor.is_ready_for_cloning(child_name)

                response = WebSocketMessageBuilder.progress(
                    "voice_clone_progress",
                    current=sample_count,
                    total=5,
                    message=f"Collecting voice samples... ({sample_count}/5)",
                )

                # Create voice clone when 5 samples are collected
                if ready_for_cloning:
                    voice_id, error = await voice_processor.create_instant_voice_clone(
                        user_id=child_name, voice_name=f"{child_name}_voice_clone"
                    )

                    if voice_id:
                        response = WebSocketMessageBuilder.success(
                            "voice_clone_ready",
                            {
                                "voice_id": voice_id,
                                "message": f"{child_name}'s voice successfully cloned!",
                            },
                        )
                        # Update session with voice ID
                        session["voice_id"] = voice_id
                    else:
                        response = WebSocketMessageBuilder.error(
                            "VOICE_CLONE_FAILED", f"Voice cloning failed: {error}"
                        )
                await websocket.send_json(response)
            else:
                await websocket.send_json(
                    WebSocketMessageBuilder.error(
                        "AUDIO_COLLECTION_FAILED", "Voice sample collection failed"
                    )
                )
        except Exception as e:
            logger.error(f"Binary audio processing failed: {e}")
            await websocket.send_json(
                WebSocketMessageBuilder.error(
                    "BINARY_PROCESSING_ERROR", f"Audio processing error: {str(e)}"
                )
            )

    async def handle_start_conversation(
        self,
        websocket: WebSocket,
        client_id: str,
        message: Dict[str, Any],
        services: Dict[str, Any],
    ) -> Dict[str, Any]:
        # Handle conversation start
        try:
            chatbot_a = services.get("chatbot_a")
            if not chatbot_a:
                return WebSocketMessageBuilder.error(
                    "CHATBOT_UNAVAILABLE", "ChatBot A not available"
                )
            payload = message.get("payload", {})
            child_name = payload.get("child_name", "friend")
            age = payload.get("age", 7)
            interests = payload.get("interests", [])

            # Initialize ChatBot A session
            greeting = await chatbot_a.initialize_chat(
                child_name=child_name, age=age, interests=interests
            )

            # Store session
            self.active_sessions[client_id] = {
                "child_name": child_name,
                "age": age,
                "interests": interests,
                "conversation_active": True,
                "conversation_history": [],
                "created_at": datetime.now(),
            }

            # Generate TTS audio
            audio_data = await self._generate_tts_audio(greeting, services)
            audio_urls = None
            if audio_data:
                audio_base64 = base64.b64encode(audio_data).decode("utf-8")
                audio_url = f"data:audio/mp3;base64, {audio_base64}"

            return WebSocketMessageBuilder.success(
                "conversation_started",
                {
                    "text": greeting,
                    "audio_url": audio_url,
                    "child_name": child_name,
                    "conversation_id": client_id,
                },
            )
        except Exception as e:
            logger.error(f"Conversation start failed: {e}")
            return WebSocketMessageBuilder.error(
                "CONVERSATION_START_ERROR", f"Conversation start error: {str(e)}"
            )

    async def handle_user_message(
        self,
        websocket: WebSocket,
        client_id: str,
        message: Dict[str, Any],
        services: Dict[str, Any],
    ) -> Dict[str, Any]:
        # Handle user message (Google STT result)
        try:
            text = message.get("text", "").strip()
            if not text:
                return WebSocketMessageBuilder.error(
                    "EMPTY_MESSAGE", "Empty message cannot be processed"
                )
            session = self.get_session(client_id)
            if not session or not session.get("conversation_active"):
                return WebSocketMessageBuilder.error(
                    "CONVERSATION_NOT_ACTIVE", "No active conversation found"
                )

            chatbot_a = services.get("chatbot_a")
            if not chatbot_a:
                return WebSocketMessageBuilder.error(
                    "CHATBOT_UNAVAILABLE", "ChatBot A not available"
                )

            # Add user message to history
            session["conversation_history"].append(
                {
                    "role": "user",
                    "context": text,
                    "timestamp": datetime.now().isoformat(),
                }
            )

            # Generate AI response
            ai_response = await chatbot_a.get_response(text)

            # Add AI response to history
            session["conversation_history"].append(
                {
                    "role": "assistant",
                    "content": ai_response,
                    "timestamp": datetime.now().isoformat(),
                }
            )

            # Generate TTS audio
            audio_data = await self._generate_tts_audio(ai_response, services)
            audio_urls = None
            if audio_data:
                audio_base64 = base64.b64encode(audio_data).decode("utf-8")
                audio_url = f"data:audio/mp3;base64,{audio_base64}"

            return WebSocketMessageBuilder.success(
                "jjong_response",
                {"text": ai_response, "audio_url": audio_url, "user_text": text},
            )
        except Exception as e:
            logger.error(f"User message processing failed: {e}")
            return WebSocketMessageBuilder.error(
                "MESSAGE_PROCESSING_ERROR", f"Message processing error: {str(e)}"
            )

    async def handle_end_conversation(
        self,
        websocket: WebSocket,
        client_id: str,
        message: Dict[str, Any],
        services: Dict[str, Any],
    ) -> Dict[str, Any]:
        try:
            session = self.get_session(client_id)
            if not session:
                return WebSocketMessageBuilder.error(
                    "SESSION_NOT_FOUND", "Session not found"
                )
            # Mark conversation as ended
            session["conversation_active"] = False

            # Start story generation asynchronously
            collaborator = services.get("collaborator")
            if collaborator and session.get("conversation_history"):
                # Send story generation start notification
                await websocket.send_json(
                    WebSocketMessageBuilder.success(
                        "story_generation_started",
                        message="Creating your fairy tale...",
                    )
                )

                # Generate story asynchronously
                asyncio.create_task(
                    self._generate_story_async(
                        websocket, client_id, session, collaborator
                    )
                )

                return WebSocketMessageBuilder.success(
                    "conversation_ended",
                    {"message": "Conversation ended. Starting story generation."},
                )
            else:
                return WebSocketMessageBuilder.success(
                    "conversation_ended", {"message": "Conversation ended"}
                )
        except Exception as e:
            logger.error(f"Conversation end processing failed : {e}")
            return WebSocketMessageBuilder.error(
                "CONVERSATION_END_ERROR", f"Conversation end error: {str(e)}"
            )

    async def handle_audio_config(
        self,
        websocket: WebSocket,
        client_id: str,
        message: Dict[str, Any],
        services: Dict[str, Any],
    ) -> Dict[str, Any]:
        # handle audio configuration
        try:
            config = message.get("config", {})
            session = self.get_session(client_id)
            if session:
                session["audio_config"] = config
            return WebSocketMessageBuilder.success(
                "audio_config_updated", {"config": config}
            )
        except Exception as e:
            logger.error(f"Audio config failed: {e}")
            return WebSocketMessageBuilder.error(
                "AUDIO_CONFIG_ERROR", f"Audio configuration error: {str(e)}"
            )

    async def _generate_tts_audio(
        self, text: str, services: Dict[str, Any]
    ) -> Optional[bytes]:
        try:
            from src.shared.utils.audio import generate_speech, initialize_elevenlabs

            elevenlabs_client = initialize_elevenlabs()

            if not elevenlabs_client:
                return None
            audio_data, _ = await generate_speech(
                client=elevenlabs_client,
                text=text,
                output_path="output/temp/audio/tts_temp",
            )

            return audio_data

        except Exception as e:
            logger.warning(f"TTS generation failed: {e}")
            return None

    async def _generate_story_async(
        self,
        websocket: WebSocket,
        client_id: str,
        session: Dict[str, Any],
        collaborator,
    ):
        try:
            story_id = await collaborator.create_story(
                child_name=session.get("child_name", "friend"),
                age=session.get("age", 7),
                conversation_data={
                    "conversation_history": session.get("conversation_history", {}),
                    "interests": session.get("interests", []),
                },
            )

            if story_id:
                await websocket.send_json(
                    WebSocketMessageBuilder.success(
                        "story_completed",
                        {
                            "story_id": story_id,
                            "message": "Your fairy tale is ready!",
                            "api_url": f"/api/v1/stories/{story_id}",
                        },
                    )
                )
            else:
                await websocket.send_json(
                    WebSocketMessageBuilder.error(
                        "STORY_GENERATION_FAILED", "Story generation failed"
                    )
                )
        except Exception as e:
            logger.error(f"Story generation failed: {e}")
            await websocket.send_json(
                WebSocketMessageBuilder.error(
                    "STORY_GENERATION_ERROR", f"Story generation error: {str(e)}"
                )
            )

    def get_session(self, client_id: str) -> Optional[Dict[str, Any]]:
        return self.active_sessions.get(client_id)

    def cleanup_session(self, client_id: str) -> None:
        if client_id in self.active_sessions:
            del self.active_sessions[client_id]
            logger.info(f"Session cleaned up : {client_id}")


audio_handler = JjongAlAudioHandler()


@router.websocket("/")
@Websocket_error_handler
async def unified_audio_websocket(
    websocket: WebSocket,
    user_info: Dict[str, Any] = Depends(verify_jwt_token_websocket),
    connection_manager: ConnectionManager = Depends(get_connection_manager),
    chatbot_a=Depends(get_chatbot_a),
    voice_processor=Depends(get_voice_cloning_processor),
    collaborator=Depends(get_jjong_ari_collaborate),
):

    client_id = f"jjongal_{user_info['user_id']}_{datetime.now().timestamp()}"

    # Prepare services for dependency injection
    services = {
        "chatbot_a": chatbot_a,
        "voice_processor": voice_processor,
        "collaborator": collaborator,
        "connection_manager": connection_manager,
    }

    async with websocket_session_context(client_id, websocket, connection_manager):
        try:
            logger.info(f"JjongAl WebSocket connected: {client_id}")

            # Connection success message
            await websocket.send_json(
                WebSocketMessageBuilder.success(
                    "connection_established",
                    {
                        "client_id": client_id,
                        "user_id": user_info.get("user_id"),
                        "message": "Connected to JjongAl Audio Service",
                    },
                )
            )

            # Message processing loop
            while True:
                try:
                    try:
                        message = await websocket.receive_json()
                        await audio_handler.handle_json_message(
                            websocket, client_id, message, services
                        )
                    except:
                        # handle binary data (voice clone audio)
                        audio_data = await websocket.receive_bytes()
                        await audio_handler.handle_binary_data(
                            websocket, client_id, audio_data, services
                        )
                except WebSocketDisconnect:
                    logger.info(f"JjongAl WebSocket disconnected normally: {client_id}")
                    break
                except Exception as e:
                    logger.error(f"Message processing error: {e}")
                    await websocket.send_json(
                        WebSocketMessageBuilder.error(
                            "MESSAGE_PROCESSING_ERROR",
                            "Message processing error occured",
                        )
                    )
        except Exception as e:
            logger.error(f"WebSocket connection error: {e}")
        finally:
            # cleanup
            audio_handler.cleanup_session(client_id=client_id)
            logger.info(f"JjongAl WebSocket session cleaned up: {client_id}")
