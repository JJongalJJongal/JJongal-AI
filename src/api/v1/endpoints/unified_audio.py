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
            "end_conversation": self.handle_end_conversation,
            "audio_config": self.handle_audio_config,
            # ⭐ NEW: Family voice handlers
            "family_voice_upload": self.handle_family_voice_upload,
            "family_voice_sample": self.handle_family_voice_sample,
            "family_voice_status": self.handle_family_voice_status,
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
                    self._generate_story_with_progress(
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

    # ⭐ NEW: Family voice message handlers
    async def handle_family_voice_upload(
        self,
        websocket: WebSocket,
        client_id: str,
        message: Dict[str, Any],
        services: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Handle family voice upload initiation"""
        try:
            data = message.get("data", {})
            family_member = data.get("family_member")  # "mom" | "dad"
            voice_name = data.get("voice_name", f"{family_member}_voice")
            
            # Validate family member
            if family_member not in ["mom", "dad"]:
                return WebSocketMessageBuilder.error(
                    "INVALID_FAMILY_MEMBER",
                    "Family member must be 'mom' or 'dad'",
                    {"family_member": family_member}
                )
            
            # Initialize family voice collection for this session
            if client_id not in self.active_sessions:
                self.active_sessions[client_id] = {}
            
            self.active_sessions[client_id][f"{family_member}_voice"] = {
                "family_member": family_member,
                "voice_name": voice_name,
                "samples_collected": 0,
                "samples_needed": 5,
                "status": "collecting_samples",
                "started_at": datetime.now().isoformat()
            }
            
            logger.info(f"Family voice upload started - client: {client_id}, member: {family_member}")
            
            return WebSocketMessageBuilder.success(
                "family_voice_upload_started",
                {
                    "family_member": family_member,
                    "voice_name": voice_name,
                    "samples_needed": 5,
                    "message": f"{family_member.title()} 음성 수집을 시작합니다. 5개 샘플이 필요합니다."
                }
            )
            
        except Exception as e:
            logger.error(f"Family voice upload handler error: {e}")
            return WebSocketMessageBuilder.error(
                "FAMILY_VOICE_UPLOAD_ERROR",
                "Family voice upload processing failed"
            )

    async def handle_family_voice_sample(
        self,
        websocket: WebSocket,
        client_id: str,
        message: Dict[str, Any],
        services: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Handle individual family voice sample upload"""
        try:
            data = message.get("data", {})
            family_member = data.get("family_member")
            sample_number = data.get("sample_number", 1)
            audio_data = data.get("audio_data")  # Base64 encoded
            
            if not audio_data:
                return WebSocketMessageBuilder.error(
                    "MISSING_AUDIO_DATA",
                    "Audio data is required for voice sample"
                )
            
            # Get voice cloning processor
            voice_processor = services.get("voice_cloning_processor")
            if not voice_processor:
                return WebSocketMessageBuilder.error(
                    "SERVICE_UNAVAILABLE",
                    "Voice processing service not available"
                )
            
            # Get user_id from session or use client_id
            user_id = client_id  # In real app, get from authenticated session
            voice_name = self.active_sessions.get(client_id, {}).get(f"{family_member}_voice", {}).get("voice_name", f"{family_member}_voice")
            
            # Decode and process audio sample
            import base64
            audio_bytes = base64.b64decode(audio_data)
            
            # Collect family voice sample
            result = await voice_processor.collect_family_voice_sample(
                user_id=user_id,
                family_member=family_member,
                voice_name=voice_name,
                audio_data=audio_bytes,
                file_format="mp3"
            )
            
            if result["success"]:
                # Update session info
                if client_id in self.active_sessions and f"{family_member}_voice" in self.active_sessions[client_id]:
                    self.active_sessions[client_id][f"{family_member}_voice"]["samples_collected"] = result["samples_collected"]
                
                # Check if ready for cloning
                if result["ready_for_cloning"]:
                    # Start background cloning process
                    asyncio.create_task(
                        self._process_family_voice_cloning(
                            websocket,
                            client_id,
                            user_id,
                            family_member,
                            voice_processor
                        )
                    )
                    
                    return WebSocketMessageBuilder.success(
                        "family_voice_cloning_started",
                        {
                            "family_member": family_member,
                            "samples_collected": result["samples_collected"],
                            "message": f"{family_member.title()} 음성 클로닝을 시작합니다..."
                        }
                    )
                else:
                    return WebSocketMessageBuilder.progress(
                        "family_voice_progress",
                        current=result["samples_collected"],
                        total=result["samples_needed"],
                        message=f"{family_member.title()} 음성 샘플 {result['samples_collected']}/{result['samples_needed']} 수집 완료"
                    )
            else:
                return WebSocketMessageBuilder.error(
                    "VOICE_SAMPLE_FAILED",
                    result["error"]
                )
                
        except Exception as e:
            logger.error(f"Family voice sample handler error: {e}")
            return WebSocketMessageBuilder.error(
                "FAMILY_VOICE_SAMPLE_ERROR",
                "Voice sample processing failed"
            )

    async def handle_family_voice_status(
        self,
        websocket: WebSocket,
        client_id: str,
        message: Dict[str, Any],
        services: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Handle family voice status request"""
        try:
            data = message.get("data", {})
            family_member = data.get("family_member")
            
            # Get voice cloning processor
            voice_processor = services.get("voice_cloning_processor")
            if not voice_processor:
                return WebSocketMessageBuilder.error(
                    "SERVICE_UNAVAILABLE",
                    "Voice processing service not available"
                )
            
            user_id = client_id  # In real app, get from authenticated session
            
            if family_member:
                # Get specific family member status
                status = voice_processor.get_family_voice_status(user_id, family_member)
            else:
                # Get all family voices status
                status = voice_processor.get_family_voice_status(user_id)
            
            return WebSocketMessageBuilder.success(
                "family_voice_status",
                status
            )
            
        except Exception as e:
            logger.error(f"Family voice status handler error: {e}")
            return WebSocketMessageBuilder.error(
                "FAMILY_VOICE_STATUS_ERROR",
                "Status retrieval failed"
            )

    async def _process_family_voice_cloning(
        self,
        websocket: WebSocket,
        client_id: str,
        user_id: str,
        family_member: str,
        voice_processor
    ):
        """Background task for family voice cloning"""
        try:
            # Send cloning progress
            await websocket.send_json(
                WebSocketMessageBuilder.progress(
                    "family_voice_cloning_progress",
                    current=1,
                    total=3,
                    message=f"{family_member.title()} 음성 분석 중..."
                )
            )
            
            await asyncio.sleep(2)  # Simulate processing time
            
            await websocket.send_json(
                WebSocketMessageBuilder.progress(
                    "family_voice_cloning_progress",
                    current=2,
                    total=3,
                    message=f"{family_member.title()} 음성 클로닝 중..."
                )
            )
            
            # Actual voice cloning
            result = await voice_processor.create_family_voice_clone(user_id, family_member)
            
            if result["success"]:
                await websocket.send_json(
                    WebSocketMessageBuilder.success(
                        "family_voice_ready",
                        {
                            "voice_id": result["voice_id"],
                            "family_member": family_member,
                            "voice_name": result["voice_name"],
                            "message": f"{family_member.title()} 음성 클로닝이 완료되었습니다!"
                        }
                    )
                )
                
                # Update session status
                if client_id in self.active_sessions and f"{family_member}_voice" in self.active_sessions[client_id]:
                    self.active_sessions[client_id][f"{family_member}_voice"]["status"] = "completed"
                    self.active_sessions[client_id][f"{family_member}_voice"]["voice_id"] = result["voice_id"]
                    
                logger.info(f"Family voice cloning completed - client: {client_id}, member: {family_member}, voice_id: {result['voice_id']}")
            else:
                await websocket.send_json(
                    WebSocketMessageBuilder.error(
                        "FAMILY_VOICE_CLONING_FAILED",
                        result["error"],
                        {"family_member": family_member}
                    )
                )
                
        except Exception as e:
            logger.error(f"Family voice cloning background task error: {e}")
            await websocket.send_json(
                WebSocketMessageBuilder.error(
                    "FAMILY_VOICE_CLONING_ERROR",
                    "Voice cloning processing failed",
                    {"family_member": family_member}
                )
            )

    async def _generate_story_with_progress(
        self,
        websocket: WebSocket,
        client_id: str,
        session: Dict[str, Any],
        collaborator,
    ):
        """Real-time story generation with detailed progress tracking"""
        try:

            await websocket.send_json(
                WebSocketMessageBuilder.progress(
                    "story_generation_progress",
                    current=1,
                    total=5,
                    message="쫑이가 대화 내용을 분석하고 있어요...",
                )
            )
            await asyncio.sleep(2)  # Simulate analysis time

            await websocket.send_json(
                WebSocketMessageBuilder.progress(
                    "story_generation_progress",
                    current=2,
                    total=5,
                    message="아리가 동화 구조를 설계하고 있어요...",
                )
            )
            await asyncio.sleep(3)

            await websocket.send_json(
                WebSocketMessageBuilder.progress(
                    "story_generation_progress",
                    current=3,
                    total=5,
                    message="등장인물과 배경을 만들고 있어요...",
                )
            )
            await asyncio.sleep(2)

            await websocket.send_json(
                WebSocketMessageBuilder.progress(
                    "story_generation_progress",
                    current=4,
                    total=5,
                    message="멋진 동화를 완성하고 있어요...",
                )
            )

            story_id = await collaborator.create_story(
                child_name=session.get("child_name", "friend"),
                age=session.get("age", 7),
                conversation_data={
                    "conversation_history": session.get("conversation_history", {}),
                    "interests": session.get("interests", []),
                },
            )

            await websocket.send_json(
                WebSocketMessageBuilder.progress(
                    "story_generation_progress",
                    current=5,
                    total=5,
                    message="그림과 음성을 추가하고 있어요...",
                )
            )
            await asyncio.sleep(2)

            if story_id:
                await websocket.send_json(
                    WebSocketMessageBuilder.success(
                        "story_completed",
                        {
                            "story_id": story_id,
                            "message": f"{session.get('child_name', '친구')}만의 특별한 동화가 완성되었어요!",
                            "api_url": f"/api/v1/stories/{story_id}",
                            "completion_time": datetime.now().isoformat(),
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
