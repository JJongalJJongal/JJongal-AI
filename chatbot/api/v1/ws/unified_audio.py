"""
쫑알쫑알 통합 WebSocket API
API 문서 v1.0 기준 구현

엔드포인트: /wss/v1/audio
프로토콜: 
1. Connection 수립 (JWT 인증)
2. Conversation 시작 (JSON)
3. 텍스트 메시지 전송 (Google STT 결과)
4. Voice Clone 오디오 전송 (Binary)
5. 서버 응답 (JSON)
6. Conversation 종료 (JSON)
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, Query
from typing import Dict, Any, Optional
import json
import asyncio
from datetime import datetime
import base64

# Dependencies
from ..dependencies import verify_jwt_token, get_connection_manager
from shared.utils import ConnectionManager
from shared.utils.logging_utils import get_module_logger

# Services
from chatbot.models.chat_bot_a import ChatBotA
from chatbot.data.vector_db.core import VectorDB
from chatbot.models.voice_ws.processors.voice_cloning_processor import VoiceCloningProcessor
from shared.utils.audio_utils import generate_speech
from shared.utils.langchain_manager import langchain_manager

logger = get_module_logger(__name__)

router = APIRouter(prefix="/audio", tags=["WebSocket", "Audio", "쫑알쫑알"])

class JjongAlAudioWebSocket:
    """쫑알쫑알 통합 WebSocket 핸들러"""
    
    def __init__(self):
        self.active_sessions = {}
        self.voice_cloning_processor = VoiceCloningProcessor()
    
    async def initialize_chatbot_session(self, client_id: str, user_info: Dict[str, Any], 
                                       start_message: Dict[str, Any]) -> ChatBotA:
        """ChatBot A 세션 초기화"""
        try:
            payload = start_message.get("payload", {})
            child_name = payload.get("child_name", "친구")
            age = payload.get("age", 7)
            interests = payload.get("interests", [])
            
            # VectorDB 초기화
            import os
            chroma_base = os.getenv("CHROMA_DB_PATH", "/app/chatbot/data/vector_db")
            vector_db_path = os.path.join(chroma_base, "main")
            
            vector_db = VectorDB(
                persist_directory=vector_db_path,
                embedding_model="nlpai-lab/KURE-v1",
                use_hybrid_mode=True
            )
            
            # ChatBot A 인스턴스 생성 (새로운 LangChain 관리자 활용)
            chatbot_a = ChatBotA(
                vector_db_instance=vector_db,
                token_limit=10000,
                use_langchain=True,
                enhanced_mode=True,
                session_id=f"session_{client_id}",
                langchain_manager=langchain_manager  # 통합 관리자 주입
            )
            
            # 초기화
            greeting = chatbot_a.initialize_chat(
                child_name=child_name,
                age=age,
                interests=interests
            )
            
            # 세션 저장
            self.active_sessions[client_id] = {
                "chatbot": chatbot_a,
                "child_name": child_name,
                "age": age,
                "interests": interests,
                "conversation_active": True
            }
            
            return chatbot_a, greeting
            
        except Exception as e:
            logger.error(f"ChatBot 세션 초기화 실패: {e}")
            raise
    
    async def process_user_message(self, client_id: str, text: str) -> Dict[str, Any]:
        """사용자 메시지 처리 (Google STT 결과)"""
        try:
            session = self.active_sessions.get(client_id)
            if not session:
                raise ValueError("활성 세션을 찾을 수 없습니다")
            
            chatbot = session["chatbot"]
            
            # ChatBot A 응답 생성
            ai_response = chatbot.get_response(text)
            
            # TTS 오디오 생성
            audio_data = await self._generate_tts_audio(ai_response)
            audio_base64 = base64.b64encode(audio_data).decode("utf-8") if audio_data else None
            
            return {
                "type": "ai_response",
                "text": ai_response,
                "audio_url": f"data:audio/mp3;base64,{audio_base64}" if audio_base64 else None,
                "user_text": text,
                "timestamp": datetime.now().isoformat(),
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"메시지 처리 실패: {e}")
            return {
                "type": "error",
                "error_message": f"메시지 처리 중 오류가 발생했습니다: {str(e)}",
                "error_code": "MESSAGE_PROCESSING_ERROR",
                "timestamp": datetime.now().isoformat()
            }
    
    async def process_voice_clone_audio(self, client_id: str, audio_data: bytes) -> Dict[str, Any]:
        """Voice Clone용 오디오 처리"""
        try:
            session = self.active_sessions.get(client_id)
            if not session:
                raise ValueError("활성 세션을 찾을 수 없습니다")
            
            child_name = session["child_name"]
            
            # 음성 샘플 저장
            sample_saved = await self.voice_cloning_processor.collect_user_audio_sample(
                user_id=child_name,
                audio_data=audio_data
            )
            
            sample_count = self.voice_cloning_processor.get_sample_count(child_name)
            ready_for_cloning = self.voice_cloning_processor.is_ready_for_cloning(child_name)
            
            # 진행상황 응답
            response = {
                "type": "voice_clone_progress",
                "sample_count": sample_count,
                "ready_for_cloning": ready_for_cloning,
                "has_cloned_voice": self.voice_cloning_processor.get_user_voice_id(child_name) is not None,
                "message": f"목소리 수집 중... ({sample_count}/5)",
                "timestamp": datetime.now().isoformat()
            }
            
            # 5개 수집 완료 시 클론 생성
            if ready_for_cloning:
                voice_id, error = await self.voice_cloning_processor.create_instant_voice_clone(
                    user_id=child_name,
                    voice_name=f"{child_name}_voice_clone"
                )
                
                if voice_id:
                    response = {
                        "type": "voice_clone_success",
                        "voice_id": voice_id,
                        "message": f"{child_name}님의 목소리가 성공적으로 복제되었어요! 이제 동화에서 주인공 목소리로 사용됩니다.",
                        "timestamp": datetime.now().isoformat()
                    }
                else:
                    response = {
                        "type": "voice_clone_error",
                        "error_message": f"음성 복제에 실패했습니다: {error}",
                        "error_code": "VOICE_CLONE_FAILED",
                        "timestamp": datetime.now().isoformat()
                    }
            
            return response
            
        except Exception as e:
            logger.error(f"Voice Clone 처리 실패: {e}")
            return {
                "type": "error",
                "error_message": f"음성 처리 중 오류가 발생했습니다: {str(e)}",
                "error_code": "VOICE_PROCESSING_ERROR",
                "timestamp": datetime.now().isoformat()
            }
    
    async def _generate_tts_audio(self, text: str) -> Optional[bytes]:
        """TTS 오디오 생성"""
        try:
            from shared.utils.audio_utils import initialize_elevenlabs
            
            elevenlabs_client = initialize_elevenlabs()
            if not elevenlabs_client:
                return None
            
            audio_data, _ = await generate_speech(
                client=elevenlabs_client,
                text=text,
                output_path="output/temp/audio/tts_temp"
            )
            
            return audio_data
            
        except Exception as e:
            logger.warning(f"TTS 생성 실패: {e}")
            return None
    
    def cleanup_session(self, client_id: str):
        """세션 정리"""
        if client_id in self.active_sessions:
            del self.active_sessions[client_id]

# 전역 핸들러 인스턴스
jjong_handler = JjongAlAudioWebSocket()

@router.websocket("/")
async def audio_websocket_endpoint(
    websocket: WebSocket,
    user_info: Dict[str, Any] = Depends(verify_jwt_token),
    connection_manager: ConnectionManager = Depends(get_connection_manager)
):
    """
    쫑알쫑알 통합 오디오 WebSocket 엔드포인트
    
    API 명세서 v1.0 기준:
    1. Connection 수립 (JWT 인증)
    2. Conversation 시작 (JSON) 
    3. 텍스트 메시지 전송 (Google STT 결과) 
    4. Voice Clone 오디오 전송 (Binary)
    5. 서버 응답 (JSON)
    6. Conversation 종료 (JSON)
    
    Query Parameters:
        token: JWT 인증 토큰 (필수)
        
    Example:
        wss://domain.com/wss/v1/audio?token=your_jwt_token
    """
    
    await websocket.accept()
    client_id = f"jjongal_{user_info['user_id']}_{datetime.now().timestamp()}"
    
    try:
        logger.info(f"쫑알쫑알 WebSocket 연결: {client_id}")
        
        # 연결 등록
        await connection_manager.connect(websocket, client_id, {
            "user_info": user_info,
            "connection_type": "jjongal_audio",
            "timestamp": datetime.now().isoformat()
        })
        
        # 메시지 처리 루프
        while True:
            try:
                # JSON 메시지 시도
                try:
                    message = await websocket.receive_json()
                    await handle_json_message(websocket, client_id, message, user_info)
                except:
                    # 바이너리 데이터 (Voice Clone용 오디오)
                    audio_data = await websocket.receive_bytes()
                    await handle_binary_audio(websocket, client_id, audio_data)
                    
            except WebSocketDisconnect:
                logger.info(f"쫑알쫑알 WebSocket 정상 연결 해제: {client_id}")
                break
            except Exception as e:
                logger.error(f"메시지 처리 오류: {e}")
                await websocket.send_json({
                    "type": "error",
                    "error_message": "메시지 처리 중 오류가 발생했습니다",
                    "error_code": "MESSAGE_PROCESSING_ERROR",
                    "timestamp": datetime.now().isoformat()
                })
                
    except Exception as e:
        logger.error(f"WebSocket 연결 오류: {e}")
        
    finally:
        # 정리
        jjong_handler.cleanup_session(client_id)
        await connection_manager.disconnect(client_id)


async def handle_json_message(websocket: WebSocket, client_id: str, 
                             message: Dict[str, Any], user_info: Dict[str, Any]):
    """JSON 메시지 처리"""
    message_type = message.get("type")
    
    try:
        if message_type == "start_conversation":
            # 대화 시작
            chatbot, greeting = await jjong_handler.initialize_chatbot_session(
                client_id, user_info, message
            )
            
            # TTS 오디오 생성
            audio_data = await jjong_handler._generate_tts_audio(greeting)
            audio_base64 = base64.b64encode(audio_data).decode("utf-8") if audio_data else None
            
            await websocket.send_json({
                "type": "conversation_started",
                "text": greeting,
                "audio_url": f"data:audio/mp3;base64,{audio_base64}" if audio_base64 else None,
                "timestamp": datetime.now().isoformat(),
                "status": "success"
            })
            
        elif message_type == "user_message":
            # 사용자 텍스트 메시지 (Google STT 결과)
            text = message.get("text", "").strip()
            if text:
                response = await jjong_handler.process_user_message(client_id, text)
                await websocket.send_json(response)
            
        elif message_type == "end_conversation":
            # 대화 종료
            jjong_handler.cleanup_session(client_id)
            await websocket.send_json({
                "type": "conversation_ended",
                "message": "대화가 종료되었습니다",
                "timestamp": datetime.now().isoformat(),
                "status": "success"
            })
            
        else:
            await websocket.send_json({
                "type": "error",
                "error_message": f"알 수 없는 메시지 타입: {message_type}",
                "error_code": "UNKNOWN_MESSAGE_TYPE",
                "timestamp": datetime.now().isoformat()
            })
            
    except Exception as e:
        logger.error(f"JSON 메시지 처리 실패: {e}")
        await websocket.send_json({
            "type": "error",
            "error_message": f"메시지 처리 중 오류가 발생했습니다: {str(e)}",
            "error_code": "MESSAGE_PROCESSING_ERROR",
            "timestamp": datetime.now().isoformat()
        })


async def handle_binary_audio(websocket: WebSocket, client_id: str, audio_data: bytes):
    """바이너리 오디오 데이터 처리 (Voice Clone용)"""
    try:
        response = await jjong_handler.process_voice_clone_audio(client_id, audio_data)
        await websocket.send_json(response)
        
    except Exception as e:
        logger.error(f"바이너리 오디오 처리 실패: {e}")
        await websocket.send_json({
            "type": "error",
            "error_message": f"오디오 처리 중 오류가 발생했습니다: {str(e)}",
            "error_code": "AUDIO_PROCESSING_ERROR",
            "timestamp": datetime.now().isoformat()
        }) 