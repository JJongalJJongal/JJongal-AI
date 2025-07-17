from typing import Dict, Optional
from fastapi import WebSocket
from datetime import datetime
import base64
import os
"""
ChatBot A Service 
"""

class ChatBotService:
    def __init__(self):
        self.sessions = {} # ChatBot A Session for each client
        
    async def start_conversation(self, data: Dict, client_id: str, websocket: WebSocket):
        """ Start Conversation """
        
        from chatbot.models.chat_bot_a import ChatBotA
        from chatbot.data.vector_db.core import VectorDB
        
        try:
            
            # VectorDB Initialize
            chroma_base = os.getenv("CHROMA_DB_PATH", "/app/chatbot/data/vector_db")
            vector_db_path = os.path.join(chroma_base, "main")
            
            vector_db = VectorDB(
                persist_directory=vector_db_path,
                embedding_model="nipal-lab/KURE-v1",
                use_hybrid_mode=True
            )
            
            # ChatBot A Instance 
            chatbot_a = ChatBotA(
                vector_db_instance=vector_db,
                token_limit=10000,
                use_langchain=True,
                enhanced_mode=True,
                session_id=f"session_{client_id}"
            )
            
            payload = data.get("payload", {})
            greeting = chatbot_a.initialize_chat(
                child_name=payload.get("child_name"),
                age=payload.get("age", 7),
                interests=payload.get("interests", [])
            )
            
            self.sessions[client_id] = chatbot_a
            
            audio_data = await self._generate_tts_bytes(greeting)
            audio_base64 = base64.b64encode(audio_data).decode("utf-8") if audio_data else None
            
            await websocket.send_json({
                "type": "conversation_started",
                "text": greeting,
                "audio": audio_base64,
                "timestamp": datetime.now().isoformat()
            })
        except Exception as e:
            await websocket.send_json({
                "type": "error",
                "error_message": str(e),
                "error_code": "INITIALIZATION_ERROR"
            })
    
    async def process_user_message(self, data: Dict, client_id: str, websocket: WebSocket):
        """ User Message Process """
        chatbot_a = self.sessions.get(client_id)
        if not chatbot_a:
            raise ValueError("ChatBot A session not found")
        
        user_text = data.get("text", "").strip()
        if not user_text:
            return        
        
        ai_response = chatbot_a.get_response(user_text)
        
        audio_data = await self._generate_tts_bytes(ai_response)
        audio_base64 = base64.b64encode(audio_data).decode("utf-8") if audio_data else None
        
        await websocket.send_json({
            "type": "ai_response",
            "text": ai_response,
            "audio": audio_base64,
            "timestamp": datetime.now().isoformat()
        })
        
        await self._stream_tts_audio(ai_response, websocket)
    
    async def _stream_tts_audio(self, text: str, websocket: WebSocket):
        """ Stream TTS Audio """
        try:
            # Base64 Streaming
            audio_data = await self._generate_tts_bytes(text)
            
            if audio_data:
                audio_base64 = base64.b64encode(audio_data).decode("utf-8")
                await websocket.send_json({
                    "type": "audio_chunk",
                    "audio": audio_base64,
                    "text": text
                })
    
    async def _generate_tts_bytes(self, text: str, client_id: str) -> Optional[str]:
        """ Generate TTS Audio """
        from shared.utils.audio_utils import generate_speech, initialize_elevenlabs
        
        elevenlabs_client = initialize_elevenlabs()
        
        if not elevenlabs_client:
            return None
        
        audio_data, file_path = await generate_speech(
            client=elevenlabs_client,
            text=text,
            output_path=f"output/temp/audio/tts_"
        )