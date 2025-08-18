# WebSocket 통신 핸들러
from websocket import WebSocket


class AudioHandler:
    def __init__(self, websocket: WebSocket):
        self.websocket = websocket
        self.conversation_history = [] # 대화 기록
        self.conversation_id = None # 대화 ID (대화 시작시 발생하는 것)
        self.is_active: bool = False # 현재 활성화 상태
        
        # Audio buffer TTS 
        self.tts_enabled: bool = True # TTS 활성화 여부
        
    async def handle_text_message(self, message: dict) -> None:
        """Text Message Processing (Frontend Google STT)"""
        
        message_type = message.get("type") # 메시지 타입 확인
        
        if message_type == "start_conversation": # 대화 시작
            await self._start_conversation(message)
        elif message_type == "end_conversation": # 대화 종료
            await self._end_conversation()
        elif message_type == "audio_config": # 오디오 설정
            await self._configure_audio(message)
        elif message_type == "user_message": # 사용자 메시지
            await self._handle_user_message(message)
        else: # 알 수 없는 메시지 타입
            await self._send_error("알 수 없는 메시지 타입")
    
    async def _handle_user_message(self, message: dict) -> None:
        
    
    async def _start_conversation(self, message: dict) -> None:
        """Conversation Start"""
        self.conversation_id = message.get("conversation_id") # 대화 ID 설정
        self.is_speaking = True # 말하고 있는 상태로 변경
        
        await self.websocket.send_json({
            "type": "conversation_started",
            "conversation_id": self.conversation_id
        })
        
    async def _end_conversation(self) -> None:
        """Conversation End"""
        self.is_speaking = False # 말하고 있지 않는 상태로 변경
        self.audio_buffer.clear() # 오디오 버퍼 초기화
        
        await self.websocket.send_json({
            "type": "conversation_ended",
            "conversation_id": self.conversation_id
        })
        
    async def _configure_audio(self, message: dict) -> None:
        """Audio Configuration"""
        
    
    async def handler_audio_websocket(
        websocket: WebSocket,
        user_info: Dict[str, Any],
        client_id: str
    )