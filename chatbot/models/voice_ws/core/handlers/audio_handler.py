# WebSocket 통신 핸들러
from websocket import WebSocket


class AudioHandler:
    def __init__(self, websocket: WebSocket):
        self.websocket = websocket
        self.audio_buffer = [] # 오디오 버퍼
        self.is_speaking = False # 현재 말하고 있는지에 대한 여부
        self.conversation_history = [] # 대화 기록
        self.conversation_id = None # 대화 ID (대화 시작시 발생하는 것)
        
        
    def 