from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter(prefix="/wss/voice", tags=["voice-websocket"]) # router 접두사 설정 -> 예시: /wss/voice/connect    

@router.websocket("/{child_name}")
async def voice_websocket(child_name: str, websocket: WebSocket):
    pass

