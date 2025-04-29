from fastapi import FastAPI, WebSocket
import whisper
from chatbot.models.chat_bot_a import StoryCollectionChatBot

# FastAPI 애플리케이션 인스턴스 생성
app = FastAPI()

# Whisper 모델 초기화
model = whisper.load_model("base")

chatbot = StoryCollectionChatBot()

# WebSocket 엔드포인트 정의
@app.websocket("/ws/audio")
async def audio_endpoint(websocket: WebSocket):
    await websocket.accept()
    audio_chunks = []
    try:
        while True:
            data = await websocket.receive_bytes() # 오디오 데이터 수신
            audio_chunks.append(data) # 오디오 데이터 모음
            
            # 오디오 데이터가 10초 이상 모이면 처리
            if len(audio_chunks) >= 10:
                # 오디오 데이터 결합하여 처리
                audio_data = b''.join(audio_chunks)
                
                
    except Exception as e:
        print(f"오디오 처리 중 오류 발생: {str(e)}")
        

