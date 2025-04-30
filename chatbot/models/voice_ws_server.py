import logging
import time
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
import whisper
from chatbot.models.chat_bot_a import StoryCollectionChatBot
import tempfile, os
from dotenv import load_dotenv
import openai
import base64
import json

# 환경 변수 Load & API Key 관리
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# FastAPI 애플리케이션 인스턴스 생성
app = FastAPI()

# Whisper 모델 초기화
model = whisper.load_model("base")

# 로깅 설정
logging.basicConfig(level=logging.INFO)

# 인증 토큰 검증 함수 검사
def validate_token(token):
    # 예시: 토큰이 "valid_token"이면 통과
    return token == "valid_token"

# Whisper 변환 함수
def transcribe_audio(model, file_path):
    """Whisper로 음성 파일을 텍스트로 변환"""
    try:
        result = model.transcribe(file_path, language="ko")
        return result["text"], None, None
    except Exception as e:
        logging.error(f"Whisper 오류: {e}")
        return "", f"Whisper 오류: {e}", "whisper_error"

# 챗봇 응답 생성 함수
def get_chatbot_response(chatbot, user_text):
    """챗봇에 텍스트를 전달해 응답 생성"""
    try:
        if user_text:
            return chatbot.get_response(user_text), None, None
        return "", "챗봇 입력 없음", "chatbot_no_input"
    except Exception as e:
        logging.error(f"챗봇 오류: {e}")
        return "", f"챗봇 오류: {e}", "chatbot_error"

# TTS 변환 함수
def synthesize_tts(ai_response):
    """AI 응답을 TTS로 변환"""
    try:
        if ai_response:
            tts_result = openai.audio.speech.create(
                model="tts-1",
                voice="alloy",
                input=ai_response
            )
            tts_audio = tts_result.content
            if len(tts_audio) < 2 * 1024 * 1024:
                return base64.b64encode(tts_audio).decode("utf-8"), "ok", None, None
            else:
                return "", "partial", "TTS 오디오 크기 초과", "tts_partial"
        return "", "error", "TTS 입력 없음", "tts_no_input"
    except Exception as e:
        logging.error(f"TTS 오류: {e}")
        return "", "error", f"TTS 오류: {e}", "tts_error"

# WebSocket 엔드포인트 정의
@app.websocket("/ws/audio")
async def audio_endpoint(websocket: WebSocket, token: str = Query(None)):
    # 1. 인증 토큰 검증
    if not validate_token(token):
        await websocket.close()
        logging.warning("인증 실패: 잘못된 토큰")
        return
    await websocket.accept()  # 클라이언트의 WebSocket 연결 수락
    audio_chunks = []        # 오디오 chunk 데이터를 저장할 리스트
    audio_bytes = 0          # 현재까지 누적된 오디오 데이터 크기(바이트)
    chunk_start_time = time.time()  # chunk 수집 시작 시간
    chatbot = StoryCollectionChatBot()  # 사용자별 챗봇 인스턴스 생성
    try:
        while True:
            data = await websocket.receive_bytes()  # 클라이언트로부터 오디오 chunk 수신
            audio_chunks.append(data)               # chunk 리스트에 추가
            audio_bytes += len(data)                # 누적 바이트 수 갱신
            chunk_time = time.time() - chunk_start_time  # chunk 수집 경과 시간(초)
            # 2초 이상 또는 128KB 이상 쌓이면 처리
            if chunk_time >= 2.0 or audio_bytes >= 128 * 1024:
                temp_file_path = None  # 임시 파일 경로 초기화
                try:
                    # 오디오 chunk를 하나의 파일로 결합하여 임시 wav 파일로 저장
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                        temp_file.write(b"".join(audio_chunks))
                        temp_file_path = temp_file.name
                    # Whisper 변환
                    user_text, error_message, error_code = transcribe_audio(model, temp_file_path)
                    # 챗봇 응답 생성
                    ai_response, cb_error_message, cb_error_code = get_chatbot_response(chatbot, user_text)
                    if cb_error_message:
                        error_message = cb_error_message
                        error_code = cb_error_code
                    # TTS 변환
                    audio_b64, tts_status, tts_error_message, tts_error_code = synthesize_tts(ai_response)
                    if tts_error_message:
                        error_message = tts_error_message
                        error_code = tts_error_code
                    # 클라이언트로 보낼 JSON 응답 패킷 구성
                    response_packet = {
                        "type": "ai_response",      # 응답 타입
                        "text": ai_response,         # AI 텍스트 응답
                        "audio": audio_b64,          # base64 인코딩된 음성 데이터
                        "status": tts_status         # TTS 상태
                    }
                    if error_message:
                        response_packet["error_message"] = error_message  # 에러 메시지 포함
                    if error_code:
                        response_packet["error_code"] = error_code        # 에러 코드 포함
                    await websocket.send_json(response_packet)  # WebSocket으로 응답 전송
                finally:
                    # 임시 파일 삭제(리소스 정리)
                    if temp_file_path and os.path.exists(temp_file_path):
                        os.remove(temp_file_path)
                    # chunk 관련 변수 초기화(다음 chunk 수집 준비)
                    audio_chunks = []
                    audio_bytes = 0
                    chunk_start_time = time.time()
    except WebSocketDisconnect:
        logging.info("Client disconnected")  # 클라이언트 연결 종료 로그
    except Exception as e:
        logging.error(f"오디오 처리 중 오류 발생: {str(e)}")  # 서버 에러 로그
        # 치명적 에러 발생 시 클라이언트에 에러 패킷 전송
        error_packet = {
            "type": "error",
            "error_message": str(e),
            "error_code": "server_error"
        }
        try:
            await websocket.send_json(error_packet)
        except Exception:
            pass
        
            
