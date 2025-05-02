import logging
import time
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
import whisper
from .chat_bot_a import StoryCollectionChatBot
import tempfile, os
from dotenv import load_dotenv
import openai
import base64
import json
from typing import List, Optional

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
async def audio_endpoint(
    websocket: WebSocket, 
    token: str = Query(None),
    child_name: str = Query(None),
    age: int = Query(None),
    interests: Optional[str] = Query(None)
):
    # 1. 인증 토큰 검증
    if not validate_token(token):
        await websocket.close()
        logging.warning("인증 실패: 잘못된 토큰")
        return
    
    # 2. 필수 파라미터 검사
    if not child_name:
        await websocket.close()
        logging.warning("필수 파라미터 누락: 아이 이름이 필요합니다")
        return
    
    if not age or not (4 <= age <= 9):
        await websocket.close()
        logging.warning("잘못된 파라미터: 나이는 4-9세 사이여야 합니다")
        return
    
    # 관심사 파싱 (interests=공룡,우주,동물 형식으로 전달)
    interests_list = interests.split(',') if interests else []
    
    await websocket.accept()  # 클라이언트의 WebSocket 연결 수락
    audio_chunks = []        # 오디오 chunk 데이터를 저장할 리스트
    audio_bytes = 0          # 현재까지 누적된 오디오 데이터 크기(바이트)
    chunk_start_time = time.time()  # chunk 수집 시작 시간
    
    # 챗봇 인스턴스 생성 및 초기화
    chatbot = StoryCollectionChatBot()  # 사용자별 챗봇 인스턴스 생성
    greeting = chatbot.initialize_chat(
        child_name=child_name,
        age=age,
        interests=interests_list,
        chatbot_name="부기"
    )
    
    # 인사말 전송
    greeting_audio_b64, tts_status, error_message, error_code = synthesize_tts(greeting)
    
    # 인사말 응답 패킷 구성
    greeting_packet = {
        "type": "ai_response",
        "text": greeting,
        "audio": greeting_audio_b64,
        "status": tts_status
    }
    
    if error_message:
        greeting_packet["error_message"] = error_message
    if error_code:
        greeting_packet["error_code"] = error_code
    
    # 인사말 전송
    await websocket.send_json(greeting_packet)
    
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
                    logging.info(f"사용자 음성 인식: {user_text}")
                    
                    # 챗봇 응답 생성
                    ai_response, cb_error_message, cb_error_code = get_chatbot_response(chatbot, user_text)
                    if cb_error_message:
                        error_message = cb_error_message
                        error_code = cb_error_code
                    
                    logging.info(f"AI 응답: {ai_response}")
                    
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
                        "status": tts_status,        # TTS 상태
                        "user_text": user_text       # 인식된 사용자 음성 텍스트
                    }
                    
                    if error_message:
                        response_packet["error_message"] = error_message  # 에러 메시지 포함
                    if error_code:
                        response_packet["error_code"] = error_code        # 에러 코드 포함
                    
                    await websocket.send_json(response_packet)  # WebSocket으로 응답 전송
                except Exception as chunk_error:
                    logging.error(f"Chunk 처리 중 오류: {chunk_error}")
                    # 특정 chunk만 처리 실패했을 경우 에러 메시지 전송
                    error_packet = {
                        "type": "error",
                        "error_message": str(chunk_error),
                        "error_code": "chunk_processing_error",
                        "status": "error"
                    }
                    await websocket.send_json(error_packet)
                finally:
                    # 임시 파일 삭제(리소스 정리)
                    if temp_file_path and os.path.exists(temp_file_path):
                        os.remove(temp_file_path)
                    # chunk 관련 변수 초기화(다음 chunk 수집 준비)
                    audio_chunks = []
                    audio_bytes = 0
                    chunk_start_time = time.time()
    except WebSocketDisconnect:
        logging.info(f"클라이언트 연결 종료: {child_name}({age}세)")
        # 대화 내용 저장
        try:
            save_dir = os.path.join("output", "conversations")
            os.makedirs(save_dir, exist_ok=True)
            chatbot.save_conversation(os.path.join(save_dir, f"{child_name}_{int(time.time())}.json"))
            logging.info(f"대화 내용 저장 완료: {child_name}")
        except Exception as save_error:
            logging.error(f"대화 내용 저장 중 오류: {save_error}")
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
        
            
