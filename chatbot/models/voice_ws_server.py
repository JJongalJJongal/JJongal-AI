import logging
import time
import traceback
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, HTTPException, status, Response
import whisper
from .chat_bot_a import StoryCollectionChatBot
import tempfile, os
from dotenv import load_dotenv
from openai import OpenAI
import base64
import json
from typing import List, Optional, Dict, Any
import asyncio
from .chat_bot_b import StoryGenerationChatBot

# 환경 변수 Load & API Key 관리
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
dotenv_path = os.path.join(project_root, '.env')
load_dotenv(dotenv_path=dotenv_path)
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    logging.warning(f"OPENAI_API_KEY environment variable not found. Looking for .env file at: {dotenv_path}")

# OpenAI 클라이언트 초기화
client = OpenAI(api_key=api_key)

# FastAPI 애플리케이션 인스턴스 생성
app = FastAPI()

# Whisper 모델 초기화
model = None
try:
    model = whisper.load_model("base")
    logging.info("Whisper 모델 로드 성공")
except Exception as e:
    logging.error(f"Whisper 모델 로드 실패: {e}")

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("server.log"),
        logging.StreamHandler()
    ]
)

# 활성 연결 관리
active_connections = {}

# 꼬기(chat_bot_b) 인스턴스 가져오기
from .chat_bot_b import StoryGenerationChatBot

# Chatbot B 인스턴스 저장
chatbot_b_instances = {}

# 실패한 요청에 대한 재시도 매커니즘
async def retry_operation(operation, max_retries=3, retry_delay=1):
    """
    작업을 재시도하는 함수
    
    Args:
        operation: 실행할 비동기 작업
        max_retries (int): 최대 재시도 횟수
        retry_delay (float): 재시도 간 대기 시간(초)
    
    Returns:
        작업 결과 또는 None (실패 시)
    """
    attempts = 0
    last_error = None
    
    while attempts < max_retries:
        try:
            return await operation()
        except Exception as e:
            attempts += 1
            last_error = e
            logging.warning(f"작업 실패 (시도 {attempts}/{max_retries}): {e}")
            
            if attempts < max_retries:
                await asyncio.sleep(retry_delay)
                retry_delay *= 1.5  # 지수 백오프
    
    logging.error(f"최대 재시도 횟수 초과: {last_error}")
    return None, f"작업 실패: {last_error}", "retry_failed"

# 인증 토큰 검증 함수 검사
def validate_token(token: str) -> bool:
    """
    인증 토큰 검증 함수
    
    Args:
        token (str): 검증할 토큰
    
    Returns:
        bool: 유효한 토큰이면 True, 아니면 False
    """
    valid_token = os.getenv("WS_AUTH_TOKEN", "valid_token")
    return token == valid_token

# Whisper 변환 함수
async def transcribe_audio(model, file_path: str):
    """
    Whisper로 음성 파일을 텍스트로 변환
    
    Args:
        model: Whisper 모델 인스턴스
        file_path (str): 오디오 파일 경로
    
    Returns:
        tuple: (텍스트, 오류 메시지, 오류 코드)
    """
    if model is None:
        logging.error("Whisper 모델이 초기화되지 않았습니다")
        return "", "Whisper 모델이 초기화되지 않았습니다", "whisper_not_initialized"
        
    try:
        logging.info(f"오디오 파일 변환 시작: {file_path}, 크기: {os.path.getsize(file_path)} 바이트")
        
        # 비동기 처리를 위해 스레드풀에서 실행
        result = await asyncio.to_thread(model.transcribe, file_path, language="ko")
        
        logging.info(f"Whisper 변환 결과: {result['text']}")
        return result["text"], None, None
    except Exception as e:
        error_detail = traceback.format_exc()
        logging.error(f"Whisper 오류: {e}\n{error_detail}")
        return "", f"Whisper 오류: {e}", "whisper_error"

# 챗봇 응답 생성 함수
async def get_chatbot_response(chatbot, user_text: str):
    """
    챗봇에 텍스트를 전달해 응답 생성
    
    Args:
        chatbot: 챗봇 인스턴스
        user_text (str): 사용자 입력 텍스트
    
    Returns:
        tuple: (챗봇 응답, 오류 메시지, 오류 코드)
    """
    try:
        if not user_text:
            return "", "챗봇 입력 없음", "chatbot_no_input"
            
        # 비동기 처리를 위해 스레드풀에서 실행
        response = await asyncio.to_thread(chatbot.get_response, user_text)
        return response, None, None
    except Exception as e:
        error_detail = traceback.format_exc()
        logging.error(f"챗봇 오류: {e}\n{error_detail}")
        return "", f"챗봇 오류: {e}", "chatbot_error"

# TTS 변환 함수
async def synthesize_tts(ai_response: str):
    """
    AI 응답을 TTS로 변환
    
    Args:
        ai_response (str): 텍스트 응답
    
    Returns:
        tuple: (base64 인코딩된 오디오, 상태, 오류 메시지, 오류 코드)
    """
    try:
        if not ai_response:
            return "", "error", "TTS 입력 없음", "tts_no_input"
            
        # TTS 요청 (비동기 방식으로 처리)
        async def tts_operation():
            # 동기 함수를 비동기적으로 실행
            tts_result = await asyncio.to_thread(
                client.audio.speech.create,
                model="tts-1",
                voice="nova",
                input=ai_response,
                speed=0.9,
                response_format="wav"
            )
            return tts_result
            
        # 재시도 메커니즘 적용
        tts_result_tuple = await retry_operation(tts_operation)
        
        # retry_operation의 반환 값은 (결과, 오류_메시지, 오류_코드) 형태의 튜플이거나,
        # 성공 시 실제 tts_result 객체일 수 있음.
        # 여기서는 retry_operation이 실패하여 (None, error_message, error_code)를 반환하는 경우를 먼저 확인합니다.
        if isinstance(tts_result_tuple, tuple) and tts_result_tuple[0] is None:
            _, error_msg, error_code_val = tts_result_tuple
            logging.error(f"TTS 생성 실패 후 재시도 모두 실패: {error_msg}")
            return "", "error", error_msg or "TTS 생성 실패", error_code_val or "tts_generation_failed"

        # 성공한 경우 (OpenAIObject 인스턴스)
        tts_result = tts_result_tuple 
        
        if tts_result is None: # 이중 확인, 실제로는 위에서 처리됨
            return "", "error", "TTS 생성 실패", "tts_generation_failed"
        
        tts_audio = tts_result.content
        
        # 오디오 크기 확인
        if len(tts_audio) < 2 * 1024 * 1024:
            return base64.b64encode(tts_audio).decode("utf-8"), "ok", None, None
        else:
            return "", "partial", "TTS 오디오 크기 초과", "tts_partial"
    except Exception as e:
        error_detail = traceback.format_exc()
        logging.error(f"TTS 오류: {e}\n{error_detail}")
        return "", "error", f"TTS 오류: {e}", "tts_error"

# 연결 관리 함수
@app.on_event("startup")
async def startup_event():
    """서버 시작 시 실행되는 함수"""
    logging.info("음성 WebSocket 서버 시작됨")
    
@app.on_event("shutdown")
async def shutdown_event():
    """서버 종료 시 실행되는 함수"""
    logging.info("음성 WebSocket 서버 종료됨")
    # 활성 연결 정리
    for client_id, connection_info in active_connections.items():
        try:
            if "websocket" in connection_info:
                await connection_info["websocket"].close()
        except Exception as e:
            logging.error(f"연결 종료 중 오류: {e}")

# 상태 확인 엔드포인트
@app.get("/health")
async def health_check():
    """서버 상태 확인 엔드포인트"""
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Whisper 모델이 초기화되지 않았습니다"
        )
    return {"status": "online", "whisper_model": "loaded"}

# 오류 처리 미들웨어
@app.middleware("http")
async def catch_exceptions_middleware(request, call_next):
    """전역 오류 처리 미들웨어"""
    try:
        return await call_next(request)
    except Exception as e:
        logging.error(f"처리되지 않은 서버 오류: {e}")
        return Response(
            status_code=500,
            content={"detail": "서버 내부 오류가 발생했습니다"}
        )

# WebSocket 연결 종료 처리
async def handle_disconnect(client_id: str):
    """
    클라이언트 연결 종료 처리
    
    Args:
        client_id (str): 클라이언트 식별자
    """
    if client_id in active_connections:
        connection_info = active_connections[client_id]
        
        # 대화 내용 저장
        if "chatbot" in connection_info:
            try:
                chatbot = connection_info["chatbot"]
                child_name = connection_info.get("child_name", "unknown")
                
                # 대화 저장 디렉토리 생성
                save_dir = os.path.join("output", "conversations")
                os.makedirs(save_dir, exist_ok=True)
                
                # 대화 내용 저장
                save_path = os.path.join(save_dir, f"{child_name}_{int(time.time())}.json")
                await asyncio.to_thread(chatbot.save_conversation, save_path)
                logging.info(f"대화 내용 저장 완료: {child_name} ({client_id})")
            except Exception as save_error:
                logging.error(f"대화 내용 저장 중 오류: {save_error}")
        
        # 임시 파일 정리
        if "temp_files" in connection_info:
            for temp_file in connection_info["temp_files"]:
                try:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                except Exception as e:
                    logging.warning(f"임시 파일 삭제 실패: {e}")
        
        # 연결 정보 삭제
        del active_connections[client_id]
        logging.info(f"클라이언트 연결 종료 처리 완료: {client_id}")

# WebSocket 엔드포인트 정의
@app.websocket("/ws/audio")
async def audio_endpoint(
    websocket: WebSocket, 
    token: str = Query(None),
    child_name: str = Query(None),
    age: int = Query(None),
    interests: Optional[str] = Query(None)
):
    """
    WebSocket 오디오 처리 엔드포인트
    
    Args:
        websocket (WebSocket): WebSocket 연결
        token (str): 인증 토큰
        child_name (str): 아이 이름
        age (int): 아이 나이
        interests (str): 아이 관심사 (쉼표로 구분)
    """
    # 1. 연결 준비 및 검증
    client_id = f"{child_name}_{int(time.time())}" if child_name else f"unknown_{int(time.time())}"
    
    # 토큰 검증
    if not validate_token(token):
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        logging.warning(f"인증 실패: 잘못된 토큰 (클라이언트: {client_id})")
        return
    
    # 필수 파라미터 검증
    if not child_name:
        await websocket.close(code=status.WS_1003_UNSUPPORTED_DATA)
        logging.warning(f"필수 파라미터 누락: 아이 이름 (클라이언트: {client_id})")
        return
    
    if not age or not (4 <= age <= 9):
        await websocket.close(code=status.WS_1003_UNSUPPORTED_DATA)
        logging.warning(f"잘못된 파라미터: 나이는 4-9세 사이여야 함 (클라이언트: {client_id}, 나이: {age})")
        return
    
    # 관심사 처리
    interests_list = interests.split(',') if interests else []
    
    # 2. 연결 수락 및 초기화
    await websocket.accept()
    logging.info(f"WebSocket 연결 수락: {client_id} ({child_name}, {age}세)")
    
    # 챗봇 인스턴스 생성
    chatbot = StoryCollectionChatBot()
    
    # 연결 정보 저장
    active_connections[client_id] = {
        "websocket": websocket,
        "chatbot": chatbot,
        "child_name": child_name,
        "age": age,
        "start_time": time.time(),
        "temp_files": []
    }
    
    # 3. 초기 인사말 전송
    try:
        # 챗봇 초기화
        greeting = await asyncio.to_thread(
            chatbot.initialize_chat,
            child_name=child_name,
            age=age,
            interests=interests_list,
            chatbot_name="부기"
        )
    
        # TTS 변환
        greeting_audio_b64, tts_status, error_message, error_code = await synthesize_tts(greeting)
    
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
        logging.info(f"인사말 전송 완료: {client_id}")
    except Exception as init_error:
        logging.error(f"초기화 중 오류: {init_error}")
        error_packet = {
            "type": "error",
            "error_message": str(init_error),
            "error_code": "initialization_error",
            "status": "error"
        }
        await websocket.send_json(error_packet)
        await handle_disconnect(client_id)
        return
    
    # 4. 메인 대화 루프
    audio_chunks = []  # 오디오 chunk 저장 리스트
    audio_bytes = 0    # 누적 오디오 데이터 크기
    chunk_start_time = time.time()  # chunk 수집 시작 시간
    
    try:
        while True:
            # 오디오 데이터 수신 (타임아웃 설정)
            try:
                data = await asyncio.wait_for(
                    websocket.receive_bytes(),
                    timeout=30.0  # 30초 타임아웃
                )
            except asyncio.TimeoutError:
                # 무응답 시 상태 확인 메시지 전송
                await websocket.send_json({
                    "type": "ping",
                    "message": "connection_check"
                })
                # 타임아웃 리셋
                chunk_start_time = time.time()
                continue
                
            # 오디오 데이터 처리
            audio_chunks.append(data)
            audio_bytes += len(data)
            chunk_time = time.time() - chunk_start_time
            
            # 처리 기준 충족 시 (2초 이상 또는 128KB 이상)
            if chunk_time >= 2.0 or audio_bytes >= 128 * 1024:
                temp_file_path = None  # 임시 파일 경로 초기화
                
                try:
                    # 임시 파일 생성 및 데이터 저장
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                        temp_file.write(b"".join(audio_chunks))
                        temp_file_path = temp_file.name
                    
                    # 임시 파일 추적
                    active_connections[client_id]["temp_files"].append(temp_file_path)
                    
                    # Whisper 음성 인식 수행
                    user_text, error_message, error_code = await transcribe_audio(model, temp_file_path)
                    logging.info(f"음성 인식 결과 ({client_id}): {user_text}")
                    
                    # 음성 인식 실패 시 
                    if error_code:
                        response_packet = {
                            "type": "error",
                            "error_message": error_message,
                            "error_code": error_code,
                            "status": "error"
                        }
                        await websocket.send_json(response_packet)
                        # 계속 진행 (치명적 오류가 아님)
                    else:
                        # 챗봇 응답 생성
                        ai_response, cb_error_message, cb_error_code = await get_chatbot_response(chatbot, user_text)
                        
                        # 챗봇 응답 실패 처리
                        if cb_error_code:
                            error_message = cb_error_message
                            error_code = cb_error_code
                            ai_response = "미안해, 지금은 대답하기 어려워. 다시 말해줄래?"
                    
                        logging.info(f"챗봇 응답 ({client_id}): {ai_response}")
                    
                        # TTS 변환
                        audio_b64, tts_status, tts_error_message, tts_error_code = await synthesize_tts(ai_response)
                        
                        # TTS 오류 발생 시
                        if tts_error_code:
                            error_message = tts_error_message
                            error_code = tts_error_code
                    
                        # 응답 패킷 구성
                        response_packet = {
                            "type": "ai_response",
                            "text": ai_response,
                            "audio": audio_b64,
                            "status": tts_status,
                            "user_text": user_text
                        }
                    
                        # 오류 정보 추가
                        if error_message:
                            response_packet["error_message"] = error_message
                        if error_code:
                            response_packet["error_code"] = error_code
                        
                        # 응답 전송
                        await websocket.send_json(response_packet)
                    
                except Exception as chunk_error:
                    logging.error(f"Chunk 처리 중 오류: {chunk_error}")
                    # 오류 패킷 전송
                    error_packet = {
                        "type": "error",
                        "error_message": str(chunk_error),
                        "error_code": "chunk_processing_error",
                        "status": "error"
                    }
                    await websocket.send_json(error_packet)
                
                finally:
                    # 임시 파일 정리
                    if temp_file_path and os.path.exists(temp_file_path):
                        try:
                            os.remove(temp_file_path)
                            # 삭제된 파일 목록에서 제거
                            if temp_file_path in active_connections[client_id]["temp_files"]:
                                active_connections[client_id]["temp_files"].remove(temp_file_path)
                        except Exception as e:
                            logging.warning(f"임시 파일 삭제 실패: {e}")
                    
                    # 청크 변수 초기화
                    audio_chunks = []
                    audio_bytes = 0
                    chunk_start_time = time.time()
    
    except WebSocketDisconnect:
        logging.info(f"WebSocket 연결 종료: {client_id}")
        await handle_disconnect(client_id)
    
    except Exception as e:
        logging.error(f"처리되지 않은 오류: {e}\n{traceback.format_exc()}")
        try:
            error_packet = {
                "type": "error",
                "error_message": f"서버 오류: {str(e)}",
                "error_code": "server_error",
                "status": "error"
            }
            await websocket.send_json(error_packet)
        except:
            pass  # 이미 연결이 끊어졌을 수 있음
        
        finally:
            await handle_disconnect(client_id)
        
# 꼬기(chat_bot_b) WebSocket 엔드포인트
@app.websocket("/ws/story_generation")
async def story_generation_endpoint(
    websocket: WebSocket, 
    token: str = Query(None),
    child_name: str = Query(None),
    age: int = Query(None),
    interests: Optional[str] = Query(None)
):
    """
    동화 생성을 위한 WebSocket 엔드포인트
    
    Args:
        websocket: WebSocket 연결
        token: 인증 토큰
        child_name: 아이 이름
        age: 아이 나이
        interests: 관심사 (쉼표로 구분)
    """
    # 클라이언트 ID 생성
    client_id = f"story_{child_name}_{int(time.time())}"
    
    # 토큰 검증
    if not validate_token(token):
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        logging.warning(f"인증 실패: {client_id}")
        return
    
    # 파라미터 검증
    if not child_name or not age:
        await websocket.close(code=status.WS_1003_UNSUPPORTED_DATA)
        logging.warning(f"필수 파라미터 누락: {client_id}")
        return
    
    # 관심사 처리
    interest_list = []
    if interests:
        interest_list = interests.split(',')
    
    # WebSocket 연결 수락
    await websocket.accept()
    logging.info(f"꼬기(chatbot_b) 연결 수락: {client_id}")
    
    # 꼬기(chatbot_b) 인스턴스 생성
    output_dir = os.path.join("output", client_id)
    os.makedirs(output_dir, exist_ok=True)
    chatbot_b = StoryGenerationChatBot(output_dir=output_dir)
    chatbot_b.set_target_age(age)
    
    # 클라이언트 정보 저장
    chatbot_b_instances[client_id] = {
        "chatbot": chatbot_b,
        "child_name": child_name,
        "age": age,
        "interests": interest_list,
        "last_activity": time.time()
    }
    
    try:
        # 인사 메시지 전송
        greeting_message = {
            "type": "greeting",
            "text": f"안녕! 나는 꼬기야. {child_name}님을 위한 멋진 동화를 만들어줄게. 부기가 알려준 동화 줄거리로 더 재미있는 이야기를 만들어볼까?",
        }
        await websocket.send_text(json.dumps(greeting_message))
        
        # 클라이언트와의 통신 처리
        while True:
            # 메시지 수신
            data = await websocket.receive()
            
            # 바이너리 데이터 처리 (오디오)
            if "bytes" in data:
                # 아직 오디오 처리 구현 안 함
                continue
            
            # 텍스트 데이터 처리 (JSON 형식)
            if "text" in data:
                try:
                    message = json.loads(data["text"])
                    message_type = message.get("type", "unknown")
                    
                    # 메시지 유형에 따른 처리
                    if message_type == "story_outline":
                        # 동화 줄거리 수신
                        story_outline = message.get("story_outline", {})
                        
                        # 줄거리 설정
                        chatbot_b.set_story_outline(story_outline)
                        
                        # 상세 스토리 생성 응답
                        await websocket.send_text(json.dumps({
                            "type": "processing",
                            "text": "동화 줄거리를 받았어! 이제 멋진 이야기를 만들어볼게. 잠시만 기다려줘~"
                        }))
                        
                        try:
                            # 상세 스토리 생성
                            detailed_story = await asyncio.to_thread(chatbot_b.generate_detailed_story)
                            
                            # 응답 전송
                            await websocket.send_text(json.dumps({
                                "type": "detailed_story",
                                "text": f"'{detailed_story['title']}'이라는 멋진 이야기를 만들었어!",
                                "detailed_story": detailed_story
                            }))
                        except Exception as e:
                            logging.error(f"상세 스토리 생성 오류: {str(e)}")
                            await websocket.send_text(json.dumps({
                                "type": "error",
                                "text": "스토리 생성 중에 문제가 발생했어. 다시 시도해볼까?",
                                "error_message": str(e)
                            }))
                    
                    elif message_type == "generate_illustrations":
                        # 일러스트 생성 요청
                        await websocket.send_text(json.dumps({
                            "type": "processing",
                            "text": "멋진 그림을 그려볼게. 잠시만 기다려줘~"
                        }))
                        
                        try:
                            # 일러스트 생성
                            images = await asyncio.to_thread(chatbot_b.generate_illustrations)
                            
                            # 이미지 경로 목록 전송
                            await websocket.send_text(json.dumps({
                                "type": "illustrations",
                                "text": f"{len(images)}개의 멋진 그림을 그렸어!",
                                "images": images
                            }))
                        except Exception as e:
                            logging.error(f"일러스트 생성 오류: {str(e)}")
                            await websocket.send_text(json.dumps({
                                "type": "error",
                                "text": "그림을 그리다가 문제가 발생했어. 다시 시도해볼까?",
                                "error_message": str(e)
                            }))
                    
                    elif message_type == "generate_voice":
                        # 내레이션 생성 요청
                        await websocket.send_text(json.dumps({
                            "type": "processing",
                            "text": "이야기를 읽어줄 목소리를 만들고 있어. 조금만 기다려줘~"
                        }))
                        
                        try:
                            # 내레이션 생성
                            voice_result = await asyncio.to_thread(chatbot_b.generate_voice)
                            
                            # 내레이션 결과 전송
                            await websocket.send_text(json.dumps({
                                "type": "voice",
                                "text": "이야기를 읽어줄 목소리를 만들었어!",
                                "voice_data": voice_result
                            }))
                        except Exception as e:
                            logging.error(f"내레이션 생성 오류: {str(e)}")
                            await websocket.send_text(json.dumps({
                                "type": "error",
                                "text": "목소리를 만드는 중에 문제가 발생했어. 다시 시도해볼까?",
                                "error_message": str(e)
                            }))
                    
                    elif message_type == "get_preview":
                        # 동화 미리보기 요청
                        try:
                            # 미리보기 생성
                            preview = await asyncio.to_thread(chatbot_b.get_story_preview)
                            
                            # 미리보기 전송
                            await websocket.send_text(json.dumps({
                                "type": "preview",
                                "text": f"'{preview['title']}' 동화 미리보기야!",
                                "preview": preview
                            }))
                        except Exception as e:
                            logging.error(f"미리보기 생성 오류: {str(e)}")
                            await websocket.send_text(json.dumps({
                                "type": "error",
                                "text": "미리보기를 만드는 중에 문제가 발생했어.",
                                "error_message": str(e)
                            }))
                    
                    elif message_type == "save_story":
                        # 동화 저장 요청
                        story_name = message.get("story_name", f"story_{int(time.time())}")
                        try:
                            # 동화 저장
                            story_data_path = os.path.join(output_dir, f"{story_name}.json")
                            await asyncio.to_thread(chatbot_b.save_story_data, story_data_path)
                            
                            # 저장 완료 응답
                            await websocket.send_text(json.dumps({
                                "type": "save_complete",
                                "text": "동화가 저장되었어!",
                                "story_path": story_data_path
                            }))
                        except Exception as e:
                            logging.error(f"동화 저장 오류: {str(e)}")
                            await websocket.send_text(json.dumps({
                                "type": "error",
                                "text": "동화를 저장하는 중에 문제가 발생했어.",
                                "error_message": str(e)
                            }))
                    
                    elif message_type == "ping":
                        # 연결 유지 핑
                        await websocket.send_text(json.dumps({
                            "type": "pong"
                        }))
                    
                    else:
                        # 알 수 없는 메시지 유형
                        logging.warning(f"알 수 없는 메시지 유형: {message_type}")
                        await websocket.send_text(json.dumps({
                            "type": "error",
                            "text": "이해할 수 없는 메시지야.",
                            "error_message": f"알 수 없는 메시지 유형: {message_type}"
                        }))
                
                except json.JSONDecodeError:
                    logging.error(f"JSON 디코딩 오류: {data['text']}")
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "text": "메시지 형식이 올바르지 않아.",
                        "error_message": "JSON 디코딩 오류"
                    }))
                
                # 마지막 활동 시간 업데이트
                chatbot_b_instances[client_id]["last_activity"] = time.time()
    
    except WebSocketDisconnect:
        logging.info(f"꼬기(chatbot_b) 연결 종료: {client_id}")
    
    except Exception as e:
        logging.error(f"꼬기(chatbot_b) 처리 중 예외 발생: {client_id}, {str(e)}")
        traceback.print_exc()
    
    finally:
        # 클라이언트 연결 종료 처리
        await handle_disconnect(client_id)

# 주기적으로 비활성 클라이언트 정리
@app.on_event("startup")
async def start_cleanup_task():
    """비활성 클라이언트 정리 태스크 시작"""
    asyncio.create_task(cleanup_inactive_clients())

async def cleanup_inactive_clients():
    """비활성 클라이언트 정리 함수"""
    while True:
        try:
            await asyncio.sleep(300)  # 5분마다 실행
            current_time = time.time()
            
            # 30분 이상 비활성인 클라이언트 정리
            inactive_timeout = 1800  # 30분
            inactive_clients = []
            
            # 비활성 클라이언트 식별
            for client_id, client_data in chatbot_b_instances.items():
                if current_time - client_data["last_activity"] > inactive_timeout:
                    inactive_clients.append(client_id)
            
            # 비활성 클라이언트 정리
            for client_id in inactive_clients:
                del chatbot_b_instances[client_id]
                logging.info(f"비활성 클라이언트 정리: {client_id}")
        
        except Exception as e:
            logging.error(f"비활성 클라이언트 정리 중 오류 발생: {str(e)}")

            
