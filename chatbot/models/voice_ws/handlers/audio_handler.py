"""
오디오 WebSocket 엔드포인트 핸들러

'/ws/audio' 경로의 WebSocket 연결 및 메시지 처리를 담당합니다.
"""
import time
import asyncio
import traceback
from typing import Optional, Dict, Any
from fastapi import WebSocket, status

from shared.utils.logging_utils import get_module_logger
from chatbot.models.chat_bot_a import ChatBotA # 부기 챗봇 import
from ..core.connection_engine import ConnectionEngine
from ..core.websocket_engine import WebSocketDisconnect # WebSocket 연결 종료 처리
from ..processors.audio_processor import AudioProcessor


logger = get_module_logger(__name__)

async def handle_audio_websocket(
    websocket: WebSocket, 
    child_name: str, 
    age: int, 
    interests_str: Optional[str],
    connection_engine: ConnectionEngine,
    audio_processor: AudioProcessor,
):
    """
    오디오 WebSocket 연결의 전체 라이프사이클을 관리합니다.
    인증, 초기화, 메시지 수신/처리, 연결 종료를 담당합니다.
    """
    client_id = f"{child_name}_{int(time.time())}" if child_name else f"unknown_{int(time.time())}"
    
    # 필수 파라미터 검증 (app.py에서도 일부 검증하지만, 핸들러 레벨에서도 확인하는 이중 검증)
    if not child_name: # 아이 이름이 누락되었을 경우
        await websocket.close(code=status.WS_1003_UNSUPPORTED_DATA, reason="아이 이름 누락") # WebSocket 연결 종료
        logger.warning(f"필수 파라미터 누락: 아이 이름 (클라이언트: {client_id})") # 로깅
        return 
    
    if not age or not (4 <= age <= 9): # 아이 연령대가 4-9세 사이가 아닐 경우
        await websocket.close(code=status.WS_1003_UNSUPPORTED_DATA, reason="나이 범위 오류 (4-9세)") # WebSocket 연결 종료
        logger.warning(f"잘못된 파라미터: 나이는 4-9세 사이여야 함 (클라이언트: {client_id}, 나이: {age})") # 로깅
        return 
    
    interests_list = interests_str.split(',') if interests_str else [] # 관심사 목록 처리
    
    try:
        await websocket.accept() # WebSocket 연결 수락
        logger.info(f"오디오 WebSocket 연결 수락: {client_id} ({child_name}, {age}세)") # 로깅
        
        # 사전 로드된 VectorDB 인스턴스 가져오기
        vector_db_instance = websocket.app.state.vector_db
        if vector_db_instance is None: # VectorDB 인스턴스 누락 시
            logger.error("사전 로드된 VectorDB 인스턴스를 찾을 수 없습니다. RAG 기능 없이 ChatBotA를 초기화합니다.") # 로깅
            await websocket.close(code=status.WS_1003_UNSUPPORTED_DATA, reason="VectorDB 인스턴스 누락") # 연결 종료
            return # 연결 종료
        
        # ChatBot A 인스턴스 생성 시 VectorDB 인스턴스 주입
        chatbot_a = ChatBotA(vector_db_instance=vector_db_instance) 
        
        # 디버그: 챗봇 객체 타입 및 메서드 확인
        logger.info(f"[DEBUG] 생성된 chatbot_a 타입: {type(chatbot_a)}, 메서드 목록: {dir(chatbot_a)}")
        if not hasattr(chatbot_a, 'get_conversation_history'):
            logger.error(f"[DEBUG] 생성된 chatbot_a 객체에 get_conversation_history 메서드가 없습니다. 실제 타입: {type(chatbot_a)}")
            await websocket.send_json({
                "type": "error",
                "error_message": "챗봇 객체에 get_conversation_history 메서드가 없습니다.",
                "error_code": "chatbot_object_invalid",
                "status": "error"
            })
            return
        
        connection_info = {
            "websocket": websocket, # WebSocket 객체 저장
            "chatbot": chatbot_a, # ChatBot A 인스턴스 저장
            "child_name": child_name, # 아이 이름 저장
            "age": age, # 아이 연령대 저장
            "interests": interests_list, # 관심사 목록 저장
            "start_time": time.time(), # 시작 시간 저장
            "temp_files": [] # 임시 파일 목록 저장
        }
        connection_engine.add_client(client_id, connection_info) # 연결 정보 추가

        # 초기 인사말 전송
        greeting = await asyncio.to_thread(
            chatbot_a.initialize_chat, # 부기 챗봇 인스턴스 초기화
            child_name=child_name, # 아이 이름
            age=age, # 아이 연령대
            interests=interests_list, # 관심사 목록
            chatbot_name="부기" # 챗봇 이름
        ) 
        greeting_audio_b64, tts_status, tts_error, tts_code = await audio_processor.synthesize_tts(greeting) # ElevenLabs로 인사말 음성 생성
        
        greeting_packet = {
            "type": "ai_response", "text": greeting, "audio": greeting_audio_b64, # 인사말 텍스트 및 음성 전송
            "status": tts_status, "error_message": tts_error, "error_code": tts_code # 음성 상태 전송
        }
        await websocket.send_json(greeting_packet) # 인사말 전송
        logger.info(f"인사말 전송 완료: {client_id}") # 로깅

        # 메인 대화 루프
        audio_chunks = [] # 오디오 chunk 목록
        audio_bytes_accumulated = 0 # 오디오 byte 누적 값
        chunk_collection_start_time = time.time() # 오디오 chunk 수집 시작 시간

        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_bytes(), timeout=30.0) # 오디오 데이터 수신
            except asyncio.TimeoutError: # 타임아웃 시
                await websocket.send_json({"type": "ping", "message": "connection_check"}) # 연결 상태 확인 메시지 전송
                chunk_collection_start_time = time.time() # 타임아웃 시 리셋
                continue
            
            audio_chunks.append(data) # 오디오 chunk 추가
            audio_bytes_accumulated += len(data) # 오디오 byte 누적
            elapsed_chunk_time = time.time() - chunk_collection_start_time # 오디오 chunk 수집 시간 계산

            if elapsed_chunk_time >= 1.0 or audio_bytes_accumulated >= 64 * 1024: # 오디오 chunk 수집 시간이 1초 이상이거나 오디오 byte 누적 값이 64KB 이상일 경우
                temp_file_path = None # 임시 파일 경로 초기화
                processing_start_time = time.time()
                logger.info(f"[AUDIO_PROCESS] 오디오 처리 시작 - client_id: {client_id}, 누적크기: {audio_bytes_accumulated}바이트, 누적시간: {elapsed_chunk_time:.2f}초")
                
                try:
                    # === 1단계: 오디오 청크 결합 ===
                    logger.info(f"[AUDIO_PROCESS] 1단계: 오디오 청크 결합 시작 - {len(audio_chunks)}개 청크")
                    full_audio_data = b"".join(audio_chunks) # 오디오 chunk 결합
                    audio_chunks = [] # 청크 리셋
                    audio_bytes_accumulated = 0 # 오디오 byte 누적 값 초기화
                    chunk_collection_start_time = time.time() # 오디오 chunk 수집 시간 리셋
                    logger.info(f"[AUDIO_PROCESS] 1단계 완료: 전체 오디오 크기 {len(full_audio_data)}바이트")

                    # === 2단계: 오디오 파일 처리 ===
                    step2_start = time.time()
                    logger.info(f"[AUDIO_PROCESS] 2단계: 오디오 파일 처리 시작")
                    temp_file_path, proc_error, proc_code = await audio_processor.process_audio_chunk(full_audio_data, client_id) # 오디오 청크 처리
                    step2_time = time.time() - step2_start
                    logger.info(f"[AUDIO_PROCESS] 2단계 완료: {step2_time:.2f}초 소요, temp_file: {temp_file_path}")
                    
                    if proc_error:
                        logger.error(f"[AUDIO_PROCESS] 2단계 오류: {proc_error} (코드: {proc_code})")
                        await websocket.send_json({"type": "error", "error_message": proc_error, "error_code": proc_code, "status": "error"}) # 오디오 처리 오류 전송
                        continue
                    
                    connection_engine.get_client_info(client_id)["temp_files"].append(temp_file_path) # 임시 파일 경로 추가
                    
                    # === 3단계: STT (음성→텍스트) ===
                    step3_start = time.time()
                    logger.info(f"[AUDIO_PROCESS] 3단계: STT 처리 시작")
                    user_text, stt_error, stt_code = await audio_processor.transcribe_audio(temp_file_path) # 오디오 파일 텍스트 변환
                    step3_time = time.time() - step3_start
                    logger.info(f"[AUDIO_PROCESS] 3단계 완료: {step3_time:.2f}초 소요, 인식결과: '{user_text}'")
                    
                    if stt_error: # 오디오 텍스트 변환 오류 시
                        logger.error(f"[AUDIO_PROCESS] 3단계 오류: {stt_error} (코드: {stt_code})")
                        await websocket.send_json({"type": "error", "error_message": stt_error, "error_code": stt_code, "status": "error", "user_text": ""}) # 오디오 텍스트 변환 오류 전송
                        continue
                    
                    # === 4단계: 빈 텍스트 처리 ===
                    if not user_text:
                        logger.warning(f"[AUDIO_PROCESS] 4단계: 빈 텍스트 감지, 기본 응답 전송")
                        response_packet = {"type": "ai_response", "text": "", "audio": "", "status": "ok", "user_text": user_text}
                        response_packet["error_message"] = "다시 말해줄래? 잘 안들렸어 미안해!" # 오류 메시지
                        response_packet["error_code"] = "no_valid_speech" # 오류 코드
                        await websocket.send_json(response_packet) # 오류 메시지 전송
                        continue
                    
                    # === 5단계: 챗봇 객체 검증 ===
                    step5_start = time.time()
                    logger.info(f"[AUDIO_PROCESS] 5단계: 챗봇 객체 검증 시작")
                    logger.info(f"[AUDIO_PROCESS] 챗봇 객체 타입: {type(chatbot_a)}, hasattr get_conversation_history: {hasattr(chatbot_a, 'get_conversation_history')}")
                    if not hasattr(chatbot_a, 'get_conversation_history'):
                        logger.error(f"[AUDIO_PROCESS] 5단계 오류: 챗봇 객체에 get_conversation_history 메서드가 없습니다. 타입: {type(chatbot_a)}")
                        await websocket.send_json({
                            "type": "error",
                            "error_message": "챗봇 객체에 get_conversation_history 메서드가 없습니다.",
                            "error_code": "chatbot_object_invalid",
                            "status": "error",
                            "user_text": user_text
                        })
                        continue
                    logger.info(f"[AUDIO_PROCESS] 5단계 완료: 챗봇 객체 검증 성공")
                    
                    # === 6단계: 챗봇 응답 생성 ===
                    step6_start = time.time()
                    logger.info(f"[AUDIO_PROCESS] 6단계: 챗봇 응답 생성 시작 - 입력텍스트: '{user_text}'")
                    try:
                        start_time = time.time()
                        response = await asyncio.to_thread(chatbot_a.get_response, user_text)
                        elapsed_time = time.time() - start_time
                        logger.info(f"ChatBot A 응답 생성 완료 (소요 시간: {elapsed_time:.2f}초)")
                    except Exception as e:
                        step6_time = time.time() - step6_start
                        logger.error(f"[AUDIO_PROCESS] 6단계 예외: {e} ({step6_time:.2f}초 소요)")
                        await websocket.send_json({"type": "error", "error_message": f"챗봇 응답 생성 중 예외: {e}", "error_code": "chatbot_response_exception", "status": "error", "user_text": user_text})
                        continue
                    
                    # === 7단계: TTS (텍스트→음성) ===
                    step7_start = time.time()
                    logger.info(f"[AUDIO_PROCESS] 7단계: TTS 처리 시작 - 텍스트: '{response[:50]}...'")
                    try:
                        bot_audio_b64, bot_tts_status, bot_tts_error, bot_tts_code = await audio_processor.synthesize_tts(response) # ElevenLabs로 챗봇 응답 음성 생성
                        step7_time = time.time() - step7_start
                        logger.info(f"[AUDIO_PROCESS] 7단계 완료: {step7_time:.2f}초 소요, 오디오크기: {len(bot_audio_b64) if bot_audio_b64 else 0}자")
                    except Exception as e:
                        step7_time = time.time() - step7_start
                        logger.error(f"[AUDIO_PROCESS] 7단계 예외: {e} ({step7_time:.2f}초 소요)")
                        bot_audio_b64, bot_tts_status, bot_tts_error, bot_tts_code = "", "error", f"TTS 처리 중 예외: {e}", "tts_exception"
                    
                    # === 8단계: 응답 전송 ===
                    step8_start = time.time()
                    logger.info(f"[AUDIO_PROCESS] 8단계: 응답 전송 시작")
                    response_packet = {
                        "type": "ai_response", "text": response, "audio": bot_audio_b64, # 챗봇 응답 텍스트 및 음성 전송
                        "status": bot_tts_status, "user_text": user_text, # 챗봇 응답 상태 및 사용자 텍스트 전송
                        "error_message": bot_tts_error, "error_code": bot_tts_code # 챗봇 응답 오류 메시지 및 코드 전송
                    }
                    await websocket.send_json(response_packet) # 챗봇 응답 전송
                    step8_time = time.time() - step8_start
                    total_time = time.time() - processing_start_time
                    logger.info(f"[AUDIO_PROCESS] 8단계 완료: {step8_time:.2f}초 소요, 전체처리시간: {total_time:.2f}초")
                    logger.info(f"[AUDIO_PROCESS] 처리 완료 - client_id: {client_id}")
                
                except WebSocketDisconnect: # WebSocket 연결 종료 시
                    logger.info(f"[AUDIO_PROCESS] WebSocket 연결 종료됨 (오디오 처리 중): {client_id}") # 로깅
                    raise # 상위 핸들러에서 처리하도록 다시 발생
                except Exception as e: # 오디오 처리 루프 오류 시
                    total_time = time.time() - processing_start_time
                    logger.error(f"[AUDIO_PROCESS] 예상치 못한 오류 ({client_id}): {e} (처리시간: {total_time:.2f}초)\n{traceback.format_exc()}") # 로깅
                    try:
                        await websocket.send_json({"type": "error", "error_message": f"오디오 처리 중 예상치 못한 오류: {str(e)}", "error_code": "audio_processing_unexpected_error", "status": "error"}) # 오류 메시지 전송
                    except:
                        logger.error(f"[AUDIO_PROCESS] 오류 응답 전송도 실패: {client_id}")
                finally:
                    if temp_file_path and temp_file_path in connection_engine.get_client_info(client_id)["temp_files"]: # 임시 파일 경로 확인
                         # 임시 파일은 disconnect 시 일괄 정리되므로 여기서 삭제 X
                        pass 
    
    except WebSocketDisconnect: # WebSocket 연결 종료 시
        logger.info(f"오디오 WebSocket 연결 종료됨: {client_id}") # 로깅
    except Exception as e: # 오디오 처리 루프 오류 시
        logger.error(f"오디오 WebSocket 핸들러 오류 ({client_id}): {e}\n{traceback.format_exc()}") # 로깅
        try:
            await websocket.send_json({"type": "error", "error_message": str(e), "error_code": "websocket_handler_error", "status": "error"}) # 오류 메시지 전송
        except: # 이미 연결이 끊겼을 수 있음
            pass
    finally:
        logger.info(f"오디오 WebSocket 연결 정리 시작: {client_id}")
        await connection_engine.handle_disconnect(client_id)
        logger.info(f"오디오 WebSocket 연결 정리 완료: {client_id}")

async def handle_chat_a_response(chatbot_a: ChatBotA, user_text: str) -> tuple:
    """
    ChatBot A (부기)로부터 응답을 가져옵니다.
    """
    try:
        if not user_text:
            return "", "챗봇 입력 없음", "chatbot_no_input"
        
        processed_text = user_text.strip()
        if not processed_text:
            return "", "챗봇 입력 없음", "chatbot_no_input"
            
        start_time = time.time()
        response = await asyncio.to_thread(chatbot_a.get_response, processed_text)
        elapsed_time = time.time() - start_time
        logger.info(f"ChatBot A 응답 생성 완료 (소요 시간: {elapsed_time:.2f}초)")
        
        return response, None, None
    except Exception as e:
        error_detail = traceback.format_exc()
        logger.error(f"ChatBot A 오류: {e}\n{error_detail}")
        return "", f"ChatBot A 오류: {e}", "chatbot_a_error" 