"""
오디오 WebSocket 엔드포인트 핸들러

'/ws/audio' 경로의 WebSocket 연결 및 메시지 처리를 담당합니다.
"""
import time
import asyncio
import traceback
from datetime import datetime
from typing import Optional, Dict, Any
from fastapi import WebSocket, status
from fastapi.websockets import WebSocketDisconnect, WebSocketState
import uuid

from shared.utils.logging_utils import get_module_logger
from chatbot.models.chat_bot_a import ChatBotA # 부기 챗봇 import
from ..core.connection_engine import ConnectionEngine
from ..core.websocket_engine import WebSocketEngine # WebSocket 연결 종료 처리
from ..processors.audio_processor import AudioProcessor
from chatbot.models.voice_ws.processors.voice_cloning_processor import VoiceCloningProcessor


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
    WebSocket을 통한 실시간 음성 처리 및 음성 클로닝
    
    새로운 기능:
    - 사용자 음성 샘플 수집
    - 일정 샘플 수집 후 자동 음성 클론 생성
    - 생성된 클론 음성을 ChatBotB에 전달
    """
    
    client_id = str(uuid.uuid4())
    logger.info(f"[WEBSOCKET] 새로운 오디오 WebSocket 연결: {client_id}")
    logger.info(f"연결 파라미터 - 아이 이름: {child_name}, 나이: {age}, 관심사: {interests_str}")
    
    # 음성 클로닝 프로세서 초기화
    voice_cloning_processor = VoiceCloningProcessor()
    voice_cloning_enabled = True  # 음성 클로닝 기능 활성화 여부
    
    # WebSocket 엔진 인스턴스 생성
    ws_engine = WebSocketEngine()
    
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
        logger.info(f"==========================================")
        logger.info(f"오디오 WebSocket 연결 수락")
        logger.info(f"클라이언트 ID: {client_id}")
        logger.info(f"아이 이름: {child_name}, 나이: {age}세")
        logger.info(f"관심사: {interests_list}")
        logger.info(f"WebSocket 상태: {websocket.client_state}")
        logger.info(f"==========================================") # 로깅
        
        # 연결 확인을 위한 초기 핑 전송
        try:
            await ws_engine.send_status(websocket, "connected", "WebSocket 연결 성공")
            logger.info(f"초기 상태 메시지 전송 완료: {client_id}")
        except Exception as e:
            logger.error(f"초기 상태 메시지 전송 실패: {client_id}, 오류: {e}")
            await websocket.close(code=status.WS_1011_INTERNAL_ERROR, reason="초기 통신 실패")
            return
        
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
            await ws_engine.send_error(websocket, "챗봇 객체에 get_conversation_history 메서드가 없습니다.", "chatbot_object_invalid")
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
        logger.info(f"[GREETING] 인사말 생성 시작: {client_id}")
        greeting = await asyncio.to_thread(
            chatbot_a.initialize_chat, # 부기 챗봇 인스턴스 초기화
            child_name=child_name, # 아이 이름
            age=age, # 아이 연령대
            interests=interests_list, # 관심사 목록
            chatbot_name="부기" # 챗봇 이름
        )
        
        logger.info(f"[GREETING] TTS 음성 생성 시작: {client_id}")
        greeting_audio_b64, tts_status, tts_error, tts_code = await audio_processor.synthesize_tts(greeting) # ElevenLabs로 인사말 음성 생성
        
        greeting_packet = {
            "type": "ai_response",
            "text": greeting,
            "audio": greeting_audio_b64,
            "status": tts_status,
            "user_text": "",  # 인사말이므로 빈 값
            "confidence": 1.0,  # 인사말은 100% 신뢰도
            "timestamp": datetime.now().isoformat(),
            "error_message": tts_error,
            "error_code": tts_code
        }
        
        # 인사말 전송 및 확인
        send_success = await ws_engine.send_json(websocket, greeting_packet) # 인사말 전송
        if not send_success:
            logger.error(f"인사말 전송 실패: {client_id}")
            await websocket.close(code=status.WS_1011_INTERNAL_ERROR, reason="인사말 전송 실패")
            return
        
        logger.info(f"[GREETING] 인사말 전송 완료: {client_id}") # 로깅
        
        # 준비 상태 알림
        await ws_engine.send_status(websocket, "ready", "음성 대화 준비 완료")

        # 메인 대화 루프
        audio_chunks = [] # 오디오 chunk 목록
        audio_bytes_accumulated = 0 # 오디오 byte 누적 값
        chunk_collection_start_time = time.time() # 오디오 chunk 수집 시작 시간
        last_ping_time = time.time() # 마지막 ping 시간
        ping_interval = 30.0 # ping 간격 (30초)

        logger.info(f"[MAIN_LOOP] 메인 대화 루프 시작: {client_id}")

        while True:
            try:
                # 짧은 타임아웃으로 변경하여 더 빠른 응답성 확보
                data = await asyncio.wait_for(websocket.receive_bytes(), timeout=10.0) # 오디오 데이터 수신
            except asyncio.TimeoutError: # 타임아웃 시
                # 주기적인 ping 전송으로 연결 유지
                current_time = time.time()
                if current_time - last_ping_time >= ping_interval:
                    try:
                        if websocket.client_state != WebSocketState.CONNECTED:
                            logger.info(f"연결 끊어짐 감지: {client_id}")
                            break
                        
                        # ping 전송으로 연결 상태 확인
                        await ws_engine.ping(websocket)
                        last_ping_time = current_time
                        logger.debug(f"Keep-alive ping 전송: {client_id}")
                        
                        # 오디오 청크 수집 시간 리셋
                        chunk_collection_start_time = time.time()
                        continue
                    except Exception as e:
                        logger.info(f"연결 상태 체크 실패, 연결 종료: {client_id}, 오류: {e}")
                        break
                else:
                    # ping 주기가 아직 안됐으면 계속 대기
                    continue
            except WebSocketDisconnect: # WebSocket 연결 종료 시
                logger.info(f"클라이언트 연결 종료됨 (메인 루프): {client_id}")
                break # 루프 종료
            except Exception as e:
                logger.error(f"예상치 못한 데이터 수신 오류: {client_id}, 오류: {e}")
                break
            
            audio_chunks.append(data) # 오디오 chunk 추가
            audio_bytes_accumulated += len(data) # 오디오 byte 누적
            elapsed_chunk_time = time.time() - chunk_collection_start_time # 오디오 chunk 수집 시간 계산
            
            # 데이터 수신이 활발하면 ping 시간 업데이트 (불필요한 ping 방지)
            last_ping_time = time.time()

            # 개선된 오디오 청크 수집 조건
            # - 시간: 1.5초 (음성 완료 대기)
            # - 크기: 32KB (WAV 품질 고려)
            # - 최소 청크 수: 3개 이상 (연속성 보장)
            should_process = (
                elapsed_chunk_time >= 1.5 or 
                audio_bytes_accumulated >= 32 * 1024 or
                (len(audio_chunks) >= 3 and elapsed_chunk_time >= 0.8)
            )
            
            if should_process: # 오디오 처리 조건 충족 시
                temp_file_path = None # 임시 파일 경로 초기화
                processing_start_time = time.time()
                logger.info(f"[AUDIO_PROCESS] 오디오 처리 시작 - client_id: {client_id}")
                logger.info(f"[AUDIO_PROCESS] 수집 정보 - 청크수: {len(audio_chunks)}, 크기: {audio_bytes_accumulated}바이트, 시간: {elapsed_chunk_time:.2f}초")
                
                try:
                    # === 1단계: 오디오 청크 결합 ===
                    logger.info(f"[AUDIO_PROCESS] 1단계: 오디오 청크 결합 시작 - {len(audio_chunks)}개 청크")
                    full_audio_data = b"".join(audio_chunks)
                    audio_chunks = []
                    audio_bytes_accumulated = 0
                    chunk_collection_start_time = time.time()
                    logger.info(f"[AUDIO_PROCESS] 1단계 완료: 전체 오디오 크기 {len(full_audio_data)}바이트")

                    # === 새로운 기능: 음성 클로닝 샘플 수집 ===
                    if voice_cloning_enabled and full_audio_data:
                        voice_collection_start = time.time()
                        logger.info(f"[VOICE_CLONING] 음성 샘플 수집 시작 - client_id: {client_id}")
                        
                        # 사용자 음성 샘플 저장
                        sample_saved = await voice_cloning_processor.collect_user_audio_sample(
                            user_id=child_name,  # 아이 이름을 user_id로 사용
                            audio_data=full_audio_data
                        )
                        
                        if sample_saved:
                            sample_count = voice_cloning_processor.get_sample_count(child_name)
                            
                            # 진행상황 WebSocket 메시지 전송 (2개씩 수집될 때마다)
                            if sample_count % 2 == 0:
                                await ws_engine.send_json(websocket, {
                                    "type": "voice_clone_progress",
                                    "sample_count": sample_count,
                                    "ready_for_cloning": voice_cloning_processor.is_ready_for_cloning(child_name),
                                    "has_cloned_voice": voice_cloning_processor.get_user_voice_id(child_name) is not None,
                                    "message": f"목소리 수집 중... ({sample_count}/5)",
                                    "timestamp": datetime.now().isoformat()
                                })
                                logger.info(f"[VOICE_CLONING] 진행상황 전송 - {child_name}: {sample_count}개 샘플")
                            
                            # 5개 샘플 수집 완료 시 음성 클론 생성
                            if voice_cloning_processor.is_ready_for_cloning(child_name) and \
                               voice_cloning_processor.get_user_voice_id(child_name) is None:
                                
                                logger.info(f"[VOICE_CLONING] 클론 생성 시작 - {child_name}")
                                
                                # WebSocket으로 클론 생성 시작 알림
                                await ws_engine.send_json(websocket, {
                                    "type": "voice_clone_starting",
                                    "message": "목소리 복제를 시작합니다...",
                                    "timestamp": datetime.now().isoformat()
                                })
                                
                                # ElevenLabs IVC API로 음성 클론 생성
                                voice_id, error_msg = await voice_cloning_processor.create_instant_voice_clone(
                                    user_id=child_name,
                                    voice_name=f"{child_name}_voice_clone"
                                )
                                
                                if voice_id:
                                    # 성공 시 ChatBotB에 클론 음성 설정
                                    try:
                                        # ChatBotB 인스턴스 가져오기 또는 생성
                                        chatbot_b_data = connection_engine.get_chatbot_b_instance(client_id)
                                        if not chatbot_b_data:
                                            # ChatBotB 인스턴스 생성
                                            from chatbot.models.chat_bot_b import ChatBotB
                                            chatbot_b = ChatBotB()
                                            chatbot_b.set_target_age(age)
                                            
                                            # ConnectionEngine에 ChatBotB 저장
                                            connection_engine.add_chatbot_b_instance(client_id, {
                                                "chatbot_b": chatbot_b,
                                                "last_activity": time.time()
                                            })
                                            logger.info(f"[VOICE_CLONING] ChatBotB 인스턴스 생성: {client_id}")
                                        else:
                                            chatbot_b = chatbot_b_data["chatbot_b"]
                                            connection_engine.update_chatbot_b_activity(client_id)
                                        
                                        # 클론된 음성을 ChatBotB에 설정
                                        chatbot_b.set_cloned_voice_info(
                                            child_voice_id=voice_id,
                                            main_character_name=child_name  # 아이 이름을 주인공 이름으로 사용
                                        )
                                        
                                        # 성공 메시지 전송
                                        await ws_engine.send_json(websocket, {
                                            "type": "voice_clone_success",
                                            "voice_id": voice_id,
                                            "message": f"{child_name}님의 목소리가 성공적으로 복제되었어요! 이제 동화에서 주인공 목소리로 사용됩니다.",
                                            "timestamp": datetime.now().isoformat()
                                        })
                                        
                                        logger.info(f"[VOICE_CLONING] 클론 생성 및 ChatBotB 설정 완료 - {child_name}: {voice_id}")
                                        
                                    except Exception as chatbot_error:
                                        logger.error(f"[VOICE_CLONING] ChatBotB 설정 실패: {chatbot_error}")
                                        await ws_engine.send_json(websocket, {
                                            "type": "voice_clone_error",
                                            "message": f"음성 복제는 성공했지만 설정 중 오류가 발생했습니다: {chatbot_error}",
                                            "timestamp": datetime.now().isoformat()
                                        })
                                else:
                                    # 실패 메시지 전송
                                    await ws_engine.send_json(websocket, {
                                        "type": "voice_clone_error",
                                        "message": f"목소리 복제에 실패했습니다: {error_msg}",
                                        "timestamp": datetime.now().isoformat()
                                    })
                                    logger.error(f"[VOICE_CLONING] 클론 생성 실패 - {child_name}: {error_msg}")
                        
                        voice_collection_time = time.time() - voice_collection_start
                        logger.info(f"[VOICE_CLONING] 음성 샘플 수집 완료 - 소요시간: {voice_collection_time:.2f}초")

                    # === 기존 2단계: 오디오 파일 처리 ===
                    step2_start = time.time()
                    logger.info(f"[AUDIO_PROCESS] 2단계: 오디오 파일 처리 시작")
                    
                    # 연결 상태 체크 및 진행 상황 알림
                    if websocket.client_state != WebSocketState.CONNECTED:
                        logger.warning(f"[AUDIO_PROCESS] 2단계 전 연결 끊어짐 감지: {client_id}")
                        break
                    
                    await ws_engine.send_status(websocket, "processing", "음성을 분석하고 있어요...")
                    temp_file_path, proc_error, proc_code = await audio_processor.process_audio_chunk(full_audio_data, client_id)
                    step2_time = time.time() - step2_start
                    logger.info(f"[AUDIO_PROCESS] 2단계 완료: {step2_time:.2f}초 소요, temp_file: {temp_file_path}")
                    
                    if proc_error:
                        logger.error(f"[AUDIO_PROCESS] 2단계 오류: {proc_error} (코드: {proc_code})")
                        await ws_engine.send_error(websocket, proc_error, proc_code)
                        continue
                    
                    connection_engine.get_client_info(client_id)["temp_files"].append(temp_file_path)
                    
                    # === 3단계: STT (음성→텍스트) ===
                    step3_start = time.time()
                    logger.info(f"[AUDIO_PROCESS] 3단계: STT 처리 시작")
                    
                    # 연결 상태 체크 및 진행 상황 알림
                    if websocket.client_state != WebSocketState.CONNECTED:
                        logger.warning(f"[AUDIO_PROCESS] 3단계 전 연결 끊어짐 감지: {client_id}")
                        break
                    
                    await ws_engine.send_status(websocket, "transcribing", "음성을 텍스트로 변환하고 있어요...")
                    user_text, stt_error, stt_code = await audio_processor.transcribe_audio(temp_file_path) # 오디오 파일 텍스트 변환
                    step3_time = time.time() - step3_start
                    logger.info(f"[AUDIO_PROCESS] 3단계 완료: {step3_time:.2f}초 소요, 인식결과: '{user_text}'")
                    
                    if stt_error: # 오디오 텍스트 변환 오류 시
                        logger.error(f"[AUDIO_PROCESS] 3단계 오류: {stt_error} (코드: {stt_code})")
                        error_response = {
                            "type": "error",
                            "error_message": stt_error,
                            "error_code": stt_code,
                            "status": "error",
                            "user_text": "",
                            "timestamp": datetime.now().isoformat()
                        }
                        await ws_engine.send_json(websocket, error_response) # 오디오 텍스트 변환 오류 전송
                        continue
                    
                    # === 4단계: 빈 텍스트 처리 ===
                    if not user_text:
                        logger.warning(f"[AUDIO_PROCESS] 4단계: 빈 텍스트 감지, 기본 응답 전송")
                        response_packet = {
                            "type": "ai_response",
                            "text": "",
                            "audio": "",
                            "status": "ok",
                            "user_text": user_text,
                            "confidence": 0.0,  # 빈 텍스트이므로 신뢰도 0
                            "timestamp": datetime.now().isoformat(),
                            "error_message": "다시 말해줄래? 잘 안들렸어 미안해!",
                            "error_code": "no_valid_speech"
                        }
                        await ws_engine.send_json(websocket, response_packet) # 오류 메시지 전송
                        continue
                    
                    # === 5단계: 챗봇 객체 검증 ===
                    step5_start = time.time()
                    logger.info(f"[AUDIO_PROCESS] 5단계: 챗봇 객체 검증 시작")
                    logger.info(f"[AUDIO_PROCESS] 챗봇 객체 타입: {type(chatbot_a)}, hasattr get_conversation_history: {hasattr(chatbot_a, 'get_conversation_history')}")
                    if not hasattr(chatbot_a, 'get_conversation_history'):
                        logger.error(f"[AUDIO_PROCESS] 5단계 오류: 챗봇 객체에 get_conversation_history 메서드가 없습니다. 타입: {type(chatbot_a)}")
                        await ws_engine.send_json(websocket, {
                            "type": "error",
                            "error_message": "챗봇 객체에 get_conversation_history 메서드가 없습니다.",
                            "error_code": "chatbot_object_invalid",
                            "status": "error",
                            "user_text": user_text,
                            "timestamp": datetime.now().isoformat()
                        })
                        continue
                    logger.info(f"[AUDIO_PROCESS] 5단계 완료: 챗봇 객체 검증 성공")
                    
                    # === 6단계: 챗봇 응답 생성 ===
                    step6_start = time.time()
                    logger.info(f"[AUDIO_PROCESS] 6단계: 챗봇 응답 생성 시작 - 입력텍스트: '{user_text}'")
                    
                    # 연결 상태 체크 및 진행 상황 알림
                    if websocket.client_state != WebSocketState.CONNECTED:
                        logger.warning(f"[AUDIO_PROCESS] 6단계 전 연결 끊어짐 감지: {client_id}")
                        break
                    
                    await ws_engine.send_status(websocket, "thinking", "답변을 준비하고 있어요...")
                    try:
                        start_time = time.time()
                        response = await asyncio.to_thread(chatbot_a.get_response, user_text)
                        elapsed_time = time.time() - start_time
                        logger.info(f"ChatBot A 응답 생성 완료 (소요 시간: {elapsed_time:.2f}초)")
                    except Exception as e:
                        step6_time = time.time() - step6_start
                        logger.error(f"[AUDIO_PROCESS] 6단계 예외: {e} ({step6_time:.2f}초 소요)")
                        await ws_engine.send_json(websocket, {
                            "type": "error",
                            "error_message": f"챗봇 응답 생성 중 예외: {e}",
                            "error_code": "chatbot_response_exception",
                            "status": "error",
                            "user_text": user_text,
                            "timestamp": datetime.now().isoformat()
                        })
                        continue
                    
                    # === 7단계: TTS (텍스트→음성) ===
                    step7_start = time.time()
                    logger.info(f"[AUDIO_PROCESS] 7단계: TTS 처리 시작 - 텍스트: '{response[:50]}...'")
                    
                    # 연결 상태 체크 및 진행 상황 알림
                    if websocket.client_state != WebSocketState.CONNECTED:
                        logger.warning(f"[AUDIO_PROCESS] 7단계 전 연결 끊어짐 감지: {client_id}")
                        break
                    
                    await ws_engine.send_status(websocket, "generating_voice", "음성을 생성하고 있어요...")
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
                        "type": "ai_response",
                        "text": response,
                        "audio": bot_audio_b64,
                        "status": bot_tts_status,
                        "user_text": user_text,
                        "confidence": 0.85,  # 임시 기본값 (나중에 실제 STT confidence로 교체)
                        "timestamp": datetime.now().isoformat(),
                        "error_message": bot_tts_error,
                        "error_code": bot_tts_code
                    }
                    await ws_engine.send_json(websocket, response_packet) # 챗봇 응답 전송
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
                        await ws_engine.send_error(websocket, f"오디오 처리 중 예상치 못한 오류: {str(e)}", "audio_processing_unexpected_error") # 오류 메시지 전송
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
            await ws_engine.send_error(websocket, str(e), "websocket_handler_error") # 오류 메시지 전송
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