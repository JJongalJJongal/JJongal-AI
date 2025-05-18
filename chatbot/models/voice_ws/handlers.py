"""
WebSocket 메시지 핸들러 모듈

이 모듈은 WebSocket 메시지 처리 핸들러를 제공합니다.
"""
import os
import time
import json
import logging
import asyncio
import traceback
from typing import List, Optional, Dict, Any
from fastapi import WebSocket, WebSocketDisconnect, status
from ..chat_bot_a import StoryCollectionChatBot
from ..chat_bot_b import StoryGenerationChatBot
from .audio import transcribe_audio, synthesize_tts, process_audio_chunk
from .connection import ConnectionManager
from .auth import validate_connection, extract_token_from_header

async def handle_chat_response(chatbot, user_text: str):
    """
    챗봇에서 응답 가져오기
    
    Args:
        chatbot: 챗봇 인스턴스
        user_text (str): 사용자 입력 텍스트
    
    Returns:
        tuple: (챗봇 응답, 오류 메시지, 오류 코드)
    """
    try:
        if not user_text:
            return "", "챗봇 입력 없음", "chatbot_no_input"
        
        # 입력 전처리 - 공백 제거 및 텍스트 정리
        processed_text = user_text.strip()
        if not processed_text:
            return "", "챗봇 입력 없음", "chatbot_no_input"
            
        # 비동기 처리 시작 시간 기록
        start_time = time.time()
        
        # 비동기 처리를 위해 스레드풀에서 실행
        response = await asyncio.to_thread(chatbot.get_response, processed_text)
        
        # 처리 시간 기록
        elapsed_time = time.time() - start_time
        logging.info(f"챗봇 응답 생성 완료 (소요 시간: {elapsed_time:.2f}초)")
        
        return response, None, None
    except Exception as e:
        error_detail = traceback.format_exc()
        logging.error(f"챗봇 오류: {e}\n{error_detail}")
        return "", f"챗봇 오류: {e}", "chatbot_error"

async def handle_audio_websocket(websocket: WebSocket, child_name: str, age: int, interests: Optional[str] = None):
    """
    오디오 WebSocket 핸들러
    
    Args:
        websocket (WebSocket): WebSocket 연결
        child_name (str): 아이 이름
        age (int): 아이 나이
        interests (Optional[str]): 아이 관심사 (쉼표로 구분)
    """
    # 1. 연결 준비 및 검증
    client_id = f"{child_name}_{int(time.time())}" if child_name else f"unknown_{int(time.time())}"
    
    # 토큰 검증 - 헤더에서 토큰 추출 및 검증은 호출 전에 이미 완료
    
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
    connection_info = {
        "websocket": websocket,
        "chatbot": chatbot,
        "child_name": child_name,
        "age": age,
        "start_time": time.time(),
        "temp_files": []
    }
    ConnectionManager.add_client(client_id, connection_info)
    
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
        await ConnectionManager.handle_disconnect(client_id)
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
                    # 오디오 데이터 처리
                    audio_data = b"".join(audio_chunks)
                    temp_file_path, process_error, error_code = await process_audio_chunk(audio_data, client_id)
                    
                    if process_error:
                        error_packet = {
                            "type": "error",
                            "error_message": process_error,
                            "error_code": error_code,
                            "status": "error"
                        }
                        await websocket.send_json(error_packet)
                        continue
                    
                    # 임시 파일 추적
                    connection_info = ConnectionManager.get_client_info(client_id)
                    if connection_info and temp_file_path:
                        connection_info["temp_files"].append(temp_file_path)
                    
                    # Whisper 음성 인식 수행
                    user_text, error_message, error_code = await transcribe_audio(temp_file_path)
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
                        ai_response, cb_error_message, cb_error_code = await handle_chat_response(chatbot, user_text)
                        
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
                            connection_info = ConnectionManager.get_client_info(client_id)
                            if connection_info and "temp_files" in connection_info and temp_file_path in connection_info["temp_files"]:
                                connection_info["temp_files"].remove(temp_file_path)
                        except Exception as e:
                            logging.warning(f"임시 파일 삭제 실패: {e}")
                    
                    # 청크 변수 초기화
                    audio_chunks = []
                    audio_bytes = 0
                    chunk_start_time = time.time()
    
    except WebSocketDisconnect:
        logging.info(f"WebSocket 연결 종료: {client_id}")
        await ConnectionManager.handle_disconnect(client_id)
    
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
            await ConnectionManager.handle_disconnect(client_id)

async def handle_story_generation_websocket(websocket: WebSocket, child_name: str, age: int, interests: Optional[str] = None):
    """
    동화 생성을 위한 WebSocket 핸들러
    
    Args:
        websocket: WebSocket 연결
        child_name: 아이 이름
        age: 아이 나이
        interests: 관심사 (쉼표로 구분)
    """
    # 클라이언트 ID 생성
    client_id = f"story_{child_name}_{int(time.time())}"
    
    # 토큰 검증 - 헤더에서 토큰 추출 및 검증은 호출 전에 이미 완료
    
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
    instance_data = {
        "chatbot": chatbot_b,
        "child_name": child_name,
        "age": age,
        "interests": interest_list,
        "last_activity": time.time()
    }
    ConnectionManager.add_chatbot_b_instance(client_id, instance_data)
    
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
                    
                    # 활동 시간 업데이트
                    ConnectionManager.update_chatbot_b_activity(client_id)
                    
                    # 메시지 유형에 따른 처리
                    if message_type == "story_outline":
                        await handle_story_outline(websocket, client_id, message)
                    elif message_type == "generate_illustrations":
                        await handle_generate_illustrations(websocket, client_id)
                    elif message_type == "generate_voice":
                        await handle_generate_voice(websocket, client_id)
                    elif message_type == "get_preview":
                        await handle_get_preview(websocket, client_id)
                    elif message_type == "save_story":
                        await handle_save_story(websocket, client_id, message)
                    elif message_type == "ping":
                        # 연결 유지 핑
                        await websocket.send_text(json.dumps({"type": "pong"}))
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
    
    except WebSocketDisconnect:
        logging.info(f"꼬기(chatbot_b) 연결 종료: {client_id}")
    
    except Exception as e:
        logging.error(f"꼬기(chatbot_b) 처리 중 예외 발생: {client_id}, {str(e)}")
        traceback.print_exc()
        try:
            await websocket.send_text(json.dumps({
                "type": "error",
                "text": "서버 오류가 발생했어.",
                "error_message": str(e)
            }))
        except:
            pass  # 이미 연결이 끊어졌을 수 있음

# 이하는 story_generation_websocket의 핸들러 함수들
async def handle_story_outline(websocket: WebSocket, client_id: str, message: dict):
    """동화 줄거리 처리"""
    try:
        # 인스턴스 가져오기
        instance_data = ConnectionManager.get_chatbot_b_instance(client_id)
        if not instance_data:
            await websocket.send_text(json.dumps({
                "type": "error",
                "text": "세션이 만료되었어.",
                "error_message": "세션 만료"
            }))
            return
            
        chatbot_b = instance_data["chatbot"]
        story_outline = message.get("story_outline", {})
        
        # 줄거리 설정
        chatbot_b.set_story_outline(story_outline)
        
        # 상세 스토리 생성 응답
        await websocket.send_text(json.dumps({
            "type": "processing",
            "text": "동화 줄거리를 받았어! 이제 멋진 이야기를 만들어볼게. 잠시만 기다려줘~"
        }))
        
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

async def handle_generate_illustrations(websocket: WebSocket, client_id: str):
    """일러스트 생성 처리"""
    try:
        # 인스턴스 가져오기
        instance_data = ConnectionManager.get_chatbot_b_instance(client_id)
        if not instance_data:
            await websocket.send_text(json.dumps({
                "type": "error",
                "text": "세션이 만료되었어.",
                "error_message": "세션 만료"
            }))
            return
            
        chatbot_b = instance_data["chatbot"]
        
        await websocket.send_text(json.dumps({
            "type": "processing",
            "text": "멋진 그림을 그려볼게. 잠시만 기다려줘~"
        }))
        
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

async def handle_generate_voice(websocket: WebSocket, client_id: str):
    """내레이션 생성 처리"""
    try:
        # 인스턴스 가져오기
        instance_data = ConnectionManager.get_chatbot_b_instance(client_id)
        if not instance_data:
            await websocket.send_text(json.dumps({
                "type": "error",
                "text": "세션이 만료되었어.",
                "error_message": "세션 만료"
            }))
            return
            
        chatbot_b = instance_data["chatbot"]
        
        await websocket.send_text(json.dumps({
            "type": "processing",
            "text": "이야기를 읽어줄 목소리를 만들고 있어. 조금만 기다려줘~"
        }))
        
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

async def handle_get_preview(websocket: WebSocket, client_id: str):
    """동화 미리보기 처리"""
    try:
        # 인스턴스 가져오기
        instance_data = ConnectionManager.get_chatbot_b_instance(client_id)
        if not instance_data:
            await websocket.send_text(json.dumps({
                "type": "error",
                "text": "세션이 만료되었어.",
                "error_message": "세션 만료"
            }))
            return
            
        chatbot_b = instance_data["chatbot"]
        
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

async def handle_save_story(websocket: WebSocket, client_id: str, message: dict):
    """동화 저장 처리"""
    try:
        # 인스턴스 가져오기
        instance_data = ConnectionManager.get_chatbot_b_instance(client_id)
        if not instance_data:
            await websocket.send_text(json.dumps({
                "type": "error",
                "text": "세션이 만료되었어.",
                "error_message": "세션 만료"
            }))
            return
            
        chatbot_b = instance_data["chatbot"]
        story_name = message.get("story_name", f"story_{int(time.time())}")
        
        # 동화 저장
        output_dir = os.path.join("output", client_id)
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