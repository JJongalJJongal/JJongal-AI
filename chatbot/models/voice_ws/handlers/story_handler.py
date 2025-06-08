"""
동화 생성 WebSocket 엔드포인트 핸들러

'/ws/story_generation' 경로의 WebSocket 연결 및 메시지 처리를 담당합니다.
"""
import json
import time
import asyncio
import traceback
from typing import Optional, Dict, Any
from fastapi import WebSocket, status
from datetime import datetime

from shared.utils.logging_utils import get_module_logger
from chatbot.models.chat_bot_b import ChatBotB # 꼬기 챗봇 import
from ..core.connection_engine import ConnectionEngine # 연결 엔진 import
from ..core.websocket_engine import WebSocketDisconnect, WebSocketEngine # WebSocket 연결 종료 처리
from ..processors.audio_processor import AudioProcessor # 오디오 처리 프로세서
from ..processors.voice_cloning_processor import VoiceCloningProcessor # 음성 클론 프로세서

logger = get_module_logger(__name__) # 로깅

async def handle_story_generation_websocket(
    websocket: WebSocket, # WebSocket 객체
    child_name: str, # 아이 이름
    age: int, # 아이 나이
    interests_str: Optional[str], # 관심사 목록
    connection_engine: ConnectionEngine, # 연결 엔진
    audio_processor: AudioProcessor # 오디오 처리 프로세서
):
    """
    동화 생성 WebSocket 연결의 전체 라이프사이클을 관리.
    """
    client_id = f"storygen_{child_name}_{int(time.time())}" if child_name else f"storygen_unknown_{int(time.time())}" # 클라이언트 ID 생성
    
    # WebSocket 엔진 인스턴스 생성
    ws_engine = WebSocketEngine()

    if not child_name or not (4 <= age <= 9): # 아이 정보 검증
        await websocket.close(code=status.WS_1003_UNSUPPORTED_DATA, reason="아이 정보 오류") # WebSocket 연결 종료
        logger.warning(f"잘못된 파라미터 (동화 생성): {client_id}, 이름: {child_name}, 나이: {age}") # 로깅
        return

    interests_list = interests_str.split(',') if interests_str else [] # 관심사 목록 처리

    try:
        await websocket.accept() # WebSocket 연결 수락
        logger.info(f"동화 생성 WebSocket 연결 수락: {client_id} ({child_name}, {age}세)") # 로깅

        # ChatBot B 인스턴스 생성 및 관리 (ConnectionEngine 사용)
        chatbot_b = ChatBotB() # 꼬기 챗봇 인스턴스 생성
        chatbot_b.set_target_age(age) # 대상 연령 설정
        chatbot_b.set_child_info(name=child_name, interests=interests_list) # 아이 정보 설정
        
        # 꼬기 챗봇 인스턴스 추가
        connection_engine.add_chatbot_b_instance(client_id, {
            "websocket": websocket, # WebSocket 객체
            "chatbot_b": chatbot_b, # 꼬기 챗봇 인스턴스
            "child_name": child_name, # 아이 이름
            "age": age, # 아이 나이 
            "last_activity": time.time() # 마지막 활동 시간
        })

        # 연결 상태 전송
        await ws_engine.send_status(websocket, "connected", "꼬기(ChatBot B)와 연결되었습니다. 이야기 개요를 보내주세요.")

        while True:
            try:
                raw_message = await asyncio.wait_for(websocket.receive_text(), timeout=60.0) # 텍스트 메시지 수신
                message = json.loads(raw_message) # 메시지 파싱
                message_type = message.get("type") # 메시지 타입
                connection_engine.update_chatbot_b_activity(client_id) # 활동 시간 갱신

                if message_type == "story_outline": # 만약 메시지 타입이 "story_outline" 이면
                    await handle_story_outline(websocket, client_id, message, connection_engine, chatbot_b, ws_engine) # 이야기 개요 처리
                elif message_type == "generate_illustrations": # 만약 메시지 타입이 "generate_illustrations" 이면
                    await handle_generate_illustrations(websocket, client_id, chatbot_b, ws_engine) # 삽화 생성 처리
                elif message_type == "generate_voice": # 만약 메시지 타입이 "generate_voice" 이면
                    await handle_generate_voice(websocket, client_id, chatbot_b, ws_engine) # 음성 생성 처리
                elif message_type == "save_story": # 만약 메시지 타입이 "save_story" 이면
                    await handle_save_story(websocket, client_id, message, chatbot_b, ws_engine) # 이야기 저장 처리
                else: # 알 수 없는 메시지 타입 처리
                    await ws_engine.send_error(websocket, f"알 수 없는 메시지 타입: {message_type}", "unknown_message_type") # 오류 메시지 전송
            
            except asyncio.TimeoutError: # 타임아웃 처리
                await ws_engine.ping(websocket) # 연결 상태 확인 메시지 전송
                connection_engine.update_chatbot_b_activity(client_id) # 활동 시간 갱신
                continue
            except json.JSONDecodeError: # JSON parsing 오류 처리
                await ws_engine.send_error(websocket, "잘못된 JSON 형식입니다.", "json_decode_error") # 오류 메시지 전송
            except WebSocketDisconnect: # WebSocket 연결 끊어진 경우
                logger.info(f"동화 생성 클라이언트 연결 종료됨 (메시지 루프): {client_id}") # 로깅
                raise
            except Exception as e: # 예외 처리
                logger.error(f"동화 생성 메시지 루프 오류 ({client_id}): {e}\n{traceback.format_exc()}") # 로깅
                await ws_engine.send_error(websocket, str(e), "story_loop_error") # 오류 메시지 전송

    except WebSocketDisconnect:
        logger.info(f"동화 생성 WebSocket 연결 종료됨: {client_id}")
    except Exception as e:
        logger.error(f"동화 생성 WebSocket 핸들러 오류 ({client_id}): {e}\n{traceback.format_exc()}")
        try:
            await ws_engine.send_error(websocket, str(e), "story_handler_error")
        except: # 이미 연결이 끊겼을 수 있음
            pass 
    finally:
        logger.info(f"동화 생성 WebSocket 연결 정리 시작: {client_id}")
        # ChatBot B 인스턴스는 ConnectionEngine의 타임아웃 로직으로 정리되거나, 명시적 disconnect시 정리될 수 있음
        # 여기서는 연결 자체에 대한 정리만 수행 (ConnectionEngine이 관리하므로 별도 호출 불필요할 수 있음)
        if connection_engine.get_chatbot_b_instance(client_id):
             # 필요하다면 여기서 chatbot_b_instances에서 제거하는 로직 추가
             pass
        # 일반 연결 해제시 로직은 connection_engine.handle_disconnect에서 처리
        logger.info(f"동화 생성 WebSocket 연결 정리 완료: {client_id}")

async def handle_story_outline(websocket: WebSocket, client_id: str, message: dict, connection_engine: ConnectionEngine, chatbot_b: ChatBotB, ws_engine: WebSocketEngine):
    """이야기 개요 처리 핸들러 (클론 음성 지원)"""
    logger.info(f"이야기 개요 수신 ({client_id}): {message.get('outline')}")
    story_outline_data = message.get("outline")
    if not story_outline_data or not isinstance(story_outline_data, dict):
        await ws_engine.send_error(websocket, "잘못된 이야기 개요 형식입니다.", "invalid_story_outline")
        return

    try:
        # ChatBot B에 개요 설정
        await asyncio.to_thread(chatbot_b.set_story_outline, story_outline_data)
        
        # 클론 음성 지원하는 generate_story 핸들러 호출
        await handle_generate_story(
            websocket=websocket,
            client_id=client_id,
            request_data={"story_outline": story_outline_data},
            connection_engine=connection_engine,
            ws_engine=ws_engine
        )
        
    except Exception as e:
        logger.error(f"이야기 개요 처리 중 오류 ({client_id}): {e}\n{traceback.format_exc()}")
        await ws_engine.send_error(websocket, f"이야기 처리 오류: {str(e)}", "story_outline_error")

async def handle_generate_illustrations(websocket: WebSocket, client_id: str, chatbot_b: ChatBotB, ws_engine: WebSocketEngine):
    """삽화 생성 요청 처리 핸들러"""
    logger.info(f"삽화 생성 요청 수신 ({client_id})")
    try:
        # 삽화 생성 (ChatBot B 내부 로직 사용)
        illustrations = await asyncio.to_thread(chatbot_b.generate_illustrations)
        if illustrations:
            await ws_engine.send_json(websocket, {"type": "illustrations_generated", "illustrations": illustrations, "status": "ok"})
            logger.info(f"삽화 생성 완료 및 전송 ({client_id})")
        else:
            await ws_engine.send_error(websocket, "삽화 생성 실패", "illustration_generation_failed")
            logger.error(f"삽화 생성 실패 ({client_id})")
    except Exception as e:
        logger.error(f"삽화 생성 중 오류 ({client_id}): {e}\n{traceback.format_exc()}")
        await ws_engine.send_error(websocket, f"삽화 생성 오류: {str(e)}", "illustration_generation_error")

async def handle_generate_voice(websocket: WebSocket, client_id: str, chatbot_b: ChatBotB, ws_engine: WebSocketEngine):
    """음성 생성 요청 처리 핸들러"""
    logger.info(f"음성 생성 요청 수신 ({client_id})")
    try:
        # 음성 생성 (ChatBot B 내부 로직 사용)
        voice_data = await asyncio.to_thread(chatbot_b.generate_voice)
        if voice_data:
            await ws_engine.send_json(websocket, {"type": "voice_generated", "voice_data": voice_data, "status": "ok"})
            logger.info(f"음성 생성 완료 및 전송 ({client_id})")
        else:
            await ws_engine.send_error(websocket, "음성 생성 실패", "voice_generation_failed")
            logger.error(f"음성 생성 실패 ({client_id})")
    except Exception as e:
        logger.error(f"음성 생성 중 오류 ({client_id}): {e}\n{traceback.format_exc()}")
        await ws_engine.send_error(websocket, f"음성 생성 오류: {str(e)}", "voice_generation_error")

async def handle_get_preview(websocket: WebSocket, client_id: str, chatbot_b: ChatBotB):
    """미리보기 요청 처리 핸들러"""
    logger.info(f"미리보기 요청 수신 ({client_id})")
    try:
        preview_data = await asyncio.to_thread(chatbot_b.get_story_preview)
        if preview_data:
            await websocket.send_json({"type": "preview_data", "preview": preview_data, "status": "ok"})
        else:
            await websocket.send_json({"type": "error", "message": "미리보기 생성 실패", "status": "error"})
    except Exception as e:
        logger.error(f"미리보기 생성 중 오류 ({client_id}): {e}\n{traceback.format_exc()}")
        await websocket.send_json({"type": "error", "message": f"미리보기 오류: {str(e)}", "status": "error"})

async def handle_save_story(websocket: WebSocket, client_id: str, message: dict, chatbot_b: ChatBotB, ws_engine: WebSocketEngine):
    """이야기 저장 요청 처리 핸들러"""
    logger.info(f"이야기 저장 요청 수신 ({client_id})")
    # file_format = message.get("format", "json") # 필요시 파일 포맷 지정
    try:
        # save_result = await asyncio.to_thread(chatbot_b.save_story_to_file, file_format=file_format)
        # ChatBotB에 저장 기능이 있다면 위와 같이 호출
        # 현재 ChatBotB에는 해당 기능이 명시적으로 없으므로, 임시로 성공 응답
        # 실제 저장 로직은 ChatBotB 또는 별도 유틸리티에 구현 필요
        
        # 임시: 저장 성공 메시지 전송 (실제 저장 로직은 ChatBotB에 구현되어야 함)
        # final_story_data = chatbot_b.get_generated_story_data() # 예시
        # if final_story_data:
        #     # 여기서 파일 저장 로직을 수행할 수 있음 (예: ws_utils.save_generated_story)
        #     pass
        
        await ws_engine.send_json(websocket, {"type": "story_saved", "message": "이야기 저장 기능은 ChatBot B에 구현 필요", "status": "ok_placeholder"})
        logger.info(f"이야기 저장 처리 완료 (플레이스홀더) ({client_id})")
        
    except Exception as e:
        logger.error(f"이야기 저장 중 오류 ({client_id}): {e}\n{traceback.format_exc()}")
        await websocket.send_json({"type": "error", "message": f"이야기 저장 오류: {str(e)}", "status": "error"}) 

async def handle_generate_story(websocket: WebSocket, client_id: str, request_data: Dict[str, Any], 
                               connection_engine: ConnectionEngine, ws_engine: WebSocketEngine):
    """동화 생성 요청 처리 핸들러 (클론 음성 지원)"""
    logger.info(f"동화 생성 요청 수신 ({client_id})")
    try:
        # 연결 정보 가져오기
        connection_info = connection_engine.get_client_info(client_id)
        if not connection_info:
            await ws_engine.send_error(websocket, "연결 정보를 찾을 수 없습니다", "connection_not_found")
            return
        
        child_name = connection_info.get("child_name", "친구")
        age = connection_info.get("age", 7)
        
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
            logger.info(f"[STORY_GEN] ChatBotB 인스턴스 생성: {client_id}")
        else:
            chatbot_b = chatbot_b_data["chatbot_b"]
            connection_engine.update_chatbot_b_activity(client_id)
        
        # 클론된 음성이 있는지 확인 및 설정
        voice_cloning_processor = VoiceCloningProcessor()
        cloned_voice_id = voice_cloning_processor.get_user_voice_id(child_name)
        if cloned_voice_id:
            chatbot_b.set_cloned_voice_info(
                child_voice_id=cloned_voice_id,
                main_character_name=child_name
            )
            logger.info(f"[STORY_GEN] 클론된 음성 설정 완료 - {child_name}: {cloned_voice_id}")
            
            # 클론 음성 사용 알림
            await ws_engine.send_json(websocket, {
                "type": "voice_clone_applied",
                "message": f"{child_name}님의 복제된 목소리를 동화에 적용했어요!",
                "voice_id": cloned_voice_id,
                "timestamp": datetime.now().isoformat()
            })
        
        # 스토리 개요 설정
        story_outline = request_data.get("story_outline", {})
        chatbot_b.set_story_outline(story_outline)
        
        # 진행 상황 콜백 함수 정의
        async def progress_callback(progress_data):
            await ws_engine.send_json(websocket, {
                "type": "story_progress",
                "progress": progress_data,
                "timestamp": datetime.now().isoformat()
            })
        
        # 동화 생성 시작 알림
        await ws_engine.send_json(websocket, {
            "type": "story_generation_started",
            "message": "동화 생성을 시작합니다...",
            "has_cloned_voice": cloned_voice_id is not None,
            "timestamp": datetime.now().isoformat()
        })
        
        # 동화 생성 (Enhanced Mode 사용)
        result = await chatbot_b.generate_detailed_story(
            progress_callback=progress_callback,
            use_websocket_voice=True  # WebSocket 스트리밍 음성 사용
        )
        
        # 생성 완료 알림
        await ws_engine.send_json(websocket, {
            "type": "story_generated",
            "result": result,
            "cloned_voice_used": cloned_voice_id is not None,
            "timestamp": datetime.now().isoformat()
        })
        
        logger.info(f"동화 생성 완료 ({client_id}) - 클론 음성 사용: {cloned_voice_id is not None}")
        
    except Exception as e:
        error_detail = traceback.format_exc()
        logger.error(f"동화 생성 중 오류 ({client_id}): {e}\n{error_detail}")
        await ws_engine.send_error(websocket, f"동화 생성 오류: {str(e)}", "story_generation_error") 