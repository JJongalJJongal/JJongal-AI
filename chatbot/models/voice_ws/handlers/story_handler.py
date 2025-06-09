"""
동화 생성 WebSocket 엔드포인트 핸들러

'/ws/story_generation' 경로의 WebSocket 연결 및 메시지 처리를 담당합니다.
"""
import json
import time
import asyncio
import traceback
import os
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
    websocket: WebSocket,
    child_name: str,
    age: int,
    interests_str: Optional[str],
    token: Optional[str]
):
    """
    스토리 생성 WebSocket 연결 처리
    
    주요 기능:
    1. ChatBot B 연결 및 스토리 생성 요청
    2. 스토리 생성 진행 상황 실시간 업데이트
    3. 완성된 멀티미디어 파일들을 WebSocket binary로 순서대로 전송
    """
    logger.info(f"스토리 생성 WebSocket 핸들러 시작: {child_name} ({age}세)")
    
    try:
        # 스토리 생성 진행 상황 전송
        await websocket.send_json({
            "type": "story_progress",
            "message": f"{child_name}님의 특별한 이야기를 만들고 있어요...",
            "progress": 10,
            "stage": "initialization"
        })
        
        # 여기에 실제 스토리 생성 로직 구현
        # ...
        
        # 🎯 완성된 멀티미디어 파일들을 binary로 순서대로 전송
        story_id = "example_story_123"
        story_title = f"{child_name}의 모험"
        
        # 1. 스토리 완성 메타데이터 전송
        story_metadata = {
            "type": "story_metadata",
            "story_id": story_id,
            "title": story_title,
            "child_name": child_name,
            "total_chapters": 2,
            "multimedia_count": {
                "images": 2,
                "audio": 3  # 내레이션 + 대화들
            },
            "sequence_total": 5,  # 총 전송할 파일 수
            "transfer_method": "websocket_binary_sequential",
            "timestamp": datetime.now().isoformat()
        }
        await websocket.send_json(story_metadata)
        logger.info(f"[STORY_META] 스토리 메타데이터 전송: {story_id}")
        
        # 2. 순서대로 파일 전송 (예시 - 실제로는 생성된 파일들을 읽어서 전송)
        sequence_order = [
            {"type": "image", "chapter": 1, "file": "/app/output/temp/images/story_123_ch1.png", "description": "첫 번째 장면"},
            {"type": "audio", "chapter": 1, "subtype": "narration", "file": "/app/output/temp/audio/story_123_narration1.mp3", "text": "옛날 옛적에..."},
            {"type": "audio", "chapter": 1, "subtype": "dialogue", "file": "/app/output/temp/audio/story_123_dialogue1.mp3", "text": "안녕하세요!", "speaker": "주인공"},
            {"type": "image", "chapter": 2, "file": "/app/output/temp/images/story_123_ch2.png", "description": "두 번째 장면"},
            {"type": "audio", "chapter": 2, "subtype": "narration", "file": "/app/output/temp/audio/story_123_narration2.mp3", "text": "그래서 모두 행복하게 살았답니다."}
        ]
        
        for seq_index, item in enumerate(sequence_order):
            try:
                # 파일이 실제로 존재하는지 확인하고 읽기
                file_path = item["file"]
                if not os.path.exists(file_path):
                    # 실제 구현에서는 파일이 없으면 스킵하거나 에러 처리
                    logger.warning(f"[STORY_FILE] 파일 없음, 더미 데이터로 대체: {file_path}")
                    continue
                
                # 파일 읽기
                with open(file_path, 'rb') as f:
                    file_data = f.read()
                
                file_size_mb = len(file_data) / (1024 * 1024)
                
                # 3. 각 파일의 메타데이터 전송
                file_metadata = {
                    "type": "story_file_metadata",
                    "story_id": story_id,
                    "sequence_index": seq_index,
                    "sequence_total": len(sequence_order),
                    "file_type": item["type"],
                    "chapter": item["chapter"],
                    "size": len(file_data),
                    "size_mb": round(file_size_mb, 2),
                    "format": "png" if item["type"] == "image" else "mp3",
                    "chunks_total": 1 if len(file_data) <= 1024*1024 else (len(file_data) // (1024*1024)) + 1,
                    "chunk_size": 1024*1024,  # 1MB 청크
                    "sequence_id": int(time.time() * 1000) + seq_index,
                    "description": item.get("description", ""),
                    "text": item.get("text", ""),
                    "speaker": item.get("speaker", ""),
                    "subtype": item.get("subtype", "")
                }
                
                await websocket.send_json(file_metadata)
                logger.info(f"[STORY_FILE] 파일 메타데이터 전송: {item['type']} ch{item['chapter']} ({file_size_mb:.2f}MB)")
                
                # 4. 파일 데이터 전송 (청킹 방식)
                if len(file_data) <= 1024*1024:
                    # 작은 파일 - 한 번에 전송
                    await websocket.send_bytes(file_data)
                    logger.info(f"[STORY_FILE] 작은 파일 전송 완료: {len(file_data)} bytes")
                else:
                    # 큰 파일 - 청킹해서 전송
                    chunk_size = 1024 * 1024  # 1MB 청크
                    total_chunks = (len(file_data) + chunk_size - 1) // chunk_size
                    
                    for chunk_index in range(total_chunks):
                        start_pos = chunk_index * chunk_size
                        end_pos = min(start_pos + chunk_size, len(file_data))
                        chunk_data = file_data[start_pos:end_pos]
                        
                        # 청크 헤더 전송
                        chunk_header = {
                            "type": "story_file_chunk_header",
                            "story_id": story_id,
                            "sequence_id": file_metadata["sequence_id"],
                            "chunk_index": chunk_index,
                            "total_chunks": total_chunks,
                            "chunk_size": len(chunk_data),
                            "is_final": chunk_index == total_chunks - 1
                        }
                        await websocket.send_json(chunk_header)
                        
                        # 청크 데이터 전송
                        await websocket.send_bytes(chunk_data)
                        
                        # 청크 간 지연
                        await asyncio.sleep(0.1)
                        
                        logger.debug(f"[STORY_CHUNK] 청크 {chunk_index+1}/{total_chunks} 전송 완료")
                    
                    logger.info(f"[STORY_FILE] 큰 파일 청킹 전송 완료: {total_chunks} 청크")
                
                # 5. 각 파일 전송 완료 신호
                file_complete = {
                    "type": "story_file_complete",
                    "story_id": story_id,
                    "sequence_id": file_metadata["sequence_id"],
                    "sequence_index": seq_index,
                    "file_type": item["type"],
                    "chapter": item["chapter"]
                }
                await websocket.send_json(file_complete)
                
                # 파일 간 지연 (순서 보장)
                await asyncio.sleep(0.2)
                
            except Exception as e:
                logger.error(f"[STORY_FILE] 파일 전송 실패: {item} - {e}")
                # 실패해도 다음 파일 계속 전송
                continue
        
        # 6. 전체 스토리 전송 완료 신호
        story_complete = {
            "type": "story_transfer_complete",
            "story_id": story_id,
            "title": story_title,
            "total_files_sent": len(sequence_order),
            "transfer_method": "websocket_binary_sequential",
            "message": f"{child_name}님의 이야기가 완성되었어요! 순서대로 감상해보세요.",
            "timestamp": datetime.now().isoformat()
        }
        await websocket.send_json(story_complete)
        logger.info(f"[STORY_COMPLETE] 스토리 전송 완료: {story_id}")
        
    except Exception as e:
        logger.error(f"스토리 생성 WebSocket 오류: {e}")
        await websocket.send_json({
            "type": "error",
            "error_message": f"스토리 생성 중 오류가 발생했습니다: {str(e)}",
            "error_code": "STORY_GENERATION_ERROR"
        })

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