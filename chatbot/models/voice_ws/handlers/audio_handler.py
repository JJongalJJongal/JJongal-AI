"""
오디오 WebSocket 엔드포인트 핸들러

'/ws/audio' 경로의 WebSocket 연결 및 메시지 처리를 담당합니다.
"""
import time
import asyncio
import traceback
import json
from datetime import datetime
from typing import Optional, Dict, Any
from fastapi import WebSocket, status
from fastapi.websockets import WebSocketDisconnect, WebSocketState
import tempfile

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
    오디오 WebSocket 연결 처리
    
    주요 기능:
    1. 실시간 오디오 스트리밍 수신
    2. 음성을 텍스트로 변환 (STT)
    3. ChatBot A를 통한 대화 처리
    4. 텍스트를 음성으로 변환 (TTS)  
    5. 충분한 정보 수집 시 ChatBot B로 자동 전환
    6. 동화 생성 완료 시 프론트엔드 알림
    """
    ws_engine = WebSocketEngine()
    client_id = f"{child_name}_{int(time.time())}"
    
    # 오디오 수집 상태
    audio_chunks = []
    
    # 연결 유지 관리
    ping_interval = 30.0  # 30초마다 ping
    last_ping_time = time.time()
    
    try:
        
        logger.info(f"오디오 WebSocket 핸들러 시작: {client_id} ({child_name}, {age}세)") # 로깅
        
        # 즉시 연결 중 메시지 전송 (타임아웃 방지)
        await ws_engine.send_status(websocket, "partial", f"안녕 {child_name}! 부기가 준비중이라 조금만 기다려줘!")
        logger.info(f"초기화 메시지 전송 완료: {client_id}")
        
        # VectorDB 인스턴스 생성 (ChatBotA에 필요)
        try:
            from chatbot.data.vector_db.core import VectorDB
            import os
            
            # .env에서 VectorDB 경로 읽기 (통일된 환경변수 사용)
            chroma_base = os.getenv("CHROMA_DB_PATH", "chatbot/data/vector_db")
            vector_db_path = os.path.join(chroma_base, "main")  # main DB 사용
            logger.info(f"VectorDB 경로 환경변수: {vector_db_path}")
            
            # VectorDB 초기화
            vector_db = VectorDB(
                persist_directory=vector_db_path,
                embedding_model="nlpai-lab/KURE-v1",
                use_hybrid_mode=True,
                memory_cache_size=1000,
                enable_lfu_cache=True
            )
            logger.info(f"VectorDB 초기화 완료: {vector_db_path}")
        
            # ChatBot A 인스턴스 생성 및 관리 (ConnectionEngine 사용)
            chatbot_a = ChatBotA(
                vector_db_instance=vector_db,
                token_limit=10000,
                use_langchain=True,
                legacy_compatibility=True,
                enhanced_mode=True,
                enable_performance_tracking=True
            )
            
            chatbot_a.update_child_info(child_name=child_name, age=age, interests=[item.strip() for item in interests_str.split(",")] if interests_str else [])
            
            # ChatBot A 인스턴스 추가 
            connection_engine.add_client(client_id, {
                "websocket": websocket,
                "chatbot_a": chatbot_a,
                "child_name": child_name,
                "age": age,
                "interests": [item.strip() for item in interests_str.split(",")] if interests_str else [],
                "last_activity": time.time()
            })
        except Exception as e:
            logger.warning(f"ChatBotA 초기화 실패: {e}, None으로 진행")
            raise
        
        # 연결 상태 전송
        await ws_engine.send_status(websocket, "connected", f"안녕 {child_name}! 부기와 함께 재미있는 이야기를 만들어보자!")
        
        while True:
            try:
                # WebSocket 메시지 수신 (10초 타임아웃)
                message = await asyncio.wait_for(websocket.receive(), timeout=10.0)
                
                # WebSocket disconnect 처리
                if message.get("type") == "websocket.disconnect":
                    logger.info(f"클라이언트 연결 종료: {client_id}")
                    break
                
                if message.get("type") == "websocket.receive":
                    if "bytes" in message:
                        # 바이너리 데이터 (오디오) 수신
                        audio_data = message["bytes"]
                        if len(audio_data) > 0:
                            audio_chunks.append(audio_data)
                            logger.debug(f"오디오 청크 수신: {len(audio_data)} bytes (총 {len(audio_chunks)} 청크)")
                        continue
                    elif "text" in message:
                        # 텍스트 메시지 수신 (제어 신호)
                        text_data = message["text"]
                        logger.debug(f"텍스트 메시지 수신: {text_data[:100]}...")
                        
                        try:
                            control_message = json.loads(text_data)
                            
                            if control_message.get("type") == "audio_end":
                                logger.info(f"[AUDIO_END] 오디오 종료 신호 수신, 처리 시작: {client_id}")
                                processing_start_time = time.time()
                                
                                if audio_chunks:
                                    logger.info(f"[AUDIO_END] 수집된 오디오 청크: {len(audio_chunks)}개")
                                    
                                    try:
                                        # 오디오 데이터 병합
                                        combined_audio = b''.join(audio_chunks)
                                        logger.info(f"[AUDIO_END] 병합된 오디오 크기: {len(combined_audio)} bytes")
                                        
                                        # 임시 파일로 저장
                                        temp_file_path = None
                                        try:
                                            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                                                temp_file.write(combined_audio)
                                                temp_file_path = temp_file.name
                                            
                                            logger.debug(f"[AUDIO_END] 임시 파일 생성: {temp_file_path}")
                                            
                                            # STT 처리
                                            text, error_msg, error_code = await audio_processor.transcribe_audio(temp_file_path)
                                            
                                            # STT 결과를 기존 형식에 맞게 변환
                                            if text and not error_msg:
                                                stt_result = {"text": text, "confidence": 0.95}  # confidence는 기본값
                                            else:
                                                logger.error(f"[STT] 오류 발생: {error_msg} (오류 코드: {error_code})")
                                                stt_result = None
                                            
                                            if stt_result and stt_result.get("text"):
                                                user_text = stt_result["text"].strip()
                                                confidence = stt_result.get("confidence", 0.0)
                                                
                                                logger.info(f"[STT] 변환 완료: '{user_text}' (신뢰도: {confidence:.2f})")
                                                
                                                # ChatBot A 응답 처리
                                                ai_response, tts_result, conversation_length = await handle_chat_a_response(chatbot_a, user_text, audio_processor)
                                                
                                                # 응답 패킷 구성
                                                response_packet = {
                                                    "type": "ai_response",
                                                    "text": ai_response,
                                                    "audio": tts_result.get("audio_data") if tts_result else None,
                                                    "user_text": user_text,
                                                    "confidence": confidence,
                                                    "conversation_length": conversation_length,
                                                    "timestamp": datetime.now().isoformat()
                                                }
                                                
                                                # 응답 전송
                                                await ws_engine.send_json(websocket, response_packet)
                                                logger.info(f"[AUDIO_END] 응답 전송 완료: {ai_response[:50]}...")
                                                
                                                # 오디오 청크 초기화
                                                audio_chunks.clear()
                                                
                                                # === 부기 → 꼬기 자동 전환 로직 ===
                                                if hasattr(chatbot_a, 'story_engine'):
                                                    story_engine = chatbot_a.story_engine
                                                    
                                                    # 충분한 정보 수집 조건 체크
                                                    is_story_ready = await check_story_completion(story_engine, conversation_length, child_name, age)
                                                    
                                                    if is_story_ready:
                                                        logger.info(f"[STORY_READY] 충분한 이야기 정보 수집 완료로 자동 전환 시작: {client_id}")
                                                        
                                                        # 이야기 준비 완료 메시지 전송
                                                        story_ready_packet = {
                                                            "type": "conversation_end",
                                                            "text": f"와! {child_name}가 들려준 이야기로 정말 멋진 동화를 만들 수 있을 것 같아요!",
                                                            "message": "충분한 이야기 정보가 모였어요. 이제 특별한 동화를 만들어드릴게요!",
                                                            "reason": "story_information_complete",
                                                            "user_text": user_text,
                                                            "story_elements": story_engine.get_story_elements(),
                                                            "timestamp": datetime.now().isoformat()
                                                        }
                                                        
                                                        send_success = await ws_engine.send_json(websocket, story_ready_packet)
                                                        if send_success:
                                                            logger.info(f"[STORY_READY] 이야기 준비 완료 메시지 전송 완료: {client_id}")
                                                        
                                                        # === 꼬기(ChatBot B) 자동 호출 (WorkflowOrchestrator 사용) ===
                                                        try:
                                                            story_id = await handle_orchestrator_story_generation(
                                                                websocket=websocket,
                                                                client_id=client_id,
                                                                chatbot_a=chatbot_a,
                                                                child_name=child_name,
                                                                age=age,
                                                                interests_list=[item.strip() for item in interests_str.split(",")] if interests_str else [],
                                                                ws_engine=ws_engine,
                                                                connection_engine=connection_engine
                                                            )
                                                            
                                                            # story_id를 프론트엔드에 전송 (status 체크용)
                                                            await ws_engine.send_json(websocket, {
                                                                "type": "story_id_assigned",
                                                                "story_id": story_id,
                                                                "message": "동화 생성이 시작되었어요! 잠시만 기다려주세요.",
                                                                "status_check_url": f"/api/v1/stories/{story_id}/completion",
                                                                "timestamp": datetime.now().isoformat()
                                                            })
                                                            
                                                        except Exception as story_gen_error:
                                                            logger.error(f"[STORY_GEN] 자동 동화 생성 중 오류: {story_gen_error}")
                                                            await ws_engine.send_error(websocket, f"동화 생성 중 오류가 발생했습니다: {str(story_gen_error)}", "story_generation_failed")
                                                        
                                                        # 연결 정리 및 종료
                                                        await connection_engine.handle_disconnect(client_id)
                                                        await websocket.close(code=status.WS_1000_NORMAL_CLOSURE, reason="이야기 정보 수집 완료 및 동화 생성 완료")
                                                        return
                                                
                                                processing_time = time.time() - processing_start_time
                                                logger.info(f"[AUDIO_END] 전체 처리 완료: {processing_time:.2f}초")
                                                
                                            else:
                                                logger.warning(f"[AUDIO_END] STT 실패 또는 빈 텍스트")
                                                await ws_engine.send_json(websocket, {
                                                    "type": "ai_response",
                                                    "text": "음성을 제대로 들을 수 없었어요. 다시 말해주세요!",
                                                    "audio": None,
                                                    "status": "stt_failed",
                                                    "user_text": "",
                                                    "confidence": 0.0,
                                                    "timestamp": datetime.now().isoformat()
                                                })
                                        
                                        except Exception as audio_processing_error:
                                            logger.error(f"[AUDIO_END] 오디오 처리 중 상세 오류: {audio_processing_error}")
                                            logger.error(f"[AUDIO_END] 오류 스택 트레이스: {traceback.format_exc()}")
                                            await ws_engine.send_error(websocket, f"오디오 처리 중 오류가 발생했습니다", "audio_processing_error")
                                            return
                                        
                                        # 부기가 충분한 정보 수집 완료 판단
                                        if hasattr(chatbot_a, 'story_engine'):
                                            story_engine = chatbot_a.story_engine
                                            conversation_length = len(chatbot_a.conversation.get_conversation_history())
                                            
                                            # 충분한 정보 수집 조건 체크
                                            is_story_ready = await check_story_completion(story_engine, conversation_length, child_name, age)
                                            
                                            if is_story_ready:
                                                logger.info(f"[STORY_READY] 충분한 이야기 정보 수집 완료로 대화 종료: {client_id}")
                                                
                                                # 이야기 준비 완료 메시지 전송
                                                story_ready_packet = {
                                                    "type": "conversation_end",
                                                    "text": f"와! {child_name}가 들려준 이야기로 정말 멋진 동화를 만들 수 있을 것 같아요!",
                                                    "message": "충분한 이야기 정보가 모였어요. 이제 특별한 동화를 만들어드릴게요!",
                                                    "reason": "story_information_complete",
                                                    "user_text": user_text,
                                                    "story_elements": story_engine.get_story_elements(),
                                                    "timestamp": datetime.now().isoformat()
                                                }
                                                
                                                send_success = await ws_engine.send_json(websocket, story_ready_packet)
                                                if send_success:
                                                    logger.info(f"[STORY_READY] 이야기 준비 완료 메시지 전송 완료: {client_id}")
                                                
                                                # 연결 정리 및 종료
                                                await connection_engine.handle_disconnect(client_id)
                                                await websocket.close(code=status.WS_1000_NORMAL_CLOSURE, reason="이야기 정보 수집 완료")
                                                return
                                        
                                        processing_time = time.time() - processing_start_time
                                        logger.info(f"[AUDIO_END] 전체 처리 완료: {processing_time:.2f}초")
                                        
                                    except Exception as e:
                                        logger.error(f"[AUDIO_END] 처리 중 오류: {e}", exc_info=True)
                                        await ws_engine.send_error(websocket, f"오디오 처리 중 오류가 발생했습니다: {str(e)}", "audio_processing_failed")
                                    
                                    finally:
                                        # 임시 파일 정리
                                        if temp_file_path and os.path.exists(temp_file_path):
                                            try:
                                                os.remove(temp_file_path)
                                                logger.debug(f"[AUDIO_END] 임시 파일 삭제: {temp_file_path}")
                                            except Exception as cleanup_error:
                                                logger.warning(f"[AUDIO_END] 임시 파일 삭제 실패: {cleanup_error}")
                                else:
                                    logger.warning(f"[AUDIO_END] 처리할 오디오 청크가 없음")
                                    await ws_engine.send_json(websocket, {
                                        "type": "ai_response", 
                                        "text": "음성이 수신되지 않았어요. 다시 시도해 주세요.",
                                        "audio": None,
                                        "status": "no_audio_received",
                                        "user_text": "",
                                        "confidence": 0.0,
                                        "timestamp": datetime.now().isoformat()
                                    })
                                continue
                            
                        
                            
                            elif control_message.get("type") == "conversation_finish":
                                logger.info(f"[CONVERSATION_FINISH] 사용자가 대화 완료 요청: {client_id}")
                                
                                # === 꼬기(ChatBot B) 자동 호출 (WorkflowOrchestrator 사용) ===
                                try:
                                    story_id = await handle_orchestrator_story_generation(
                                        websocket=websocket,
                                        client_id=client_id,
                                        chatbot_a=chatbot_a,
                                        child_name=child_name,
                                        age=age,
                                        interests_list=[item.strip() for item in interests_str.split(",")] if interests_str else [],
                                        ws_engine=ws_engine,
                                        connection_engine=connection_engine
                                    )
                                    
                                    # story_id를 프론트엔드에 전송 (status 체크용)
                                    await ws_engine.send_json(websocket, {
                                        "type": "story_id_assigned",
                                        "story_id": story_id,
                                        "message": "동화 생성이 시작되었어요! 잠시만 기다려주세요.",
                                        "status_check_url": f"/api/v1/stories/{story_id}/completion",
                                        "timestamp": datetime.now().isoformat()
                                    })
                                    
                                except Exception as story_gen_error:
                                    logger.error(f"[STORY_GEN] 수동 동화 생성 중 오류: {story_gen_error}")
                                    await ws_engine.send_error(websocket, f"동화 생성 중 오류가 발생했습니다: {str(story_gen_error)}", "story_generation_failed")
                                
                                # 연결 정리 및 종료
                                await connection_engine.handle_disconnect(client_id)
                                await websocket.close(code=status.WS_1000_NORMAL_CLOSURE, reason="사용자 요청으로 대화 종료 및 동화 생성 완료")
                                return
                            else:
                                logger.info(f"[CONTROL] 알 수 없는 제어 메시지: {control_message}")
                                continue
                        except json.JSONDecodeError as e:
                            logger.warning(f"텍스트 메시지가 JSON이 아님: {text_data[:50]}..., 오류: {e}")
                            continue
                    elif control_message.get("type") == "websocket.disconnect":
                        logger.info(f"[WEBSOCKET_DISCONNECT] 클라이언트 연결 종료: {client_id}")
                        await connection_engine.handle_disconnect(client_id)
                        await websocket.close(code=status.WS_1000_NORMAL_CLOSURE, reason="클라이언트 연결 종료")
                        return
                    else:
                        logger.warning(f"알 수 없는 메시지 타입: {client_id}, 메시지: {message}")
                        continue
                else:
                    logger.warning(f"예상치 못한 WebSocket 메시지: {client_id}, 타입: {message.get('type')}")
                    continue
                    
            except asyncio.TimeoutError:
                logger.debug(f"[TIMEOUT] WebSocket 메시지 수신 타임아웃 (10초): {client_id}")
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
            except WebSocketDisconnect:
                logger.info(f"클라이언트 연결 종료됨 (메인 루프): {client_id}")
                break
            except RuntimeError as e:
                if "Cannot call \"receive\" once a disconnect message has been received" in str(e):
                    logger.info(f"클라이언트가 이미 연결을 끊었음: {client_id}")
                    break
                else:
                    logger.error(f"RuntimeError 발생: {client_id}, 오류: {e}")
                    break
            except Exception as e:
                logger.error(f"예상치 못한 데이터 수신 오류: {client_id}, 오류: {e}")
                logger.error(f"오류 세부사항: {traceback.format_exc()}")
                break
    
    except WebSocketDisconnect:
        logger.info(f"오디오 WebSocket 연결 종료됨: {client_id}")
    except Exception as e:
        logger.error(f"오디오 WebSocket 핸들러 오류 ({client_id}): {e}\n{traceback.format_exc()}")
        try:
            await ws_engine.send_error(websocket, str(e), "websocket_handler_error")
        except:
            pass
    finally:
        logger.info(f"오디오 WebSocket 연결 정리 시작: {client_id}")
        await connection_engine.handle_disconnect(client_id)
        logger.info(f"오디오 WebSocket 연결 정리 완료: {client_id}")

async def handle_orchestrator_story_generation(
    websocket: WebSocket,
    client_id: str,
    chatbot_a,
    child_name: str,
    age: int,
    interests_list: list,
    ws_engine: WebSocketEngine,
    connection_engine: ConnectionEngine
) -> str:
    """
    WorkflowOrchestrator를 사용한 동화 생성 (REST API 연동)
    
    Returns:
        str: 생성된 story_id
    """
    logger.info(f"[ORCHESTRATOR] WorkflowOrchestrator를 통한 동화 생성 시작: {client_id}")
    
    try:
        # 1. WorkflowOrchestrator 가져오기
        from chatbot.app import orchestrator
        from chatbot.workflow.story_schema import ChildProfile
        
        if not orchestrator:
            raise RuntimeError("WorkflowOrchestrator가 초기화되지 않았습니다")
        
        # 2. 부기에서 대화 데이터 추출
        conversation_history = chatbot_a.conversation.get_conversation_history()
        conversation_data = {
            "messages": [
                {"role": msg.get("role", "user"), "content": msg.get("content", "")}
                for msg in conversation_history
            ],
            "child_name": child_name,
            "interests": interests_list,
            "total_turns": len(conversation_history)
        }
        
        # 3. ChildProfile 생성
        child_profile = ChildProfile(
            name=child_name,
            age=age,
            interests=interests_list,
            language_level="basic"
        )
        
        # 4. 동화 생성 시작 알림
        await ws_engine.send_json(websocket, {
            "type": "orchestrator_story_started",
            "message": "WorkflowOrchestrator가 동화를 생성하고 있어요...",
            "child_name": child_name,
            "timestamp": datetime.now().isoformat()
        })
        
    
        # 5. 실제 동화 생성 실행 (실시간)
        story_schema = await orchestrator.create_story(
            child_profile=child_profile,
            conversation_data=conversation_data,
            story_preferences=None
        )
        
        actual_story_id = story_schema.metadata.story_id
        logger.info(f"[ORCHESTRATOR] 동화 생성 완료: {actual_story_id}")
        
        # 완료 알림
        try:
            await ws_engine.send_json(websocket, {
                "type": "orchestrator_story_completed",
                "story_id": actual_story_id,
                "message": "🎉 동화가 완성되었어요! 이제 확인해보세요.",
                "files_ready": True,
                "completion_url": f"/api/v1/stories/{actual_story_id}/completion",
                "timestamp": datetime.now().isoformat()
            })
        except Exception as e:
            logger.warning(f"[ORCHESTRATOR] 완료 알림 전송 실패: {e}")
        
        return actual_story_id
        
    except Exception as e:
        logger.error(f"[ORCHESTRATOR] WorkflowOrchestrator 동화 생성 실패: {e}")
        raise

async def handle_automatic_story_generation(
    websocket: WebSocket,
    client_id: str,
    chatbot_a,
    child_name: str,
    age: int,
    interests_list: list,
    ws_engine: WebSocketEngine,
    connection_engine: ConnectionEngine
):
    """
    부기에서 꼬기로 자동 전환하여 동화 생성 처리
    
    Args:
        websocket: WebSocket 연결
        client_id: 클라이언트 ID
        chatbot_a: ChatBot A 인스턴스
        child_name: 아이 이름
        age: 아이 나이
        interests_list: 관심사 목록
        ws_engine: WebSocket 엔진
        connection_engine: 연결 엔진
    """
    logger.info(f"[AUTO_STORY_GEN] 자동 동화 생성 시작: {client_id}")
    
    try:
        # 1. 부기에서 이야기 개요 추출
        logger.info(f"[AUTO_STORY_GEN] 부기에서 이야기 개요 추출 중...")
        
        story_outline = chatbot_a.get_story_outline_for_chatbot_b()
        logger.info(f"[AUTO_STORY_GEN] 이야기 개요 추출 완료: {story_outline.get('title', 'Unknown')}")
        
        # 2. 꼬기(ChatBot B) 인스턴스 생성
        from chatbot.models.chat_bot_b import ChatBotB
        
        chatbot_b = ChatBotB()
        chatbot_b.set_target_age(age)
        chatbot_b.set_child_info(name=child_name, interests=interests_list)
        
        # 3. 클론된 음성 확인 및 설정
        voice_cloning_processor = VoiceCloningProcessor()
        cloned_voice_id = voice_cloning_processor.get_user_voice_id(child_name)
        if cloned_voice_id:
            chatbot_b.set_cloned_voice_info(
                child_voice_id=cloned_voice_id,
                main_character_name=child_name
            )
            logger.info(f"[AUTO_STORY_GEN] 클론된 음성 설정 완료 - {child_name}: {cloned_voice_id}")
            
            # 클론 음성 사용 알림
            await ws_engine.send_json(websocket, {
                "type": "voice_clone_applied",
                "message": f"{child_name}님의 복제된 목소리를 동화에 적용했어요!",
                "voice_id": cloned_voice_id,
                "timestamp": datetime.now().isoformat()
            })
        
        # 4. 꼬기에 이야기 개요 설정
        chatbot_b.set_story_outline(story_outline)
        logger.info(f"[AUTO_STORY_GEN] 꼬기에 이야기 개요 설정 완료")
        
        # 5. 진행 상황 콜백 함수 정의
        async def progress_callback(progress_data):
            await ws_engine.send_json(websocket, {
                "type": "story_progress",
                "progress": progress_data,
                "timestamp": datetime.now().isoformat()
            })
            logger.info(f"[AUTO_STORY_GEN] 진행 상황 업데이트: {progress_data.get('step', 'unknown')}")
        
        # 6. 동화 생성 시작 알림
        await ws_engine.send_json(websocket, {
            "type": "story_generation_started",
            "message": "동화 생성을 시작합니다...",
            "has_cloned_voice": cloned_voice_id is not None,
            "story_title": story_outline.get('title', '멋진 이야기'),
            "timestamp": datetime.now().isoformat()
        })
        
        # 7. 꼬기로 동화 생성 (Enhanced Mode)
        logger.info(f"[AUTO_STORY_GEN] 꼬기로 동화 생성 시작...")
        generation_start_time = time.time()
        
        result = await chatbot_b.generate_detailed_story(
            progress_callback=progress_callback,
            use_websocket_voice=True  # WebSocket 스트리밍 음성 사용
        )
        
        generation_time = time.time() - generation_start_time
        logger.info(f"[AUTO_STORY_GEN] 꼬기 동화 생성 완료: {generation_time:.2f}초")
        
        # 8. 생성 완료 알림 (프론트엔드로)
        completion_packet = {
            "type": "story_generated",
            "message": f"🎉 {child_name}님만의 특별한 동화가 완성되었어요!",
            "result": result,
            "cloned_voice_used": cloned_voice_id is not None,
            "generation_time": generation_time,
            "story_title": story_outline.get('title', '멋진 이야기'),
            "chapters_count": len(result.get('story_data', {}).get('chapters', [])),
            "timestamp": datetime.now().isoformat()
        }
        
        await ws_engine.send_json(websocket, completion_packet)
        logger.info(f"[AUTO_STORY_GEN] 동화 완성 알림 전송 완료: {client_id}")
        
        # 9. 최종 성공 메시지
        await ws_engine.send_json(websocket, {
            "type": "workflow_completed",
            "message": "부기와 꼬기가 함께 만든 동화가 완성되었어요! 이제 읽거나 들어보세요.",
            "success": True,
            "timestamp": datetime.now().isoformat()
        })
        
        logger.info(f"[AUTO_STORY_GEN] 전체 워크플로우 완료: {client_id}")
        
    except Exception as e:
        logger.error(f"[AUTO_STORY_GEN] 자동 동화 생성 실패: {e}")
        logger.error(f"[AUTO_STORY_GEN] 스택 트레이스: {traceback.format_exc()}")
        
        # 오류 발생 시 사용자에게 알림
        await ws_engine.send_json(websocket, {
            "type": "story_generation_failed",
            "message": "동화 생성 중 문제가 발생했어요. 다시 시도해주세요.",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        })
        
        raise

async def check_story_completion(story_engine, conversation_length: int, child_name: str, age: int) -> bool:
    """
    충분한 이야기 정보가 수집되었는지 확인
    
    Args:
        story_engine: 이야기 엔진
        conversation_length: 대화 길이
        child_name: 아이 이름
        age: 아이 나이
        
    Returns:
        bool: 이야기 생성 준비 완료 여부
    """
    try:
        # 최소 대화 턴 수 확인 (5턴 이상)
        if conversation_length < 10:  # user + assistant = 2턴이므로 최소 5회 대화
            return False
        
        # 이야기 요소별 수집 상태 확인
        story_elements = story_engine.get_story_elements()
        
        # 각 단계별 충분한 정보 수집 여부 체크
        character_ready = story_elements.get("character", {}).get("count", 0) >= 2
        setting_ready = story_elements.get("setting", {}).get("count", 0) >= 1
        problem_ready = story_elements.get("problem", {}).get("count", 0) >= 1
        
        # 기본 요소들이 모두 수집되었는지 확인
        basic_elements_ready = character_ready and setting_ready and problem_ready
        
        # 현재 단계가 resolution에 도달했는지 확인
        current_stage = story_engine.story_stage
        is_in_final_stage = current_stage == "resolution"
        
        # 품질 점수 확인 (평균 0.6 이상)
        quality_scores = getattr(story_engine, 'quality_scores', [])
        avg_quality = sum(quality_scores[-5:]) / len(quality_scores[-5:]) if quality_scores else 0.5
        quality_threshold_met = avg_quality >= 0.6
        
        # 대화 길이가 충분히 길어진 경우 (15턴 이상)
        is_long_conversation = conversation_length >= 30
        
        # 종료 조건들
        conditions = {
            "basic_elements": basic_elements_ready,
            "final_stage": is_in_final_stage,
            "quality_ok": quality_threshold_met,
            "long_conversation": is_long_conversation,
            "min_length": conversation_length >= 10
        }
        
        logger.info(f"[STORY_CHECK] {child_name}의 이야기 완성도 체크: {conditions}")
        
        # 완료 조건: (기본 요소 + 최종 단계) 또는 (기본 요소 + 긴 대화) 또는 (매우 긴 대화)
        is_ready = (
            (basic_elements_ready and is_in_final_stage) or
            (basic_elements_ready and is_long_conversation) or
            (conversation_length >= 40)  # 매우 긴 대화는 무조건 종료
        )
        
        if is_ready:
            logger.info(f"[STORY_CHECK] 이야기 수집 완료! 조건: {[k for k, v in conditions.items() if v]}")
        
        return is_ready
        
    except Exception as e:
        logger.error(f"[STORY_CHECK] 이야기 완성도 체크 중 오류: {e}")
        # 오류 시 긴 대화면 종료하도록 처리
        return conversation_length >= 40

async def handle_chat_a_response(chatbot_a: ChatBotA, user_text: str, audio_processor: AudioProcessor) -> tuple:
    """
    ChatBot A 응답 처리
    
    Args:
        chatbot_a: ChatBot A 인스턴스
        user_text: 사용자 입력 텍스트
        
    Returns:
        tuple: (ai_response, tts_result, conversation_length)
    """
    try:
        # ChatBot A 응답 생성
        ai_response = await asyncio.to_thread(chatbot_a.get_response, user_text)
        logger.info(f"[CHAT_A] 부기 응답 생성 완료: {ai_response[:50]}...")
        
        # TTS 처리 (음성 생성)
        tts_result = None
        try:
            # AudioProcessor의 synthesize_tts 메서드 사용
            audio_data, status, error_msg, error_code = await audio_processor.synthesize_tts(ai_response)
            if status != "error" and audio_data:
                tts_result = {"audio_data": audio_data}
                logger.info(f"[TTS] 음성 생성 완료: {len(audio_data)} bytes")
            else:
                logger.warning(f"[TTS] 음성 생성 실패: {error_msg}")
                tts_result = None
        except Exception as tts_error:
            logger.warning(f"[TTS] 음성 생성 실패: {tts_error}")
            tts_result = None
        
        # 대화 길이 확인
        conversation_length = len(chatbot_a.conversation.get_conversation_history())
        
        return ai_response, tts_result, conversation_length
        
    except Exception as e:
        logger.error(f"[CHAT_A] ChatBot A 응답 처리 중 오류: {e}")
        raise 