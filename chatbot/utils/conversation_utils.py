"""
대화 관련 유틸리티 함수 모음
"""
import os
import json
import traceback
from datetime import datetime
from typing import Any

from shared.utils.logging_utils import get_module_logger
logger = get_module_logger(__name__)

async def save_conversation(chatbot: Any, child_name: str, client_id: str):
    """챗봇의 대화 내용을 파일로 저장합니다."""
    try:
        # 상세한 디버그 로깅
        logger.info(f"[SAVE_CONV] save_conversation 호출됨 - client_id: {client_id}")
        logger.info(f"[SAVE_CONV] chatbot 타입: {type(chatbot)}")
        logger.info(f"[SAVE_CONV] chatbot None 여부: {chatbot is None}")
        logger.info(f"[SAVE_CONV] hasattr get_conversation_history: {hasattr(chatbot, 'get_conversation_history')}")
        
        if chatbot is None:
            logger.error(f"[SAVE_CONV] chatbot이 None입니다 ({client_id=})")
            return
            
        if not hasattr(chatbot, 'get_conversation_history'):
            logger.error(f"[SAVE_CONV] chatbot 객체에 get_conversation_history 메서드가 없습니다 ({client_id=}). 타입: {type(chatbot)}")
            return
            
        logger.info(f"[SAVE_CONV] get_conversation_history 메서드 호출 시도")
        history = chatbot.get_conversation_history() # 챗봇 객체의 대화 내용 가져오기
        logger.info(f"[SAVE_CONV] 대화 기록 가져오기 성공, 메시지 개수: {len(history) if history else 0}")
        if not history:
            logger.info(f"저장할 대화 내용 없음 ({client_id=})")
            return

        now = datetime.now() # 현재 시간 가져오기
        
        # 안전한 디렉토리 생성
        base_output_dir = os.getenv("MULTIMEDIA_OUTPUT_DIR", "output")
        date_folder = now.strftime("%Y-%m-%d")
        output_dir = os.path.join(base_output_dir, "conversations", date_folder) # 저장 경로 설정
        
        # 디렉토리 안전하게 생성
        try:
        os.makedirs(output_dir, exist_ok=True) # 저장 경로 생성
            logger.debug(f"[SAVE_CONV] 대화 저장 디렉토리 확인/생성: {output_dir}")
        except PermissionError as e:
            logger.error(f"[SAVE_CONV] 디렉토리 생성 권한 오류: {output_dir} - {e}")
            return
        except OSError as e:
            logger.error(f"[SAVE_CONV] 디렉토리 생성 실패: {output_dir} - {e}")
            return
        except Exception as e:
            logger.error(f"[SAVE_CONV] 예상치 못한 디렉토리 생성 오류: {output_dir} - {e}")
            return

        filename = f"{child_name}_{now.strftime('%Y%m%d_%H%M%S')}_{client_id}.json" # 파일 이름 설정
        filepath = os.path.join(output_dir, filename)

        conversation_data = { # 대화 내용 저장
            "client_id": client_id, # 클라이언트 ID
            "child_name": child_name, # 아이 이름
            "timestamp": now.isoformat(), # 현재 시간
            "history": history # 대화 내용
        }

        with open(filepath, "w", encoding="utf-8") as f: # 파일 저장
            json.dump(conversation_data, f, ensure_ascii=False, indent=4) # 대화 내용 저장
        
        logger.info(f"대화 내용 저장 완료: {filepath}") # 로깅

    except AttributeError as ae: # get_conversation_history가 없을 경우 대비
        logger.error(f"[SAVE_CONV] AttributeError 발생: {ae}")
        logger.error(f"[SAVE_CONV] 챗봇 객체에 get_conversation_history 메서드가 없습니다 ({client_id=}). 타입: {type(chatbot)}")
        logger.error(f"[SAVE_CONV] 사용 가능한 메서드: {[m for m in dir(chatbot) if not m.startswith('_')]}")
    except Exception as e: # 예외 처리
        error_detail = traceback.format_exc() # 예외 상세 정보 가져오기
        logger.error(f"대화 내용 저장 실패 ({client_id=}): {e}\n{error_detail}") # 로깅