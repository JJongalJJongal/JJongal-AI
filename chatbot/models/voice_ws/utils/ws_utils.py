"""
WebSocket 관련 유틸리티 함수 모음
"""
import os
import json
import asyncio
import logging
import traceback
from datetime import datetime

# --- 로깅 설정 --- (기존 setup_logging)
DEFAULT_LOG_LEVEL = logging.INFO
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

def setup_logging(log_level: int = DEFAULT_LOG_LEVEL, log_format: str = LOG_FORMAT) -> logging.Logger:
    """
    기본 로깅 설정을 수행하고 루트 로거를 반환합니다.
    shared.utils.logging_utils.get_module_logger를 사용하는 것이 권장되지만,
    모듈의 최상위 레벨에서 단일 로거가 필요한 경우 사용할 수 있습니다.
    """
    logging.basicConfig(level=log_level, format=log_format)
    logger = logging.getLogger("voice_ws_server") # 또는 적절한 이름
    logger.info(f"로깅 설정 완료 (레벨: {logging.getLevelName(log_level)})")
    return logger

# --- 재시도 로직 --- (기존 retry_operation)
async def retry_operation(operation, max_retries=3, delay=1, allowed_exceptions=(Exception,)):
    """
    주어진 비동기 작업을 최대 횟수만큼 재시도합니다.
    
    Args:
        operation: 실행할 비동기 함수 (인자 없이 호출 가능해야 함).
        max_retries: 최대 재시도 횟수.
        delay: 재시도 간 대기 시간 (초).
        allowed_exceptions: 재시도를 트리거할 예외 유형의 튜플.
    
    Returns:
        성공 시 작업 결과, 모든 재시도 실패 시 (None, 에러 메시지, 에러 코드).
    """
    for attempt in range(max_retries):
        try:
            return await operation()
        except allowed_exceptions as e:
            error_code = e.__class__.__name__ # 예외 클래스 이름을 코드로 사용
            if attempt == max_retries - 1:
                logging.error(f"작업 실패 (최대 재시도 도달): {e} (코드: {error_code})")
                return None, str(e), error_code
            logging.warning(f"작업 중 오류 발생 (시도 {attempt + 1}/{max_retries}): {e}. {delay}초 후 재시도합니다.")
            await asyncio.sleep(delay)
    return None, "알 수 없는 재시도 오류", "unknown_retry_error" # 이론상 도달하지 않음

# --- 임시 파일 정리 --- (기존 cleanup_temp_files)
def cleanup_temp_files(file_paths: list):
    """임시 파일 목록을 정리합니다."""
    for file_path in file_paths:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logging.info(f"임시 파일 삭제: {file_path}")
        except Exception as e:
            logging.error(f"임시 파일 삭제 실패 ({file_path}): {e}")

# --- 대화 내용 저장 --- (기존 save_conversation)
async def save_conversation(chatbot, child_name: str, client_id: str):
    """챗봇의 대화 내용을 파일로 저장합니다."""
    try:
        history = chatbot.get_chat_history()
        if not history:
            logging.info(f"저장할 대화 내용 없음: {client_id}")
            return

        now = datetime.now()
        output_dir = os.path.join(os.getcwd(), "output", "conversations", now.strftime("%Y-%m-%d"))
        os.makedirs(output_dir, exist_ok=True)

        filename = f"{child_name}_{now.strftime('%Y%m%d_%H%M%S')}_{client_id}.json"
        filepath = os.path.join(output_dir, filename)

        conversation_data = {
            "client_id": client_id,
            "child_name": child_name,
            "timestamp": now.isoformat(),
            "history": history
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(conversation_data, f, ensure_ascii=False, indent=4)
        
        logging.info(f"대화 내용 저장 완료: {filepath}")

    except Exception as e:
        error_detail = traceback.format_exc()
        logging.error(f"대화 내용 저장 실패 ({client_id}): {e}\n{error_detail}") 