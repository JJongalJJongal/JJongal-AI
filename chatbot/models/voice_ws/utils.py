"""
유틸리티 함수 모듈

이 모듈은 다양한 유틸리티 함수를 제공합니다.
"""
import os
import time
import logging
import asyncio
from typing import Callable, Any

# 로깅 설정
def setup_logging():
    """로깅 설정 초기화"""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    log_level = os.getenv("LOG_LEVEL", "INFO")
    log_file = os.getenv("LOG_FILE", "server.log")
    
    logging.basicConfig(
        level=getattr(logging, log_level),
        format=log_format,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logging.info("로깅 설정 완료")

# 실패한 요청에 대한 재시도 매커니즘
async def retry_operation(operation, max_retries=3, retry_delay=1):
    """
    작업을 재시도하는 함수
    
    Args:
        operation: 실행할 비동기 작업
        max_retries (int): 최대 재시도 횟수
        retry_delay (float): 재시도 간 대기 시간(초)
    
    Returns:
        작업 결과 또는 (None, error_message, error_code) 튜플
    """
    attempts = 0
    last_error = None
    
    while attempts < max_retries:
        try:
            result = await operation()
            logging.debug(f"작업 성공 (시도 {attempts+1}/{max_retries})")
            return result
        except Exception as e:
            attempts += 1
            last_error = e
            logging.warning(f"작업 실패 (시도 {attempts}/{max_retries}): {e}")
            
            if attempts < max_retries:
                # 지수 백오프 적용 (더 안정적인 재시도)
                adjusted_delay = retry_delay * (1.5 ** (attempts - 1))
                logging.info(f"{adjusted_delay:.2f}초 후 재시도...")
                await asyncio.sleep(adjusted_delay)
                
    logging.error(f"최대 재시도 횟수 초과: {last_error}")
    error_detail = str(last_error) if last_error else "알 수 없는 오류"
    return None, f"작업 실패: {error_detail}", "retry_failed"

# 시간 측정 데코레이터
def timing_decorator(func):
    """
    함수 실행 시간을 측정하는 데코레이터
    
    Args:
        func: 측정할 함수
    
    Returns:
        실행 시간을 로그로 출력하는 래퍼 함수
    """
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        end_time = time.time()
        logging.debug(f"{func.__name__} 실행 시간: {end_time - start_time:.4f}초")
        return result
    return wrapper

# 대화 내용 저장
async def save_conversation(chatbot, child_name, client_id):
    """
    대화 내용을 파일로 저장
    
    Args:
        chatbot: 챗봇 인스턴스
        child_name (str): 아이 이름
        client_id (str): 클라이언트 ID
    
    Returns:
        tuple: (저장 경로, 오류 메시지)
    """
    try:
        # 대화 저장 디렉토리 생성
        save_dir = os.path.join("output", "conversations")
        os.makedirs(save_dir, exist_ok=True)
        
        # 대화 내용 저장
        save_path = os.path.join(save_dir, f"{child_name}_{int(time.time())}.json")
        await asyncio.to_thread(chatbot.save_conversation, save_path)
        logging.info(f"대화 내용 저장 완료: {child_name} ({client_id})")
        
        return save_path, None
    except Exception as e:
        logging.error(f"대화 내용 저장 중 오류: {e}")
        return None, f"대화 내용 저장 실패: {e}"

# 임시 파일 정리
def cleanup_temp_files(temp_files):
    """
    임시 파일 목록을 삭제
    
    Args:
        temp_files (list): 임시 파일 경로 목록
    """
    for temp_file in temp_files:
        try:
            if os.path.exists(temp_file):
                os.remove(temp_file)
                logging.debug(f"임시 파일 삭제 완료: {temp_file}")
        except Exception as e:
            logging.warning(f"임시 파일 삭제 실패: {e}")
            
# 배치 처리 도우미
async def process_in_batches(items, batch_size, process_func):
    """
    항목들을 배치로 처리
    
    Args:
        items (list): 처리할 항목 목록
        batch_size (int): 배치 크기
        process_func (callable): 각 배치를 처리할 함수
        
    Returns:
        list: 처리 결과 목록
    """
    results = []
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        batch_results = await process_func(batch)
        results.extend(batch_results)
    return results 