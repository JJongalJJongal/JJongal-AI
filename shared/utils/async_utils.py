"""
비동기 관련 유틸리티 함수 모음
"""
import asyncio
from typing import Callable, Awaitable, Any, Optional, Tuple

from shared.utils.logging_utils import get_module_logger 
logger = get_module_logger(__name__) 


async def retry_operation(
    operation: Callable[[], Awaitable[Any]], 
    max_retries: int = 3, 
    delay: int = 1, 
    allowed_exceptions: Tuple[type[Exception], ...] = (Exception,)
) -> Tuple[Optional[Any], Optional[str], Optional[str]]:
    """
    주어진 비동기 작업을 최대 횟수만큼 재시도합니다.
    
    Args:
        operation: 실행할 비동기 함수 (인자 없이 호출 가능해야 함).
        max_retries: 최대 재시도 횟수.
        delay: 재시도 간 대기 시간 (초).
        allowed_exceptions: 재시도를 트리거할 예외 유형의 튜플.
    
    Returns:
        Array[Optional[Any], Optional[str], Optional[str]]: 
            - 성공 시 (결과, None, None)
            - 모든 재시도 실패 시 (None, 에러 메시지, 에러 코드).
    """
    for attempt in range(max_retries):
        try:
            result = await operation()
            return result, None, None  # 성공 시 결과와 함께 None 반환
        except allowed_exceptions as e:
            error_code = e.__class__.__name__ # 예외 클래스 이름을 코드로 사용
            if attempt == max_retries - 1:
                logger.error(f"작업 실패 (최대 재시도 도달): {e} (코드: {error_code})")
                return None, str(e), error_code
            logger.warning(f"작업 중 오류 발생 (시도 {attempt + 1}/{max_retries}): {e}. {delay}초 후 재시도합니다.")
            await asyncio.sleep(delay)
    # 루프가 정상적으로 완료된 경우 (이론적으로는 max_retries 시도에서 반환되어야 함)
    logger.error("retry_operation: 최대 재시도 후에도 결과나 예외 없이 루프 종료됨. 이는 예상치 못한 상황입니다.")
    return None, "알 수 없는 재시도 오류: 최대 재시도 후 루프 종료", "unknown_retry_loop_ended" 