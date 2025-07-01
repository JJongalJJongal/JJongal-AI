"""
기본 생성기 클래스

모든 Generator의 공통 Interface 와 기능 정의

"""
from shared.utils.logging_utils import get_module_logger
import asyncio
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Callable
from enum import Enum
import time

# Logging 설정
logger = get_module_logger(__name__)

class GeneratorStatus(Enum):
    """Generator 상태 정의"""
    IDLE = "idle"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    
class BaseGenerator(ABC):
    """ 기본 생성기 클래스 """
    
    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0, timeout: float = 300.0):
        """
        Args:
            max_retries: 최대 재시도 횟수
            retry_delay: 재시도 간격 (초)
            timeout: 타임아웃 시간 (초)
        """
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout
        
        # 상태 관리
        self.status = GeneratorStatus.IDLE
        self.current_task_id = None
        self.last_error = None
        self.generation_count = 0
        
        # 성능 metric
        self.total_generation_time = 0.0
        self.average_generation_time = 0.0
        
    @abstractmethod
    async def generate(self, input_data: Dict[str, Any],
                       progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        
        """
        추상 method : 각 Generator 에서 구현
        
        Args:
            input_data: 생성에 필요한 입력 데이터
            progress_callback: 진행 상태 콜백 함수
            
        Returns:
            Dict: 생성 결과
        """
        pass
    
    async def generate_with_retry(self,
                                  input_data: Dict[str, Any],
                                  progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        재시도 로직이 포함된 생성 메서드
        """
        
        self.status = GeneratorStatus.PROCESSING
        start_time = time.time()
        
        for attempt in range(self.max_retries + 1): # 최대 재시도 횟수 + 1
            try:
                if progress_callback: # 진행 상태 호출
                    await progress_callback({
                        "generator": self.__class__.__name__, # 생성기 이름
                        "status": "processing", # 상태
                        "attempt": attempt + 1, # 시도 횟수
                        "max_attempts": self.max_retries + 1 # 최대 시도 횟수
                    })
                
                # 실제 생성 작업 수행
                result = await asyncio.wait_for(
                    self.generate(input_data, progress_callback),
                    timeout = self.timeout
                )
                

                # 성공 시 metric update
                generation_time = time.time() - start_time
                self._update_metrics(generation_time)
                
                self.status = GeneratorStatus.COMPLETED
                self.last_error = None
                
                if progress_callback: # 진행 상태 호출
                    await progress_callback({
                        "generator": self.__class__.__name__, # 생성기 이름
                        "status": "completed", # 상태
                        "generation_time": generation_time, # 생성 시간
                    })
                
                return result
            # 타임아웃 시 재시도
            except asyncio.TimeoutError:
                error_msg = f"생성 작업 타임아웃 (시도 {attempt + 1}/{self.max_retries + 1})" # 에러 메시지
                logger.warning(error_msg) # 로깅
                self.last_error = error_msg # 마지막 에러 저장
                
            except Exception as e:
                error_msg = f"생성 작업 실패 (시도 {attempt + 1}/{self.max_retries + 1}) : {str(e)}" # 에러 메시지
                logger.warning(error_msg) # 로깅
                self.last_error = str(e) # 마지막 에러 저장
                
            # 마지막 시도가 아니면 재시도
            if attempt < self.max_retries:
                self.status = GeneratorStatus.RETRYING # 재시도 상태 설정
                if progress_callback: # 진행 상태 호출
                    await progress_callback({
                        "generator": self.__class__.__name__, # 생성기 이름
                        "status": "retrying", # 상태
                        "error": self.last_error, # 마지막 에러 메시지
                        "attempt": attempt + 1, # 시도 횟수
                    })
                    
                await asyncio.sleep(self.retry_delay) # 재시도 간격 대기
                    
        # 모든 재시도 실패
        self.status = GeneratorStatus.FAILED
        error_msg = f"모든 재시도 실패. 마지막 오류 : {self.last_error}" # 에러 메시지
        logger.error(error_msg) # 로깅
        
        if progress_callback: # 진행 상태 호출
            await progress_callback({
                "generator": self.__class__.__name__, # 생성기 이름
                "status": "failed", # 상태
                "error": error_msg, # 마지막 에러 메시지
            })
        
        raise Exception(error_msg) # 예외 발생
    
    def _update_metrics(self, generation_time: float):
        """ 성능 metric 업데이트 """
        self.generation_count += 1 # 생성 횟수 증가
        self.total_generation_time += generation_time # 총 생성 시간 증가
        self.average_generation_time = self.total_generation_time / self.generation_count # 평균 생성 시간 계산
        
    def get_status(self) -> Dict[str, Any]:
        """현재 상태 반환"""
        return {
            "status": self.status.value, # 상태
            "generation_count": self.generation_count, # 생성 횟수
            "total_generation_time": self.total_generation_time, # 총 생성 시간
            "average_generation_time": self.average_generation_time, # 평균 생성 시간
            "current_task_id": self.current_task_id, # 현재 작업 ID
            "last_error": self.last_error, # 마지막 에러 메시지
        }
    
    def reset_metrics(self):
        """metric 초기화"""
        self.generation_count = 0 # 생성 횟수 초기화
        self.total_generation_time = 0.0 # 총 생성 시간 초기화
        self.average_generation_time = 0.0 # 평균 생성 시간 초기화
        self.last_error = None # 마지막 에러 메시지 초기화
        self.status = GeneratorStatus.IDLE # 상태 초기화
        
    async def health_check(self) -> bool:
        """Generator 상태 확인"""
        try:
            # 각 Generator 에서 override 가능
            return self.status != GeneratorStatus.FAILED
        except Exception as e:
            logger.error(f"Health Check 실패 : {e}")
            return False
    