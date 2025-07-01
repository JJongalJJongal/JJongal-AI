"""
부기 (ChatBot A) 기본 프로세서 추상 클래스
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from shared.utils.logging_utils import get_module_logger

logger = get_module_logger(__name__)

class BaseProcessor(ABC):
    """
    모든 프로세서의 기본 추상 클래스
    
    이 클래스는 부기 챗봇의 모든 처리 컴포넌트가 상속받아야 하는
    공통 인터페이스와 기본 기능을 제공합니다.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        기본 프로세서 초기화
        
        Args:
            config (Optional[Dict[str, Any]]): 프로세서 설정
        """
        self.config = config or {}
        self.is_initialized = False
        
    @abstractmethod
    def initialize(self) -> bool:
        """
        프로세서 초기화
        
        Returns:
            bool: 초기화 성공 여부
        """
        pass
    
    @abstractmethod
    def process(self, input_data: Any) -> Any:
        """
        입력 데이터 처리
        
        Args:
            input_data (Any): 처리할 입력 데이터
            
        Returns:
            Any: 처리된 결과 데이터
        """
        pass
    
    def validate_input(self, input_data: Any) -> bool:
        """
        입력 데이터 유효성 검증
        
        Args:
            input_data (Any): 검증할 입력 데이터
            
        Returns:
            bool: 유효성 검증 결과
        """
        return input_data is not None
    
    def get_status(self) -> Dict[str, Any]:
        """
        프로세서 상태 정보 반환
        
        Returns:
            Dict[str, Any]: 프로세서 상태 정보
        """
        return {
            "is_initialized": self.is_initialized,
            "config": self.config,
            "processor_type": self.__class__.__name__
        }
    
    def cleanup(self) -> None:
        """
        프로세서 정리 작업
        """
        logger.info(f"{self.__class__.__name__} 프로세서 정리 완료")
        self.is_initialized = False 