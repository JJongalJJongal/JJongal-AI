"""
로깅 관련 유틸리티 모듈
"""
import os
import logging
from typing import Optional, Union
from pathlib import Path

from .file_utils import ensure_directory


def setup_logger(
    name: str, 
    log_level: Union[str, int] = "INFO", 
    log_file: Optional[str] = None,
    console: bool = True
) -> logging.Logger:
    """
    로거 설정
    
    Args:
        name (str): 로거 이름
        log_level (Union[str, int]): 로깅 레벨 (문자열 또는 정수)
        log_file (Optional[str]): 로그 파일 경로
        console (bool): 콘솔 출력 여부
        
    Returns:
        logging.Logger: 설정된 로거 객체
    """
    # 로깅 레벨 설정
    if isinstance(log_level, str):
        level = getattr(logging, log_level.upper())
    else:
        level = log_level
        
    # 로거 생성
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 기존 핸들러 제거
    if logger.handlers:
        logger.handlers.clear()
    
    # 로그 포맷 설정
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # 파일 핸들러 설정
    if log_file:
        # 로그 디렉토리 확인
        log_dir = Path(log_file).parent
        ensure_directory(log_dir)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # 콘솔 핸들러 설정
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger


def setup_root_logger(
    log_level: Union[str, int] = "INFO", 
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    루트 로거 설정
    
    Args:
        log_level (Union[str, int]): 로깅 레벨 (문자열 또는 정수)
        log_file (Optional[str]): 로그 파일 경로
        
    Returns:
        logging.Logger: 설정된 루트 로거 객체
    """
    return setup_logger("", log_level, log_file, True)


def get_module_logger(name: str) -> logging.Logger:
    """
    모듈별 로거 생성 (이미 설정된 루트 로거 상속)
    
    Args:
        name (str): 모듈 이름
        
    Returns:
        logging.Logger: 모듈 로거 객체
    """
    return logging.getLogger(name) 