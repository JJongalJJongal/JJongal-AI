"""
파일 처리 관련 유틸리티 모듈
"""
import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

logger = logging.getLogger(__name__)


def ensure_directory(path: Union[str, Path]) -> Path:
    """
    디렉토리 존재 확인 및 생성
    
    Args:
        path (Union[str, Path]): 디렉토리 경로
        
    Returns:
        Path: 생성된 디렉토리 경로
    """
    dir_path = Path(path) if isinstance(path, str) else path
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def save_json(data: Dict[str, Any], file_path: Union[str, Path], indent: int = 4) -> bool:
    """
    JSON 데이터 저장
    
    Args:
        data (Dict[str, Any]): 저장할 데이터
        file_path (Union[str, Path]): 저장할 파일 경로
        indent (int): JSON 들여쓰기 수준
        
    Returns:
        bool: 성공 여부
    """
    try:
        # 디렉토리 확인
        dir_path = Path(file_path).parent if isinstance(file_path, str) else file_path.parent
        ensure_directory(dir_path)
        
        # 파일 저장
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=indent)
            
        logger.info(f"JSON 파일 저장 완료: {file_path}")
        return True
    except Exception as e:
        logger.error(f"JSON 파일 저장 실패: {e}")
        return False


def load_json(file_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """
    JSON 데이터 로드
    
    Args:
        file_path (Union[str, Path]): 로드할 파일 경로
        
    Returns:
        Optional[Dict[str, Any]]: 로드된 데이터 (실패 시 None)
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        logger.info(f"JSON 파일 로드 완료: {file_path}")
        return data
    except FileNotFoundError:
        logger.error(f"JSON 파일이 존재하지 않습니다: {file_path}")
        return None
    except json.JSONDecodeError:
        logger.error(f"JSON 파일 형식이 올바르지 않습니다: {file_path}")
        return None
    except Exception as e:
        logger.error(f"JSON 파일 로드 실패: {e}")
        return None


def list_files(directory: Union[str, Path], pattern: str = "*") -> List[Path]:
    """
    디렉토리 내 파일 목록 반환
    
    Args:
        directory (Union[str, Path]): 디렉토리 경로
        pattern (str): 파일명 패턴 (glob 형식)
        
    Returns:
        List[Path]: 파일 경로 목록
    """
    try:
        dir_path = Path(directory) if isinstance(directory, str) else directory
        return list(dir_path.glob(pattern))
    except Exception as e:
        logger.error(f"디렉토리 내 파일 목록 조회 실패: {e}")
        return []


def get_project_root() -> Path:
    """
    프로젝트 루트 디렉토리 경로 반환
    
    Returns:
        Path: 프로젝트 루트 디렉토리 경로
    """
    # 현재 파일 기준 상위 디렉토리 탐색
    current_dir = Path(__file__).resolve().parent  # /shared/utils
    return current_dir.parent.parent  # /CCB_AI 