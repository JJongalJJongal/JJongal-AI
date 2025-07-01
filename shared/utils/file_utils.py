"""
파일 처리 관련 유틸리티 모듈
"""
import os
import shutil
import base64
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

from shared.utils.logging_utils import get_module_logger
logger = get_module_logger(__name__)


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

def save_image_from_base64(base64_str, file_path):
    """Base64 인코딩된 Image 를 파일로 저장"""
    try:
        # Base64 header 제거
        if ',' in base64_str:
            base64_str = base64_str.split(',')[1]
            
        # directory 생성
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Base64 decoding * file save
        with open(file_path, "wb") as f:
            f.write(base64.b64decode(base64_str))
            
        return file_path
    
    except Exception as e:
        print(f"이미지 저장 중 오류 발생 : {e}")
        return None


def save_audio(audio_data, file_path, mode="wb"):
    """
    오디오 데이터를 파일로 저장

    Args:
        audio_data (bytes): 저장할 오디오 데이터 (binary)
        file_path (str): 저장할 파일 경로
        mode (str): 파일 열기 모드 (wb)
    
    Returns:
        str: 저장된 파일 경로
    """
    
    try:
        # directory 생성
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # 오디오 데이터 저장
        with open(file_path, mode) as f:
            f.write(audio_data)
            
        return file_path
    
    except Exception as e:
        print(f"오디오 저장 중 오류 발생 : {e}")
        return None

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

def ensure_directory(directory_path: Union[str, Path]) -> str:
    """
    디렉토리가 존재하는지 확인하고, 없으면 생성

    Args:
        directory_path (Union[str, Path]): 확인할 디렉토리 경로

    Returns:
        str: 생성된 디렉토리 경로
    """
    if not isinstance(directory_path, Path):
        directory_path = Path(directory_path)
        
    directory_path.mkdir(parents=True, exist_ok=True)
    return str(directory_path)
    
def copy_file(source_path: Union[str, Path], target_path: Union[str, Path]) -> Optional[str]:
    """
    파일 복사

    Args:
        source_path: 원본 파일 경로
        target_path: 대상 파일 경로

    Returns:
        복사된 파일 경로 또는 오류 시 None
    """
    try:
        # 대상 디렉토리 생성
        ensure_directory(os.path.dirname(target_path))
        
        # 파일 복사
        shutil.copy2(source_path, target_path)
        
        return str(target_path)
    
    except Exception as e:
        logger.error(f"파일 복사 중 오류 발생: {e}")
        return None

def get_project_root() -> Path:
    """
    프로젝트 루트 디렉토리 경로 반환
    
    Returns:
        Path: 프로젝트 루트 디렉토리 경로
    """
    # 현재 파일 기준 상위 디렉토리 탐색
    current_dir = Path(__file__).resolve().parent  # /shared/utils
    return current_dir.parent.parent  # /CCB_AI 

def file_exists(file_path: Union[str, Path]) -> bool:
    """
    파일 존재 여부 확인
    
    Args:
        file_path (Union[str, Path]): 확인할 파일 경로
        
    Returns:
        bool: 파일 존재 여부
    """
    
    return os.path.isfile(file_path)

def get_file_size(file_path: Union[str, Path]) -> int:
    """
    파일 크기를 바이트 단위로 반환함.
    
    Args:
        file_path (Union[str, Path]): 파일 경로
        
    Returns:
        int: 파일 크기 (바이트)
    """
    if not file_exists(file_path):
        return 0
    return os.path.getsize(file_path)

def cleanup_temp_files(file_paths: List[Union[str, Path]], logger_instance: Optional[logging.Logger] = None):
    """임시 파일 목록을 정리합니다."""
    current_logger = logger_instance if logger_instance else logger # 외부 로거 또는 현재 모듈 로거 사용
    for file_path_item in file_paths:
        try:
            file_to_delete = Path(file_path_item) # Path 객체로 변환
            if file_to_delete.exists() and file_to_delete.is_file():
                file_to_delete.unlink() # Path.unlink()를 사용하여 파일 삭제
                current_logger.info(f"임시 파일 삭제: {file_to_delete}")
            elif not file_to_delete.exists():
                current_logger.warning(f"삭제할 임시 파일이 존재하지 않음: {file_to_delete}")
            elif not file_to_delete.is_file():
                current_logger.warning(f"삭제 대상이 파일이 아님 (디렉토리 등): {file_to_delete}")
        except Exception as e:
            current_logger.error(f"임시 파일 삭제 실패 ({file_path_item}): {e}")