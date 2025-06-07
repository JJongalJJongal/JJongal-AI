"""
파일 권한 관리 유틸리티 모듈

Docker 컨테이너 환경에서 파일과 디렉토리의 권한을 적절히 설정하여
프론트엔드에서 접근 가능하게 만드는 기능을 제공합니다.
"""
import os
import stat
import logging
from pathlib import Path
from typing import Union, Optional

logger = logging.getLogger(__name__)


def set_file_permissions(file_path: Union[str, Path], 
                        mode: Optional[int] = None,
                        make_readable: bool = True) -> bool:
    """
    파일의 권한을 설정합니다.
    
    Args:
        file_path: 파일 경로
        mode: 권한 모드 (기본값: 644)
        make_readable: 읽기 권한 보장 여부
        
    Returns:
        bool: 권한 설정 성공 여부
    """
    try:
        file_path = Path(file_path)
        
        if not file_path.exists():
            logger.warning(f"파일이 존재하지 않음: {file_path}")
            return False
            
        # 기본 권한 모드 설정
        if mode is None:
            if file_path.is_dir():
                mode = 0o755  # 디렉토리: rwxr-xr-x
            else:
                mode = 0o644  # 파일: rw-r--r--
                
        # 읽기 권한 보장
        if make_readable:
            if file_path.is_dir():
                mode |= stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH  # 읽기 권한
                mode |= stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH  # 실행 권한 (디렉토리 접근용)
            else:
                mode |= stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH  # 읽기 권한
                
        os.chmod(file_path, mode)
        logger.debug(f"파일 권한 설정 완료: {file_path} -> {oct(mode)}")
        return True
        
    except Exception as e:
        logger.error(f"파일 권한 설정 실패: {file_path} - {e}")
        return False


def set_directory_permissions(dir_path: Union[str, Path], 
                             mode: Optional[int] = None,
                             recursive: bool = True,
                             make_readable: bool = True) -> bool:
    """
    디렉토리와 하위 파일들의 권한을 설정합니다.
    
    Args:
        dir_path: 디렉토리 경로
        mode: 권한 모드 (기본값: 디렉토리 755, 파일 644)
        recursive: 재귀적으로 하위 파일 처리 여부
        make_readable: 읽기 권한 보장 여부
        
    Returns:
        bool: 권한 설정 성공 여부
    """
    try:
        dir_path = Path(dir_path)
        
        if not dir_path.exists():
            logger.warning(f"디렉토리가 존재하지 않음: {dir_path}")
            return False
            
        if not dir_path.is_dir():
            logger.warning(f"디렉토리가 아님: {dir_path}")
            return False
            
        success = True
        
        # 디렉토리 자체 권한 설정
        dir_mode = mode if mode is not None else 0o755
        if not set_file_permissions(dir_path, dir_mode, make_readable):
            success = False
            
        # 재귀적으로 하위 항목 처리
        if recursive:
            for item in dir_path.rglob("*"):
                try:
                    if item.is_dir():
                        item_mode = mode if mode is not None else 0o755
                    else:
                        item_mode = mode if mode is not None else 0o644
                        
                    if not set_file_permissions(item, item_mode, make_readable):
                        success = False
                        
                except Exception as e:
                    logger.error(f"하위 항목 권한 설정 실패: {item} - {e}")
                    success = False
                    
        logger.info(f"디렉토리 권한 설정 완료: {dir_path} (재귀: {recursive})")
        return success
        
    except Exception as e:
        logger.error(f"디렉토리 권한 설정 실패: {dir_path} - {e}")
        return False


def ensure_readable_output() -> bool:
    """
    output 폴더와 하위 파일들이 읽기 가능하도록 권한을 설정합니다.
    
    Returns:
        bool: 권한 설정 성공 여부
    """
    try:
        output_paths = [
            "/app/output",
            "./output"  # 로컬 환경 대비
        ]
        
        success = True
        
        for output_path in output_paths:
            path = Path(output_path)
            if path.exists():
                logger.info(f"출력 폴더 권한 설정 시작: {path}")
                if not set_directory_permissions(path, recursive=True, make_readable=True):
                    success = False
                else:
                    logger.info(f"출력 폴더 권한 설정 완료: {path}")
                break
        else:
            logger.warning("출력 폴더를 찾을 수 없음")
            success = False
            
        return success
        
    except Exception as e:
        logger.error(f"출력 폴더 권한 설정 실패: {e}")
        return False


def create_file_with_permissions(file_path: Union[str, Path], 
                                content: Union[str, bytes] = "",
                                mode: Optional[int] = None,
                                encoding: str = "utf-8") -> bool:
    """
    파일을 생성하고 적절한 권한을 설정합니다.
    
    Args:
        file_path: 파일 경로
        content: 파일 내용
        mode: 권한 모드 (기본값: 644)
        encoding: 텍스트 인코딩
        
    Returns:
        bool: 파일 생성 및 권한 설정 성공 여부
    """
    try:
        file_path = Path(file_path)
        
        # 디렉토리 생성
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 부모 디렉토리 권한 설정
        set_directory_permissions(file_path.parent, recursive=False)
        
        # 파일 생성
        if isinstance(content, str):
            file_path.write_text(content, encoding=encoding)
        else:
            file_path.write_bytes(content)
            
        # 파일 권한 설정
        file_mode = mode if mode is not None else 0o644
        set_file_permissions(file_path, file_mode, make_readable=True)
        
        logger.debug(f"파일 생성 및 권한 설정 완료: {file_path}")
        return True
        
    except Exception as e:
        logger.error(f"파일 생성 및 권한 설정 실패: {file_path} - {e}")
        return False


def check_file_accessibility(file_path: Union[str, Path]) -> dict:
    """
    파일의 접근 가능성을 확인합니다.
    
    Args:
        file_path: 파일 경로
        
    Returns:
        dict: 접근성 정보
    """
    try:
        file_path = Path(file_path)
        
        result = {
            "exists": file_path.exists(),
            "is_file": file_path.is_file(),
            "is_dir": file_path.is_dir(),
            "readable": False,
            "writable": False,
            "executable": False,
            "permissions": None,
            "size": None
        }
        
        if file_path.exists():
            stat_info = file_path.stat()
            result["permissions"] = oct(stat_info.st_mode)[-3:]
            result["size"] = stat_info.st_size
            
            result["readable"] = os.access(file_path, os.R_OK)
            result["writable"] = os.access(file_path, os.W_OK)
            result["executable"] = os.access(file_path, os.X_OK)
            
        return result
        
    except Exception as e:
        logger.error(f"파일 접근성 확인 실패: {file_path} - {e}")
        return {"error": str(e)} 