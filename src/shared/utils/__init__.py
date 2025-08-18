"""
CCB_AI 프로젝트 공통 유틸리티 패키지
"""

# Korean 유틸리티
from .korean import has_final_consonant, get_josa, format_with_josa

# 파일 유틸리티
from .file_utils import (
    ensure_directory, save_json, load_json, 
    list_files, get_project_root
)

# 로깅 유틸리티
from .logging import setup_logger, setup_root_logger, get_module_logger

# OpenAI 유틸리티
from .openai import initialize_client, track_token_usage, generate_chat_completion

# 오디오 유틸리티
from .audio import initialize_elevenlabs, transcribe_audio, generate_speech

# WebSocket 유틸리티
from .websocket import validate_token, ConnectionManager

# S3 유틸리티
from ...data.storage.s3_manager import S3Manager

__all__ = [
    # 한국어 유틸리티
    "has_final_consonant", "get_josa", "format_with_josa",
    # 파일 유틸리티
    "ensure_directory", "save_json", "load_json", "list_files", "get_project_root",
    # 로깅 유틸리티
    "setup_logger", "setup_root_logger", "get_module_logger",
    # OpenAI 유틸리티
    "initialize_client", "track_token_usage", "generate_chat_completion",
    # 오디오 유틸리티
    "initialize_elevenlabs", "transcribe_audio", "generate_speech",
    # WebSocket 유틸리티
    "validate_token", "ConnectionManager",
    # S3 관리
    "S3Manager"
]
