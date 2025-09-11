"""
애플리케이션 환경 설정 모듈
"""
import os
from typing import Dict, Any
from dotenv import load_dotenv
from pathlib import Path


def get_project_root() -> Path:
    """
    프로젝트 루트 디렉토리 경로 반환
    
    Returns:
        Path: 프로젝트 루트 디렉토리 경로
    """
    # 현재 파일 기준으로 4단계 위로 이동 (src/shared/configs/app.py -> JJongal-AI)
    current_path = Path(__file__).resolve()
    return current_path.parent.parent.parent.parent


def initialize_env() -> Dict[str, str]:
    """
    환경 변수 초기화
    
    Returns:
        Dict[str, str]: 환경 변수 딕셔너리
    """
    project_root = get_project_root()
    dotenv_path = project_root / '.env'
    
    # .env 파일 로드
    load_dotenv(dotenv_path=str(dotenv_path))
    
    return {}  # 기본 빈 딕셔너리 반환


def get_env_vars() -> Dict[str, str]:
    """
    환경 변수 반환
    
    Returns:
        Dict[str, str]: 환경 변수 딕셔너리
    """
    return {
        # API 키
        "openai_api_key": os.getenv("OPENAI_API_KEY", ""),
        "elevenlabs_api_key": os.getenv("ELEVENLABS_API_KEY", ""),
        
        # 모델 설정
        "openai_model": os.getenv("OPENAI_MODEL", "gpt-4o"),
        "elevenlabs_model": os.getenv("ELEVENLABS_MODEL", "eleven_multilingual_v2"),
        "whisper_model": os.getenv("WHISPER_MODEL", "base"),
        
        # 웹소켓 설정
        "ws_auth_token": os.getenv("WS_AUTH_TOKEN", "valid_token"),
        
        # 로깅 설정
        "log_level": os.getenv("LOG_LEVEL", "INFO"),
        "log_file": os.getenv("LOG_FILE", "server.log"),
        
        # 토큰 제한
        "token_limit": os.getenv("TOKEN_LIMIT", "10000"),
        
        # 파일 경로
        "output_dir": os.getenv("OUTPUT_DIR", "output"),
        "data_dir": os.getenv("DATA_DIR", "data"),
    }


def get_app_settings() -> Dict[str, Any]:
    """
    애플리케이션 설정 반환
    
    Returns:
        Dict[str, Any]: 애플리케이션 설정 딕셔너리
    """
    env_vars = get_env_vars()
    
    return {
        # 일반 설정
        "app_name": "쫑알쫑알 AI",
        "app_version": "1.0.0",
        "app_description": "아동 동화 생성 시스템",
        
        # API 설정
        "api_host": "0.0.0.0",
        "api_port": int(os.getenv("API_PORT", "8000")),
        
        # 환경 변수
        "env_vars": env_vars,
        
        # 토큰 제한
        "token_limit": int(env_vars.get("token_limit", "10000")),
        
        # 웹소켓 설정
        "ws_connection_timeout": 30 * 60,  # 30분
    }


class AppConfig:
    """Application configuration class"""
    
    def __init__(self):
        initialize_env()
        self.env_vars = get_env_vars()
        self.app_settings = get_app_settings()
    
    @property
    def openai_api_key(self) -> str:
        return self.env_vars.get("openai_api_key", "")
    
    @property
    def elevenlabs_api_key(self) -> str:
        return self.env_vars.get("elevenlabs_api_key", "")
    
    @property
    def ws_auth_token(self) -> str:
        return self.env_vars.get("ws_auth_token", "")
    
    @property
    def log_level(self) -> str:
        return self.env_vars.get("log_level", "INFO")
    
    @property
    def token_limit(self) -> int:
        return int(self.env_vars.get("token_limit", "10000"))


# 환경 변수 미리 초기화
initialize_env() 