"""
CCB_AI 프로젝트의 공통 모듈 패키지
"""

from .utils import (
    # Korean 유틸리티
    has_final_consonant, get_josa, format_with_josa,
    
    # 파일 유틸리티
    ensure_directory, save_json, load_json, list_files, get_project_root,
    
    # 로깅 유틸리티
    setup_logger, setup_root_logger, get_module_logger,
    
    # OpenAI 유틸리티
    initialize_client, track_token_usage, generate_chat_completion,
    
    # 오디오 유틸리티
    initialize_elevenlabs, transcribe_audio, generate_speech,
    
    # WebSocket 유틸리티
    validate_token, ConnectionManager
)

from .configs import (
    # 애플리케이션 설정
    get_env_vars, get_app_settings, get_project_root, initialize_env,
    
    # 프롬프트 설정
    load_chatbot_a_prompts, load_chatbot_b_prompts,
    get_default_chatbot_a_prompts, get_default_chatbot_b_prompts
) 