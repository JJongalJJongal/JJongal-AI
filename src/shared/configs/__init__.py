"""
CCB_AI 프로젝트 설정 패키지
"""

# 애플리케이션 설정
from .app import get_env_vars, get_app_settings, get_project_root, initialize_env

# 프롬프트 설정
from .prompts import (
    load_chatbot_a_prompts,
    load_chatbot_b_prompts,
    get_default_chatbot_a_prompts,
    get_default_chatbot_b_prompts
) 