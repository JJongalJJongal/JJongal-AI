"""
Modern Ari (ChatBot B) - API v2.0 Compliant Story Generation

Modern version featuring:
- API v2.0 compliant story generation
- LCEL-based LangChain chains for efficiency
- Multimedia content generation (text, images, voice)
- Age-appropriate content adaptation
- Voice configuration support

주요 구성 요소:
- ModernAri: API v2.0 호환 메인 인터페이스 (아리/Ari)
- Generators: 텍스트, 이미지, 음성 생성기들
"""

# 생성자 모듈
from .generators import (
    BaseGenerator,
    TextGenerator,
    ImageGenerator,
    VoiceGenerator
)

# Modern 메인 인터페이스 클래스
from .chat_bot_b import ModernAri, ChatBotB

__all__ = [
    # Modern 메인 인터페이스
    "ModernAri",
    "ChatBotB",  # Backward compatibility
    
    # 생성자 모듈
    "TextGenerator",
    "ImageGenerator", 
    "VoiceGenerator",
    "BaseGenerator"
]
