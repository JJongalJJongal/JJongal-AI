"""
꼬기 (ChatBot B) - 동화 생성 챗봇 모듈

이 모듈은 부기(ChatBot A)에서 수집한 이야기 요소를 바탕으로
완전한 멀티미디어 동화를 생성하는 시스템.

주요 구성 요소:
- StoryGenerationEngine: 동화 생성 핵심 엔진
- ContentPipeline: 텍스트 -> 이미지 -> 음성 파이프라인
- Generators: 텍스트, 이미지, 음성 생성기들
"""

# 핵심 모듈
from .core import StoryGenerationEngine, ContentPipeline

# 생성자 모듈
from .generators import (
    TextGenerator,
    ImageGenerator, 
    VoiceGenerator,
    BaseGenerator
)

# 메인 인터페이스 클래스
from .chat_bot_b import ChatBotB

__all__ = [
    # 메인 인터페이스 클래스
    "ChatBotB",
    
    # 핵심 엔진
    "StoryGenerationEngine",
    "ContentPipeline",
    
    # 생성자 모듈
    "TextGenerator",
    "ImageGenerator", 
    "VoiceGenerator",
    "BaseGenerator"
]
