"""
Enhanced Ari (ChatBot B) - LangChain-기반 동화 생성 시스템

Enhanced version featuring:
- LangChain manager integration for optimized AI operations  
- Multi-chain architecture (planning→generation→evaluation→enhancement)
- Advanced RAG system with cultural context
- Multi-language support with localization
- Performance optimization and intelligent caching
- Age-specific prompt engineering

주요 구성 요소:
- EnhancedAri: LangChain 기반 메인 인터페이스 (ChatBotB 호환)
- StoryGenerationEngine: 동화 생성 핵심 엔진
- ContentPipeline: 텍스트 -> 이미지 -> 음성 파이프라인
- Generators: 텍스트, 이미지, 음성 생성기들
"""

# 핵심 모듈
from .core import StoryGenerationEngine, ContentPipeline

# 생성자 모듈
from .generators import (
    BaseGenerator,
    TextGenerator,
    ImageGenerator,
    VoiceGenerator
)

# Enhanced 메인 인터페이스 클래스
from .chat_bot_b import EnhancedAri, ChatBotB

__all__ = [
    # Enhanced 메인 인터페이스
    "EnhancedAri",
    "ChatBotB",  # Backward compatibility
    
    # 핵심 엔진
    "StoryGenerationEngine",
    "ContentPipeline",
    
    # 생성자 모듈
    "TextGenerator",
    "ImageGenerator", 
    "VoiceGenerator",
    "BaseGenerator"
]
