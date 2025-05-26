"""
부기 (ChatBot A) Processors Module

이 모듈은 부기 챗봇의 처리 컴포넌트들을 포함합니다:
- BaseProcessor: 모든 프로세서의 추상 기본 클래스
- MessageProcessor: 메시지 형식 처리 및 포맷팅
- LanguageProcessor: 한국어 특화 언어 처리
- VoiceProcessor: 음성 처리 (향후 구현 예정)
"""

from .base_processor import BaseProcessor
from .message_processor import MessageProcessor
from .language_processor import LanguageProcessor

__all__ = [
    "BaseProcessor",
    "MessageProcessor", 
    "LanguageProcessor"
] 