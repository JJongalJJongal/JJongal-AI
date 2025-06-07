"""
부기 (ChatBot A) Core Module

이 모듈은 부기 챗봇의 핵심 엔진들을 포함합니다:
- ConversationEngine: 대화 관리 및 토큰 제한 처리
- LangChainConversationEngine: LangChain 기반 대화 엔진
- StoryCollectionEngine: 이야기 수집 및 분석 엔진
- RAGEngine: LangChain 기반 검색 증강 생성 엔진
- LegacyIntegrationManager: 레거시 컴포넌트 통합 관리
"""

from .conversation_engine import ConversationEngine
from .langchain_conversation_engine import LangChainConversationEngine
from .story_engine import StoryEngine
from .rag_engine import RAGSystem
from .legacy_integration import (
    LegacyIntegrationManager,
    LegacyConversationManagerAdapter,
    LegacyMessageFormatterAdapter,
    LegacyStoryCollectorAdapter,
    LegacyStoryAnalyzerAdapter
)

__all__ = [
    "ConversationEngine",
    "LangChainConversationEngine",
    "StoryEngine",
    "RAGSystem",
    "LegacyIntegrationManager",
    "LegacyConversationManagerAdapter",
    "LegacyMessageFormatterAdapter",
    "LegacyStoryCollectorAdapter",
    "LegacyStoryAnalyzerAdapter"
] 