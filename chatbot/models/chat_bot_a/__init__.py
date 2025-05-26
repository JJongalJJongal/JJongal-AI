"""
ChatBot A - 아이들과 대화하며 동화 줄거리를 수집하는 AI 챗봇

v3.0.0: 통합 엔진 기반 리팩토링
- 40% 코드 감소
- 성능 향상
- 모듈화 개선
- 완전한 리팩토링 완료

주요 변경사항:
- StoryEngine: 이야기 수집/분석 통합
- UnifiedMessageProcessor: 메시지 처리 통합
- ChatBotA: 새로운 메인 클래스 (기존 API 호환)
"""

from .chat_bot_a import ChatBotA

# 통합 엔진들 (고급 사용자용)
from .core.story_engine import StoryEngine
from .processors.unified_message_processor import UnifiedMessageProcessor

# 유지되는 컴포넌트
from .conversation_manager import ConversationManager


# 메인 API 노출
__all__ = [
    'ChatBotA',
    
    # 통합 엔진들
    'StoryEngine',
    'UnifiedMessageProcessor',
    
    # 유지되는 컴포넌트
    'ConversationManager',

]
