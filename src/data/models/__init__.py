"""
Data Models Module for JJongal-AI

Pydantic models and data structures for stories, conversations,
and other core data entities in the storytelling pipeline.
"""

from .story import (
    Story,
    StoryStatus,
    StoryMetadata,
    StoryElement,
    StoryResponse,
    CreateStoryRequest,
    UpdateStoryRequest
)

try:
    from .conversation import (
        ConversationSession,
        ConversationMessage,
        MessageType,
        StoryElementProgress,
        SessionStatus
    )
    _conversation_available = True
except ImportError:
    _conversation_available = False

__all__ = [
    "Story",
    "StoryStatus", 
    "StoryMetadata",
    "StoryElement",
    "StoryResponse",
    "CreateStoryRequest",
    "UpdateStoryRequest"
]

if _conversation_available:
    __all__.extend([
        "ConversationSession",
        "ConversationMessage",
        "MessageType",
        "StoryElementProgress",
        "SessionStatus"
    ])

__version__ = "1.0.0"