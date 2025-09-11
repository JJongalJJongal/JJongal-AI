"""
API Models Module for JJongal-AI v1

Pydantic models for API request/response structures,
data validation, and standardized API communication.
"""

from .requests import (
    StoryGenerationRequest,
    AudioUploadRequest,
    VoiceCloneRequest,
    ChatRequest
)

try:
    from .responses import (
        BaseResponse,
        StoryResponse,
        AudioResponse,
        VoiceCloneResponse,
        ChatResponse,
        ErrorResponse,
        HealthResponse
    )
    _responses_available = True
except ImportError:
    _responses_available = False

__all__ = [
    "StoryGenerationRequest",
    "AudioUploadRequest", 
    "VoiceCloneRequest",
    "ChatRequest"
]

if _responses_available:
    __all__.extend([
        "BaseResponse",
        "StoryResponse",
        "AudioResponse",
        "VoiceCloneResponse", 
        "ChatResponse",
        "ErrorResponse",
        "HealthResponse"
    ])

__version__ = "1.0.0"