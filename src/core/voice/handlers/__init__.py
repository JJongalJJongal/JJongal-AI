"""
Voice Handlers Module for JJongal-AI

Audio processing handlers for WebSocket endpoints and real-time
voice interaction in the storytelling system.
"""

try:
    from .audio_handler import AudioHandler, get_audio_handler
    _audio_handler_available = True
except ImportError:
    _audio_handler_available = False

__all__ = []

if _audio_handler_available:
    __all__.extend(["AudioHandler", "get_audio_handler"])

__version__ = "1.0.0"