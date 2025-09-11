"""
Voice Processing Module for JJongal-AI

Real-time voice processing, cloning, and audio handling for
the storytelling conversation system.
"""

from .cloning import VoiceCloner, VoiceCloneStatus, get_voice_cloner
from .processors.auth_processor import VoiceAuthProcessor, get_auth_processor

try:
    from .handlers.audio_handler import AudioHandler, get_audio_handler
    _audio_handler_available = True
except ImportError:
    _audio_handler_available = False
    AudioHandler = None
    get_audio_handler = None

__all__ = [
    "VoiceCloner",
    "VoiceCloneStatus", 
    "get_voice_cloner",
    "VoiceAuthProcessor",
    "get_auth_processor"
]

if _audio_handler_available:
    __all__.extend(["AudioHandler", "get_audio_handler"])

__version__ = "1.0.0"