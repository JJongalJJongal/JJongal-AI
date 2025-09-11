"""
API Endpoints Module for JJongal-AI v1

FastAPI endpoints for story generation, audio processing,
authentication, and health monitoring.
"""

try:
    from .auth import auth_router
    _auth_available = True
except ImportError:
    _auth_available = False

try:
    from .stories import stories_router
    _stories_available = True
except ImportError:
    _stories_available = False

try:
    from .health import health_router
    _health_available = True
except ImportError:
    _health_available = False

try:
    from .unified_audio import audio_router
    _audio_available = True
except ImportError:
    _audio_available = False

try:
    from .family_voices import family_voices_router
    _family_voices_available = True
except ImportError:
    _family_voices_available = False

__all__ = []

if _auth_available:
    __all__.append("auth_router")

if _stories_available:
    __all__.append("stories_router")

if _health_available:
    __all__.append("health_router")

if _audio_available:
    __all__.append("audio_router")

if _family_voices_available:
    __all__.append("family_voices_router")

__version__ = "1.0.0"