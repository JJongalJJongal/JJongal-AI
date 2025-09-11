"""
Middleware Module for JJongal-AI API v1

Authentication, CORS, error handling, and other middleware
components for the FastAPI application.
"""

try:
    from .auth import AuthMiddleware, get_current_user
    _auth_available = True
except ImportError:
    _auth_available = False

try:
    from .cors import setup_cors
    _cors_available = True
except ImportError:
    _cors_available = False

__all__ = []

if _auth_available:
    __all__.extend(["AuthMiddleware", "get_current_user"])

if _cors_available:
    __all__.append("setup_cors")

__version__ = "1.0.0"