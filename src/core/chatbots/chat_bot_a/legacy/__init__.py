"""
Legacy Module for ChatBot A (Jjongi)

Backward compatibility adapters and legacy support for
transitioning from older ChatBot A implementations.
"""

try:
    from .chat_bot_a_adapter import LegacyChatBotAAdapter
    _adapter_available = True
except ImportError:
    _adapter_available = False

__all__ = []

if _adapter_available:
    __all__.append("LegacyChatBotAAdapter")

__version__ = "1.0.0"