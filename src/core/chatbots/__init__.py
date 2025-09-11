"""
Chatbots Module for JJongal-AI

Core chatbot implementations for the storytelling system:
- ChatBot A (Jjongi): Story collection through conversation
- ChatBot B (Ari): Complete story generation with multimedia
- Collaboration: Cooperation between chatbots
"""

from .chat_bot_a.chat_bot_a import ChatBotA
from .chat_bot_b.chat_bot_b import ChatBotB

try:
    from .collaboration.jjong_ari_collaborator import JjongAriCollaborator
    _collaboration_available = True
except ImportError:
    _collaboration_available = False

__all__ = [
    "ChatBotA",
    "ChatBotB"
]

if _collaboration_available:
    __all__.append("JjongAriCollaborator")

__version__ = "1.0.0"