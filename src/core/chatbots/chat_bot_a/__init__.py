"""
ChatBot A (Jjongi) Module

Voice-interactive story collection bot that gathers story elements
from children through real-time conversation with voice cloning.
"""

from .chat_bot_a import ChatBotA
from .chains.conversation_chain import ConversationChain
from .chains.rag_chain import ModernRAGChain, RAGChain

try:
    from .memory.conversation_memory import ConversationMemoryManager
    _memory_available = True
except ImportError:
    _memory_available = False

try:
    from .tools.memory_tool import MemoryTool, get_memory_tool
    _tools_available = True
except ImportError:
    _tools_available = False

__all__ = [
    "ChatBotA",
    "ConversationChain",
    "ModernRAGChain",
    "RAGChain"
]

if _memory_available:
    __all__.append("ConversationMemoryManager")

if _tools_available:
    __all__.extend(["MemoryTool", "get_memory_tool"])

__version__ = "1.0.0"