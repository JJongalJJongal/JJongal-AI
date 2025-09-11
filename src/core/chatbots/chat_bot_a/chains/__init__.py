"""
LangChain Chains Module for ChatBot A (Jjongi)

Modern LangChain-based conversation and RAG chains for
enhanced story collection with retrieval-augmented generation.
"""

from .conversation_chain import ConversationChain
from .rag_chain import ModernRAGChain, RAGChain, create_rag_chain

__all__ = [
    "ConversationChain",
    "ModernRAGChain", 
    "RAGChain",
    "create_rag_chain"
]

__version__ = "1.0.0"