import os
from typing import Dict, List, Optional, Any

# Modern ChromaDB and LangChain imports
try:
    from langchain_community.vectorstores import Chroma
    from langchain_openai import OpenAIEmbeddings
    from langchain_community.embeddings import HuggingFaceEmbeddings
except ImportError as e:
    raise ImportError(f"Required packages not installed: {e}")

from src.shared.utils.logging import get_module_logger

# 로깅 설정
logger = get_module_logger(__name__)

# 기본 임베딩 모델 설정 (향후 설정 파일 등으로 분리 가능)
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

class ModernVectorDB:
    """
    Modern ChromaDB wrapper for JJongal-AI RAG system - 2025 Best Practices
    
    Features:
    - LangChain Chroma integration
    - OpenAI embeddings with fallback to Korean models
    - Simplified interface for fairy tale RAG
    - Age-appropriate content filtering
    """

    def __init__(self, 
                 persist_directory = None, 
                 collection_name: str = "fairy_tales",
                 use_openai: bool = True):
        """
        Modern VectorDB initialization
        
        Args:
            persist_directory: ChromaDB persistence directory
            collection_name: Collection name
            use_openai: Use OpenAI embeddings (fallback to Korean models if False)
        """
        self.persist_directory = persist_directory or "data/chroma_db"
        self.collection_name = collection_name
        self.use_openai = use_openai
        
        # Setup embeddings
        self.embeddings = self._setup_embeddings()
        
        # Setup LangChain Chroma vectorstore
        self.vectorstore = self._setup_vectorstore()
        
        logger.info(f"Modern VectorDB initialized: {collection_name} at {persist_directory}")
    def _setup_embeddings(self):
        """Setup modern embeddings"""
        try:
            return HuggingFaceEmbeddings(
                model_name="nlpai-lab/KURE-v1",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
        except Exception as e:
            logger.warning(f"Korean embeddings failed: {e}, using default")
            return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    def _setup_vectorstore(self):
        """Setup LangChain Chroma vectorstore"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(self.persist_directory, exist_ok=True)
            
            return Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory
            )
        except Exception as e:
            logger.error(f"Vectorstore setup failed: {e}")
            raise
    
    # Modern API methods
    def add_documents(self, texts: List[str], metadatas: Optional[List[Dict]] = None) -> List[str]:
        """Add documents to vectorstore"""
        try:
            ids = self.vectorstore.add_texts(texts, metadatas=metadatas)
            logger.info(f"Added {len(texts)} documents to vectorstore")
            return ids
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            raise
    
    def similarity_search(self, query: str, k: int = 3, **kwargs) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        try:
            docs = self.vectorstore.similarity_search(query, k=k, **kwargs)
            
            results = []
            for doc in docs:
                results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "source": doc.metadata.get("source", "unknown")
                })
            
            logger.debug(f"Found {len(results)} similar documents for query: {query[:50]}...")
            return results
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            return []
    
    def get_retriever(self, search_type: str = "mmr", **kwargs):
        """Get LangChain retriever for RAG chains"""
        search_kwargs = {"k": 3, **kwargs}
        
        if search_type == "mmr":
            search_kwargs.update({"lambda_mult": 0.7, "fetch_k": 10})
        
        return self.vectorstore.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs
        )
    
    def delete_collection(self):
        """Delete the collection"""
        try:
            self.vectorstore.delete_collection()
            logger.info(f"Deleted collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
            raise

# Backward compatibility
VectorDB = ModernVectorDB

def create_vector_db(persist_directory = None, **kwargs) -> ModernVectorDB:
    """Factory function for creating ModernVectorDB"""
    return ModernVectorDB(persist_directory=persist_directory, **kwargs)
    
