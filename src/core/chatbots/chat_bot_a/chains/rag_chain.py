"""
Modern RAG Chain for ChatBot A (쫑이/Jjongi) - 2025 LangChain Best Practices

Simplified and optimized RAG implementation using LCEL (LangChain Expression Language).
Focus on essential functionality with modern patterns.
"""

from typing import Dict, Any, List, Optional
import asyncio

# 2025 LangChain imports
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ..monitoring.langsmith_config import trace_chatbot_method
from src.shared.utils.logging import get_module_logger
from src.shared.configs.prompts import load_chatbot_a_prompts

logger = get_module_logger(__name__)

# Modern ChromaDB package - 2025 Best Practices (after logger definition)
try:
    from langchain_chroma import Chroma
    logger.info("✅ RAG Chain: 최신 langchain-chroma 사용")
except ImportError:
    # Fallback to legacy import for backward compatibility
    from langchain_community.vectorstores import Chroma
    logger.warning("⚠️ RAG Chain: 구버전 langchain_community 사용 - 업그레이드 필요")

class ModernRAGChain:
    """
    Modern RAG Chain for ChatBot A - 2025 Best Practices
    
    Simplified features:
    - LCEL chains for efficiency
    - Modern ChromaDB integration  
    - Streamlined retrieval logic
    - Age-appropriate filtering
    """
    
    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.7,
        vector_db_path: Optional[str] = None
    ):
        """Initialize Modern RAG Chain"""
        self.model_name = model_name
        self.temperature = temperature
        
        # Modern LLM setup
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            top_p=0.9,
            frequency_penalty=0.1
        )
        
        # Load configuration
        self.prompts = load_chatbot_a_prompts()
        
        # Setup modern vector store
        self._setup_vector_store(vector_db_path)
        
        # Setup LCEL chains
        self._setup_modern_chains()
        
        logger.info(f"Modern RAG Chain initialized with {model_name}")
        
    def _setup_vector_store(self, vector_db_path: Optional[str]):
        """Setup LFU Hybrid vector store for advanced RAG capabilities"""
        try:
            if vector_db_path:
                from src.data.vector_db.core import LFUHybridVectorDB, SearchWeights
                
                logger.info("🚀 RAG Chain: Initializing LFU Hybrid Vector Store...")
                
                # Initialize LFU Hybrid VectorDB for RAG - ChatBot A uses Summary DB
                self.lfu_vector_db = LFUHybridVectorDB(
                    persist_directory=vector_db_path,
                    collection_name="fairy_tales_summary",  # ChatBot A uses Summary collection
                    use_openai=False,  # Use KURE-v1 Korean embeddings
                    memory_cache_size=300,  # Smaller cache for RAG chain
                    disk_cache_size=1000,
                    enable_bm25=True,   # Enable keyword search for better RAG
                    search_weights=SearchWeights(
                        vector_weight=0.6,    # Higher vector weight for RAG
                        bm25_weight=0.3,      # Keyword matching
                        metadata_weight=0.1   # Light metadata filtering
                    )
                )
                
                # For backward compatibility, expose vectorstore
                self.vectorstore = self.lfu_vector_db.vectorstore
                
                # Create hybrid retriever that uses both vector and keyword search
                self.retriever = self._create_hybrid_retriever()
                
                self.rag_enabled = True
                logger.info("✅ LFU Hybrid Vector Store initialized for RAG")
                
                # Log cache stats
                cache_stats = self.lfu_vector_db.get_cache_stats()
                logger.info(f"📊 RAG Cache: {cache_stats['cache_stats']['cache_size']} entries, BM25: {cache_stats['bm25_enabled']}")
                
            else:
                self.rag_enabled = False
                logger.info("RAG disabled - no vector DB path provided")
                
        except Exception as e:
            logger.error(f"LFU Hybrid vector store setup failed: {e}")
            self.rag_enabled = False
    
    def _create_hybrid_retriever(self):
        """Create hybrid retriever using LFU system"""
        class LFUHybridRetriever:
            def __init__(self, lfu_vector_db):
                self.lfu_vector_db = lfu_vector_db
            
            def get_relevant_documents(self, query: str, k: int = 3):
                """Get documents using hybrid search"""
                try:
                    # Use hybrid search for better results
                    hybrid_results = self.lfu_vector_db.hybrid_search(
                        query=query, 
                        k=k, 
                        use_cache=True
                    )
                    
                    # Convert to LangChain Document format
                    from langchain_core.documents import Document
                    documents = []
                    
                    for result in hybrid_results:
                        doc = Document(
                            page_content=result.content,
                            metadata={
                                **result.metadata,
                                'hybrid_score': result.final_score,
                                'vector_score': result.vector_score,
                                'bm25_score': result.bm25_score
                            }
                        )
                        documents.append(doc)
                    
                    return documents
                    
                except Exception as e:
                    logger.error(f"Hybrid retrieval failed: {e}")
                    return []
        
        return LFUHybridRetriever(self.lfu_vector_db)
        
    def _setup_modern_chains(self):
        """Setup modern LCEL chains"""
        
        # Context retrieval chain (LCEL style)
        if self.rag_enabled:
            self.context_chain = (
                {"query": RunnablePassthrough()}
                | RunnableLambda(self._get_context)
            )
        else:
            self.context_chain = RunnableLambda(lambda x: "")
        
        # Main RAG conversation chain
        self.rag_prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 '쫑이'입니다. 아이와 친근하게 대화하며 이야기 요소를 수집합니다.

참고 동화 지식:
{context}

규칙:
1. 간단하고 친근한 말투 사용
2. 아이의 창의성 격려  
3. 한 번에 하나의 질문만
4. 연령에 맞는 어휘 사용
5. 반말 사용"""),
            ("human", "아이 입력: {user_input}\n목표 요소: {target_element}")
        ])
        
        # Modern LCEL chain
        self.conversation_chain = (
            RunnableParallel({
                "context": self.context_chain,
                "user_input": RunnablePassthrough(),
                "target_element": RunnablePassthrough()
            })
            | self.rag_prompt
            | self.llm
            | StrOutputParser()
        )
    
    def _get_context(self, inputs: Dict[str, str]) -> str:
        """Get RAG context using modern retriever"""
        if not self.rag_enabled:
            return ""
        
        try:
            query = inputs.get("query", "")
            if not query:
                return ""
            
            docs = self.retriever.get_relevant_documents(query)
            
            if not docs:
                return ""
            
            # Format context from top documents
            context_parts = []
            for i, doc in enumerate(docs[:2], 1):
                content = doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content
                context_parts.append(f"{i}. {content}")
            
            return "\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"Error getting context: {e}")
            return ""
    @trace_chatbot_method("generate_rag_response")
    async def generate_rag_response(
        self,
        user_input: str,
        target_element: str = "character"
    ) -> str:
        """
        Generate RAG-enhanced response using modern LCEL chain
        
        Args:
            user_input: Child's input
            target_element: Target story element to focus on
            
        Returns:
            RAG-enhanced response as 쫑이
        """
        try:
            response = await self.conversation_chain.ainvoke({
                "query": user_input,
                "user_input": user_input,
                "target_element": target_element
            })
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating RAG response: {e}")
            return f"재미있는 이야기네! {target_element}에 대해 더 말해줄래?"
    
    async def get_similar_stories(self, query: str, top_k: int = 3) -> List[str]:
        """Get similar stories for inspiration (simplified)"""
        if not self.rag_enabled:
            return []
        
        try:
            docs = self.retriever.get_relevant_documents(query)
            return [doc.page_content[:100] + "..." for doc in docs[:top_k]]
        except Exception as e:
            logger.error(f"Error getting similar stories: {e}")
            return []

# Factory function for backward compatibility
def create_rag_chain(vector_db_path: Optional[str] = None) -> ModernRAGChain:
    """Create ModernRAGChain instance"""
    return ModernRAGChain(vector_db_path=vector_db_path)

# Backward compatibility alias  
RAGChain = ModernRAGChain