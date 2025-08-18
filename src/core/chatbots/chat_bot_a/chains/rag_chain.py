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
from langchain_community.vectorstores import Chroma

from ..monitoring.langsmith_config import trace_chatbot_method
from src.shared.utils.logging import get_module_logger
from src.shared.configs.prompts import load_chatbot_a_prompts

logger = get_module_logger(__name__)

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
            model_kwargs={"top_p": 0.9, "frequency_penalty": 0.1}
        )
        
        # Load configuration
        self.prompts = load_chatbot_a_prompts()
        
        # Setup modern vector store
        self._setup_vector_store(vector_db_path)
        
        # Setup LCEL chains
        self._setup_modern_chains()
        
        logger.info(f"Modern RAG Chain initialized with {model_name}")
        
    def _setup_vector_store(self, vector_db_path: Optional[str]):
        """Setup modern ChromaDB vector store"""
        try:
            if vector_db_path:
                embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
                
                self.vectorstore = Chroma(
                    persist_directory=vector_db_path,
                    embedding_function=embeddings,
                    collection_name="fairy_tales"
                )
                
                # Modern retriever with MMR
                self.retriever = self.vectorstore.as_retriever(
                    search_type="mmr",
                    search_kwargs={"k": 3, "lambda_mult": 0.7}
                )
                
                self.rag_enabled = True
                logger.info("Modern ChromaDB vector store initialized")
            else:
                self.rag_enabled = False
                logger.info("RAG disabled - no vector DB path provided")
                
        except Exception as e:
            logger.error(f"Vector store setup failed: {e}")
            self.rag_enabled = False
        
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