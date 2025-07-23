"""
RAG Chain for ChatBot A (쫑이/Jjongi)

Enhanced conversation chain with Retrieval-Augmented Generation (RAG)
using vector database to enrich story collection with existing fairy tale knowledge.
"""

from typing import Dict, Any, List, Optional, Tuple
import asyncio
import random

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain.schema import BaseMessage, HumanMessage, AIMessage

from ....data.vector_db.core import VectorDB
from ..monitoring.langsmith_config import trace_chatbot_method
from shared.utils.logging_utils import get_module_logger
from shared.configs.prompts_config import load_chatbot_a_prompts

logger = get_module_logger(__name__)

class RAGChain:
    """
    RAG Chain for ChatBot A
    
    Features:
    - Vector database integration for fairy tale knowledge
    - Context-aware story element suggestions
    - Age-appropriate content retrieval
    - Enhanced conversation guidance
    """
    
    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.7,
        vector_db: VectorDB = None
    ):
        """Initialize RAG Chain with vector database connection"""
        self.model_name = model_name
        self.temperature = temperature
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            model_kwargs={"top_p": 0.9}
        )
        
        # Load prompts from JSON
        self.prompts = load_chatbot_a_prompts()
        
        # Initialize vector database
        self.vector_db = vector_db or VectorDB()
        
        # Setup RAG Chains
        self._setup_chains()
        
        logger.info(f"RAG Chain initialized with model: {model_name}")
        
    def _setup_chains(self):
        """Setup LangChain RAG chains for different purposes"""
        
        # Get RAG enhanced prompts from JSON
        rag_prompts = self.prompts["rag_enhanced_prompts"]["conversation_with_context"]
        
        # Story element enhancement chain
        self.rag_conversation_prompt = ChatPromptTemplate.from_messages([
            ("system", rag_prompts["system"]),
            ("human", rag_prompts["human"])
        ])
        
        self.rag_conversation_chain = (
            self.rag_conversation_prompt |
            self.llm |
            StrOutputParser()
        )
    
    # RAG Search Chains
    @trace_chatbot_method("retrieve_context")
    async def retrieve_context(
        self,
        query: str,
        target_element: str = None,
        child_age: int = 6,
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant fairy tale context from vector database
        
        Args:
            query: User input or search query
            target_element: Target story element (character, setting, problem, resolution)
            child_age: Child's age for filtering appropriate content
            top_k: Number of results to retrieve
            
        Returns:
            List of relevant documents with metadata
        """
        
        try:
            # Build enhanced search query using JSON keywords
            search_query = self._build_search_query(query, target_element, child_age)
            
            # Search vector database
            results = await asyncio.to_thread(
                self.vector_db.search_similar,
                query=search_query,
                top_k=top_k,
                collection_name="fairy_tales"
            )
            
            logger.debug(f"Retrieved {len(results)} documents for query: {search_query[:50]}...")
            return results
        
        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            return []
        
    def _build_search_query(self, query: str, target_element: str = None, child_age: int = 6) -> str:
        """Build enhanced search query using JSON keywords"""
        
        search_parts = [query]
        
        # Add element-specific keywords from JSON
        if target_element and target_element in self.prompts["rag_search_keywords"]:
            keywords = self.prompts["rag_search_keywords"][target_element]
            search_parts.extend(keywords[:3]) # Use top 3 keywords
        
        # Add age-appropriate context
        age_context = "유아용 쉬운 동화" if child_age <= 7 else "초등학생용 동화"
        search_parts.append(age_context)
        
        return " ".join(search_parts)
    
    @trace_chatbot_method("generate_rag_response")
    async def generate_rag_response(
        self,
        user_input: str,
        child_name: str,
        target_element: str,
        session_context: Dict[str, Any] = None
    ) -> str:
        """
        Generate enhanced conversation response using RAG
        
        Args:
            user_input: Child's input
            child_name: Child's name
            child_age: Child's age
            target_element: Target story element to foucs on
            session_context: Current session context
            
        Returns:
            RAG-enhanced response as 쫑이
        """
        try:
            # Retrieve relevant fairy tale context
            retrieved_docs = await self.retrieve_context(
                query=user_input,
                target_element=target_element,
                child_age=child_name,
                top_k=3
            )
            
            # Format retrieved context
            retrieved_context = self._format_retrieved_context(retrieved_docs)
            
            # Generate RAG-enhanced response
            response = await self.rag_conversation_chain.ainvoke({
                "user_input": user_input,
                "child_name": child_name,
                "target_element": target_element,
                "retrieved_context": retrieved_context
            })
            
            return response
        
        except Exception as e:
            logger.error(f"Error generating RAG response: {e}")
            # Fallback to JSON-based response
            return self._get_fallback_response(child_name, target_element)
    
    @trace_chatbot_method("generate_story_guidance")
    async def generate_story_guidance(
        self,
        child_name: str,
        child_age: int,
        target_element: str,
        current_elements: Dict[str, List] = None
    ) -> str:
        """
        Generate story guidance for missing elements using RAG
        
        Args:
            child_name: Child's name
            child_age: Child's age
            target_element: Missing story element to collect
            current_elements: Already collected story elements
            
        Returns:
            RAG-enhanced guidance question
        """
        try:
            # Create context query based on current elements
            context_query = self._build_context_query(current_elements)
            
            # Retrieve relevant examples for the target element
            retrieved_docs = await self.retrieve_context(
                query=context_query,
                target_element=target_element,
                child_age=child_age,
                top_k=2
            )
            
            # Format retrieved context
            retrieved_context = self._format_retrieved_context(retrieved_docs)
            
            # Generate guidance using RAG
            guidance = await self.rag_conversation_chain.ainvoke({
                "user_input": f"{target_element}에 대해 더 이야기해줘!",
                "child_name": child_name,
                "child_age": child_age,
                "target_element": target_element,
                "retrieved_context": retrieved_context
            })
            
            return guidance
        
        except Exception as e:
            logger.error(f"Error generating story guidance: {e}")
            # Fallback to JSON story prompting questions
            return self._get_fallback_guidance(child_name, target_element)
        
    def _build_context_query(self, current_elements: Dict[str, List]) -> str:
        """Build context query from current story elements"""
        if not current_elements:
            return "동화 이야기"
        
        query_parts = []
        
        for element_type, elements in current_elements.items():
            if elements:
                contents = [
                    elem.get('content', str(elem)) if isinstance(elem, dict) else str(elem)
                    for elem in elements
                ]
                query_parts.append(contents)
        
        return " ".join(query_parts) if query_parts else "동화 이야기"
    
    def _format_retrieved_context(self, docs: List[Dict[str, Any]]) -> str:
        """Format retrieved documents into readable context"""
        if not docs:
            return "관련 동화 내용을 찾을 수 없어서, 네 상상력에만 의존할게!"
        
        context_parts = []
        for i, doc in enumerate(docs, 1):
            context = doc.get("document", '')
            metadata = doc.get("metadata", {})
            
            # Extract useful metadata
            title = metadata.get("title", f"동화 예시 {i}")
            
            # Limit content length for readability
            content_preview = context[:150] + "..." if len(context) > 150 else context
            context_parts.append(f"[{title}] {content_preview}")
            
        return "\n".join(context_parts)
    
    def _get_fallback_response(self, child_name: str, target_element: str) -> str:
        """Get fallback response using JSON encouragemenet phrases"""
        try:
            encouragemenets = self.prompts["encouragement_phrases"]
            selected = random.choice(encouragemenets)
            return selected.format(name=child_name)
        
        except:
            return f"{child_name}아, 네 이야기가 정말 재미있어! 더 들려줄래?"
        
    def _get_fallback_story_question(self, child_name: str, target_element: str) -> str:
        """Get fallback story question using JSON prompts"""
        try:
            story_questions = self.prompts["story_prompting_questions"].get(target_element, [])
            if story_questions:
                selected = random.choice(story_questions)
                return selected.format(name=child_name)
            
        except:
            pass
        
        # Ultimate fallback using follow_up_questions
        try:
            follow_ups = self.prompts["follow_up_questions"]
            selected = random.choice(follow_ups)
            return selected.format(name=child_name)
        except:
            return f"{child_name}아, 네 이야기가 정말 재미있어! 더 들려줄래?"
    
    async def search_similar_stories(
        self,
        story_elements: Dict[str, List],
        child_age: int = 7,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for similar stories based on collected elements
        
        Args:
            story_elements: Collected story elements
            child_age: Child's age
            top_k: Number of results
            
        Returns:
            Similar stories for inspiration
        """
        
        try:
            # Build comprehensive search query from story elements
            query_parts = []
            
            for elements in story_elements.items():
                if elements:
                    element_contents = [
                        elem.get("content", str(elem)) if isinstance(elem, dict) else str(elem)
                        for elem in elements
                    ]
                    query_parts.append(" ".join(element_contents))
                
                search_query = " ".join(query_parts) if query_parts else "동화 이야기"
                
                # Search for similar stories
                similar_stories = await self.retrieve_context(
                    query=search_query,
                    child_age=child_age,
                    top_k=top_k
                )
                
                return similar_stories
        except Exception as e:
            logger.error(f"Error searching similar stories: {e}")
            return []
        
    def get_contextual_encouragement(self, child_name: str, element: str) -> str:
        """
        Get contextual encouragement using existing JSON encouragement phrases
        
        Args:
            child_name: Child's name
            element: Story element being discussed
            
        Returns:
            Contextual encouragement message
        """
        try:
            encouragements = self.prompts["encouragement_phrases"]
            selected = random.choice(encouragements)
            return selected.format(name=child_name)
        except:
            return f"정말 대단해, {child_name}아! 네 {element}에 대한 생각이 너무 창의적이야!"
        
    async def get_age_appropriate_vocabulary(self, child_age: int) -> Dict[str, Any]:
        """Get age-appropriate vocabulary and concepts from JSON"""
        try:
            age_group = "4-7" if child_age <= 7 else "8-9"
            return self.prompts["age_appropriate_language"][age_group]
        except:
            # Fallback for age-appropriate language
            return {
                "vocabulary": ["친구", "이야기", "재미있다", "좋아하다"],
                "sentence_length": "3-5 단어" if child_age <= 7 else "7-12 단어",
                "concepts": ["기본 감정", "단순한 관계"] if child_age <= 7 else ["복잡한 감정", "다양한 관점"]
            }
        
        
        
            
            