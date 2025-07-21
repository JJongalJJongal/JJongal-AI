"""
LangSmith Configuration and Tracing for ChatBot A

Advanced monitoring and observability setup for tracking AI model performance,
conversation quality, and story collection effectiveness.
"""

import os
from typing import Dict, Any, Optional, List
from datetime import datetime
import uuid

from langsmith import Client, traceable
from langsmith.run_helpers import trace
from langchain.callbacks import LangChainTracer
from langchain.schema import BaseMessage

from shared.utils.logging_utils import get_module_logger

logger = get_module_logger(__name__)


class ChatBotATracer:
    """
    Custom tracer for ChatBot A with specialized metrics for story collection
    and child conversation monitoring.
    """
    
    def __init__(self, project_name: str = "ChatBot-A-Jjongi"):
        self.project_name = project_name
        self.client = None
        self.session_id = None
        self.child_profile = {}
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize LangSmith client with proper configuration"""
        try:
            # Check for LangSmith API key
            api_key = os.getenv("LANGCHAIN_API_KEY") or os.getenv("LANGSMITH_API_KEY")
            if not api_key:
                logger.warning("LangSmith API key not found. Monitoring will be disabled.")
                return
            
            # Initialize client
            self.client = Client(
                api_key=api_key,
                api_url=os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
            )
            
            # Set up tracing environment
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_PROJECT"] = self.project_name
            
            logger.info(f"LangSmith tracing initialized for project: {self.project_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize LangSmith client: {e}")
            self.client = None
    
    def start_conversation_session(self, child_name: str, age: int, interests: List[str]) -> str:
        """Start a new conversation session with child profile tracking"""
        self.session_id = f"session_{uuid.uuid4().hex[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.child_profile = {
            "name": child_name,
            "age": age,
            "interests": interests,
            "session_start": datetime.now().isoformat()
        }
        
        if self.client:
            try:
                # Create session metadata
                session_metadata = {
                    "session_id": self.session_id,
                    "child_profile": self.child_profile,
                    "chatbot": "ChatBot_A_Jjongi",
                    "version": "2025_LangChain_Enhanced"
                }
                
                # Log session start
                self.client.create_run(
                    name="conversation_session_start",
                    run_type="llm",
                    inputs={"child_profile": self.child_profile},
                    session_name=self.session_id,
                    extra=session_metadata
                )
                
                logger.info(f"LangSmith session started: {self.session_id}")
                
            except Exception as e:
                logger.error(f"Failed to start LangSmith session: {e}")
        
        return self.session_id
    
    @traceable(name="conversation_turn")
    def trace_conversation_turn(
        self, 
        user_input: str, 
        ai_response: str, 
        intent: str,
        story_elements_found: List[Dict[str, Any]],
        processing_time: float,
        turn_number: int
    ) -> Dict[str, Any]:
        """Trace individual conversation turn with detailed metrics"""
        
        turn_data = {
            "session_id": self.session_id,
            "turn_number": turn_number,
            "user_input": user_input,
            "ai_response": ai_response,
            "intent": intent,
            "story_elements_found": story_elements_found,
            "processing_time_ms": processing_time * 1000,
            "timestamp": datetime.now().isoformat(),
            "child_profile": self.child_profile
        }
        
        # Add conversation quality metrics
        turn_data.update(self._calculate_turn_metrics(user_input, ai_response, story_elements_found))
        
        return turn_data
    
    @traceable(name="story_collection_progress")
    def trace_story_progress(
        self, 
        collected_elements: Dict[str, Any], 
        completion_percentage: float,
        current_stage: str
    ) -> Dict[str, Any]:
        """Trace story collection progress with detailed analytics"""
        
        progress_data = {
            "session_id": self.session_id,
            "collected_elements": collected_elements,
            "completion_percentage": completion_percentage,
            "current_stage": current_stage,
            "elements_count": len(collected_elements.get("elements", [])),
            "timestamp": datetime.now().isoformat(),
            "child_age": self.child_profile.get("age")
        }
        
        # Add story quality metrics
        progress_data.update(self._calculate_story_metrics(collected_elements))
        
        return progress_data
    
    @traceable(name="rag_query")
    def trace_rag_query(
        self, 
        query: str, 
        retrieved_docs: List[Dict], 
        relevance_scores: List[float],
        retrieval_time: float
    ) -> Dict[str, Any]:
        """Trace RAG retrieval with relevance metrics"""
        
        rag_data = {
            "session_id": self.session_id,
            "query": query,
            "docs_retrieved": len(retrieved_docs),
            "avg_relevance_score": sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0,
            "retrieval_time_ms": retrieval_time * 1000,
            "timestamp": datetime.now().isoformat()
        }
        
        return rag_data
    
    def _calculate_turn_metrics(self, user_input: str, ai_response: str, story_elements: List) -> Dict[str, Any]:
        """Calculate conversation quality metrics for each turn"""
        return {
            "user_input_length": len(user_input.split()),
            "ai_response_length": len(ai_response.split()),
            "story_elements_extracted": len(story_elements),
            "response_appropriateness_score": self._assess_age_appropriateness(ai_response),
            "engagement_indicators": self._detect_engagement_indicators(user_input),
            "korean_language_quality": self._assess_korean_quality(ai_response)
        }
    
    def _calculate_story_metrics(self, collected_elements: Dict) -> Dict[str, Any]:
        """Calculate story collection quality metrics"""
        elements = collected_elements.get("elements", [])
        
        # Count elements by type
        element_counts = {}
        for element in elements:
            element_type = element.get("element_type", "unknown")
            element_counts[element_type] = element_counts.get(element_type, 0) + 1
        
        return {
            "character_count": element_counts.get("character", 0),
            "setting_count": element_counts.get("setting", 0),
            "problem_count": element_counts.get("problem", 0),
            "resolution_count": element_counts.get("resolution", 0),
            "story_creativity_score": self._assess_creativity(elements),
            "story_coherence_score": self._assess_coherence(elements)
        }
    
    def _assess_age_appropriateness(self, response: str) -> float:
        """Simple age appropriateness assessment"""
        # Basic implementation - can be enhanced with more sophisticated analysis
        age = self.child_profile.get("age", 5)
        
        word_count = len(response.split())
        sentence_count = response.count('.') + response.count('!') + response.count('?')
        avg_words_per_sentence = word_count / max(sentence_count, 1)
        
        if age <= 7:
            # Younger children need shorter sentences
            ideal_length = 8
        else:
            # Older children can handle longer sentences
            ideal_length = 12
        
        # Score based on sentence length appropriateness
        score = max(0.0, 1.0 - abs(avg_words_per_sentence - ideal_length) / ideal_length)
        return round(score, 2)
    
    def _detect_engagement_indicators(self, user_input: str) -> List[str]:
        """Detect engagement indicators in user input"""
        indicators = []
        
        if len(user_input.split()) > 10:
            indicators.append("detailed_response")
        if "!" in user_input:
            indicators.append("excitement")
        if "?" in user_input:
            indicators.append("curiosity")
        if any(word in user_input.lower() for word in ["재미있", "좋아", "신나"]):
            indicators.append("positive_emotion")
        if any(word in user_input.lower() for word in ["그리고", "그래서", "다음에"]):
            indicators.append("story_continuation")
            
        return indicators
    
    def _assess_korean_quality(self, response: str) -> float:
        """Assess Korean language quality and naturalness"""
        # Basic implementation - checks for proper Korean structure
        korean_particles = ["은", "는", "이", "가", "을", "를", "에", "에서", "로", "으로"]
        korean_endings = ["요", "야", "어", "아", "지", "까", "네", "군"]
        
        particle_count = sum(1 for particle in korean_particles if particle in response)
        ending_count = sum(1 for ending in korean_endings if response.endswith(ending))
        
        # Simple scoring based on Korean linguistic features
        score = min(1.0, (particle_count + ending_count) / len(response.split()) * 10)
        return round(score, 2)
    
    def _assess_creativity(self, elements: List[Dict]) -> float:
        """Assess creativity of story elements"""
        if not elements:
            return 0.0
        
        # Count unique concepts and creative descriptors
        unique_concepts = set()
        creative_words = ["마법", "모험", "신비", "특별한", "재미있는", "용감한", "친절한"]
        
        creativity_score = 0
        for element in elements:
            content = element.get("content", "").lower()
            unique_concepts.add(content[:20])  # First 20 chars as concept identifier
            
            for word in creative_words:
                if word in content:
                    creativity_score += 0.1
        
        # Normalize score
        base_score = len(unique_concepts) / max(len(elements), 1)
        final_score = min(1.0, base_score + creativity_score)
        
        return round(final_score, 2)
    
    def _assess_coherence(self, elements: List[Dict]) -> float:
        """Assess coherence between story elements"""
        if len(elements) < 2:
            return 0.5  # Default score for insufficient data
        
        # Simple coherence check based on element relationships
        character_elements = [e for e in elements if e.get("element_type") == "character"]
        setting_elements = [e for e in elements if e.get("element_type") == "setting"]
        
        # Check if characters and settings are logically related
        coherence_score = 0.5  # Base score
        
        if character_elements and setting_elements:
            # Additional points for having both characters and settings
            coherence_score += 0.3
        
        return round(min(1.0, coherence_score), 2)
    
    def end_conversation_session(self, final_story_outline: Dict[str, Any]) -> Dict[str, Any]:
        """End conversation session with comprehensive summary"""
        if not self.client:
            return {}
        
        try:
            session_summary = {
                "session_id": self.session_id,
                "child_profile": self.child_profile,
                "final_story_outline": final_story_outline,
                "session_end": datetime.now().isoformat(),
                "total_duration": self._calculate_session_duration()
            }
            
            # Log session end
            self.client.create_run(
                name="conversation_session_end",
                run_type="llm",
                inputs={"session_summary": session_summary},
                outputs={"story_outline": final_story_outline},
                session_name=self.session_id
            )
            
            logger.info(f"LangSmith session ended: {self.session_id}")
            return session_summary
            
        except Exception as e:
            logger.error(f"Failed to end LangSmith session: {e}")
            return {}
    
    def _calculate_session_duration(self) -> float:
        """Calculate total session duration in minutes"""
        if "session_start" not in self.child_profile:
            return 0.0
        
        start_time = datetime.fromisoformat(self.child_profile["session_start"])
        duration = (datetime.now() - start_time).total_seconds() / 60
        return round(duration, 2)


def setup_langsmith_tracing(project_name: str = "ChatBot-A-Jjongi") -> Optional[ChatBotATracer]:
    """
    Setup LangSmith tracing for ChatBot A
    
    Returns:
        ChatBotATracer instance if successful, None if failed
    """
    try:
        tracer = ChatBotATracer(project_name)
        
        if tracer.client:
            logger.info("LangSmith tracing setup completed successfully")
            return tracer
        else:
            logger.warning("LangSmith tracing setup failed - running without monitoring")
            return None
            
    except Exception as e:
        logger.error(f"LangSmith setup error: {e}")
        return None


# Decorator for easy tracing of ChatBot A methods
def trace_chatbot_method(method_name: str):
    """Decorator to easily trace ChatBot A methods"""
    def decorator(func):
        @traceable(name=f"chatbot_a_{method_name}")
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator