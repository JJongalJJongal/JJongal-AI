"""
Unified Langsmith monitoring system for entire JjongAlJjonal
application
"""

from optparse import Option
import os
import asyncio
import time
import json
import uuid
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from contextlib import asynccontextmanager
from dataclasses import dataclass, asdict
from enum import Enum

from langsmith import Client
from langchain.callbacks import LangChainTracer
from shared.utils.logging_utils import get_module_logger

logger = get_module_logger(__name__)

class StoryGenerationPhase(Enum):
    CONVERSATION_START = "conversation_start"
    STORY_COLLECTION = "story_collection"
    STORY_ANALYSIS = "story_analysis"
    STORY_GENERATION = "story_generation"
    IMAGE_GENERATION = "image_generation"
    VOICE_GENERATION = "voice_generation"
    STORY_COMPLETION = "story_completion"

@dataclass
class ChildProfile:
    name: str
    age: int
    interests: List[str]
    session_start: str
    voice_samples_count: int = 0
    voice_clone_ready: bool = False

@dataclass
class SessionMetrics:
    total_interactions: int = 0
    successful_interactions: int = 0
    failed_interactions: int = 0
    total_processing_time: float = 0.0
    avg_processing_time: float = 0.0
    rag_queries: int = 0
    story_elements_collected: int = 0
    quality_scores: List[float] = None

    def __post_init__(self):
        if self.quality_scores is None:
            self.quality_scores = []

class UnifiedLangSmithMonitor:
    def __init__(self, 
                project_name: str = "JjongAlJjongAl-Production",
                enable_quality_analysis: bool = True):
        
        self.project_name = project_name
        self.enable_quality_analysis = enable_quality_analysis
        
        self.client = None
        self.tracer = None
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.session_metrics: Dict[str, SessionMetrics] = {}

        self.global_metrics = {
            "total_sessions": 0,
            "successful_stories": 0,
            "failed_stories": 0,
            "avg_story_generation_time": 0.0
        }

        self._initialize_clinet()
        
    def _initialize_clinet(self):
        """Initialize LangSmith client"""
        try:
            api_key = os.getenv("LANGCHAIN_API_KEY")
            if not api_key:
                logger.error("LANGCHAIN_API_KEY not found. Monitoring disabled")
                return
            
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_PROJECT"] = self.project_name

            self.client = Client(api_key=api_key)
            self.tracer = LangChainTracer(project_name=self.project_name, client=self.client)

            logger.info(f"LangSmith monitoring initialized: {self.project_name}")

        except Exception as e:
            logger.error(f"Failed to initialized LangSmith: {e}")
            self.client = None

    @asynccontextmanager
    async def create_story_session(
        self,
        child_profile: ChildProfile,
        story_theme: Optional[str] = None,
        metadata: Dict[str, Any] = None
    ):
        """Create comprehensive story generation session"""

        # Generate unique session ID
        session_id = f"story_{uuid.uuid4().hex[:8]}_{datetime.now().strftime("%Y%m%d_%H%M%S")}"

        # Initialize session data
        session_data = {
            "session_id": session_id,
            "child_profile": asdict(child_profile),
            "story_theme": story_theme,
            "created_at": datetime.now().isoformat(),
            "current_phase": StoryGenerationPhase.CONVERSATION_START,
            "phases_completed": [],
            "metadata": metadata or {}
        }

        self.session_metrics[session_id] = SessionMetrics()
        self.active_sessions[session_id] = session_data

        # Create LangSmith session
        langsmith_session = None
        if self.client:
            try:
                langsmith_session = await asyncio.to_thread(
                    self.client.create_session,
                    session_name=f"Story_{child_profile.name}_{session_id[:8]}",
                    description=f"Story generation for {child_profile.name} (age {child_profile.age})"
                )

                await asyncio.to_thread(
                    self.client.create_run,
                    session_id=str(langsmith_session.id),
                    name="story_session_start",
                    run_type="chain",
                    inputs={
                        "child_profile": asdict(child_profile),
                        "story_theme": story_theme
                    },
                    tags=["session_start", "jjongal_system", f"age_{child_profile.age}"]
                )

            except Exception as e:
                logger.error(f"Failed to create LangSmith session: {e}")
            
            session_context = {
                "session_id": session_id,
                "langsmith_session_id": str(langsmith_session.id) if langsmith_session else None,
                "tracer": self.tracer,
                "child_profile": child_profile,
                "start_time": time.time()
            }

            try:
                yield session_context
            except Exception as e:
                await self._log_session_error(session_id, str(e))
                raise
            finally:
                await self._finalize_session(session_id, session_context)


    async def log_chatbot_a_interaction(
        self,
        session_id: str,
        user_input: str,
        ai_response: str,
        extracted_elements: List[Dict[str, Any]],
        processing_time: float,
        turn_number: int):
        """Log ChatBot A (Jjongi) interaction"""

        if session_id not in self.session_metrics:
            logger.warning(f"Session {session_id} not found for ChatBot A logging")
            return
        
        # Update session metrics
        metrics = self.session_metrics[session_id]
        metrics.total_interactions += 1
        metrics.total_processing_time += processing_time
        metrics.avg_processing_time = metrics.total_processing_time / metrics.total_interactions
        metrics.story_elements_collected += len(extracted_elements)

        # Quality analysis
        quality_score = 0.0
        if self.enable_quality_analysis:
            quality_score = await self._analyze_interaction_quality(
                session_id,
                user_input,
                ai_response,
                extracted_elements
            )
            metrics.quality_scores.append(quality_score)
        
        # Log to LangSmith
        if self.client and session_id in self.active_sessions:
            try:
                langsmith_session_id = self.active_sessions[session_id].get("langsmith_session_id")
                if langsmith_session_id:
                    await asyncio.to_thread(
                        self.client.create.run,
                        session_id=langsmith_session_id,
                        name=f"chatbot_a_turn_{turn_number}",
                        run_type="llm",
                        inputs={
                            "user_input": user_input,
                            "turn_number": turn_number
                        },
                        outputs={
                            "ai_response": ai_response,
                            "extracted_elements": extracted_elements,
                            "processiong_time_ms": processing_time * 1000,
                            "quality_score": quality_score
                        },
                        tags=["chatbot_a", "story_collection", f"turn_{turn_number}"]
                    )

            except Exception as e:
                logger.error(f"Failed to log ChatBot A interaction: {e}")

    async def log_chatbot_b_generation(
        self,
        session_id: str,
        story_outline: Dict[str, Any],
        generated_story: Dict[str, Any],
        generation_metrics: Dict[str, Any],
        multimedia_assets: Dict[str, Any] = None):
        """Log ChatBot B (Ari) story generation"""

        if session_id not in self.session_metrics:
            logger.warning(f"Session {session_id} not found for ChatBot B logging")
            return
        
        # Calculate story statistics
        story_stats = self._calculate_story_statistics(generated_story)

        # Quality assessment
        story_quality = 0.0
        if self.enable_quality_analysis:
            story_quality = await self._analyze_story_quality(session_id, generated_story)

        # Log to LangSmith
        if self.client and session_id in self.active_sessions:
            try:
                langsmith_session_id = self.active_sessions[session_id].get("langsmith_session_id")
                if langsmith_session_id:
                    await asyncio.to_thread(
                        self.client.create_run,
                        session_id=langsmith_session_id,
                        name="chatbot_b_story_generation",
                        run_type="chain",
                        inputs={
                            "story_outline", story_outline
                        },
                        outputs={
                            "story_statistics": story_stats,
                            "generation_metrics": generation_metrics,
                            "multimedia_assets": multimedia_assets or {},
                            "story_quality_assets": story_quality
                        },
                        tags=["chatbot_b", "story_generation", "multimedia"]
                    )
            except Exception as e:
                logger.error(f"Failed to log ChatBot B generation: {e}")

    async def log_rag_retrieval(
        self,
        session_id: str,
        query: str,
        retrieved_docs: List[Dict[str, Any]],
        similarity_scores: List[float],
        processing_time: float):
        """Log RAG system retrieval"""

        if session_id not in self.session_metrics:
