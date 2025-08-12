<<<<<<< Updated upstream
from typing import Dict, Any, List
=======
<<<<<<< HEAD
"""
Ari (ChatBot B) - Unified Fairy Tale Generation Chatbot

Main class for generating complete multimedia fairy tales based on story elements 
collected from Jjongi (ChatBot A)
- Advanced prompt system (v3.0) applied
- Age-specific generation (4-7 years, 8-9 years)
- Performance tracking and optimization
"""

from shared.utils.logging_utils import get_module_logger
import os
import time
from typing import Dict, Any, Optional, Callable
=======
from typing import Dict, Any, List
>>>>>>> develop
>>>>>>> Stashed changes
from pathlib import Path
from shared.utils.logging_utils import get_module_logger

logger = get_module_logger(__name__)

class ChatBotB:
    """
<<<<<<< Updated upstream
    ChatBot B (Ari) - story generation chatbot
=======
<<<<<<< HEAD
    Ari - Unified Fairy Tale Generation Chatbot Main Class
    
    Based on story elements collected from Jjongi:
    1. Generate detailed story text (age-specific)
    2. Generate chapter-wise images (DALL-E 3)
    3. Generate character voices (ElevenLabs with voice cloning)
    4. Create complete multimedia fairy tales
>>>>>>> Stashed changes
    
    Features:
    - Constructor-based initialization
    - Age-specific story generation (4-7, 8-9)
    - Automatic content type optimization
    - Performance tracking and optimization
    """
    
    def __init__(self,
                 target_age: int = None,
                 story_outline: Dict[str, Any] = None,
                 child_voice_id: str = None,
                 output_dir: str = "output"):
        """
        Initialize ChatBot B
        
        Args:
            target_age: Child's age (4-9)
            story_outline: Story outline from ChatBot A
            child_voice_id: Optional cloned voice ID
            output_dir: Output directory path
<<<<<<< Updated upstream
        """
=======
            vector_db_path: ChromaDB vector database path
            collection_name: ChromaDB collection name
            enable_performance_tracking: Enable performance tracking
=======
    ChatBot B (Ari) - story generation chatbot
    
    Features:
    - Constructor-based initialization
    - Age-specific story generation (4-7, 8-9)
    - Automatic content type optimization
    - Performance tracking and optimization
    """
    
    def __init__(self,
                 target_age: int = None,
                 story_outline: Dict[str, Any] = None,
                 child_voice_id: str = None,
                 output_dir: str = "output"):
        """
        Initialize ChatBot B
        
        Args:
            target_age: Child's age (4-9)
            story_outline: Story outline from ChatBot A
            child_voice_id: Optional cloned voice ID
            output_dir: Output directory path
>>>>>>> develop
        """
>>>>>>> Stashed changes
        self.target_age = target_age
        self.story_outline = story_outline
        self.child_voice_id = child_voice_id
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
<<<<<<< Updated upstream
        # Extract voice settings
        self.child_voice_id = self.voice_config.get("child_voice_id")
        self.parent_voice_id = self.voice_config.get("parent_voice_id")
=======
<<<<<<< HEAD
        self.enable_performance_tracking = enable_performance_tracking
        
        # Set unified vector DB path
        if vector_db_path is None:
            import os
            chroma_base = os.getenv("CHROMA_DB_PATH", "/app/chatbot/data/vector_db")
            vector_db_path = os.path.join(chroma_base, "main")  # Default: use main DB
            logger.info(f"Vector DB path not specified. Setting from env var: {vector_db_path}")
        
        self.vector_db_path = vector_db_path
        self.collection_name = collection_name
        
        # Client initialization
        self.openai_client = None
        self.elevenlabs_api_key = None
        
        # Story settings
        self.target_age = None
        self.story_outline = None
        
        # Core engines
        self.story_engine = None
        self.content_pipeline = None
        
        # Generators
        self.text_generator = None
        self.image_generator = None
        self.voice_generator = None
        
        # Performance metrics
        self.performance_metrics = {
            "total_stories_generated": 0,
            "successful_generations": 0,
            "average_generation_time": 0,
            "age_group_statistics": {}
        }
>>>>>>> Stashed changes
        
=======
        # Extract voice settings
        self.child_voice_id = self.voice_config.get("child_voice_id")
        self.parent_voice_id = self.voice_config.get("parent_voice_id")
        
>>>>>>> develop
        # Initialize components
        self._initialize_clients()
        self._initialize_engines()
<<<<<<< Updated upstream
        
        # Configure voice mapping
        self._configure_voice_mapping()
        
    def _configure_voice_mapping(self):
        """Configure voice mapping for different story roles"""
        if not self.voice_generator:
            return
        
        voice_mapping = {}
        
        # Configure main characters voice
        
=======
        
        # Configure voice mapping
        self._configure_voice_mapping()
        
    def _configure_voice_mapping(self):
        """Configure voice mapping for different story roles"""
        if not self.voice_generator:
            return
        
        voice_mapping = {}
        
        # Configure main characters voice
        
<<<<<<< HEAD
    def _initialize_clients(self):
        """Initialize API clients"""
        try:
            # Initialize OpenAI client
            self.openai_client = initialize_client()
            
            # Load ElevenLabs API key
            raw_key = os.getenv("ELEVENLABS_API_KEY")
            if raw_key:
                logger.info(f"ElevenLabs API key loaded successfully (length: {len(raw_key)})")
            else:
                logger.warning("ElevenLabs API key not set in environment variables")
            
            self.elevenlabs_api_key = raw_key
            
            logger.info("API clients initialization completed")
            
        except Exception as e:
            logger.error(f"API clients initialization failed: {e}")
            raise
            
    def _initialize_engines(self, vector_db_path: str, collection_name: str):
        """Initialize engines and generators"""
        try:
            # Set unified temp path
            base_temp_path = self.output_dir / "temp"
            base_temp_path.mkdir(parents=True, exist_ok=True)
            
            # 1. Initialize generators
            self.text_generator = TextGenerator(
                openai_client=self.openai_client,
                vector_db_path=vector_db_path,
                collection_name=collection_name,
                enable_performance_tracking=self.enable_performance_tracking
            )
            
            self.image_generator = ImageGenerator( 
                openai_client=self.openai_client,
                model_name="dall-e-3",
                temp_storage_path=str(base_temp_path / "images"),
                enable_performance_tracking=self.enable_performance_tracking
            )
            
            self.voice_generator = VoiceGenerator(
                elevenlabs_api_key=self.elevenlabs_api_key,
                temp_storage_path=str(base_temp_path / "audio"),
                voice_id="xi3rF0t7dg7uN2M0WUhr",  # Yuna (default narrator voice)
                model_id="eleven_multilingual_v2",  # Default model ID (Korean support)
                voice_settings=None,  # Voice settings (stability, similarity_boost, style, use_speaker_boost)
                max_retries=3,  # Maximum retry count
                enable_chunking=True,  # Enable text chunking (prevent large audio files)
                max_chunk_length=500  # Maximum chunk length (character count)
            )
            
            # 2. Initialize story generation engine
            self.story_engine = StoryGenerationEngine(
                openai_client=self.openai_client,
                elevenlabs_client=None,
                output_dir=str(self.output_dir)  # Pass basic output path only (excluding temp)
            )
            
            # Inject generators into engine
            self.story_engine.set_generators(
                text_generator=self.text_generator,
                image_generator=self.image_generator,
                voice_generator=self.voice_generator,
                rag_enhancer=None
            )
            
            # 3. Initialize content pipeline
            self.content_pipeline = ContentPipeline(
                openai_client=self.openai_client,
                vector_db_path=vector_db_path,
                collection_name=collection_name
            )
            
            logger.info(f"Engines and generators initialization completed (unified temp path: {base_temp_path})")
            
        except Exception as e:
            logger.error(f"Engine initialization failed: {e}")
            raise
>>>>>>> Stashed changes
    
    def _identify_adult_characters(self, story_elements: Dict) -> List[str]:
        """Identify adult characters from story elements"""
        adult_keywords = ["엄마", "아빠", "선생님", "왕", "여왕", "마법사"]
        grandparent_keywords = ["할머니", "할아버지"]
        
        adult_characters = []
        grandparent_characters = []
        
<<<<<<< Updated upstream
=======
        Args:
            use_websocket_voice: Enable WebSocket voice streaming (default True)
            progress_callback: Progress callback function
            
        Returns:
            Dict: Generated story data with performance metrics
        """
        if not self.story_outline:
            raise ValueError("Story outline not set. Call set_story_outline() first.")
        
        if not self.target_age:
            raise ValueError("Target age not set. Call set_target_age() first.")
        
        start_time = time.time()
        
        result = {}
        try:
            # Start performance tracking
            self.performance_metrics["total_stories_generated"] += 1
            
            # Update age group statistics
            age_group_key = self._get_age_group_key(self.target_age)
            if age_group_key not in self.performance_metrics["age_group_statistics"]:
                self.performance_metrics["age_group_statistics"][age_group_key] = 0
            self.performance_metrics["age_group_statistics"][age_group_key] += 1
        
            # Add age information to story outline
            enhanced_outline = {
                **self.story_outline,
                "age_group": self.target_age,
                "target_age": self.target_age,
                "websocket_voice": use_websocket_voice
            }
            
            # Generate story with current generators
            result = await self._generate_story(
                enhanced_outline,
                use_websocket_voice,
                progress_callback
            )
        
            # Update performance metrics
            generation_time = time.time() - start_time
            self.performance_metrics["successful_generations"] += 1
            self._update_average_generation_time(generation_time)
            
            # Add metadata to result
            result["metadata"] = result.get("metadata", {})
            result["metadata"].update({
                "websocket_voice": use_websocket_voice,
                "generation_time": generation_time,
                "prompt_version": "3.0_unified",
                "age_group": age_group_key
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Story generation failed: {e}")
            raise

    async def _generate_story(self, 
                            enhanced_outline: Dict[str, Any],
                            use_websocket_voice: bool,
                            progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Generate complete story with all generators"""
        
        # 1. Text generation
        if progress_callback:
            await progress_callback({
                "step": "story_generation",
                "status": "starting",
                "phase": "text"
            })
        
        story_data = await self.text_generator.generate(
            enhanced_outline, progress_callback
        )
        
        # 2. Image generation
        if progress_callback:
            await progress_callback({
                "step": "image_generation",
                "status": "starting",
                "chapters": len(story_data.get("chapters", []))
            })
        
        # Pass original analysis data from ChatBot A to image generator
        image_input_data = {
            "story_data": story_data,
            "story_id": story_data.get("story_id"),
            # Additional analysis data from ChatBot A
            "conversation_summary": enhanced_outline.get("conversation_summary", ""),
            "extracted_keywords": enhanced_outline.get("extracted_keywords", []),
            "conversation_analysis": enhanced_outline.get("conversation_analysis", {}),
            "child_profile": enhanced_outline.get("child_profile", {}),
            "story_generation_method": "unified"
        }
        
        image_data = await self.image_generator.generate(
            image_input_data, progress_callback
        )
        
        # 3. Voice generation (with WebSocket support)
        if progress_callback:
            await progress_callback({
                "step": "voice_generation",
                "status": "starting",
                "websocket_enabled": use_websocket_voice
            })
        
        # Pass original analysis data from ChatBot A to voice generator
        voice_input_data = {
            "story_data": story_data,
            "story_id": story_data.get("story_id"),
            # Additional analysis data from ChatBot A
            "conversation_summary": enhanced_outline.get("conversation_summary", ""),
            "extracted_keywords": enhanced_outline.get("extracted_keywords", []),
            "conversation_analysis": enhanced_outline.get("conversation_analysis", {}),
            "child_profile": enhanced_outline.get("child_profile", {}),
            "story_generation_method": "unified"
        }
        
        voice_data = await self.voice_generator.generate(
            voice_input_data, progress_callback, use_websocket=use_websocket_voice
        )
        
        # 4. Combine results
        return {
            "story_data": story_data,
            "image_paths": [img.get("image_path") for img in image_data.get("images", [])],
            "audio_paths": voice_data.get("audio_files", []),
            "story_id": story_data.get("story_id"),
            "status": "complete",
            "generator_metadata": {
                "text_metrics": self.text_generator.get_performance_metrics() if hasattr(self.text_generator, 'get_performance_metrics') else {},
                "image_metrics": self.image_generator.get_performance_metrics() if hasattr(self.image_generator, 'get_performance_metrics') else {}
            },
            "voice_metadata": voice_data.get("metadata", {})
        }
    
    
    async def generate_text_only(self, 
                                progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Generate text only (without images and audio)"""
        if not self.story_outline or not self.target_age:
            raise ValueError("Story outline and target age must be set first.")
        
        enhanced_outline = {
            **self.story_outline,
            "age_group": self.target_age,
            "target_age": self.target_age
        }
        
        story_data = await self.text_generator.generate(enhanced_outline, progress_callback)
        
        return {
            "story_data": story_data,
            "image_paths": [],
            "audio_paths": [],
            "story_id": story_data.get("story_id"),
            "status": "text_only",
            "metadata": {
                "prompt_version": "3.0_unified"
            }
        } 
    
    async def generate_with_pipeline(self, 
                                   progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Generate story through content pipeline"""
        if not self.story_outline or not self.target_age:
            raise ValueError("Story outline and target age must be set first.")
        
        enhanced_outline = {
            **self.story_outline,
            "age_group": self.target_age,
            "target_age": self.target_age
        }
        
        # Generate through content pipeline
        result = await self.content_pipeline.execute_pipeline(enhanced_outline, progress_callback=progress_callback)
        
        return result
    
    def _get_age_group_key(self, age: int) -> str:
        """Return age group key"""
        if 4 <= age <= 7:
            return "age_4_7"
        elif 8 <= age <= 9:
            return "age_8_9"
        else:
            return "age_other"
    
    def _update_average_generation_time(self, new_time: float):
        """Update average generation time"""
        current_avg = self.performance_metrics["average_generation_time"]
        successful_count = self.performance_metrics["successful_generations"]
        
        if successful_count == 1:
            self.performance_metrics["average_generation_time"] = new_time
        else:
            # Incremental average calculation
            self.performance_metrics["average_generation_time"] = (
                (current_avg * (successful_count - 1) + new_time) / successful_count
            )
    
    def get_generation_status(self) -> Dict[str, Any]:
        """Get generator status"""
        status = {
            "story_engine_status": "ready" if self.story_engine else "not_initialized",
            "text_generator_status": "ready" if self.text_generator else "not_initialized",
            "image_generator_status": "ready" if self.image_generator else "not_initialized",
            "voice_generator_status": "ready" if self.voice_generator else "not_initialized",
            "target_age_set": self.target_age is not None,
            "story_outline_set": self.story_outline is not None,
            "performance_metrics": self.performance_metrics
        }
        
        return status
    
    async def health_check(self) -> Dict[str, bool]:
        """Health check for all components"""
        health_status = {
            "openai_client": bool(self.openai_client),
            "elevenlabs_api_key": bool(self.elevenlabs_api_key),
            "text_generator": bool(self.text_generator),
            "image_generator": bool(self.image_generator),
            "voice_generator": bool(self.voice_generator),
            "story_engine": bool(self.story_engine),
            "content_pipeline": bool(self.content_pipeline)
        }
        
        # Individual generator health checks
        if self.text_generator and hasattr(self.text_generator, 'health_check'):
            text_health = await self.text_generator.health_check()
            health_status["text_generator_detailed"] = text_health
            
        if self.image_generator and hasattr(self.image_generator, 'health_check'):
            image_health = await self.image_generator.health_check()
            health_status["image_generator_detailed"] = image_health
        
        # Overall health status
        core_components = ["openai_client", "text_generator", "image_generator", "voice_generator"]
        health_status["overall_healthy"] = all([health_status[key] for key in core_components])
        
        return health_status
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        metrics = {
            "chatbot_metrics": self.performance_metrics
        }
        
        # Add generator metrics if available
        if self.text_generator and hasattr(self.text_generator, 'get_performance_metrics'):
            metrics["text_metrics"] = self.text_generator.get_performance_metrics()
        
        if self.image_generator and hasattr(self.image_generator, 'get_performance_metrics'):
            metrics["image_metrics"] = self.image_generator.get_performance_metrics()
        
        return metrics
    
    def cleanup(self):
        """Resource cleanup"""
        logger.info("ChatBot B resource cleanup started")
        
        # Cleanup generators
        if self.story_engine:
            self.story_engine = None
        
        if self.content_pipeline:
            self.content_pipeline = None
        
        if self.text_generator:
            self.text_generator = None
        
        if self.image_generator:
            self.image_generator = None
        
        if self.voice_generator:
            self.voice_generator = None
        
        logger.info("ChatBot B resource cleanup completed")
=======
    
    def _identify_adult_characters(self, story_elements: Dict) -> List[str]:
        """Identify adult characters from story elements"""
        adult_keywords = ["엄마", "아빠", "선생님", "왕", "여왕", "마법사"]
        grandparent_keywords = ["할머니", "할아버지"]
        
        adult_characters = []
        grandparent_characters = []
        
>>>>>>> Stashed changes
        # Check all story elements for adult character mentions
        for element_list in story_elements.values():
            if isinstance(element_list, list):
                for element in element_list:
                    content = str(element.get("content", "")) if isinstance(element, dict) else str(element)
<<<<<<< Updated upstream
                    for keyword in 
=======
                    for keyword in 
>>>>>>> develop
>>>>>>> Stashed changes
