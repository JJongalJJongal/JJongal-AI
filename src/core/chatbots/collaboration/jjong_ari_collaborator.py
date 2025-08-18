

"""
Modern Jjong-Ari Collaboration System - 2025 LangChain Best Practices

Streamlined collaboration workflow between ChatBot A (쫑이) and ChatBot B (아리)
for complete fairy tale generation using modern LangChain patterns.
"""

import asyncio
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, Callable

# 2025 LangChain imports
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import JsonOutputParser

from src.shared.utils.logging import get_module_logger
from ..chat_bot_a.chat_bot_a import ChatBotA
from ..chat_bot_b.chat_bot_b import ModernAri

logger = get_module_logger(__name__)

class ModernJjongAriCollaborator:
    """Modern Jjong-Ari Collaboration System using 2025 LangChain patterns"""
    
    def __init__(self, output_dir: str = "output"):
        self.output_dir = output_dir
        logger.info("Modern JjongAri Collaborator initialized")
        
    async def collaborate_story_creation(
        self,
        jjong_instance: ChatBotA,
        session_id: str,
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Modern story collaboration workflow between 쫑이 and 아리
        
        Args:
            jjong_instance: ChatBot A instance (쫑이)
            session_id: Active conversation session ID
            progress_callback: Optional progress callback
            
        Returns:
            Complete story generation result
        """
        story_id = str(uuid.uuid4())
        start_time = datetime.now()
        
        try:
            logger.info(f"Starting modern Jjong-Ari collaboration: {story_id}")
            
            # Progress callback
            if progress_callback:
                await progress_callback({"step": "collaboration_start", "story_id": story_id})
            
            # Phase 1: 쫑이 prepares story outline for 아리
            if progress_callback:
                await progress_callback({"step": "story_outline_preparation", "status": "starting"})
            
            story_outline = await jjong_instance.get_story_outline_for_chatbot_b(session_id)
            
            # Phase 2: Initialize 아리 and prepare StoryRequest
            if progress_callback:
                await progress_callback({"step": "ari_initialization", "status": "starting"})
            
            ari_instance = ModernAri(output_dir=self.output_dir)
            
            # Convert story outline to API v2.0 StoryRequest format
            story_request = {
                "child_name": story_outline["child_profile"]["name"],
                "age": story_outline["child_profile"]["age"],
                "interests": story_outline["child_profile"]["interests"],
                "conversation_summary": story_outline.get("conversation_summary", ""),
                "story_elements": story_outline["story_elements"],
                "voice_config": {
                    "narrator_voice": "default"
                }
            }
            
            # Phase 3: 아리 generates complete story using API v2.0 method
            if progress_callback:
                await progress_callback({"step": "story_generation", "status": "starting"})
            
            story_result = await ari_instance.generate_story_from_request(story_request, progress_callback)
            
            # Combine results
            collaboration_result = {
                "story_id": story_id,
                "jjong_session_id": session_id,
                "story_outline": story_outline,
                "generated_story": story_result,
                "collaboration_metadata": {
                    "start_time": start_time.isoformat(),
                    "completion_time": datetime.now().isoformat(),
                    "total_duration_seconds": (datetime.now() - start_time).total_seconds()
                }
            }
            
            logger.info(f"Modern collaboration completed: {story_id}")
            return collaboration_result
            
        except Exception as e:
            logger.error(f"Collaboration failed for {story_id}: {e}")
            raise
    
    async def create_story_from_request(self, story_request: Dict[str, Any], progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Create story directly from API v2.0 StoryRequest (bypass ChatBot A)
        
        Args:
            story_request: StoryRequest format from API v2.0
            progress_callback: Optional progress callback
            
        Returns:
            API v2.0 compliant Story format
        """
        try:
            logger.info("Creating story directly from API v2.0 StoryRequest")
            
            if progress_callback:
                await progress_callback({"step": "direct_story_creation", "status": "starting"})
            
            # Initialize Modern Ari
            ari_instance = ModernAri(output_dir=self.output_dir)
            
            # Generate story using API v2.0 method
            story_result = await ari_instance.generate_story_from_request(story_request, progress_callback)
            
            logger.info(f"Direct story creation completed: {story_result['story_id']}")
            return story_result
            
        except Exception as e:
            logger.error(f"Direct story creation failed: {e}")
            raise
    
    async def get_collaboration_status(self, story_id: str) -> Dict[str, Any]:
        """Get collaboration status for a story"""
        # Simple status check - could be extended with actual storage
        return {
            "story_id": story_id,
            "status": "completed",  # This would be dynamic in real implementation
            "last_updated": datetime.now().isoformat()
        }

# Backward compatibility
JjongAriCollaborator = ModernJjongAriCollaborator

def create_collaborator(output_dir: str = "output") -> ModernJjongAriCollaborator:
    """Factory function for creating collaborator"""
    return ModernJjongAriCollaborator(output_dir=output_dir)