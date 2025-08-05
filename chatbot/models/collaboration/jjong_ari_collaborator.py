"""
Jjong-Ari Collaboration System
Internal collaboration between Jjong and Ari for complete fairy tale generation
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from pathlib import Path

from shared.utils.logging_utils import get_module_logger
from .story_storage import TempStoryStorage

logger = get_module_logger(__name__)

class JjongAriCollaborator:
    """Jjong-Ari Collaboration System"""
    
    def __init__(self, temp_storage_dir: str = "output/temp_stories"):
        self.storage = TempStoryStorage(temp_storage_dir)
        
    async def collaborate_story_creation(
        self,
        jjong_instance, # ChatBot A Instance
        conversation_data: Dict[str, Any]
    ) -> str:
        """
        Generate story through Jjong-Ari collaboration
        
        Args:
            jjong_instance: ChatBot A Instance
            conversation_data: Conversation data with child
            
        Returns:
            story_id: Generated story ID
        """
        
        story_id = str(uuid.uuid4())
        
        try:
            logger.info(f"Starting Jjong-Ari collaboration for story: {story_id}")
            
            # 1. Jjong analyzes and organizes story elements
            story_elements = await self._jjong_analyze_conversation(jjong_instance, conversation_data)
            
            # 2. Jjong prepares story generation request for Ari
            ari_request = await self._jjong_prepare_ari_request(jjong_instance, story_elements)
            
            # 3. Ari generates complete multimedia fairy tale
            complete_story = await self._ari_generate_story(ari_request)
            
            # 4. Store result in temporary storage
            await self.storage.store_story(story_id, complete_story)
            
            logger.info(f"Jjong-Ari collaboration completed for story: {story_id}")
            return story_id

        except Exception as e:
            logger.error(f"Jjong-Ari collaboration completed for story: {story_id}: {e}")
            raise
    
    async def _jjong_analyze_conversation(
        self,
        jjong_instance,
        conversation_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Jjong analyzes conversation content and extracts story elements"""
        
        # Utilize Jjong's conversation analysis functionality
        if hasattr(jjong_instance, "extract_story_elements"):
            story_elements = jjong_instance.extract_story_elements()
        else:
            # Default analysis logic
            story_elements = {
                "main_characters": conversation_data.get("main_characters", ""),
                "setting": conversation_data.get("setting", ""),
                "theme": conversation_data.get("theme", "")
            }
            
        return {
            "conversation_summary": conversation_data.get("summary", ""),
            "story_elements": story_elements,
            "child_profile": conversation_data.get("child_profile", {}),
            "extracted_keywords": conversation_data.get("keywords", []),
            "emotional_tone": conversation_data.get("tone", ""),
            "interests": conversation_data.get("interests", [])
        }
        
    async def _jjong_prepare_ari_request(
        self,
        jjong_instance,
        story_elements: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Jjong prepares request data for Ari"""
        
        # Get voice cloning information
        voice_info = {}
        
        if hasattr(jjong_instance, "get_voice_cloning_info"):
            voice_info = jjong_instance.get_voice_cloning_info()
        
        return {
            **story_elements,
            "child_voice_id": voice_info.get("voice_id"),
            "main_character_name": story_elements.get("story_elements", {}).get("main_character", ""),
            "generation_preferences": {
                "use_enhanced": True,
                "use_websocket_voice""
            }
        }