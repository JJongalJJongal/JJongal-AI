"""
Story Generation Core for JJongal-AI

Core story generation functionality with templates and AI integration.
"""

import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime
from src.shared.utils.logging import get_module_logger

logger = get_module_logger(__name__)

class StoryGenerator:
    """Core story generation system"""
    
    def __init__(self):
        """Initialize story generator"""
        self.default_templates = {
            "adventure": {
                "opening": "옛날 옛적에 {character}가 살고 있었습니다.",
                "development": "{character}는 {setting}에서 모험을 시작했습니다.",
                "climax": "그런데 {conflict}가 일어났습니다!",
                "resolution": "{character}는 {solution}으로 문제를 해결했습니다.",
                "ending": "그래서 모든 사람들이 행복하게 살았답니다."
            }
        }
        logger.info("Story Generator initialized")
    
    async def generate_story(
        self, 
        story_elements: Dict[str, Any], 
        genre: str = "adventure",
        child_age: int = 6
    ) -> Dict[str, Any]:
        """
        Generate complete story from elements
        
        Args:
            story_elements: Collected story elements
            genre: Story genre
            child_age: Child's age for appropriate content
            
        Returns:
            Dict: Generated story data
        """
        try:
            template = self.default_templates.get(genre, self.default_templates["adventure"])
            
            # Fill template with story elements
            story_parts = {}
            for part, template_text in template.items():
                story_parts[part] = self._fill_template(template_text, story_elements)
            
            # Create complete story
            complete_story = f"""
            {story_parts['opening']}
            
            {story_parts['development']}
            
            {story_parts['climax']}
            
            {story_parts['resolution']}
            
            {story_parts['ending']}
            """
            
            return {
                "story_id": f"story_{int(datetime.now().timestamp())}",
                "title": f"{story_elements.get('character', '주인공')}의 모험",
                "content": complete_story.strip(),
                "genre": genre,
                "age_group": "YOUNG" if child_age <= 6 else "OLD",
                "created_at": datetime.now(),
                "story_elements": story_elements
            }
            
        except Exception as e:
            logger.error(f"Story generation failed: {e}")
            raise
    
    def _fill_template(self, template: str, elements: Dict[str, Any]) -> str:
        """Fill template with story elements"""
        filled = template
        
        # Default values
        defaults = {
            "character": "용감한 주인공",
            "setting": "마법의 숲",
            "conflict": "어려운 문제",
            "solution": "지혜로운 방법"
        }
        
        # Fill with provided elements or defaults
        for key, default in defaults.items():
            value = elements.get(key, default)
            if isinstance(value, list) and value:
                value = value[0]
            filled = filled.replace("{" + key + "}", str(value))
        
        return filled

_story_generator = None

def get_story_generator() -> StoryGenerator:
    """Get global story generator instance"""
    global _story_generator
    if _story_generator is None:
        _story_generator = StoryGenerator()
    return _story_generator