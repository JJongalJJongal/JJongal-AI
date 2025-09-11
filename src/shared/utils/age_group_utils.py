"""
Age Group Management Utilities

Handles age-appropriate content and conversation management for different child age groups.
"""

from typing import Dict, List, Optional
from enum import Enum

class AgeGroup(Enum):
    """Age group classifications for content appropriate conversations"""
    AGE_4_TO_7 = "4-7"
    AGE_8_TO_9 = "8-9"
    UNKNOWN = "unknown"

class AgeGroupManager:
    """Manages age group specific settings and content filtering"""
    
    def __init__(self):
        self.age_group_settings = {
            AgeGroup.AGE_4_TO_7: {
                "vocabulary_level": "simple",
                "max_story_length": 500,
                "conversation_style": "very_friendly",
                "content_complexity": "basic",
                "interaction_prompts": [
                    "재미있는 이야기를 만들어볼까?",
                    "어떤 동물이 좋아?",
                    "무엇을 하고 싶어?"
                ]
            },
            AgeGroup.AGE_8_TO_9: {
                "vocabulary_level": "intermediate",
                "max_story_length": 800,
                "conversation_style": "friendly",
                "content_complexity": "moderate",
                "interaction_prompts": [
                    "어떤 모험 이야기가 좋을까?",
                    "주인공은 어떤 성격이면 좋겠어?",
                    "이야기에서 어떤 일이 일어났으면 좋겠어?"
                ]
            }
        }
    
    def classify_age_group(self, age: Optional[int]) -> AgeGroup:
        """Classify age into appropriate age group"""
        if age is None:
            return AgeGroup.UNKNOWN
        
        if 4 <= age <= 7:
            return AgeGroup.AGE_4_TO_7
        elif 8 <= age <= 9:
            return AgeGroup.AGE_8_TO_9
        else:
            return AgeGroup.UNKNOWN
    
    def get_settings(self, age_group: AgeGroup) -> Dict:
        """Get settings for specific age group"""
        return self.age_group_settings.get(age_group, self.age_group_settings[AgeGroup.AGE_4_TO_7])
    
    def get_vocabulary_level(self, age_group: AgeGroup) -> str:
        """Get vocabulary level for age group"""
        settings = self.get_settings(age_group)
        return settings.get("vocabulary_level", "simple")
    
    def get_max_story_length(self, age_group: AgeGroup) -> int:
        """Get maximum story length for age group"""
        settings = self.get_settings(age_group)
        return settings.get("max_story_length", 500)
    
    def get_conversation_style(self, age_group: AgeGroup) -> str:
        """Get conversation style for age group"""
        settings = self.get_settings(age_group)
        return settings.get("conversation_style", "very_friendly")
    
    def get_interaction_prompts(self, age_group: AgeGroup) -> List[str]:
        """Get age-appropriate interaction prompts"""
        settings = self.get_settings(age_group)
        return settings.get("interaction_prompts", [])
    
    def is_content_appropriate(self, content: str, age_group: AgeGroup) -> bool:
        """Check if content is appropriate for age group (basic implementation)"""
        # Basic content filtering - can be expanded
        inappropriate_words = ["scary", "무서운", "위험한", "violent"]
        
        if age_group == AgeGroup.AGE_4_TO_7:
            # More strict filtering for younger children
            return not any(word.lower() in content.lower() for word in inappropriate_words)
        
        return True  # More lenient for older children