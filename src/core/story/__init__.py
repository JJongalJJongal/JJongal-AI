"""
Story Generation Module for JJongal-AI

Story creation, template management, and content generation
for personalized children's fairy tales.
"""

from .generation import StoryGenerator, get_story_generator
from .templates import (
    StoryTemplateManager, 
    StoryTemplate,
    StoryGenre,
    AgeGroup,
    get_template_manager
)

__all__ = [
    "StoryGenerator",
    "get_story_generator",
    "StoryTemplateManager",
    "StoryTemplate", 
    "StoryGenre",
    "AgeGroup",
    "get_template_manager"
]

__version__ = "1.0.0"