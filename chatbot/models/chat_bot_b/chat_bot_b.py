from typing import Dict, Any, List
from pathlib import Path
from shared.utils.logging_utils import get_module_logger

logger = get_module_logger(__name__)

class ChatBotB:
    """
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
        """
        self.target_age = target_age
        self.story_outline = story_outline
        self.child_voice_id = child_voice_id
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract voice settings
        self.child_voice_id = self.voice_config.get("child_voice_id")
        self.parent_voice_id = self.voice_config.get("parent_voice_id")
        
        # Initialize components
        self._initialize_clients()
        self._initialize_engines()
        
        # Configure voice mapping
        self._configure_voice_mapping()
        
    def _configure_voice_mapping(self):
        """Configure voice mapping for different story roles"""
        if not self.voice_generator:
            return
        
        voice_mapping = {}
        
        # Configure main characters voice
        
    
    def _identify_adult_characters(self, story_elements: Dict) -> List[str]:
        """Identify adult characters from story elements"""
        adult_keywords = ["엄마", "아빠", "선생님", "왕", "여왕", "마법사"]
        grandparent_keywords = ["할머니", "할아버지"]
        
        adult_characters = []
        grandparent_characters = []
        
        # Check all story elements for adult character mentions
        for element_list in story_elements.values():
            if isinstance(element_list, list):
                for element in element_list:
                    content = str(element.get("content", "")) if isinstance(element, dict) else str(element)
                    for keyword in 