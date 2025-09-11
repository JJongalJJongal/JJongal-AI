"""
Voice Options Manager for ChatBot A (쫑이)
Manages multiple voice options for 쫑이's speech synthesis
"""

from typing import Dict, List, Optional, Any
from enum import Enum
from dataclasses import dataclass
from src.shared.utils.logging import get_module_logger
from src.shared.configs.settings import get_settings

logger = get_module_logger(__name__)

class JjongiVoiceType(Enum):
    """Available voice types for 쫑이"""
    CHEERFUL = "cheerful"      # 밝고 활발한 목소리
    GENTLE = "gentle"          # 부드럽고 따뜻한 목소리  
    PLAYFUL = "playful"        # 장난기 있고 재미있는 목소리

@dataclass
class VoiceOption:
    """Voice option configuration"""
    voice_id: str
    voice_type: JjongiVoiceType
    display_name: str
    description: str
    voice_settings: Optional[Dict[str, Any]] = None
    is_active: bool = True

class JjongiVoiceManager:
    """
    Voice Options Manager for 쫑이 Chatbot
    
    Features:
    - Multiple pre-configured voice options
    - Dynamic voice switching during conversation
    - Voice quality optimization for children
    - ElevenLabs API integration
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.voice_options = self._initialize_voice_options()
        self.current_voice = JjongiVoiceType.CHEERFUL  # Default voice
        
        logger.info(f"JjongiVoiceManager initialized with {len(self.voice_options)} voice options")
    
    def _initialize_voice_options(self) -> Dict[JjongiVoiceType, VoiceOption]:
        """Initialize voice options with ElevenLabs Voice IDs"""
        
        # 이 부분은 사용자가 제공할 3개의 ElevenLabs Voice ID로 업데이트됩니다
        voice_options = {
            JjongiVoiceType.CHEERFUL: VoiceOption(
                voice_id="7KVKb7yXXL7DeiHwlzUv",  
                voice_type=JjongiVoiceType.CHEERFUL,
                display_name="밝은 쫑이",
                description="활발하고 밝은 목소리로 아이들과 즐겁게 대화함",
                voice_settings={
                    "stability": 0.5,  # eleven_v3: 0.0=Creative, 0.5=Natural, 1.0=Robust
                    "similarity_boost": 0.85,
                    "style": 0.2,
                    "use_speaker_boost": True
                }
            ),
            JjongiVoiceType.GENTLE: VoiceOption(
                voice_id="RR3x4e4e5I5OzZ4sFPTY",  
                voice_type=JjongiVoiceType.GENTLE,
                display_name="따뜻한 쫑이",
                description="부드럽고 차분한 목소리로 아이들을 편안하게 해줌",
                voice_settings={
                    "stability": 1.0,  # eleven_v3: Robust for gentle voice
                    "similarity_boost": 0.80,
                    "style": 0.1,
                    "use_speaker_boost": False
                }
            ),
            JjongiVoiceType.PLAYFUL: VoiceOption(
                voice_id="PJywxc95pczekscvFWGI",  
                voice_type=JjongiVoiceType.PLAYFUL,
                display_name="장난꾸러기 쫑이",
                description="재미있고 장난기 넘치는 목소리로 아이들과 놀아줌",
                voice_settings={
                    "stability": 0.0,  # eleven_v3: Creative for playful voice
                    "similarity_boost": 0.90,
                    "style": 0.4,
                    "use_speaker_boost": True
                }
            )
        }
        
        return voice_options
    
    def get_voice_options(self) -> List[Dict[str, Any]]:
        """Get list of available voice options for API response"""
        options = []
        for voice_type, option in self.voice_options.items():
            if option.is_active:
                options.append({
                    "voice_type": voice_type.value,
                    "display_name": option.display_name,
                    "description": option.description,
                    "is_current": voice_type == self.current_voice
                })
        return options
    
    def set_voice_option(self, voice_type: str) -> bool:
        """Set current voice option"""
        try:
            voice_enum = JjongiVoiceType(voice_type)
            if voice_enum in self.voice_options:
                old_voice = self.current_voice
                self.current_voice = voice_enum
                logger.info(f"Voice changed from {old_voice.value} to {voice_enum.value}")
                return True
            else:
                logger.warning(f"Voice option not found: {voice_type}")
                return False
        except ValueError:
            logger.error(f"Invalid voice type: {voice_type}")
            return False
    
    def get_current_voice_config(self) -> Dict[str, Any]:
        """Get current voice configuration for TTS"""
        option = self.voice_options[self.current_voice]
        return {
            "voice_id": option.voice_id,
            "voice_settings": option.voice_settings,
            "voice_type": option.voice_type.value,
            "display_name": option.display_name
        }
    
    def update_voice_ids(self, voice_id_mapping: Dict[str, str]) -> None:
        """Update ElevenLabs Voice IDs (사용자가 제공한 실제 Voice ID로 업데이트)"""
        updated_count = 0
        
        for voice_type_str, voice_id in voice_id_mapping.items():
            try:
                voice_type = JjongiVoiceType(voice_type_str)
                if voice_type in self.voice_options:
                    old_id = self.voice_options[voice_type].voice_id
                    self.voice_options[voice_type].voice_id = voice_id
                    logger.info(f"Updated {voice_type.value} voice ID: {old_id[:8]}... → {voice_id[:8]}...")
                    updated_count += 1
            except ValueError:
                logger.error(f"Invalid voice type in mapping: {voice_type_str}")
        
        logger.info(f"Updated {updated_count} voice IDs")
    
    def validate_voice_ids(self) -> Dict[str, bool]:
        """Validate that all voice IDs are properly configured"""
        validation_results = {}
        
        for voice_type, option in self.voice_options.items():
            is_placeholder = option.voice_id.startswith("PLACEHOLDER_")
            is_valid = not is_placeholder and len(option.voice_id) > 10  # ElevenLabs IDs are typically longer
            
            validation_results[voice_type.value] = is_valid
            
            if is_placeholder:
                logger.warning(f"{voice_type.value} voice still uses placeholder ID")
            elif not is_valid:
                logger.warning(f"{voice_type.value} voice ID appears invalid: {option.voice_id}")
        
        return validation_results
    
    def get_voice_config_for_conversation(self, preference: Optional[str] = None) -> Dict[str, Any]:
        """Get voice configuration optimized for conversation context"""
        
        # Use preference if provided and valid
        if preference:
            try:
                preferred_voice = JjongiVoiceType(preference)
                if preferred_voice in self.voice_options:
                    self.current_voice = preferred_voice
            except ValueError:
                logger.warning(f"Invalid voice preference: {preference}, using current voice")
        
        config = self.get_current_voice_config()
        
        # Add conversation-specific optimizations
        config["conversation_optimizations"] = {
            "speaking_rate": 0.95,  # Slightly slower for clarity
            "pause_duration": 0.8,  # Appropriate pauses
            "emotion_intensity": 0.7,  # Moderate emotion for consistency
            "child_friendly": True
        }
        
        return config

# Global instance for easy access
jjongi_voice_manager = JjongiVoiceManager()