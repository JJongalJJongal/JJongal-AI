import os
import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from elevenlabs.client import ElevenLabs
from elevenlabs import Voice, VoiceSettings
from shared.utils import get_module_logger

logger = get_module_logger(__name__)


class VoiceCloningProcessor:
    def __init__(self, sample_base_dir="output/temp/voice_samples"):
       """
       Initialize voice cloning processor
       
       Args:
           sample_base_dir: Base directory for storing voice samples
       """ 
       self.samples_base_dir = Path(sample_base_dir)
       self.samples_base_dir.mkdir(parents=True, exist_ok=True)
       
       # Initialize ElevenLabs client
       self.elevenlabs_client = self._initialize_elevenlabs_client()
       
       # User-specific voice sample and clone information management
       self.user_samples: Dict[str, List[str]] = {} # user_id: [sample_paths]
       self.user_voice_ids: Dict[str, str] = {} # user_id: voice_id
       self.clone_metadata: Dict[str, Dict] = {} # user_id: metadata
       
       # Configuration
       self.required_samples = 5
       self.max_sample_size = 10 * 1024 * 1024 # 10MB
       self.min_sample_size = 1000 # 1KB
       self.supported_formats = [".mp3", ".wav", ".m4a"]
       
       logger.info(f"VoiceCloningProcessor initialized - samples_dir: {self.samples_base_dir}")
    
    def _initialize_elevenlabs_client(self) -> Optional[ElevenLabs]:
        """Initialize ElevenLabs client"""
        try:
            api_key = os.getenv("ELEVENLABS_API_KEY")
            if not api_key:
                logger.error("ELEVENLABS_API_KEY environment variable not set")
                return None
        
            client = ElevenLabs(api_key=api_key)
            logger.info("ElevenLabs client initialized successfully")
            return client
        
        except Exception as e:
            logger.error(f"Failed to initialize ElevenLabs client: {e}")
            return None
        
    
    # Core methods
    async def collect_user_audio_sample(self, user_id : str, audio_data : bytes, file_format="mp3") -> bool:
        """
        Collect user voice sample
        
        Args:
            user_id: User identifier
            audio_data: Voice data (bytes)
            file_format: Audio file format
            
        Returns:
            bool: Collection success status
        """
        try:
            # Check file size
            if len(audio_data) > self.max_sample_size:
                logger.warning(f"Voice sample size exceeded: {len(audio_data)} bytes (max: {self.max_sample_size})")
                return False
            
            if len(audio_data) < self.min_sample_size:
                logger.warning(f"Voice sample size too small: {len(audio_data)} bytes (min: {self.min_sample_size})")
                return False
            
            # Create user-specific directory
            user_dir = self.samples_base_dir / user_id
            user_dir.mkdir(parents=True, exist_ok=True)
            
            # Create current sample count
            if user_id not in self.user_samples:
                self.user_samples[user_id] = []
                
            current_count = len(self.user_samples[user_id])
            if current_count >= self.required_samples:
                logger.info(f"User {user_id} already has sufficient samples ({current_count}/{self.required_samples})")
                return True
            
            # Save file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filename = f"sample_{current_count + 1}_{timestamp}.{file_format}"
            file_path = user_dir / filename
            
            with open(file_path, "wb") as f:
                f.write(audio_data)
                
            self.user_samples[user_id].append(str(file_path))
            
            logger.info(f"Voice sample saved: {file_path} ({len(audio_data)} bytes)")
            logger.info(f"User {user_id} sample progress:
                        {len(self.user_samples[user_id])}/{self.required_samples}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to collect user voice sample - user_id: {user_id}, error: {e}")
            return False
            
    async def create_instanct_voice_clone(self, user_id: str, voice_name: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Create instant voice clone
        
        Args:
            user_id: User identifier
            voice_name: Name for the voice to create
        
        Returns:
            Tuple[voice_id, error_message]: Generated voice ID and error message
        """
        try:
            if not self.elevenlabs_client:
                return None, "ElevenLabs client not initialized"
        
            if not self._is_ready_for_cloning(user_id):
                return None, f"Insufficient samples ({self.get_sample_count(user_id)}/{self.required_samples})"
            
            # Check if voice already exists
            if user_id in self.user_voice_ids:
                logger.info(f"Voice already exists for user {user_id}: {self.user_voice_ids[user_id]}")
                return self.user_voice_ids[user_id], None

            sample_paths = self.user_samples[user_id]
            
            # Read voice files and prepare
            audio_files = []
            for sample_path in sample_paths:
                if os.path.exists(sample_path):
                    with open(sample_path, "rb") as f:\
                        audio_files.append(f.read())
                else:
                    logger.warning(f"Sample file not found: {sample_path}")
                
            
            if len(audio_files) < self.required_samples:
                return None, f"Insufficient valid samples ({len(audio_files)}/{self.required_samples})"
            
            logger.info(f"Starting voice cloning - user_id: {user_id}, samples: {len(audio_files)}")
            
            # Create voice clone with ElevenLabs API
            voice = await asyncio.to_thread(
                self._create_voice_with_samples,
                voice_name,
                audio_files
            )
            
            if voice and hasattr(voice, "voice_id"):
                voice_id = voice.voice_id
                
                # Store generated voice ID
                self.user_voice_ids[user_id] = voice_id
                
                # Store metadata
                self.clone_metadata[user_id] = {
                    "voice_id": voice_id,
                    "voice_name": voice_name,
                    "created_at": datetime.now().isoformat(),
                    "sample_count": len(audio_files),
                    "sample_paths": sample_paths
                }
                
                # Save metadata to file
                await self._save_clone_metadata(user_id)
                
                logger.info(f"Voice cloning completed - user_id: {user_id}, voice_id: {voice_id}")
                return voice_id, None
            else:
                logger.error(f"Voice cloning failed - invalid response: {voice}")
                return None, "Voice cloning API response invalid"
            
        except Exception as e:
            error_msg = f"Error during voice cloning: {str(e)}"
            logger.error(f"{error_msg} - user_id: {user_id}")
            return None, error_msg
    
    def _create_voice_with_samples(self, voice_name: str, audio_files: List[bytes]) -> Any:
        """
        Create voice with ElevenLabs API (sync function)
        
        Args:
            voice_name: Voice name
            audio_files: List of voice file data
            
        Returns:
            Created voice object
        """    
        try:
            # Call ElevenLabs API
            voice = self.elevenlabs_client.voices.add(
                name=voice_name,
                files=audio_files,
                settings=VoiceSettings(
                    stability=0.75,
                    similarity_boost=0.75,
                    style=0.0,
                    use_speaker_boost=True
                )
            )
            return voice
        
        except Exception as e:
            logger.error(f"ElevenLabs voice creation API call failed: {e}")
            raise
            
    async def generate_cloned_speech(self, user_id: str, text: str, output_path: Optional[str] = None) -> Tuple[Optional[bytes], Optional[str]]:
        """
        Generate speech with cloned voice
        
        Args:
            user_id: User identifier
            text: Text to synthesize
            output_path: File path to save (optional)
            
        Returns:
            Tuple[audio_data, file_path]: Audio data and file path
        """
        
        try:
            voice_id = self.get_user_voice_id(user_id)
            if not voice_id:
                logger.warning(f"Cloned voice not found for user {user_id}")
                return None, None
            
            if not self.elevenlabs_client:
                logger.error("ElevenLabs client not initialized")
                return None, None
            
            # Generate speech
            audio = await asyncio.to_thread(
                self.elevenlabs_client.text_to_speech.convert,
                text=text,
                voice_id=voice_id,
                model_id="eleven_multilingual_v2",
                output_format="wav_44100"
            )
            
            # Save file if needed
            if output_path:
                if not os.path.isabs(output_path):
                    output_path = str(self.samples_base_dir / output_path)
                
                with open(output_path, "wb") as f:
                    f.write(audio)
                    
                logger.info(f"Cloned voice file saved: {output_path}")
                return audio, output_path
            
            return audio, None
        
        except Exception as e:
            logger.error(f"Cloned speech generation failed - user_id: {user_id}, error: {e}")
            return None, None
    
    # State methods
    def get_sample_count(self, user_id: str) -> int:
        """Get current sample count for user"""
        return len(self.user_samples.get(user_id, []))
    
    def is_ready_for_cloning(self, user_id: str) -> bool:
        """Check if ready for voice cloning"""
        return self.get_sample_count(user_id) >= self.required_samples
    
    def get_user_voice_id(self, user_id: str) -> Optional[str]:
        """Get generated voice ID for user"""
        sample_count = self.get_sample_count(user_id)
        voice_id = self.get_user_voice_id(user_id)
        metadata = self.clone_metadata.get(user_id, {})
        
        return {
            "user_id": user_id,
            "sample_count": sample_count,
            "required_samples": self.required_samples,
            "ready_for_cloning": self.is_ready_for_cloning(user_id),
            "has_cloned_voice": voice_id is not None,
            "voice_id": voice_id,
            "created_at": metadata.get("created_at"),
            "voice_name": metadata.get("voice_name")
        }
        
    # Management methods
    async def _save_clone_metadata(self, user_id: str):
        """Save voice clone metadata"""
        
        try:
            if user_id not in self.clone_metadata:
                return
            
            metadata_dir = self.samples_base_dir / user_id
            metadata_file = metadata_dir / "clone_metadata.json"
            
            with open(metadata_file, "w", encoding='utf-8') as f:
                json.dump(self.clone_metadata[user_id], f, ensure_ascii=False, indent=2)
                
            logger.info(f"Voice clone metadata saved: {metadata_file}")
        
        except Exception as e:
            logger.error(f"Failed to save metadata - user_id: {user_id}, error: {e}")
    
    async def cleanup_user_samples(self, user_id: str, keep_metadata: bool = True):
        """
        Clean up user sample files
        
        Args:
            user_id: User identifier
            keep_metadata: Whether to preserve metadata
        """
        try:
            user_dir = self.samples_base_dir / user_id
            if not user_dir.exists():
                return
            
            # Delete sample files
            if user_id in self.user_samples:
                for sample_path in self.user_samples[user_id]:
                    if os.path.exists(sample_path):
                        os.remove(sample_path)
                        logger.debug(f"Sample file deleted: {sample_path}")
                
                del self.user_samples[user_id]
                
            # Delete metadata if not keeping
            if not keep_metadata:
                if user_id in self.clone_metadata:
                    del self.clone_metadata[user_id]
                
                metadata_file = user_dir / "clone_metadata.json"
                if metadata_file.exists():
                    os.remove(metadata_file)
                
                if user_id in self.user_voice_ids:
                    del self.user_voice_ids[user_id]
            
            # Remove directory if empty
            if user_dir.exists() and not any(user_dir.iterdir()):
                user_dir.rmdir()
            
            logger.info(f"User sample cleanup completed - user_id: {user_id}")
            
        except Exception as e:
            logger.error(f"Sample cleanup failed - user_id: {user_id}, error: {e}")
        
    
        
    
    