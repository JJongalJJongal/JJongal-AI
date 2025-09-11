import os
import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from elevenlabs.client import ElevenLabs
from elevenlabs import Voice, VoiceSettings
from src.shared.utils.logging import get_module_logger

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
       
       # ⭐ NEW: Family voice management
       self.family_samples: Dict[str, Dict[str, List[str]]] = {} # user_id: {family_member: [sample_paths]}
       self.family_voice_ids: Dict[str, Dict[str, str]] = {} # user_id: {family_member: voice_id}
       self.family_metadata: Dict[str, Dict[str, Dict]] = {} # user_id: {family_member: metadata}
       self.supported_family_members = ["mom", "dad"] # 지원되는 가족 구성원
       self.max_voices_per_family_member = 2 # 가족 구성원당 최대 음성 수
       
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
        
    
    # ⭐ NEW: Family voice management methods
    async def collect_family_voice_sample(self, user_id: str, family_member: str, voice_name: str, audio_data: bytes, file_format="mp3") -> Dict[str, Any]:
        """
        Collect family member voice sample
        
        Args:
            user_id: User identifier
            family_member: Family member type ("mom" | "dad")
            voice_name: Custom voice name
            audio_data: Voice data (bytes)
            file_format: Audio file format
            
        Returns:
            Dict with collection result and progress info
        """
        try:
            # Validate family member
            if family_member not in self.supported_family_members:
                return {
                    "success": False,
                    "error": f"Unsupported family member: {family_member}. Supported: {self.supported_family_members}"
                }
            
            # Check file size
            if len(audio_data) > self.max_sample_size:
                return {
                    "success": False,
                    "error": f"Voice sample too large: {len(audio_data)} bytes (max: {self.max_sample_size})"
                }
            
            if len(audio_data) < self.min_sample_size:
                return {
                    "success": False,
                    "error": f"Voice sample too small: {len(audio_data)} bytes (min: {self.min_sample_size})"
                }
            
            # Initialize user's family structure
            if user_id not in self.family_samples:
                self.family_samples[user_id] = {}
                self.family_voice_ids[user_id] = {}
                self.family_metadata[user_id] = {}
            
            # Initialize family member structure
            if family_member not in self.family_samples[user_id]:
                self.family_samples[user_id][family_member] = []
                
            current_count = len(self.family_samples[user_id][family_member])
            if current_count >= self.required_samples:
                return {
                    "success": True,
                    "message": f"Already has sufficient samples ({current_count}/{self.required_samples})",
                    "samples_collected": current_count,
                    "samples_needed": self.required_samples,
                    "ready_for_cloning": True
                }
            
            # Create family member directory
            family_dir = self.samples_base_dir / user_id / family_member
            family_dir.mkdir(parents=True, exist_ok=True)
            
            # Save file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filename = f"{family_member}_sample_{current_count + 1}_{timestamp}.{file_format}"
            file_path = family_dir / filename
            
            with open(file_path, "wb") as f:
                f.write(audio_data)
                
            self.family_samples[user_id][family_member].append(str(file_path))
            
            new_count = len(self.family_samples[user_id][family_member])
            ready_for_cloning = new_count >= self.required_samples
            
            # Store voice name for later use
            if family_member not in self.family_metadata[user_id]:
                self.family_metadata[user_id][family_member] = {}
            self.family_metadata[user_id][family_member]["voice_name"] = voice_name
            
            logger.info(f"Family voice sample saved: {file_path} ({len(audio_data)} bytes)")
            logger.info(f"User {user_id} {family_member} sample progress: {new_count}/{self.required_samples}")
            
            return {
                "success": True,
                "message": "Sample collected successfully",
                "samples_collected": new_count,
                "samples_needed": self.required_samples,
                "ready_for_cloning": ready_for_cloning,
                "file_path": str(file_path)
            }
        
        except Exception as e:
            error_msg = f"Failed to collect family voice sample - user_id: {user_id}, family_member: {family_member}, error: {e}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg
            }

    async def create_family_voice_clone(self, user_id: str, family_member: str) -> Dict[str, Any]:
        """
        Create family member voice clone
        
        Args:
            user_id: User identifier
            family_member: Family member type ("mom" | "dad")
        
        Returns:
            Dict with cloning result
        """
        try:
            if not self.elevenlabs_client:
                return {
                    "success": False,
                    "error": "ElevenLabs client not initialized"
                }
        
            if not self._is_family_ready_for_cloning(user_id, family_member):
                current_count = self._get_family_sample_count(user_id, family_member)
                return {
                    "success": False,
                    "error": f"Insufficient samples ({current_count}/{self.required_samples})"
                }
            
            # Check if voice already exists
            if (user_id in self.family_voice_ids and 
                family_member in self.family_voice_ids[user_id]):
                existing_voice_id = self.family_voice_ids[user_id][family_member]
                logger.info(f"Voice already exists for {user_id} {family_member}: {existing_voice_id}")
                return {
                    "success": True,
                    "voice_id": existing_voice_id,
                    "message": "Voice already exists",
                    "status": "completed"
                }

            sample_paths = self.family_samples[user_id][family_member]
            voice_name = self.family_metadata[user_id][family_member].get("voice_name", f"{family_member}_voice")
            
            # Read voice files and prepare
            audio_files = []
            for sample_path in sample_paths:
                if os.path.exists(sample_path):
                    with open(sample_path, "rb") as f:
                        audio_files.append(f.read())
                else:
                    logger.warning(f"Sample file not found: {sample_path}")
                
            if len(audio_files) < self.required_samples:
                return {
                    "success": False,
                    "error": f"Insufficient valid samples ({len(audio_files)}/{self.required_samples})"
                }
            
            logger.info(f"Starting family voice cloning - user_id: {user_id}, family_member: {family_member}, samples: {len(audio_files)}")
            
            # Create voice clone with ElevenLabs API
            voice = await asyncio.to_thread(
                self._create_voice_with_samples,
                f"{voice_name}_{family_member}",
                audio_files
            )
            
            if voice and hasattr(voice, "voice_id"):
                voice_id = voice.voice_id
                
                # Store generated voice ID
                self.family_voice_ids[user_id][family_member] = voice_id
                
                # Update metadata
                self.family_metadata[user_id][family_member].update({
                    "voice_id": voice_id,
                    "created_at": datetime.now().isoformat(),
                    "sample_count": len(audio_files),
                    "sample_paths": sample_paths,
                    "status": "completed"
                })
                
                # Save metadata to file
                await self._save_family_metadata(user_id, family_member)
                
                logger.info(f"Family voice cloning completed - user_id: {user_id}, family_member: {family_member}, voice_id: {voice_id}")
                return {
                    "success": True,
                    "voice_id": voice_id,
                    "family_member": family_member,
                    "voice_name": voice_name,
                    "status": "completed",
                    "samples_used": len(audio_files)
                }
            else:
                logger.error(f"Family voice cloning failed - invalid response: {voice}")
                return {
                    "success": False,
                    "error": "Voice cloning API response invalid"
                }
            
        except Exception as e:
            error_msg = f"Error during family voice cloning: {str(e)}"
            logger.error(f"{error_msg} - user_id: {user_id}, family_member: {family_member}")
            return {
                "success": False,
                "error": error_msg
            }
    
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
            logger.info(f"User {user_id} sample progress: {len(self.user_samples[user_id])}/{self.required_samples}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to collect user voice sample - user_id: {user_id}, error: {e}")
            return False
            
    async def create_instant_voice_clone(self, user_id: str, voice_name: str) -> Tuple[Optional[str], Optional[str]]:
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
        
            if not self.is_ready_for_cloning(user_id):
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
                    with open(sample_path, "rb") as f:
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
    
    async def generate_family_voice_speech(self, user_id: str, family_member: str, text: str, output_path: Optional[str] = None) -> Tuple[Optional[bytes], Optional[str]]:
        """
        Generate speech with family member's cloned voice
        
        Args:
            user_id: User identifier
            family_member: Family member type ("mom" | "dad")
            text: Text to synthesize
            output_path: File path to save (optional)
            
        Returns:
            Tuple[audio_data, file_path]: Audio data and file path
        """
        try:
            voice_id = self.get_family_voice_id(user_id, family_member)
            if not voice_id:
                logger.warning(f"Family voice not found for user {user_id} {family_member}")
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
                    
                logger.info(f"Family voice file saved: {output_path}")
                return audio, output_path
            
            return audio, None
        
        except Exception as e:
            logger.error(f"Family voice speech generation failed - user_id: {user_id}, family_member: {family_member}, error: {e}")
            return None, None

    # ⭐ NEW: Family voice utility methods
    def _is_family_ready_for_cloning(self, user_id: str, family_member: str) -> bool:
        """Check if family member voice is ready for cloning"""
        return self._get_family_sample_count(user_id, family_member) >= self.required_samples
    
    def _get_family_sample_count(self, user_id: str, family_member: str) -> int:
        """Get current sample count for family member"""
        if (user_id not in self.family_samples or 
            family_member not in self.family_samples[user_id]):
            return 0
        return len(self.family_samples[user_id][family_member])
    
    def get_family_voice_id(self, user_id: str, family_member: str) -> Optional[str]:
        """Get generated voice ID for family member"""
        if (user_id not in self.family_voice_ids or 
            family_member not in self.family_voice_ids[user_id]):
            return None
        return self.family_voice_ids[user_id][family_member]
    
    def get_family_voice_status(self, user_id: str, family_member: str = None) -> Dict[str, Any]:
        """Get family voice status - for specific member or all family members"""
        if family_member:
            # Single family member status
            sample_count = self._get_family_sample_count(user_id, family_member)
            voice_id = self.get_family_voice_id(user_id, family_member)
            metadata = {}
            if (user_id in self.family_metadata and 
                family_member in self.family_metadata[user_id]):
                metadata = self.family_metadata[user_id][family_member]
            
            return {
                "user_id": user_id,
                "family_member": family_member,
                "sample_count": sample_count,
                "required_samples": self.required_samples,
                "ready_for_cloning": self._is_family_ready_for_cloning(user_id, family_member),
                "has_cloned_voice": voice_id is not None,
                "voice_id": voice_id,
                "voice_name": metadata.get("voice_name"),
                "status": metadata.get("status", "not_started"),
                "created_at": metadata.get("created_at")
            }
        else:
            # All family members status
            status = {
                "user_id": user_id,
                "family_voices": {},
                "total_voices": 0,
                "completed_voices": 0
            }
            
            for member in self.supported_family_members:
                member_status = self.get_family_voice_status(user_id, member)
                status["family_voices"][member] = member_status
                if member_status["has_cloned_voice"]:
                    status["completed_voices"] += 1
                if member_status["sample_count"] > 0:
                    status["total_voices"] += 1
            
            return status
    
    def list_family_voices(self, user_id: str) -> List[Dict[str, Any]]:
        """List all family voices for user"""
        voices = []
        
        if user_id not in self.family_voice_ids:
            return voices
        
        for family_member, voice_id in self.family_voice_ids[user_id].items():
            metadata = self.family_metadata.get(user_id, {}).get(family_member, {})
            
            voices.append({
                "voice_id": voice_id,
                "family_member": family_member,
                "voice_name": metadata.get("voice_name", f"{family_member}_voice"),
                "status": metadata.get("status", "completed"),
                "created_at": metadata.get("created_at"),
                "sample_count": metadata.get("sample_count", 0)
            })
        
        return voices
    
    async def _save_family_metadata(self, user_id: str, family_member: str):
        """Save family voice metadata"""
        try:
            if (user_id not in self.family_metadata or 
                family_member not in self.family_metadata[user_id]):
                return
            
            metadata_dir = self.samples_base_dir / user_id / family_member
            metadata_file = metadata_dir / "family_voice_metadata.json"
            
            with open(metadata_file, "w", encoding='utf-8') as f:
                json.dump(self.family_metadata[user_id][family_member], f, ensure_ascii=False, indent=2)
                
            logger.info(f"Family voice metadata saved: {metadata_file}")
        
        except Exception as e:
            logger.error(f"Failed to save family metadata - user_id: {user_id}, family_member: {family_member}, error: {e}")
    
    async def delete_family_voice(self, user_id: str, family_member: str) -> Dict[str, Any]:
        """Delete family member voice and cleanup files"""
        try:
            voice_id = self.get_family_voice_id(user_id, family_member)
            
            if not voice_id:
                return {
                    "success": False,
                    "error": f"No voice found for {user_id} {family_member}"
                }
            
            # Delete from ElevenLabs (if needed - ElevenLabs API might not support this)
            # Note: ElevenLabs might not provide voice deletion API
            
            # Cleanup local data
            if user_id in self.family_samples and family_member in self.family_samples[user_id]:
                # Delete sample files
                for sample_path in self.family_samples[user_id][family_member]:
                    if os.path.exists(sample_path):
                        os.remove(sample_path)
                        logger.debug(f"Sample file deleted: {sample_path}")
                
                del self.family_samples[user_id][family_member]
            
            if user_id in self.family_voice_ids and family_member in self.family_voice_ids[user_id]:
                del self.family_voice_ids[user_id][family_member]
            
            if user_id in self.family_metadata and family_member in self.family_metadata[user_id]:
                del self.family_metadata[user_id][family_member]
            
            # Delete metadata file
            metadata_dir = self.samples_base_dir / user_id / family_member
            metadata_file = metadata_dir / "family_voice_metadata.json"
            if metadata_file.exists():
                os.remove(metadata_file)
            
            # Remove directory if empty
            if metadata_dir.exists() and not any(metadata_dir.iterdir()):
                metadata_dir.rmdir()
            
            logger.info(f"Family voice deleted - user_id: {user_id}, family_member: {family_member}, voice_id: {voice_id}")
            
            return {
                "success": True,
                "message": f"Family voice deleted successfully",
                "deleted_voice_id": voice_id,
                "family_member": family_member
            }
            
        except Exception as e:
            error_msg = f"Failed to delete family voice - user_id: {user_id}, family_member: {family_member}, error: {e}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg
            }

    # State methods
    def get_sample_count(self, user_id: str) -> int:
        """Get current sample count for user"""
        return len(self.user_samples.get(user_id, []))
    
    def is_ready_for_cloning(self, user_id: str) -> bool:
        """Check if ready for voice cloning"""
        return self.get_sample_count(user_id) >= self.required_samples
    
    def get_user_voice_id(self, user_id: str) -> Optional[str]:
        """Get generated voice ID for user"""
        return self.user_voice_ids.get(user_id)
    
    def get_user_voice_status(self, user_id: str) -> Dict[str, Any]:
        """Get user voice cloning status"""
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
        
    
        
    
    