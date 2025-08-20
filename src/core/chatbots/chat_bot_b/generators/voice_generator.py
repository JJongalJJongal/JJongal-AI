import asyncio
import time
import uuid
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List

from elevenlabs.client import ElevenLabs
from elevenlabs import Voice, VoiceSettings

from .base_generator import BaseGenerator
from src.shared.configs.prompts import load_prompts_config
from src.shared.utils.logging import get_module_logger

logger = get_module_logger(__name__)

class VoiceGenerator(BaseGenerator):
    def __init__(self, elevenlabs_api_key: Optional[str] = None, output_dir: str = "output/temp/audio", max_retries: int = 3):
        super().__init__(max_retries=max_retries, timeout=60.0)

        self.elevenlabs_api_key = elevenlabs_api_key
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.prompts_config = load_prompts_config("chatbot_b")

        self._setup_elevenlabs_client()

        logger.info(f"VoiceGenerator initialization complete : {output_dir}")

    def _setup_elevenlabs_client(self) -> None:
        try:
            if not self.elevenlabs_api_key:
                logger.warning("Elevenlabs API Key not found")
                self.client = None
                return
            self.client = ElevenLabs(api_key=self.elevenlabs_api_key)
            logger.info("ElevenLabs Client initialization success")
        except Exception as e:
            logger.error(f"ElevenLabs Client initialization failed : {e}")
            self.client = None
    
    async def generate(self, input_data: Dict[str, Any], progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        start_time = time.time()

        try:
            if self.client is None:
                return self._empty_result("ElevenLabs API key not configured")
            
            story_data = input_data.get("story_data", {})
            voice_config = input_data.get("voice_config", {})
            target_age = input_data.get("target_age", 7)
            story_id = input_data.get("story_id", str(uuid.uuid4()))
            chapters = story_data.get("chapters", [])

            if not chapters:
                return self._empty_result()
            
            if progress_callback:
                await progress_callback({"step": "voice_generation_start", "total_chapters": len(chapters)})

            # voice generation for chapter
            audio_results = []
            total_duration = 0.0

            for i, chapter in enumerate(chapters):
                try:
                    if progress_callback:
                        await progress_callback({
                            "step": "generating_chapter_audio",
                            "chapter": i+1, "total": len(chapters)
                        })
                    audio_result = await self._generate_chapter_audio(
                        chapter, voice_config, target_age, story_id, i
                    )

                    audio_results.append(audio_result)
                    total_duration += audio_result.get("duration", 0)

                    if i < len(chapters) - 1:
                        await asyncio.sleep(1)
                except Exception as e:
                    logger.error(f"Chapter {i+1} voice generation failed: {e}")
                    audio_results.append(self._error_result(i+1, str(e)))

            return {
                "audio_files": audio_results,
                "metadata": {
                    "total_duration": round(total_duration, 2),
                    "successful_audio": len([r for r in audio_results if r.get("audio_path")]),
                    "model_used": "eleven_v3",
                    "generation_time": round(time.time() - start_time, 2)
                }
            }
        except Exception as e:
            logger.error(f"Voice generation failed: {e}")
            return self._empty_result(str(e))

    async def _generate_chapter_audio(self, chapter, voice_config, target_age, story_id, chapter_index):
        try:
            chapter_title = chapter.get("title", "")
            chapter_content = chapter.get("content", "")
            full_text = f"{chapter_title}. {chapter_content}" if chapter_title else chapter_content

            if not full_text.strip():
                return self._error_result(chapter_index + 1, " ")
            
            # Voice setting
            voice_id = self._determine_voice_id(voice_config, target_age)
            voice_settings = self._get_age_specific_settings(target_age)

            logger.info(f"Chapter {chapter_index + 1} voice generate...")

            if self.client is None:
                raise Exception("ElevenLabs Client not initialized.")
            
            audio = await asyncio.to_thread(
                self.client.text_to_speech.convert,
                text=full_text,
                voice_id=voice_id,
                model_id="eleven",
                voice_settings=voice_settings,
                output_format="mp3_44100_128"
            )

            # voice file save
            audio_path = await self._save_audio_file(
                audio, story_id, chapter.get("chapter_number", chapter_index + 1)
            )

            estimation_duration = len(full_text.split()) / 2.5 # 150 word per minute

            return {
                "chapter_number": chapter.get("chapter_number", chapter_index + 1),
                "audio_path": str(audio_path),
                "duration": round(estimation_duration, 2),
                "voice_type": "narrator",
                "voice_id": voice_id,
                "status": "success"
            }
        
        except Exception as e:
            logger.error(f"Chapter voice generation failed: {e}")
            return self._error_result(chapter.get("chapter_number", chapter_index + 1), str(e))

    def _determine_voice_id(self, voice_config: Dict[str, Any], target_age: int) -> str:
        
        # voice_config check
        child_voice_id = voice_config.get("child_voice_id")
        if child_voice_id:
            return child_voice_id
    
        # parent_voice_id = voice_config.get("parent_voice_id")
        # if parent_voice_id:
        #     return parent_voice_id

        if target_age <= 7:
            return "default_child_friendly_voice_id"
        else:
            return "default_mature_child_voice_id"
    
    def _get_age_specific_settings(self, target_age: int) -> VoiceSettings:

        if target_age <= 7:
            return VoiceSettings(
                stability=0.7,
                similarity_boost=0.8,
                style=0.3,
                use_speaker_boost=True
            )
        else:
            return VoiceSettings(
                stability=0.6,
                similarity_boost=0.9,
                style=0.5, # Expression
                use_speaker_boost=True
            )
    
    async def _save_audio_file(self, audio_data, story_id: str, chapter_number: int) -> Path:
        story_dir = self.output_dir / story_id[:8]
        story_dir.mkdir(parents=True, exist_ok=True)
        audio_path = story_dir / f"chapter_{chapter_number}_{story_id[:8]}.mp3"

        try:
            with open(audio_path, 'wb') as f:
                if hasattr(audio_data, 'read'):
                    for chunk in audio_data:
                        f.write(chunk)
                else:
                    f.write(audio_data)
            
            logger.info(f"Voice file save complete : {audio_path}")
            return audio_path
        
        except Exception as e:
            logger.error(f"Voice file save failed: {e}")
            raise
    
    def _empty_result(self, error: Optional[str] = None) -> Dict[str, Any]:
        result = {
            "audio_files": [],
            "metadata": {
                "total_duration": 0,
                "successful_audio": 0,
                "model_used": "eleven_v3",
                "generation_time": 0
            }
        }

        if error:
            result["metadata"]["error"] = error
        
        return result
    
    def _error_result(self, chapter_number: int, error: str) -> Dict[str, Any]:
        return {
            "chapter_number": chapter_number,
            "audio_path": None,
            "duration": 0,
            "voice_type": "error",
            "status": "error",
            "error": error
        }
    
    async def health_check(self) -> bool:
        try:
            checks = [
                bool(self.prompts_config),
                self.client is not None,
                self.output_dir.exists(),
                bool(self.elevenlabs_api_key)
            ]
            return all(checks)
        except Exception as e:
            logger.error(f"Healthcheck failed : {e}")
            return False