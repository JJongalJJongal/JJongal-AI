"""
Enhanced Audio Processing Utilities for JJongal-AI

Comprehensive audio processing with robust error handling,
validation, and performance optimization.
"""
import os
import logging
import asyncio
import traceback
import time
from pathlib import Path
from typing import Tuple, Optional, Any, Dict, Union
from elevenlabs.client import ElevenLabs
from elevenlabs import play, stream, save, Voice, VoiceSettings

from ..configs.app import get_env_vars
from .logging import get_module_logger

logger = get_module_logger(__name__)


class AudioError(Exception):
    """Custom exception for audio-related errors"""
    pass


class ElevenLabsClientError(AudioError):
    """Exception for ElevenLabs API client errors"""
    pass


def ensure_audio_directory() -> Path:
    """
    /output/temp/audio 디렉토리가 존재하는지 확인하고 생성
    
    Returns:
        Path: 오디오 저장 디렉토리 경로
    """
    # 통일된 temp 경로 구조 사용 - 중복 제거
    audio_dir = Path("output") / "temp" / "audio"  # output/temp/audio (중복 제거)
    audio_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"오디오 저장 디렉토리 준비 완료: {audio_dir}")
    return audio_dir


def initialize_elevenlabs() -> Optional[ElevenLabs]:
    """
    Enhanced ElevenLabs client initialization with validation
    
    Returns:
        Optional[ElevenLabs]: Initialized client or None if failed
    """
    try:
        # Try multiple environment variable names for flexibility
        api_key = (
            os.getenv("ELEVENLABS_API_KEY") or 
            get_env_vars().get("elevenlabs_api_key")
        )
        
        if not api_key:
            logger.warning("ELEVENLABS_API_KEY not found in environment variables")
            return None
            
        # Check for placeholder values
        if api_key in ["sk_REPLACE_WITH_YOUR_ACTUAL_ELEVENLABS_API_KEY", "", "your-api-key-here"]:
            logger.warning("ELEVENLABS_API_KEY appears to be a placeholder value")
            return None
        
        # Validate API key format (ElevenLabs keys typically start with 'sk_')
        if not api_key.startswith('sk_'):
            logger.warning(f"ELEVENLABS_API_KEY format seems invalid (should start with 'sk_')")
        
        client = ElevenLabs(api_key=api_key)
        
        # Test the connection with a simple API call
        try:
            voices = client.voices.get_all()
            voice_count = len(voices.voices) if voices and voices.voices else 0
            logger.info(f"ElevenLabs client initialized successfully. Available voices: {voice_count}")
            return client
            
        except Exception as e:
            logger.error(f"ElevenLabs API key validation failed: {e}")
            # Return None instead of raising to allow graceful degradation
            return None
            
    except Exception as e:
        logger.error(f"Failed to initialize ElevenLabs client: {e}")
        return None


async def transcribe_audio(model: Any, file_path: str, language: str = "ko") -> Tuple[str, Optional[str], Optional[str]]:
    """
    Whisper로 음성 파일을 텍스트로 변환
    
    Args:
        model: Whisper 모델 인스턴스
        file_path (str): 오디오 파일 경로
        language (str): 인식할 언어 코드 (기본값: 'ko' - 한국어)
    
    Returns:
        tuple: (텍스트, 오류 메시지, 오류 코드)
    """
    if model is None:
        logger.error("Whisper 모델이 초기화되지 않았습니다")
        return "", "Whisper 모델이 초기화되지 않았습니다", "whisper_not_initialized"
        
    try:
        file_size = os.path.getsize(file_path)
        logger.info(f"오디오 파일 변환 시작: {file_path}, 크기: {file_size} 바이트")
        
        # 파일이 너무 작으면 무시
        if file_size < 1000:  # 1KB 미만
            logger.warning("오디오 파일이 너무 작습니다")
            return "", "오디오 파일이 너무 작습니다", "audio_too_small"
        
        # 비동기 처리를 위해 스레드풀에서 실행
        # 언어 추가 지정으로 정확도 향상
        result = await asyncio.to_thread(
            model.transcribe, 
            file_path, 
            language=language,
            fp16=False  # 호환성 향상을 위해 fp16 비활성화
        )
        
        text = result.get("text", "").strip()
        logger.info(f"Whisper 변환 결과: {text}")
        
        # 결과가 빈 문자열이면 처리
        if not text:
            logger.warning("Whisper가 텍스트를 인식하지 못했습니다")
            return "", "음성을 인식하지 못했습니다. 더 명확하게 말씀해주세요.", "no_speech_detected"
            
        return text, None, None
        
    except Exception as e:
        error_detail = traceback.format_exc()
        logger.error(f"Whisper 오류: {e}\n{error_detail}")
        return "", f"Whisper 오류: {e}", "whisper_error"


async def generate_speech(
    client: Optional[ElevenLabs], 
    text: str, 
    voice_id: Optional[str] = None,
    output_path: Optional[str] = None,
    voice_settings: Optional[VoiceSettings] = None
) -> Tuple[Optional[bytes], Optional[str]]:
    """
    Enhanced speech generation with comprehensive error handling
    
    Args:
        client (Optional[ElevenLabs]): ElevenLabs client
        text (str): Text to synthesize
        voice_id (Optional[str]): Voice ID (uses default if None)
        output_path (Optional[str]): Path to save audio file
        voice_settings (Optional[VoiceSettings]): Voice settings
        
    Returns:
        Tuple[Optional[bytes], Optional[str]]: Audio data and file path
        
    Raises:
        AudioError: If speech generation fails
    """
    try:
        # Input validation
        if client is None:
            raise ElevenLabsClientError("ElevenLabs client not initialized")
            
        if not text or not text.strip():
            raise AudioError("Text cannot be empty")
            
        # Text length validation (ElevenLabs has limits)
        if len(text) > 5000:
            logger.warning(f"Text length ({len(text)}) exceeds recommended limit (5000)")
            text = text[:5000] + "..."
        
        # Use default voice if none specified
        if voice_id is None:
            try:
                voices = await asyncio.to_thread(client.voices.get_all)
                if voices and voices.voices:
                    # Prefer Korean voices if available
                    korean_voices = [v for v in voices.voices if 'korean' in v.name.lower() or '한국' in v.name.lower()]
                    if korean_voices:
                        voice_id = korean_voices[0].voice_id
                        logger.debug(f"Using Korean voice: {voice_id}")
                    else:
                        voice_id = voices.voices[0].voice_id
                        logger.debug(f"Using default voice: {voice_id}")
                else:
                    raise AudioError("No voices available")
            except Exception as e:
                logger.error(f"Failed to get default voice: {e}")
                voice_id = "21m00Tcm4TlvDq8ikWAM"  # Fallback to known working voice
        
        # Set default voice settings if not provided
        if voice_settings is None:
            voice_settings = VoiceSettings(
                stability=0.7,
                similarity_boost=0.8,
                style=0.3,
                use_speaker_boost=True
            )
        
        logger.debug(f"Generating speech: {len(text)} chars with voice {voice_id}")
        
        # Generate audio with timeout
        try:
            audio_data = await asyncio.wait_for(
                asyncio.to_thread(
                    client.text_to_speech.convert,
                    text=text,
                    voice_id=voice_id,
                    voice_settings=voice_settings,
                    model_id="eleven_multilingual_v2",
                    output_format="mp3_44100_128"
                ),
                timeout=60.0  # 60 second timeout
            )
        except asyncio.TimeoutError:
            raise AudioError("Speech generation timeout")
        except Exception as e:
            logger.error(f"ElevenLabs API call failed: {e}")
            raise ElevenLabsClientError(f"Speech generation failed: {e}")
        
        if not audio_data:
            raise AudioError("Empty audio data received from API")
        
        # Validate audio data
        if len(audio_data) < 1000:  # Less than 1KB seems suspicious
            logger.warning(f"Generated audio is very small: {len(audio_data)} bytes")
        
        # Save to file if path provided
        file_path = None
        if output_path:
            try:
                # Handle relative paths
                if not os.path.isabs(output_path):
                    audio_dir = ensure_audio_directory()
                    output_path = str(audio_dir / output_path)
                
                # Ensure parent directory exists
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                
                with open(output_path, "wb") as f:
                    f.write(audio_data)
                
                file_path = output_path
                logger.info(f"Audio saved successfully: {file_path} ({len(audio_data)} bytes)")
                
            except Exception as e:
                logger.error(f"Failed to save audio to {output_path}: {e}")
                # Don't raise here, return audio data anyway
        
        logger.debug(f"Speech generation completed: {len(audio_data)} bytes")
        return audio_data, file_path
        
    except AudioError:
        raise  # Re-raise our custom errors
    except Exception as e:
        logger.error(f"Unexpected error in speech generation: {e}")
        raise AudioError(f"Speech generation failed: {e}")


def validate_audio_data(audio_data: bytes) -> Dict[str, Any]:
    """
    Validate audio data and extract basic information
    
    Args:
        audio_data: Audio data bytes
        
    Returns:
        Dict with validation results and metadata
    """
    validation_result = {
        "valid": False,
        "size_bytes": 0,
        "estimated_duration": 0.0,
        "format": "unknown",
        "issues": []
    }
    
    try:
        if not audio_data:
            validation_result["issues"].append("Empty audio data")
            return validation_result
        
        size_bytes = len(audio_data)
        validation_result["size_bytes"] = size_bytes
        
        # Basic size validation
        if size_bytes < 1000:  # Less than 1KB
            validation_result["issues"].append("Audio data suspiciously small")
        elif size_bytes > 50 * 1024 * 1024:  # More than 50MB
            validation_result["issues"].append("Audio data very large")
        
        # Format detection based on header
        if audio_data.startswith(b'ID3') or audio_data.startswith(b'\xff\xfb'):
            validation_result["format"] = "mp3"
            # Rough duration estimation for MP3 at 128kbps
            validation_result["estimated_duration"] = round(size_bytes / (128 * 1024 / 8), 2)
        elif audio_data.startswith(b'RIFF'):
            validation_result["format"] = "wav"
            # Rough duration estimation for WAV (assuming 44.1kHz 16-bit)
            validation_result["estimated_duration"] = round(size_bytes / (44100 * 2), 2)
        else:
            validation_result["issues"].append("Unknown audio format")
        
        # Mark as valid if no critical issues
        critical_issues = ["Empty audio data"]
        has_critical = any(issue in validation_result["issues"] for issue in critical_issues)
        validation_result["valid"] = not has_critical
        
        return validation_result
        
    except Exception as e:
        logger.error(f"Audio validation error: {e}")
        validation_result["issues"].append(f"Validation error: {e}")
        return validation_result


async def cleanup_temp_audio_files(
    directory: Union[str, Path],
    max_age_hours: int = 24,
    dry_run: bool = False
) -> Dict[str, Any]:
    """
    Clean up temporary audio files with proper error handling
    
    Args:
        directory: Directory to clean
        max_age_hours: Maximum age in hours for files to keep
        dry_run: If True, only report what would be deleted
        
    Returns:
        Cleanup summary
    """
    cleanup_summary = {
        "files_found": 0,
        "files_deleted": 0,
        "bytes_freed": 0,
        "errors": []
    }
    
    try:
        directory = Path(directory)
        if not directory.exists():
            cleanup_summary["errors"].append(f"Directory does not exist: {directory}")
            return cleanup_summary
        
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        audio_extensions = {'.mp3', '.wav', '.ogg', '.m4a', '.flac'}
        
        for file_path in directory.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in audio_extensions:
                cleanup_summary["files_found"] += 1
                
                try:
                    file_age = current_time - file_path.stat().st_mtime
                    
                    if file_age > max_age_seconds:
                        file_size = file_path.stat().st_size
                        
                        if not dry_run:
                            file_path.unlink()
                            cleanup_summary["files_deleted"] += 1
                            cleanup_summary["bytes_freed"] += file_size
                            logger.debug(f"Deleted old audio file: {file_path}")
                        else:
                            logger.info(f"Would delete: {file_path} ({file_size} bytes)")
                            
                except Exception as e:
                    error_msg = f"Failed to process {file_path}: {e}"
                    cleanup_summary["errors"].append(error_msg)
                    logger.error(error_msg)
        
        if not dry_run:
            logger.info(f"Audio cleanup completed: {cleanup_summary['files_deleted']} files deleted, "
                       f"{cleanup_summary['bytes_freed']} bytes freed")
        
        return cleanup_summary
        
    except Exception as e:
        error_msg = f"Audio cleanup failed: {e}"
        cleanup_summary["errors"].append(error_msg)
        logger.error(error_msg)
        return cleanup_summary 