"""
오디오 처리 관련 유틸리티 모듈
"""
import os
import logging
import asyncio
import traceback
from pathlib import Path
from typing import Tuple, Optional, Any, Dict
from elevenlabs.client import ElevenLabs
from elevenlabs import play, stream, save

from ..configs.app_config import get_env_vars

logger = logging.getLogger(__name__)


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
    ElevenLabs 클라이언트 초기화
    
    Returns:
        Optional[ElevenLabs]: 초기화된 ElevenLabs 클라이언트 (실패 시 None)
    """
    env_vars = get_env_vars()
    api_key = env_vars.get("elevenlabs_api_key")
    
    if not api_key:
        logger.warning("ELEVENLABS_API_KEY가 설정되지 않았습니다.")
        return None
    
    try:
        client = ElevenLabs(api_key=api_key)
        logger.info("ElevenLabs 클라이언트 초기화 성공")
        return client
    except Exception as e:
        logger.error(f"ElevenLabs 클라이언트 초기화 실패: {e}")
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
    voice_id: str = "21m00Tcm4TlvDq8ikWAM",
    output_path: Optional[str] = None
) -> Tuple[Optional[bytes], Optional[str]]:
    """
    ElevenLabs를 사용하여 음성 생성
    
    Args:
        client (Optional[ElevenLabs]): ElevenLabs 클라이언트
        text (str): 음성으로 변환할 텍스트
        voice_id (str): 사용할 음성 ID (기본값: "21m00Tcm4TlvDq8ikWAM" - 한국어 여성 음성)
        output_path (Optional[str]): 저장할 파일 경로
        
    Returns:
        Tuple[Optional[bytes], Optional[str]]: 오디오 데이터와 파일 경로 (실패 시 None, None)
    """
    if client is None:
        logger.error("ElevenLabs 클라이언트가 초기화되지 않았습니다")
        return None, None
        
    try:
        if not text.strip():
            logger.warning("변환할 텍스트가 비어있습니다")
            return None, None
            
        # 음성 생성
        audio = await asyncio.to_thread(
            client.text_to_speech.convert,
            text=text,
            voice_id=voice_id,
            model_id="eleven_multilingual_v2",
            output_format="wav_44100"
        )
        
        # 파일로 저장 (필요한 경우)
        if output_path:
            # output_path가 상대 경로인 경우 /output/temp/audio 기준으로 처리
            if not os.path.isabs(output_path):
                audio_dir = ensure_audio_directory()
                output_path = str(audio_dir / output_path)
            
            with open(output_path, "wb") as f:
                f.write(audio)
            logger.info(f"음성 파일 저장 완료: {output_path}")
            return audio, output_path
            
        return audio, None
        
    except Exception as e:
        logger.error(f"음성 생성 중 오류 발생: {e}")
        return None, None 