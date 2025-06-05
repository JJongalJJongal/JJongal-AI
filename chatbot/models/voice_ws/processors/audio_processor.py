"""
오디오 처리 프로세서

Whisper STT 및 OpenAI TTS 기능을 제공합니다.
"""
import os
import base64
import asyncio
import traceback
import tempfile
import whisper
from openai import OpenAI
from dotenv import load_dotenv

from shared.utils.logging_utils import get_module_logger
from shared.utils.async_utils import retry_operation

logger = get_module_logger(__name__)

# 환경 변수 로드 (프로젝트 루트 기준)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
dotenv_path = os.path.join(project_root, '.env')
load_dotenv(dotenv_path=dotenv_path)

# OpenAI API 키 확인
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    logger.warning(f"OPENAI_API_KEY 환경 변수를 찾을 수 없습니다. .env 파일: {dotenv_path}")

# OpenAI 클라이언트 초기화
try:
    openai_client = OpenAI(api_key=api_key)
    logger.info("OpenAI 클라이언트 초기화 성공")
except Exception as e:
    logger.error(f"OpenAI 클라이언트 초기화 실패: {e}")
    openai_client = None

# Whisper 모델 초기화
whisper_model = None
whisper_model_name = os.getenv("WHISPER_MODEL", "base")
try:
    whisper_model = whisper.load_model(whisper_model_name)
    logger.info(f"Whisper 모델({whisper_model_name}) 로드 성공")
except Exception as e:
    logger.error(f"Whisper 모델 로드 실패: {e}")

class AudioProcessor:
    """
    오디오 처리를 담당하는 프로세서
    """
    def __init__(self):
        self.openai_client = openai_client
        self.whisper_model = whisper_model
        logger.info("AudioProcessor 초기화 완료")

    async def transcribe_audio(self, file_path: str, language: str = "ko"):
        """
        Whisper로 음성 파일을 텍스트로 변환
        
        Args:
            file_path (str): 오디오 파일 경로
            language (str): 인식할 언어 코드 (기본값: 'ko' - 한국어)
        
        Returns:
            tuple: (텍스트, 오류 메시지, 오류 코드)
        """
        if self.whisper_model is None:
            logger.error("Whisper 모델이 초기화되지 않았습니다")
            return "", "Whisper 모델이 초기화되지 않았습니다", "whisper_not_initialized"
            
        try:
            file_size = os.path.getsize(file_path)
            logger.info(f"오디오 파일 변환 시작: {file_path}, 크기: {file_size} 바이트")
            
            if file_size < 1000:  # 1KB 미만
                logger.warning("오디오 파일이 너무 작습니다")
                return "", "오디오 파일이 너무 작습니다", "audio_too_small"
            
            result = await asyncio.to_thread(
                self.whisper_model.transcribe, 
                file_path, 
                language=language,
                fp16=False
            )
            
            text = result.get("text", "").strip()
            logger.info(f"Whisper 변환 결과: {text}")
            
            if not text:
                logger.warning("Whisper가 텍스트를 인식하지 못했습니다")
                return "", "음성을 인식하지 못했습니다. 더 명확하게 말씀해주세요.", "no_speech_detected"
                
            return text, None, None
            
        except Exception as e:
            error_detail = traceback.format_exc()
            logger.error(f"Whisper 오류: {e}\n{error_detail}")
            return "", f"Whisper 오류: {e}", "whisper_error"

    async def synthesize_tts(self, text: str, voice: str = "nova"):
        """
        텍스트를 TTS로 변환
        
        Args:
            text (str): 텍스트 응답
            voice (str): 사용할 음성 (기본값: 'nova')
        
        Returns:
            tuple: (base64 인코딩된 오디오, 상태, 오류 메시지, 오류 코드)
        """
        try:
            if not text:
                return "", "error", "TTS 입력 없음", "tts_no_input"
            
            if self.openai_client is None:
                return "", "error", "OpenAI 클라이언트가 초기화되지 않았습니다", "openai_client_not_initialized"
                
            async def tts_operation():
                tts_result = await asyncio.to_thread(
                    self.openai_client.audio.speech.create,
                    model="tts-1-hd", # 최신 모델 사용 (tts-1-hd, tts-1, tts-1-large)
                    voice=voice, # 음성 선택 (nova, alloy, echo, fable, onyx, nova-2, shimmer)
                    input=text, # 텍스트 입력
                    speed=0.9, # 속도 조절 (0.0-2.0)
                    response_format="mp3" # 오디오 형식 (mp3, opus, aac, flac)
                )
                return tts_result # type: ignore
                
            tts_audio, error_msg, error_code = await retry_operation(tts_operation)
            
            if tts_audio is None:
                logger.error(f"TTS 생성 실패 후 재시도 모두 실패 : {error_msg}")
                return "", "error", error_msg or "TTS 생성 실패", error_code or "tts_generation_failed"            
        
            # tts_audio가 실제 오디오 데이터, 바이트 데이터로 변환
            if hasattr(tts_audio, 'read') and callable(tts_audio.read):
                tts_audio = tts_audio.read()
            elif hasattr(tts_audio, 'content') and isinstance(tts_audio.content, bytes):
                tts_audio = tts_audio.content
            elif not isinstance(tts_audio, bytes):
                logger.error(f"추출된 tts_audio가 바이트 형식이 아닙니다. 타입: {type(tts_audio)}")
                return "", "error", "추출된 오디오 데이터 형식 오류", "tts_audio_not_bytes"
            
            if len(tts_audio) < 2 * 1024 * 1024: # 2MB 미만
                return base64.b64encode(tts_audio).decode("utf-8"), "ok", None, None
            else:
                logger.warning(f"TTS 오디오 크기 초과: {len(tts_audio)} 바이트")
                return base64.b64encode(tts_audio).decode("utf-8"), "large_file", "오디오 크기가 큽니다", "tts_large_file"
        except Exception as e:
            error_detail = traceback.format_exc()
            logger.error(f"TTS 오류: {e}\n{error_detail}")
            return "", "error", f"TTS 오류: {e}", "tts_error"

    async def process_audio_chunk(self, audio_data, client_id):
        """
        오디오 청크 처리
        
        Args:
            audio_data (bytes): 오디오 데이터
            client_id (str): 클라이언트 ID
        
        Returns:
            tuple: (임시 파일 경로, 오류 메시지, 오류 코드)
        """
        temp_file_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
                temp_file.write(audio_data)
                temp_file_path = temp_file.name
                
            return temp_file_path, None, None
        except Exception as e:
            error_detail = traceback.format_exc()
            logger.error(f"오디오 청크 처리 오류 ({client_id}): {e}\n{error_detail}")
            
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                except:
                    pass
                    
            return None, f"오디오 처리 오류: {e}", "audio_processing_error" 