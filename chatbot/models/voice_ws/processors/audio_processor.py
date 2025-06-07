"""
오디오 처리 프로세서

Whisper STT 및 ElevenLabs TTS 기능을 제공합니다.
"""
import os
import base64
import asyncio
import traceback
import tempfile
import whisper
import aiohttp
from dotenv import load_dotenv

from shared.utils.logging_utils import get_module_logger
from shared.utils.async_utils import retry_operation

logger = get_module_logger(__name__)

# 환경 변수 로드 (프로젝트 루트 기준)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
dotenv_path = os.path.join(project_root, '.env')
load_dotenv(dotenv_path=dotenv_path)

# ElevenLabs API 키 확인
elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")
if not elevenlabs_api_key:
    logger.error(f"ELEVENLABS_API_KEY 환경 변수를 찾을 수 없습니다. .env 파일: {dotenv_path}")

# ElevenLabs API 설정
ELEVENLABS_BASE_URL = "https://api.elevenlabs.io/v1"

def get_elevenlabs_headers(api_key: str):
    """ElevenLabs API 헤더 생성"""
    return {
        "Accept": "audio/mp3",
        "Content-Type": "application/json",
        "xi-api-key": api_key
    }

# 부기(ChatBot A)용 음성 설정 - 아이와 대화하기 적합한 따뜻한 음성
BUGI_VOICE_CONFIG = {
    "voice_id": "AW5wrnG1jVizOYY7R1Oo",  # Jiyoung (따뜻하고 친근한 한국어 음성)
    "model_id": "eleven_multilingual_v2",  # 한국어 지원 모델
    "voice_settings": {
        "stability": 0.6,  # 안정성 (0.0-1.0)
        "similarity_boost": 0.8,  # 유사성 증가
        "style": 0.2,  # 약간의 스타일
        "use_speaker_boost": True  # 스피커 부스트
    }
}

# 부기의 다양한 음성 옵션 (상황별 사용 가능)
BUGI_VOICE_OPTIONS = {
    "default": "AW5wrnG1jVizOYY7R1Oo",  # Jiyoung (기본, 따뜻함)
    "cheerful": "UvkXHIJzOBYWOI51BDKp",  # Jeong-Ah (밝고 활기참)
    "gentle": "21m00Tcm4TlvDq8ikWAM",     # Rachel (부드럽고 차분함)
    "storytelling": "xi3rF0t7dg7uN2M0WUhr"  # Yuna (이야기하기 좋음)
}

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
        self.elevenlabs_api_key = elevenlabs_api_key
        self.whisper_model = whisper_model
        self.bugi_voice_config = BUGI_VOICE_CONFIG
        logger.info(f"AudioProcessor 초기화 완료 (ElevenLabs API: {'확인' if elevenlabs_api_key else '불가능'})")

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

    async def synthesize_tts(self, text: str, voice: str = None):
        """
        ElevenLabs API를 사용한 텍스트를 TTS로 변환
        
        Args:
            text (str): 텍스트 응답
            voice (str): 사용할 음성 (기본값: None - 부기 기본 음성 사용)
        
        Returns:
            tuple: (base64 인코딩된 오디오, 상태, 오류 메시지, 오류 코드)
        """
        try:
            # 텍스트 검증 강화
            if not text or not isinstance(text, str):
                logger.warning(f"TTS 입력이 유효하지 않음: {repr(text)}")
                return "", "error", "TTS 입력 없음", "tts_no_input"
            
            text = text.strip()
            if not text:
                logger.warning("TTS 입력이 빈 문자열입니다")
                return "", "error", "TTS 입력이 빈 문자열입니다", "tts_empty_input"
            
            if not self.elevenlabs_api_key:
                return "", "error", "ElevenLabs API 키가 설정되지 않았습니다", "elevenlabs_api_key_not_set"
            
            # 음성 ID 결정 (텍스트 내용에 따른 지능적 선택 또는 직접 지정)
            if voice:
                voice_id = voice  # 직접 지정된 음성 사용
            else:
                voice_id = self.get_voice_for_content(text)  # 텍스트 내용에 따라 선택
            
            # 텍스트 정리 (음성 생성에 적합하게)
            cleaned_text = self._clean_text_for_speech(text)
            
            # 정리된 텍스트 재검증
            if not cleaned_text or not cleaned_text.strip():
                logger.warning(f"텍스트 정리 후 빈 내용됨. 원본: {repr(text)}, 정리후: {repr(cleaned_text)}")
                return "", "error", "텍스트 정리 후 빈 내용이 되었습니다", "tts_cleaned_empty"
            
            async def elevenlabs_tts_operation():
                # ElevenLabs API 요청 데이터
                data = {
                    "text": cleaned_text,
                    "model_id": self.bugi_voice_config["model_id"],
                    "voice_settings": self.bugi_voice_config["voice_settings"]
                }
                
                url = f"{ELEVENLABS_BASE_URL}/text-to-speech/{voice_id}"
                headers = get_elevenlabs_headers(self.elevenlabs_api_key)
                
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, headers=headers, json=data) as response:
                        if response.status != 200:
                            error_text = await response.text()
                            raise Exception(f"ElevenLabs API 오류 ({response.status}): {error_text}")
                        
                        # 오디오 데이터 읽기
                        audio_data = await response.read()
                        return audio_data
                
            tts_audio, error_msg, error_code = await retry_operation(elevenlabs_tts_operation)
            
            if tts_audio is None:
                logger.error(f"ElevenLabs TTS 생성 실패 후 재시도 모두 실패: {error_msg}")
                return "", "error", error_msg or "TTS 생성 실패", error_code or "tts_generation_failed"
            
            # 오디오 데이터 검증
            if not isinstance(tts_audio, bytes):
                logger.error(f"ElevenLabs TTS 응답이 바이트 형식이 아닙니다. 타입: {type(tts_audio)}")
                return "", "error", "TTS 응답 데이터 형식 오류", "tts_audio_not_bytes"
            
            # 오디오 크기 확인 및 base64 인코딩
            audio_size_mb = len(tts_audio) / (1024 * 1024)
            
            if audio_size_mb < 2.0:  # 2MB 미만
                encoded_audio = base64.b64encode(tts_audio).decode("utf-8")
                logger.info(f"ElevenLabs TTS 생성 성공: {len(cleaned_text)}자 → {audio_size_mb:.2f}MB")
                return encoded_audio, "ok", None, None
            else:
                logger.warning(f"TTS 오디오 크기 초과: {audio_size_mb:.2f}MB")
                encoded_audio = base64.b64encode(tts_audio).decode("utf-8")
                return encoded_audio, "large_file", "오디오 크기가 큽니다", "tts_large_file"
                
        except Exception as e:
            error_detail = traceback.format_exc()
            logger.error(f"ElevenLabs TTS 오류: {e}\n{error_detail}")
            return "", "error", f"TTS 오류: {e}", "elevenlabs_tts_error"
    
    def _clean_text_for_speech(self, text: str) -> str:
        """음성 생성을 위한 텍스트 정리 (부기 전용)"""
        # 기본 정리
        text = text.strip()
        
        # 특수 문자 처리 (부기가 아이와 대화할 때 자연스럽게 하기 위해 추가)
        replacements = {
            "**": "",  # 마크다운 볼드 제거
            "*": "",   # 마크다운 이탤릭 제거
            "_": "",   # 언더스코어 제거
            "#": "",   # 해시태그 제거
            "`": "",   # 백틱 제거
            "---": ". ",  # 구분선을 마침표로
            "...": ".. ",  # 말줄임표 정리
            "ㅋㅋ": "크크",  # 웃음 표현 자연스럽게
            "ㅎㅎ": "하하",  # 웃음 표현 자연스럽게
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        # 연속된 공백 정리
        import re
        text = re.sub(r'\s+', ' ', text)
        
        # 연속된 마침표 정리
        text = re.sub(r'\.{3,}', '.. ', text)
        
        # ElevenLabs 텍스트 길이 제한 (5000자)
        if len(text) > 4900:
            logger.warning(f"텍스트가 너무 깁니다 ({len(text)}자). 4900자로 자릅니다.")
            text = text[:4897] + "..."
        
        return text.strip()
    
    def get_voice_for_content(self, text: str) -> str:
        """텍스트 내용에 따라 적절한 음성 선택 (부기 전용)"""
        text_lower = text.lower()
        
        # 감정이나 상황에 따른 음성 선택
        if any(keyword in text_lower for keyword in ["안녕", "반가워", "좋아", "재미", "신나", "와!"]):
            return BUGI_VOICE_OPTIONS["cheerful"]  # 밝고 활기찬 음성
        elif any(keyword in text_lower for keyword in ["이야기", "옛날", "동화", "들려줄게", "한번"]):
            return BUGI_VOICE_OPTIONS["storytelling"]  # 이야기하기 좋은 음성
        elif any(keyword in text_lower for keyword in ["괜찮", "천천히", "걱정", "무서워", "힘들"]):
            return BUGI_VOICE_OPTIONS["gentle"]  # 부드럽고 차분한 음성
        else:
            return BUGI_VOICE_OPTIONS["default"]  # 기본 따뜻한 음성
    
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