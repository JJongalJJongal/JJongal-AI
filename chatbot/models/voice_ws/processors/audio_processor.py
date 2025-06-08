"""
오디오 처리 프로세서

Whisper STT 및 ElevenLabs TTS 기능을 제공합니다.
"""
import os
import ssl
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
        "Accept": "audio/wav",
        "Content-Type": "application/json",
        "xi-api-key": api_key
    }

# 부기(ChatBot A) 전용 음성 설정 - 한국어 동화 최적화
BUGI_VOICE_CONFIG = {
    "voice_id": "AW5wrnG1jVizOYY7R1Oo",  # Jiyoung - 한국어 지원 여성 음성
    "model_id": "eleven_multilingual_v2",  # 한국어 최적화 모델
    "voice_settings": {
        "stability": 0.50,  # 한국어: 안정성과 표현력의 균형
        "similarity_boost": 0.85,  # 한국어 발음 명확성 강화
        "style": 0.15,  # 한국어 동화 톤에 맞게 조정
        "use_speaker_boost": True
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
        
        # 클라이언트별 음성 매핑 지원 추가
        self.user_voice_mappings = {}  # {client_id: {"voice_id": str, "voice_settings": dict}}
        
        logger.info(f"AudioProcessor 초기화 완료 (ElevenLabs API: {'확인' if elevenlabs_api_key else '불가능'})")

    def set_user_voice_mapping(self, client_id: str, voice_id: str, voice_settings: dict = None):
        """
        클라이언트별 음성 매핑 설정
        
        Args:
            client_id (str): 클라이언트 식별자
            voice_id (str): 사용할 음성 ID (클론 음성 ID 포함)
            voice_settings (dict): 음성 설정 (옵션)
        """
        # 기본 음성 설정 사용 (클론 음성에 최적화)
        default_voice_settings = {
            "stability": 0.7,  # 클론 음성을 위한 안정성 증가
            "similarity_boost": 0.8,  # 유사성 증대
            "style": 0.3,  # 약간의 스타일
            "use_speaker_boost": True
        }
        
        # 사용자 설정이 있으면 기본값과 병합
        final_voice_settings = default_voice_settings.copy()
        if voice_settings:
            final_voice_settings.update(voice_settings)
        
        self.user_voice_mappings[client_id] = {
            "voice_id": voice_id,
            "voice_settings": final_voice_settings
        }
        
        logger.info(f"클라이언트 {client_id}의 음성 매핑 설정: {voice_id}")
    
    def get_user_voice_config(self, client_id: str) -> dict:
        """
        클라이언트별 음성 설정 반환
        
        Args:
            client_id (str): 클라이언트 식별자
            
        Returns:
            dict: 음성 설정 (voice_id, voice_settings 포함)
        """
        if client_id in self.user_voice_mappings:
            return self.user_voice_mappings[client_id]
        
        # 기본 부기 음성 설정 반환
        return {
            "voice_id": self.bugi_voice_config["voice_id"],
            "voice_settings": self.bugi_voice_config["voice_settings"]
        }
    
    def remove_user_voice_mapping(self, client_id: str):
        """
        클라이언트의 음성 매핑 제거
        
        Args:
            client_id (str): 클라이언트 식별자
        """
        if client_id in self.user_voice_mappings:
            del self.user_voice_mappings[client_id]
            logger.info(f"클라이언트 {client_id}의 음성 매핑 제거됨")

    async def transcribe_audio(self, file_path: str, language: str = "ko"):
        """
        Whisper로 음성 파일을 텍스트로 변환
        
        Args:
            file_path (str): 오디오 파일 경로
            language (str): 인식할 언어 코드 (기본값: 'ko' - 한국어)
        
        Returns:
            tuple: (텍스트, 오류 메시지, 오류 코드, 품질 정보)
        """
        if self.whisper_model is None:
            logger.error("Whisper 모델이 초기화되지 않았습니다")
            return "", "Whisper 모델이 초기화되지 않았습니다", "whisper_not_initialized", None
            
        try:
            file_size = os.path.getsize(file_path)
            file_ext = os.path.splitext(file_path)[1].lower()
            logger.info(f"오디오 파일 변환 시작: {file_path}")
            logger.info(f"파일 정보 - 크기: {file_size} 바이트, 확장자: {file_ext}")
            
            if file_size < 1000:  # 1KB 미만
                logger.warning("오디오 파일이 너무 작습니다")
                return "", "오디오 파일이 너무 작습니다", "audio_too_small", None
            
            # 지원되는 오디오 형식 확인
            supported_formats = ['.wav', '.mp3', '.m4a', '.ogg', '.webm', '.flac']
            if file_ext not in supported_formats:
                logger.warning(f"지원되지 않는 오디오 형식: {file_ext}")
                # Whisper는 대부분의 형식을 지원하므로 경고만 출력하고 계속 진행
            
            logger.info(f"Whisper 모델({whisper_model_name})로 변환 시작...")
            result = await asyncio.to_thread(
                self.whisper_model.transcribe, 
                file_path, 
                language=language,
                fp16=False
            )
            
            text = result.get("text", "").strip()
            
            # 강화된 품질 검증
            quality_info = self._analyze_stt_quality(result, text, file_size)
            
            logger.info(f"Whisper 변환 결과: '{text}' (길이: {len(text)}자)")
            logger.info(f"품질 분석: {quality_info}")
            
            if hasattr(result, 'segments') and result.segments:
                logger.info(f"세그먼트 수: {len(result.segments)}")
            
            # 다단계 검증
            validation_result = self._validate_stt_result(text, quality_info)
            
            if not validation_result["is_valid"]:
                logger.warning(f"STT 결과 품질 부족: {validation_result['reason']}")
                return "", validation_result["reason"], validation_result["error_code"], quality_info
            
            return text, None, None, quality_info
            
        except Exception as e:
            error_detail = traceback.format_exc()
            logger.error(f"Whisper 오류: {e}\n{error_detail}")
            return "", f"Whisper 오류: {e}", "whisper_error", None
    
    def _analyze_stt_quality(self, whisper_result: dict, text: str, file_size: int) -> dict:
        """
        Whisper STT 결과의 품질 분석
        
        Args:
            whisper_result (dict): Whisper 결과 딕셔너리
            text (str): 인식된 텍스트
            file_size (int): 원본 오디오 파일 크기
            
        Returns:
            dict: 품질 분석 정보
        """
        quality_info = {
            "text_length": len(text),
            "word_count": len(text.split()) if text else 0,
            "file_size_kb": file_size / 1024,
            "segments_count": 0,
            "avg_confidence": 0.0,
            "min_confidence": 1.0,
            "has_no_speech_segments": False,
            "quality_score": 0.0
        }
        
        # 세그먼트 분석 (Whisper segments 정보 활용)
        segments = whisper_result.get("segments", [])
        if segments:
            quality_info["segments_count"] = len(segments)
            
            confidences = []
            no_speech_count = 0
            
            for segment in segments:
                # 세그먼트별 신뢰도 추출 (일부 Whisper 버전에서 제공)
                if "avg_logprob" in segment:
                    # avg_logprob을 신뢰도로 변환 (대략적)
                    confidence = max(0.0, min(1.0, (segment["avg_logprob"] + 1.0)))
                    confidences.append(confidence)
                
                # no_speech_prob 체크
                if segment.get("no_speech_prob", 0.0) > 0.7:
                    no_speech_count += 1
            
            if confidences:
                quality_info["avg_confidence"] = sum(confidences) / len(confidences)
                quality_info["min_confidence"] = min(confidences)
            
            quality_info["has_no_speech_segments"] = no_speech_count > 0
        
        # 전체 품질 점수 계산 (0.0 ~ 1.0)
        score = 0.0
        
        # 텍스트 길이 점수 (30%)
        if quality_info["text_length"] >= 3:
            text_score = min(1.0, quality_info["text_length"] / 50.0)  # 50자 기준
            score += text_score * 0.3
        
        # 단어 수 점수 (20%)
        if quality_info["word_count"] >= 1:
            word_score = min(1.0, quality_info["word_count"] / 10.0)  # 10단어 기준
            score += word_score * 0.2
        
        # 신뢰도 점수 (40%)
        if quality_info["avg_confidence"] > 0:
            score += quality_info["avg_confidence"] * 0.4
        else:
            # 신뢰도 정보가 없으면 기본 점수
            score += 0.6 * 0.4
        
        # 파일 크기 점수 (10%)
        if quality_info["file_size_kb"] >= 10:  # 10KB 이상
            size_score = min(1.0, quality_info["file_size_kb"] / 100.0)  # 100KB 기준
            score += size_score * 0.1
        
        # no_speech 세그먼트가 있으면 점수 감점
        if quality_info["has_no_speech_segments"]:
            score *= 0.8
        
        quality_info["quality_score"] = score
        
        return quality_info
    
    def _validate_stt_result(self, text: str, quality_info: dict) -> dict:
        """
        STT 결과 검증 (동화 프로젝트용 - 더 관대한 기준)
        
        Args:
            text (str): 인식된 텍스트
            quality_info (dict): 품질 분석 정보
            
        Returns:
            dict: 검증 결과
        """
        # 기본 검증
        if not text:
            return {
                "is_valid": False,
                "reason": "목소리가 잘 안 들려요. 더 크게 말해줄 수 있어요?",
                "error_code": "no_speech_detected"
            }
        
        # 매우 짧은 텍스트 검증 (완화됨)
        if len(text) < 1:  # 1글자로 완화 (기존 2글자)
            return {
                "is_valid": False,
                "reason": "조금 더 말해줄 수 있어요?",
                "error_code": "text_too_short"
            }
        
        # 품질 점수 기반 검증 (매우 관대함)
        if quality_info.get("quality_score", 1.0) < 0.1:  # 0.3 → 0.1로 완화
            return {
                "is_valid": False,
                "reason": "잘 안 들려요. 다시 말해줄 수 있어요?",
                "error_code": "low_quality_audio"
            }
        
        # 신뢰도 기반 검증 (매우 관대함)
        if quality_info.get("avg_confidence", 1.0) > 0 and quality_info["avg_confidence"] < 0.2:  # 0.4 → 0.2로 완화
            return {
                "is_valid": False,
                "reason": "잘 안 들려요. 다시 말해줄 수 있어요?",
                "error_code": "low_confidence"
            }
        
        # 모든 검증 통과 (대부분의 음성 허용)
        return {
            "is_valid": True,
            "reason": None,
            "error_code": None
        }

    async def synthesize_tts(self, text: str, voice: str = None, client_id: str = None):
        """
        ElevenLabs API를 사용한 텍스트를 TTS로 변환
        
        Args:
            text (str): 텍스트 응답
            voice (str): 사용할 음성 (직접 지정, 옵션)
            client_id (str): 클라이언트 식별자 (클론 음성 사용을 위해 추가)
        
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
            
            # 음성 설정 결정 (우선순위: 직접 지정 > 클라이언트별 설정 > 기본 설정)
            if voice:
                # 직접 지정된 음성 사용
                voice_id = voice
                voice_settings = self.bugi_voice_config["voice_settings"]
                model_id = self.bugi_voice_config["model_id"]
                logger.info(f"직접 지정된 음성 사용: {voice_id}")
            elif client_id:
                # 클라이언트별 음성 설정 사용 (클론 음성 포함)
                user_config = self.get_user_voice_config(client_id)
                voice_id = user_config["voice_id"]
                voice_settings = user_config["voice_settings"]
                model_id = self.bugi_voice_config["model_id"]  # 모델은 기본값 사용
                
                if client_id in self.user_voice_mappings:
                    logger.info(f"클라이언트 {client_id}의 클론 음성 사용: {voice_id}")
                else:
                    logger.info(f"클라이언트 {client_id}의 기본 부기 음성 사용: {voice_id}")
            else:
                # 기본 부기 음성 설정 사용
                voice_id = self.bugi_voice_config["voice_id"]
                voice_settings = self.bugi_voice_config["voice_settings"]
                model_id = self.bugi_voice_config["model_id"]
                logger.info(f"기본 부기 음성 사용: {voice_id}")
            
            # 텍스트 전처리 (부기 스타일)
            cleaned_text = self._prepare_bugi_text_for_speech(text)
            
            # 정리된 텍스트 재검증
            if not cleaned_text or not cleaned_text.strip():
                logger.warning(f"텍스트 정리 후 빈 내용됨. 원본: {repr(text)}, 정리후: {repr(cleaned_text)}")
                return "", "error", "텍스트 정리 후 빈 내용이 되었습니다", "tts_cleaned_empty"
            
            async def elevenlabs_tts_operation():
                # ElevenLabs API 요청 데이터
                data = {
                    "text": cleaned_text,
                    "model_id": model_id,
                    "voice_settings": voice_settings,
                    "output_format": "wav_44100"
                }
                
                url = f"{ELEVENLABS_BASE_URL}/text-to-speech/{voice_id}"
                headers = get_elevenlabs_headers(self.elevenlabs_api_key)
                
                # SSL 컨텍스트 설정
                ssl_context = ssl.create_default_context()
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE
                
                timeout = aiohttp.ClientTimeout(total=120)
                
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(url, headers=headers, json=data, ssl=ssl_context) as response:
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
    
    def _prepare_bugi_text_for_speech(self, text: str) -> str:
        """부기(ChatBot A) 전용 텍스트 전처리 - 동화 친화적 스타일"""
        
        # 기본 정리
        text = text.strip()
        if not text:
            return ""
        
        # 부기의 친근한 말투 최적화
        # 감정 표현 강화
        if "안녕" in text:
            text = f"[cheerful] {text}"
        elif "미안" in text or "죄송" in text:
            text = f"[apologetic] {text}"
        elif "와!" in text or "우와" in text:
            text = f"[excited] {text}"
        elif "음..." in text or "글쎄" in text:
            text = f"[thoughtful] {text}"
        elif "?" in text:
            text = f"[curious] {text}"
        elif "!" in text and any(word in text for word in ["좋아", "재미", "신나", "멋져"]):
            text = f"[enthusiastic] {text}"
        elif "그래" in text or "맞아" in text:
            text = f"[agreeing] {text}"
        else:
            # 기본적으로 친근한 톤
            text = f"{text} (친근하게)"
        
        # 길이 제한
        if len(text) > 500:  # 부기는 짧고 명확하게
            text = text[:497] + "..."
        
        return text
    
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
    
    def _detect_audio_format(self, audio_data: bytes) -> str:
        """
        오디오 데이터의 헤더를 분석해서 형식을 감지
        
        Args:
            audio_data (bytes): 오디오 데이터
        
        Returns:
            str: 파일 확장자 (.wav, .mp3, .webm, .ogg)
        """
        if len(audio_data) < 12:
            return ".wav"  # 기본값
        
        # WAV 헤더 감지 (RIFF...WAVE)
        if audio_data[:4] == b'RIFF' and audio_data[8:12] == b'WAVE':
            return ".wav"
        
        # MP3 헤더 감지 (ID3 태그 또는 프레임 헤더)
        if audio_data[:3] == b'ID3' or (audio_data[0:2] == b'\xff\xfb') or (audio_data[0:2] == b'\xff\xf3'):
            return ".mp3"
        
        # WebM 헤더 감지
        if audio_data[:4] == b'\x1a\x45\xdf\xa3':
            return ".webm"
        
        # OGG 헤더 감지
        if audio_data[:4] == b'OggS':
            return ".ogg"
        
        # M4A/AAC 헤더 감지
        if audio_data[4:8] == b'ftyp' and b'M4A' in audio_data[:20]:
            return ".m4a"
        
        # 감지 실패 시 WAV 기본값 (Whisper가 잘 처리함)
        return ".wav"

    async def process_audio_chunk(self, audio_data, client_id):
        """
        오디오 청크 처리 (개선된 형식 감지 및 성능 최적화)
        
        Args:
            audio_data (bytes): 오디오 데이터
            client_id (str): 클라이언트 ID
        
        Returns:
            tuple: (임시 파일 경로, 오류 메시지, 오류 코드)
        """
        temp_file_path = None
        try:
            # 기본 검증
            if not audio_data or len(audio_data) < 100:
                logger.warning(f"오디오 데이터가 너무 작습니다 ({client_id}): {len(audio_data) if audio_data else 0} bytes")
                return None, "오디오 데이터가 너무 작습니다", "audio_data_too_small"
            
            # 오디오 형식 자동 감지
            audio_format = self._detect_audio_format(audio_data)
            
            logger.info(f"오디오 형식 감지 결과 ({client_id}): {audio_format}, 크기: {len(audio_data)} bytes")
            
            # 임시 디렉토리 확인 및 생성
            import tempfile
            temp_dir = tempfile.gettempdir()
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir, exist_ok=True)
            
            # 감지된 형식에 맞는 확장자로 임시 파일 생성
            with tempfile.NamedTemporaryFile(delete=False, suffix=audio_format, prefix=f"audio_{client_id[:8]}_") as temp_file:
                temp_file.write(audio_data)
                temp_file_path = temp_file.name
            
            # 파일 생성 검증
            if not os.path.exists(temp_file_path):
                logger.error(f"임시 파일 생성 실패 ({client_id}): {temp_file_path}")
                return None, "임시 파일 생성 실패", "temp_file_creation_failed"
            
            file_size = os.path.getsize(temp_file_path)
            logger.info(f"오디오 임시 파일 생성 완료 ({client_id}): {temp_file_path} ({file_size} bytes)")
            
            return temp_file_path, None, None
            
        except Exception as e:
            error_detail = traceback.format_exc()
            logger.error(f"오디오 청크 처리 오류 ({client_id}): {e}\n{error_detail}")
            
            # 실패 시 임시 파일 정리
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                    logger.info(f"실패한 임시 파일 정리 완료: {temp_file_path}")
                except Exception as cleanup_error:
                    logger.warning(f"임시 파일 정리 실패: {cleanup_error}")
                    
            return None, f"오디오 처리 오류: {e}", "audio_processing_error" 