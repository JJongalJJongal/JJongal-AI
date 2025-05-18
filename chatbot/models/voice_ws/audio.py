"""
오디오 처리 모듈

이 모듈은 Whisper를 이용한 STT(Speech-to-Text)와 
OpenAI TTS(Text-to-Speech) 기능을 제공합니다.
"""
import os
import base64
import logging
import asyncio
import traceback
import tempfile
import whisper
from openai import OpenAI
from dotenv import load_dotenv
from .utils import retry_operation

# 환경 변수 로드
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
dotenv_path = os.path.join(project_root, '.env')
load_dotenv(dotenv_path=dotenv_path)

# OpenAI API 키 확인
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    logging.warning(f"OPENAI_API_KEY 환경 변수를 찾을 수 없습니다. .env 파일: {dotenv_path}")

# OpenAI 클라이언트 초기화
try:
    openai_client = OpenAI(api_key=api_key)
    logging.info("OpenAI 클라이언트 초기화 성공")
except Exception as e:
    logging.error(f"OpenAI 클라이언트 초기화 실패: {e}")
    openai_client = None

# Whisper 모델 초기화
whisper_model = None
whisper_model_name = os.getenv("WHISPER_MODEL", "base")
try:
    whisper_model = whisper.load_model(whisper_model_name)
    logging.info(f"Whisper 모델({whisper_model_name}) 로드 성공")
except Exception as e:
    logging.error(f"Whisper 모델 로드 실패: {e}")

async def transcribe_audio(file_path: str, language: str = "ko"):
    """
    Whisper로 음성 파일을 텍스트로 변환
    
    Args:
        file_path (str): 오디오 파일 경로
        language (str): 인식할 언어 코드 (기본값: 'ko' - 한국어)
    
    Returns:
        tuple: (텍스트, 오류 메시지, 오류 코드)
    """
    if whisper_model is None:
        logging.error("Whisper 모델이 초기화되지 않았습니다")
        return "", "Whisper 모델이 초기화되지 않았습니다", "whisper_not_initialized"
        
    try:
        file_size = os.path.getsize(file_path)
        logging.info(f"오디오 파일 변환 시작: {file_path}, 크기: {file_size} 바이트")
        
        # 파일이 너무 작으면 무시
        if file_size < 1000:  # 1KB 미만
            logging.warning("오디오 파일이 너무 작습니다")
            return "", "오디오 파일이 너무 작습니다", "audio_too_small"
        
        # 비동기 처리를 위해 스레드풀에서 실행
        result = await asyncio.to_thread(
            whisper_model.transcribe, 
            file_path, 
            language=language,
            fp16=False  # 호환성 향상을 위해 fp16 비활성화
        )
        
        text = result.get("text", "").strip()
        logging.info(f"Whisper 변환 결과: {text}")
        
        # 결과가 빈 문자열이면 처리
        if not text:
            logging.warning("Whisper가 텍스트를 인식하지 못했습니다")
            return "", "음성을 인식하지 못했습니다. 더 명확하게 말씀해주세요.", "no_speech_detected"
            
        return text, None, None
        
    except Exception as e:
        error_detail = traceback.format_exc()
        logging.error(f"Whisper 오류: {e}\n{error_detail}")
        return "", f"Whisper 오류: {e}", "whisper_error"

async def synthesize_tts(text: str, voice: str = "nova"):
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
        
        # OpenAI 클라이언트 확인
        if openai_client is None:
            return "", "error", "OpenAI 클라이언트가 초기화되지 않았습니다", "openai_client_not_initialized"
            
        # TTS 요청 (비동기 방식으로 처리)
        async def tts_operation():
            # 동기 함수를 비동기적으로 실행
            tts_result = await asyncio.to_thread(
                openai_client.audio.speech.create,
                model="tts-1-hd",  # 더 높은 품질의 TTS 사용
                voice=voice,
                input=text,
                speed=0.9,
                response_format="mp3"  # mp3 형식으로 변경 (더 작은 파일 크기)
            )
            return tts_result
            
        # 재시도 메커니즘 적용
        tts_result = await retry_operation(tts_operation)
        
        # retry_operation의 반환값 확인
        if isinstance(tts_result, tuple) and tts_result[0] is None:
            _, error_msg, error_code = tts_result
            logging.error(f"TTS 생성 실패 후 재시도 모두 실패: {error_msg}")
            return "", "error", error_msg or "TTS 생성 실패", error_code or "tts_generation_failed"
        
        # 성공한 경우의 처리
        tts_audio = tts_result.content
        
        # 오디오 크기 확인
        if len(tts_audio) < 2 * 1024 * 1024:  # 2MB 미만
            return base64.b64encode(tts_audio).decode("utf-8"), "ok", None, None
        else:
            # 큰 오디오 파일 처리 개선
            logging.warning(f"TTS 오디오 크기 초과: {len(tts_audio)} 바이트")
            
            # 큰 파일이지만 일부 반환
            return base64.b64encode(tts_audio).decode("utf-8"), "large_file", "오디오 크기가 큽니다", "tts_large_file"
    except Exception as e:
        error_detail = traceback.format_exc()
        logging.error(f"TTS 오류: {e}\n{error_detail}")
        return "", "error", f"TTS 오류: {e}", "tts_error"

async def process_audio_chunk(audio_data, client_id):
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
        # 임시 파일 생성 및 데이터 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_file.write(audio_data)
            temp_file_path = temp_file.name
            
        return temp_file_path, None, None
    except Exception as e:
        error_detail = traceback.format_exc()
        logging.error(f"오디오 청크 처리 오류: {e}\n{error_detail}")
        
        # 임시 파일 삭제 시도
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except:
                pass
                
        return None, f"오디오 처리 오류: {e}", "audio_processing_error" 