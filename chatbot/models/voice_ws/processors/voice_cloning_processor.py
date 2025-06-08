"""
실시간 음성 클로닝 프로세서

사용자 음성 샘플을 수집하고 ElevenLabs Voice Cloning API를 사용하여
새로운 음성 클론을 생성합니다.
"""
import os
import ssl
import asyncio
import aiohttp
import tempfile
import traceback
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv

from shared.utils.logging_utils import get_module_logger
from elevenlabs import ElevenLabs

logger = get_module_logger(__name__)

# 환경 변수 로드
load_dotenv()

class VoiceCloningProcessor:
    """
    실시간 음성 클로닝을 위한 프로세서
    
    기능:
    1. 사용자 음성 샘플 수집 및 저장
    2. ElevenLabs Instant Voice Cloning API 호출
    3. 생성된 음성 ID 반환 및 관리
    """
    
    def __init__(self, elevenlabs_api_key: str = None):
        """
        음성 클로닝 프로세서 초기화
        
        Args:
            elevenlabs_api_key: ElevenLabs API 키
        """
        self.logger = get_module_logger(__name__)
        
        # ElevenLabs 클라이언트 설정
        if elevenlabs_api_key:
            self.client = ElevenLabs(api_key=elevenlabs_api_key)
        else:
            api_key = os.getenv("ELEVENLABS_API_KEY")
            if not api_key:
                self.logger.error("ElevenLabs API 키가 설정되지 않았습니다")
                raise ValueError("ElevenLabs API 키가 필요합니다")
            self.client = ElevenLabs(api_key=api_key)
        
        # 음성 샘플 저장소 - 통일된 temp 경로 사용
        self.temp_audio_dir = Path("output") / "temp" / "voice_samples"  # 중복 제거
        self.temp_audio_dir.mkdir(parents=True, exist_ok=True)
        
        # 클로닝 설정
        self.min_samples_required = 5  # 최소 필요 샘플 수
        self.max_sample_duration = 30  # 최대 샘플 길이 (초)
        self.min_sample_duration = 3   # 최소 샘플 길이 (초)
        
        # 사용자별 음성 데이터 관리
        self.user_voice_data = {}  # {child_name: {"samples": [...], "voice_id": "...", "clone_status": "..."}}
        
        self.logger.info(f"음성 클로닝 프로세서 초기화 완료 (샘플 저장소: {self.temp_audio_dir})")
    
    async def collect_user_audio_sample(self, user_id: str, audio_data: bytes) -> bool:
        """
        사용자 음성 샘플 수집 및 저장
        
        Args:
            user_id: 사용자 식별자
            audio_data: 오디오 바이트 데이터
            
        Returns:
            bool: 저장 성공 여부
        """
        try:
            # 사용자별 폴더 생성
            user_audio_dir = self.temp_audio_dir / user_id
            user_audio_dir.mkdir(exist_ok=True)
            
            # 타임스탬프 기반 파일명
            import time
            timestamp = int(time.time())
            audio_file_path = user_audio_dir / f"sample_{timestamp}.mp3"
            
            # 오디오 파일 저장
            with open(audio_file_path, 'wb') as f:
                f.write(audio_data)
            
            # 사용자 샘플 목록에 추가
            if user_id not in self.user_voice_data:
                self.user_voice_data[user_id] = {"samples": [], "voice_id": None, "clone_status": "ready"}
            self.user_voice_data[user_id]["samples"].append(str(audio_file_path))
            
            logger.info(f"사용자 {user_id} 음성 샘플 저장 완료: {audio_file_path} ({len(audio_data)} bytes)")
            logger.info(f"현재 {user_id}의 샘플 수: {len(self.user_voice_data[user_id]['samples'])}")
            
            return True
            
        except Exception as e:
            logger.error(f"음성 샘플 저장 실패: {e}")
            return False
    
    async def create_instant_voice_clone(self, user_id: str, voice_name: str = None) -> Tuple[Optional[str], Optional[str]]:
        """
        사용자 음성 샘플로 Instant Voice Clone 생성
        
        Args:
            user_id: 사용자 식별자
            voice_name: 생성할 음성의 이름 (기본값: user_id 기반)
            
        Returns:
            Tuple[voice_id, error_message]: 생성된 음성 ID와 에러 메시지
        """
        if not self.client:
            return None, "ElevenLabs 클라이언트가 설정되지 않았습니다"
        
        # 음성 샘플 확인
        if user_id not in self.user_voice_data or not self.user_voice_data[user_id]["samples"]:
            return None, f"사용자 {user_id}의 음성 샘플이 없습니다"
        
        voice_name = voice_name or f"{user_id}_voice_clone"
        
        try:
            # SSL 컨텍스트 설정
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            
            timeout = aiohttp.ClientTimeout(total=300)  # 5분 타임아웃
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
                # Multipart form data 준비
                data = aiohttp.FormData()
                data.add_field('name', voice_name)
                data.add_field('description', f'실시간 생성된 {user_id} 음성 클론')
                data.add_field('remove_background_noise', 'true')
                
                # 음성 파일들 추가 (최대 3개만 사용)
                sample_files = self.user_voice_data[user_id]["samples"][-3:]  # 최근 3개 샘플
                for i, file_path in enumerate(sample_files):
                    with open(file_path, 'rb') as f:
                        file_data = f.read()
                        data.add_field('files', file_data, filename=f'sample_{i}.mp3', content_type='audio/mpeg')
                
                # ElevenLabs IVC API 호출
                url = f"{self.client.base_url}/voices/add"
                headers = {"xi-api-key": self.client.api_key}
                
                async with session.post(url, headers=headers, data=data, ssl=ssl_context) as response:
                    response_text = await response.text()
                    
                    if response.status == 200:
                        result = await response.json()
                        voice_id = result.get("voice_id")
                        
                        if voice_id:
                            # 생성된 음성 ID 저장
                            self.user_voice_data[user_id]["voice_id"] = voice_id
                            self.user_voice_data[user_id]["clone_status"] = "ready"
                            logger.info(f"사용자 {user_id} 음성 클론 생성 성공: {voice_id}")
                            return voice_id, None
                        else:
                            return None, "음성 ID가 응답에 포함되지 않았습니다"
                    else:
                        error_msg = f"ElevenLabs API 오류 ({response.status}): {response_text}"
                        logger.error(error_msg)
                        return None, error_msg
                        
        except Exception as e:
            error_detail = traceback.format_exc()
            logger.error(f"음성 클론 생성 실패: {e}\n{error_detail}")
            return None, f"음성 클론 생성 중 오류: {e}"
    
    def get_user_voice_id(self, user_id: str) -> Optional[str]:
        """
        사용자의 클론된 음성 ID 반환
        
        Args:
            user_id: 사용자 식별자
            
        Returns:
            Optional[str]: 클론된 음성 ID (없으면 None)
        """
        return self.user_voice_data.get(user_id, {}).get("voice_id")
    
    def get_sample_count(self, user_id: str) -> int:
        """
        사용자의 수집된 음성 샘플 수 반환
        
        Args:
            user_id: 사용자 식별자
            
        Returns:
            int: 음성 샘플 수
        """
        return len(self.user_voice_data.get(user_id, {}).get("samples", []))
    
    def is_ready_for_cloning(self, user_id: str, min_samples: int = 5) -> bool:
        """
        음성 클로닝 준비 완료 여부 확인
        
        Args:
            user_id: 사용자 식별자
            min_samples: 최소 필요 샘플 수
            
        Returns:
            bool: 클로닝 준비 완료 여부
        """
        sample_count = self.get_sample_count(user_id)
        return sample_count >= min_samples
    
    async def cleanup_user_samples(self, user_id: str) -> bool:
        """
        사용자 음성 샘플 정리
        
        Args:
            user_id: 사용자 식별자
            
        Returns:
            bool: 정리 성공 여부
        """
        try:
            # 파일 시스템에서 샘플 파일들 삭제
            if user_id in self.user_voice_data:
                for file_path in self.user_voice_data[user_id]["samples"]:
                    try:
                        Path(file_path).unlink(missing_ok=True)
                    except Exception as e:
                        logger.warning(f"샘플 파일 삭제 실패: {file_path} - {e}")
                
                # 메모리에서 제거
                del self.user_voice_data[user_id]
                logger.info(f"사용자 {user_id} 음성 샘플 정리 완료")
            
            return True
            
        except Exception as e:
            logger.error(f"사용자 샘플 정리 실패: {e}")
            return False 