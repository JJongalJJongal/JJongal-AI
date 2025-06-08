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
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv

from shared.utils.logging_utils import get_module_logger
from elevenlabs import ElevenLabs

# 고급 오디오 분석을 위한 라이브러리
try:
    import librosa
    import soundfile as sf
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    librosa = None
    sf = None

logger = get_module_logger(__name__)

# 환경 변수 로드
load_dotenv()

class VoiceCloningProcessor:
    """
    실시간 음성 클로닝을 위한 프로세서
    
    기능:
    1. 사용자 음성 샘플 수집 및 저장
    2. 고급 오디오 품질 분석 (SNR, 주파수 분석)
    3. ElevenLabs Instant Voice Cloning API 호출
    4. 생성된 음성 ID 반환 및 관리
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
        
        # 클로닝 설정 (강화된 품질 기준)
        self.min_samples_required = 5  # 최소 필요 샘플 수
        self.max_sample_duration = 30  # 최대 샘플 길이 (초)
        self.min_sample_duration = 3   # 최소 샘플 길이 (초)
        
        # 품질 기준 (새로 추가)
        self.min_snr_db = 10.0  # 최소 SNR (Signal-to-Noise Ratio) 10dB
        self.min_quality_score = 0.6  # 최소 품질 점수
        self.max_noise_level = 0.3  # 최대 노이즈 레벨
        
        # 사용자별 음성 데이터 관리
        self.user_voice_data = {}
        
        # librosa 가용성 확인
        if LIBROSA_AVAILABLE:
            self.logger.info(f"음성 클로닝 프로세서 초기화 완료 (고급 품질 분석 활성화, 샘플 저장소: {self.temp_audio_dir})")
        else:
            self.logger.warning(f"librosa 라이브러리가 없어 기본 품질 분석만 사용합니다. (샘플 저장소: {self.temp_audio_dir})")
    
    async def collect_user_audio_sample(self, user_id: str, audio_data: bytes) -> bool:
        """
        사용자 음성 샘플 수집 및 저장 (강화된 품질 검증 포함)
        
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
            
            # 오디오 형식 감지
            if audio_data[:4] == b'RIFF' and audio_data[8:12] == b'WAVE':
                extension = '.wav'
            elif audio_data[:3] == b'ID3' or (audio_data[0:2] == b'\xff\xfb') or (audio_data[0:2] == b'\xff\xf3'):
                extension = '.mp3'
            else:
                extension = '.wav'  # 기본값
            
            audio_file_path = user_audio_dir / f"sample_{timestamp}{extension}"
            
            # 오디오 파일 저장
            with open(audio_file_path, 'wb') as f:
                f.write(audio_data)
            
            # 고급 품질 분석 수행
            quality_analysis = await self._analyze_audio_quality(audio_file_path, audio_data)
            
            # 품질 기준 검증
            if not self._validate_audio_quality(quality_analysis):
                logger.warning(f"사용자 {user_id} 음성 샘플 품질 부족: {quality_analysis}")
                # 품질이 낮은 샘플은 삭제
                os.remove(audio_file_path)
                return False
            
            # 사용자 샘플 목록에 추가 (품질 정보 포함)
            if user_id not in self.user_voice_data:
                self.user_voice_data[user_id] = {"samples": [], "voice_id": None, "clone_status": "ready"}
            
            self.user_voice_data[user_id]["samples"].append({
                "path": str(audio_file_path),
                "quality": quality_analysis,
                "timestamp": timestamp
            })
            
            logger.info(f"사용자 {user_id} 고품질 음성 샘플 저장 완료: {audio_file_path}")
            logger.info(f"품질 분석: SNR={quality_analysis.get('snr_db', 'N/A')}dB, 점수={quality_analysis.get('quality_score', 'N/A')}")
            logger.info(f"현재 {user_id}의 고품질 샘플 수: {len(self.user_voice_data[user_id]['samples'])}")
            
            return True
            
        except Exception as e:
            logger.error(f"음성 샘플 저장 실패: {e}")
            return False
    
    async def _analyze_audio_quality(self, audio_file_path: Path, audio_data: bytes) -> Dict[str, Any]:
        """
        고급 오디오 품질 분석
        
        Args:
            audio_file_path: 오디오 파일 경로
            audio_data: 오디오 바이트 데이터
            
        Returns:
            Dict: 품질 분석 결과
        """
        quality_analysis = {
            "file_size_kb": len(audio_data) / 1024,
            "duration_seconds": 0.0,
            "sample_rate": 0,
            "snr_db": 0.0,
            "noise_level": 0.0,
            "spectral_centroid_mean": 0.0,
            "zero_crossing_rate": 0.0,
            "rms_energy": 0.0,
            "quality_score": 0.0,
            "has_clipping": False,
            "analysis_method": "basic"
        }
        
        if not LIBROSA_AVAILABLE:
            # librosa가 없으면 기본 분석만
            quality_analysis["quality_score"] = 0.7 if len(audio_data) > 10000 else 0.3
            return quality_analysis
        
        try:
            # librosa로 오디오 로드
            y, sr = librosa.load(str(audio_file_path), sr=None)
            quality_analysis["sample_rate"] = sr
            quality_analysis["duration_seconds"] = len(y) / sr
            quality_analysis["analysis_method"] = "advanced"
            
            # 1. SNR 계산 (Signal-to-Noise Ratio)
            snr_db = self._calculate_snr(y)
            quality_analysis["snr_db"] = snr_db
            
            # 2. RMS 에너지 (음성 강도)
            rms = librosa.feature.rms(y=y)[0]
            quality_analysis["rms_energy"] = float(np.mean(rms))
            
            # 3. 스펙트럴 센트로이드 (음성 밝기/명확도)
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            quality_analysis["spectral_centroid_mean"] = float(np.mean(spectral_centroids))
            
            # 4. 제로 크로싱 레이트 (음성 활동성)
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            quality_analysis["zero_crossing_rate"] = float(np.mean(zcr))
            
            # 5. 클리핑 감지
            max_amplitude = np.max(np.abs(y))
            quality_analysis["has_clipping"] = max_amplitude > 0.95
            
            # 6. 노이즈 레벨 추정
            noise_level = self._estimate_noise_level(y, sr)
            quality_analysis["noise_level"] = noise_level
            
            # 7. 종합 품질 점수 계산
            quality_score = self._calculate_quality_score(quality_analysis)
            quality_analysis["quality_score"] = quality_score
            
            logger.debug(f"고급 오디오 분석 완료: {audio_file_path}")
            
        except Exception as e:
            logger.error(f"고급 오디오 분석 실패: {e}, 기본 분석 사용")
            quality_analysis["quality_score"] = 0.5  # 분석 실패 시 중간 점수
        
        return quality_analysis
    
    def _calculate_snr(self, y: np.ndarray) -> float:
        """
        SNR (Signal-to-Noise Ratio) 계산
        
        Args:
            y: 오디오 신호 배열
            
        Returns:
            float: SNR 값 (dB)
        """
        try:
            # 음성 활동 구간과 노이즈 구간 분리 (간단한 방법)
            rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
            rms_threshold = np.percentile(rms, 30)  # 하위 30%를 노이즈로 간주
            
            signal_frames = rms > rms_threshold
            noise_frames = rms <= rms_threshold
            
            if np.sum(signal_frames) == 0 or np.sum(noise_frames) == 0:
                return 10.0  # 기본값
            
            signal_power = np.mean(rms[signal_frames] ** 2)
            noise_power = np.mean(rms[noise_frames] ** 2)
            
            if noise_power == 0:
                return 30.0  # 매우 높은 SNR
            
            snr_linear = signal_power / noise_power
            snr_db = 10 * np.log10(snr_linear)
            
            return float(np.clip(snr_db, -10.0, 50.0))  # -10dB ~ 50dB 범위로 제한
            
        except:
            return 10.0  # 계산 실패 시 기본값
    
    def _estimate_noise_level(self, y: np.ndarray, sr: int) -> float:
        """
        노이즈 레벨 추정
        
        Args:
            y: 오디오 신호 배열
            sr: 샘플링 레이트
            
        Returns:
            float: 노이즈 레벨 (0.0 ~ 1.0)
        """
        try:
            # 고주파 노이즈 분석
            stft = librosa.stft(y)
            magnitude = np.abs(stft)
            
            # 고주파 대역의 에너지 (4kHz 이상)
            freq_bins = librosa.fft_frequencies(sr=sr, n_fft=2048)
            high_freq_mask = freq_bins > 4000
            
            if np.sum(high_freq_mask) > 0:
                high_freq_energy = np.mean(magnitude[high_freq_mask, :])
                total_energy = np.mean(magnitude)
                
                noise_ratio = high_freq_energy / (total_energy + 1e-8)
                return float(np.clip(noise_ratio, 0.0, 1.0))
            
            return 0.1  # 기본 노이즈 레벨
            
        except:
            return 0.1
    
    def _calculate_quality_score(self, analysis: Dict[str, Any]) -> float:
        """
        종합 품질 점수 계산
        
        Args:
            analysis: 품질 분석 결과
            
        Returns:
            float: 품질 점수 (0.0 ~ 1.0)
        """
        score = 0.0
        
        # SNR 점수 (40%)
        snr_db = analysis.get("snr_db", 0)
        snr_score = np.clip((snr_db - 5) / 20.0, 0.0, 1.0)  # 5dB~25dB 범위
        score += snr_score * 0.4
        
        # RMS 에너지 점수 (20%)
        rms_energy = analysis.get("rms_energy", 0)
        rms_score = np.clip(rms_energy / 0.1, 0.0, 1.0)  # 0.1 기준
        score += rms_score * 0.2
        
        # 노이즈 레벨 점수 (20%)
        noise_level = analysis.get("noise_level", 1.0)
        noise_score = 1.0 - np.clip(noise_level, 0.0, 1.0)
        score += noise_score * 0.2
        
        # 지속 시간 점수 (10%)
        duration = analysis.get("duration_seconds", 0)
        duration_score = np.clip(duration / 10.0, 0.0, 1.0)  # 10초 기준
        score += duration_score * 0.1
        
        # 파일 크기 점수 (10%)
        file_size_kb = analysis.get("file_size_kb", 0)
        size_score = np.clip(file_size_kb / 100.0, 0.0, 1.0)  # 100KB 기준
        score += size_score * 0.1
        
        # 클리핑 패널티
        if analysis.get("has_clipping", False):
            score *= 0.7
        
        return float(np.clip(score, 0.0, 1.0))
    
    def _validate_audio_quality(self, quality_analysis: Dict[str, Any]) -> bool:
        """
        오디오 품질 검증
        
        Args:
            quality_analysis: 품질 분석 결과
            
        Returns:
            bool: 품질 기준 통과 여부
        """
        # 기본 크기 및 지속 시간 체크
        if quality_analysis["file_size_kb"] < 10:  # 10KB 미만
            logger.debug("품질 검증 실패: 파일 크기 너무 작음")
            return False
        
        if quality_analysis["duration_seconds"] < self.min_sample_duration:
            logger.debug(f"품질 검증 실패: 지속 시간 부족 ({quality_analysis['duration_seconds']}s < {self.min_sample_duration}s)")
            return False
        
        # 고급 분석이 가능한 경우
        if quality_analysis["analysis_method"] == "advanced":
            # SNR 체크
            if quality_analysis["snr_db"] < self.min_snr_db:
                logger.debug(f"품질 검증 실패: SNR 부족 ({quality_analysis['snr_db']}dB < {self.min_snr_db}dB)")
                return False
            
            # 노이즈 레벨 체크
            if quality_analysis["noise_level"] > self.max_noise_level:
                logger.debug(f"품질 검증 실패: 노이즈 레벨 과다 ({quality_analysis['noise_level']} > {self.max_noise_level})")
                return False
            
            # 클리핑 체크
            if quality_analysis["has_clipping"]:
                logger.debug("품질 검증 실패: 오디오 클리핑 감지")
                return False
        
        # 종합 품질 점수 체크
        if quality_analysis["quality_score"] < self.min_quality_score:
            logger.debug(f"품질 검증 실패: 품질 점수 부족 ({quality_analysis['quality_score']} < {self.min_quality_score})")
            return False
        
        logger.debug("품질 검증 통과")
        return True
    
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
                        
                        # 파일 확장자에 따른 content_type 설정
                        if file_path.endswith('.wav'):
                            content_type = 'audio/wav'
                            filename = f'sample_{i}.wav'
                        elif file_path.endswith('.mp3'):
                            content_type = 'audio/mpeg'
                            filename = f'sample_{i}.mp3'
                        else:
                            content_type = 'audio/wav'  # 기본값
                            filename = f'sample_{i}.wav'
                        
                        data.add_field('files', file_data, filename=filename, content_type=content_type)
                
                # ElevenLabs IVC API 호출
                url = f"{self.client.base_url}/voices/add"
                headers = {"xi-api-key": self.client.api_key}
                
                async with session.post(url, headers=headers, data=data, ssl=ssl_context) as response:
                    response_text = await response.text()
                    logger.info(f"ElevenLabs API 응답 상태: {response.status}")
                    logger.debug(f"ElevenLabs API 응답 내용: {response_text[:500]}...")
                    
                    if response.status == 200:
                        try:
                            result = await response.json()
                            voice_id = result.get("voice_id")
                            
                            if voice_id:
                                # 생성된 음성 ID 저장
                                self.user_voice_data[user_id]["voice_id"] = voice_id
                                self.user_voice_data[user_id]["clone_status"] = "ready"
                                logger.info(f"사용자 {user_id} 음성 클론 생성 성공: {voice_id}")
                                return voice_id, None
                            else:
                                return None, f"음성 ID가 응답에 포함되지 않았습니다. 응답: {result}"
                        except Exception as json_error:
                            return None, f"응답 JSON 파싱 실패: {json_error}, 응답: {response_text}"
                    elif response.status == 201:
                        # ElevenLabs는 종종 201 Created를 반환
                        try:
                            result = await response.json()
                            voice_id = result.get("voice_id")
                            
                            if voice_id:
                                self.user_voice_data[user_id]["voice_id"] = voice_id
                                self.user_voice_data[user_id]["clone_status"] = "ready"
                                logger.info(f"사용자 {user_id} 음성 클론 생성 성공 (201): {voice_id}")
                                return voice_id, None
                            else:
                                return None, f"음성 ID가 응답에 포함되지 않았습니다. 응답: {result}"
                        except Exception as json_error:
                            return None, f"응답 JSON 파싱 실패: {json_error}, 응답: {response_text}"
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
        사용자의 고품질 샘플 수 반환
        
        Args:
            user_id: 사용자 식별자
            
        Returns:
            int: 고품질 샘플 수
        """
        if user_id in self.user_voice_data:
            return len(self.user_voice_data[user_id]["samples"])
        return 0
    
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
                        Path(file_path["path"]).unlink(missing_ok=True)
                    except Exception as e:
                        logger.warning(f"샘플 파일 삭제 실패: {file_path['path']} - {e}")
                
                # 메모리에서 제거
                del self.user_voice_data[user_id]
                logger.info(f"사용자 {user_id} 음성 샘플 정리 완료")
            
            return True
            
        except Exception as e:
            logger.error(f"사용자 샘플 정리 실패: {e}")
            return False 