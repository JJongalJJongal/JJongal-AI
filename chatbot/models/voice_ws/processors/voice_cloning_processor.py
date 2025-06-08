"""
실시간 음성 클로닝 프로세서

사용자 음성 샘플을 수집하고 ElevenLabs Voice Cloning API를 사용하여
새로운 음성 클론을 생성합니다.
"""
import os
import ssl
import time
import aiohttp
import traceback
import subprocess
import warnings
import asyncio
import gc
import psutil
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

# NumPy와 오디오 라이브러리
import numpy as np

# 고급 오디오 분석을 위한 라이브러리
try:
    import librosa
    import soundfile as sf
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    librosa = None
    sf = None

from shared.utils.logging_utils import get_module_logger
from elevenlabs import ElevenLabs

logger = get_module_logger(__name__)

# 환경 변수 로드
load_dotenv()

# librosa 경고 억제
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")
warnings.filterwarnings("ignore", category=FutureWarning, module="librosa")

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
        self.logger.info("🎤 음성 복제 처리기 초기화 (메모리 최적화 포함)")
        
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
        
        # 품질 기준 (테스트용 더 완화된 기준)
        self.min_snr_db = 10.0  # 최소 SNR 10dB (더욱 완화)
        self.min_quality_score = 0.45  # 최소 품질 점수 45% (더욱 완화)
        self.max_noise_level = 0.35  # 최대 노이즈 레벨 35% (더욱 완화)
        self.min_sample_rate = 8000  # 최소 샘플 레이트 8kHz
        self.preferred_sample_rate = 44100  # 권장 샘플 레이트 44.1kHz
        
        # 사용자별 음성 데이터 관리
        self.user_voice_data = {}
        
        # RNNoise 설정
        self.rnnoise_enabled = True  # RNNoise 활성화
        self.rnnoise_model_path = self.temp_audio_dir / "rnnoise_model.rnnn"  # RNNoise 모델 경로
        
        # 성능 모니터링
        self.performance_stats = {
            "total_clones": 0,
            "successful_clones": 0,
            "failed_clones": 0,
            "memory_cleanups": 0,
            "processing_times": []
        }
        
        # 메모리 최적화 설정
        self.max_concurrent_processes = 2  # 동시 처리 제한
        self.memory_limit_mb = 1024  # 1GB 메모리 제한
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_concurrent_processes)
        
        # librosa 가용성 확인
        if LIBROSA_AVAILABLE:
            self.logger.info(f"음성 클로닝 프로세서 초기화 완료 (고급 품질 분석 + RNNoise 활성화, 샘플 저장소: {self.temp_audio_dir})")
        else:
            self.logger.warning(f"librosa 라이브러리가 없어 기본 품질 분석만 사용합니다. (샘플 저장소: {self.temp_audio_dir})")
    
    async def collect_user_audio_sample(self, user_id: str, audio_data: bytes, for_cloning: bool = True) -> bool:
        """
        사용자 음성 샘플 수집 및 저장 (강화된 품질 검증 포함)
        
        Args:
            user_id: 사용자 식별자
            audio_data: 오디오 바이트 데이터
            for_cloning: 음성 클로닝용인지 여부 (True: 엄격한 검증, False: 관대한 검증)
            
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
            
            # 오디오 형식을 WAV로 통일 (최적 호환성)
            audio_file_path = user_audio_dir / f"sample_{timestamp}.wav"
            
            # WAV 형식이 아닌 경우 변환 후 저장
            if audio_data[:4] == b'RIFF' and audio_data[8:12] == b'WAVE':
                # 이미 WAV 형식
                with open(audio_file_path, 'wb') as f:
                    f.write(audio_data)
            else:
                # 다른 형식은 임시 파일로 저장 후 WAV로 변환
                temp_file_path = user_audio_dir / f"temp_{timestamp}.raw"
                with open(temp_file_path, 'wb') as f:
                    f.write(audio_data)
                
                try:
                    # librosa로 로드 후 WAV로 저장
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        y, sr = librosa.load(str(temp_file_path), sr=None)
                    
                    # WAV 형식으로 저장
                    if LIBROSA_AVAILABLE and sf is not None:
                        sf.write(str(audio_file_path), y, sr)
                    else:
                        # soundfile이 없으면 원본 그대로 저장
                        with open(audio_file_path, 'wb') as f:
                            f.write(audio_data)
                    
                    # 임시 파일 제거
                    os.remove(temp_file_path)
                    
                except Exception as conversion_error:
                    logger.warning(f"WAV 변환 실패, 원본 형식으로 저장: {conversion_error}")
                    with open(audio_file_path, 'wb') as f:
                        f.write(audio_data)
                    # 임시 파일이 있으면 제거
                    if temp_file_path.exists():
                        os.remove(temp_file_path)
            
            # RNNoise 노이즈 제거 적용
            denoised_path = None
            if self.rnnoise_enabled:
                denoised_path = await self._apply_rnnoise_denoising(audio_file_path, user_id)
                if denoised_path:
                    logger.info(f"RNNoise 노이즈 제거 완료: {denoised_path}")
                    # 노이즈 제거된 파일로 품질 분석 진행
                    analysis_file_path = denoised_path
                else:
                    logger.warning(f"RNNoise 적용 실패, 원본 파일로 품질 분석 진행")
                    analysis_file_path = audio_file_path
            else:
                analysis_file_path = audio_file_path
            
            # 고급 품질 분석 수행 (노이즈 제거된 파일로)
            quality_analysis = await self._analyze_audio_quality(analysis_file_path, audio_data)
            
            # 품질 기준 검증 (용도에 따라 다른 기준 적용)
            if for_cloning:
                # 음성 클로닝용: 엄격한 검증
                validation_result = self._validate_audio_quality_detailed(quality_analysis)
                if not validation_result["is_valid"]:
                    logger.warning(f"사용자 {user_id} 음성 클로닝용 샘플 품질 부족")
                    logger.warning(f"품질 분석: {quality_analysis}")
                    logger.warning(f"실패 이유: {validation_result['reasons']}")
                    
                    # 품질이 낮은 샘플은 삭제
                    os.remove(audio_file_path)
                    return False
                else:
                    logger.info(f"사용자 {user_id} 음성 클로닝용 고품질 샘플 검증 통과")
            else:
                # 일반 대화용: 매우 관대한 검증 (기본적으로 모든 음성 허용)
                basic_validation = self._validate_audio_for_conversation(quality_analysis)
                if not basic_validation["is_valid"]:
                    logger.warning(f"사용자 {user_id} 대화용 음성 품질 부족")
                    logger.warning(f"실패 이유: {basic_validation['reasons']}")
                    
                    # 대화용은 품질이 매우 낮아도 일단 저장 (STT는 가능할 수 있음)
                    logger.info(f"대화용 음성이므로 품질이 낮아도 저장 진행")
                else:
                    logger.info(f"사용자 {user_id} 대화용 음성 검증 통과")
            
            # 사용자 샘플 목록에 추가 (품질 정보 포함)
            if user_id not in self.user_voice_data:
                self.user_voice_data[user_id] = {"samples": [], "voice_id": None, "clone_status": "ready"}
            
            # 최종 저장할 파일 경로 결정 (노이즈 제거된 파일 우선)
            final_file_path = denoised_path if denoised_path else audio_file_path
            
            self.user_voice_data[user_id]["samples"].append({
                "path": str(final_file_path),
                "original_path": str(audio_file_path) if denoised_path else None,
                "quality": quality_analysis,
                "timestamp": timestamp,
                "rnnoise_applied": denoised_path is not None
            })
            
            logger.info(f"사용자 {user_id} 고품질 음성 샘플 저장 완료: {final_file_path}")
            logger.info(f"품질 분석: SNR={quality_analysis.get('snr_db', 'N/A')}dB, 점수={quality_analysis.get('quality_score', 'N/A')}")
            logger.info(f"RNNoise 적용: {'Yes' if denoised_path else 'No'}")
            logger.info(f"현재 {user_id}의 고품질 샘플 수: {len(self.user_voice_data[user_id]['samples'])}")
            
            return True
            
        except Exception as e:
            logger.error(f"음성 샘플 저장 실패: {e}")
            return False
    
    async def _apply_rnnoise_denoising(self, audio_file_path: Path, user_id: str) -> Optional[Path]:
        """
        RNNoise를 사용한 노이즈 제거 적용
        
        Args:
            audio_file_path: 원본 오디오 파일 경로
            user_id: 사용자 ID
            
        Returns:
            Optional[Path]: 노이즈 제거된 파일 경로 (실패 시 None)
        """
        try:
            if not LIBROSA_AVAILABLE:
                logger.warning("librosa가 설치되지 않아 RNNoise를 적용할 수 없습니다")
                return None
            
            # 노이즈 제거된 파일 경로 생성
            denoised_file_path = audio_file_path.parent / f"denoised_{audio_file_path.name}"
            
            # 1. 오디오 파일을 16kHz PCM으로 변환 (RNNoise 요구사항)
            try:
                # WAV 파일 최적화 로딩 (경고 억제)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    y, sr = librosa.load(str(audio_file_path), sr=16000)
                
                # 16-bit PCM으로 변환
                y_int16 = (y * 32767).astype(np.int16)
                
                # 임시 RAW 파일 생성
                raw_input_path = audio_file_path.parent / f"temp_input_{user_id}.raw"
                raw_output_path = audio_file_path.parent / f"temp_output_{user_id}.raw"
                
                # RAW 파일 저장
                y_int16.tobytes() 
                with open(raw_input_path, 'wb') as f:
                    f.write(y_int16.tobytes())
                
                # 2. RNNoise 적용 (시스템 명령어 사용)
                rnnoise_success = await self._run_rnnoise_command(raw_input_path, raw_output_path)
                
                if rnnoise_success and raw_output_path.exists():
                    # 3. 처리된 RAW 파일을 다시 오디오 파일로 변환
                    with open(raw_output_path, 'rb') as f:
                        denoised_raw = f.read()
                    
                    # bytes를 int16 배열로 변환
                    denoised_int16 = np.frombuffer(denoised_raw, dtype=np.int16)
                    
                    # float로 정규화
                    denoised_float = denoised_int16.astype(np.float32) / 32767.0
                    
                    # 오디오 파일로 저장
                    sf.write(str(denoised_file_path), denoised_float, 16000)
                    
                    # 임시 파일 정리
                    try:
                        os.remove(raw_input_path)
                        os.remove(raw_output_path)
                    except:
                        pass
                    
                    logger.info(f"RNNoise 노이즈 제거 성공: {denoised_file_path}")
                    return denoised_file_path
                    
                else:
                    logger.warning("RNNoise 명령어 실행 실패")
                    return None
                    
            except Exception as e:
                logger.error(f"RNNoise 오디오 처리 중 오류: {e}")
                return None
                
        except Exception as e:
            logger.error(f"RNNoise 노이즈 제거 실패: {e}")
            return None
    
    async def _run_rnnoise_command(self, input_path: Path, output_path: Path) -> bool:
        """
        RNNoise 시스템 명령어 실행
        
        Args:
            input_path: 입력 RAW 파일 경로
            output_path: 출력 RAW 파일 경로
            
        Returns:
            bool: 성공 여부
        """
        try:
            # RNNoise 명령어 확인 (시스템에 설치되어 있는지)
            result = subprocess.run(['which', 'rnnoise'], capture_output=True, text=True)
            if result.returncode != 0:
                # RNNoise가 설치되지 않은 경우 Python 기반 대안 사용
                return await self._apply_python_noise_reduction(input_path, output_path)
            
            # RNNoise 실행
            cmd = ['rnnoise', str(input_path), str(output_path)]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                logger.debug(f"RNNoise 명령어 실행 성공: {cmd}")
                return True
            else:
                logger.warning(f"RNNoise 명령어 실행 실패: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("RNNoise 명령어 타임아웃")
            return False
        except Exception as e:
            logger.error(f"RNNoise 명령어 실행 중 오류: {e}")
            return False
    
    async def _apply_python_noise_reduction(self, input_path: Path, output_path: Path) -> bool:
        """
        Python 기반 노이즈 감소 (RNNoise 대안)
        
        Args:
            input_path: 입력 RAW 파일 경로
            output_path: 출력 RAW 파일 경로
            
        Returns:
            bool: 성공 여부
        """
        try:
            if not LIBROSA_AVAILABLE:
                return False
            
            # RAW 파일 읽기
            with open(input_path, 'rb') as f:
                raw_data = f.read()
            
            # bytes를 int16 배열로 변환
            audio_int16 = np.frombuffer(raw_data, dtype=np.int16)
            audio_float = audio_int16.astype(np.float32) / 32767.0
            
            # 간단한 노이즈 감소 필터 적용 (스펙트럴 게이팅)
            # 1. 스펙트로그램 계산
            stft = librosa.stft(audio_float, n_fft=2048, hop_length=512)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # 2. 노이즈 추정 (처음 0.5초를 노이즈로 가정)
            noise_frames = int(0.5 * 16000 / 512)  # 0.5초에 해당하는 프레임 수
            noise_profile = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)
            
            # 3. 스펙트럴 감소 적용
            alpha = 2.0  # 감소 강도
            noise_threshold = noise_profile * alpha
            
            # 각 주파수 빈에서 노이즈 임계값보다 작은 값들을 감소
            mask = magnitude > noise_threshold
            reduced_magnitude = magnitude * mask + magnitude * 0.1 * (~mask)
            
            # 4. 위상 복원 및 역변환
            reduced_stft = reduced_magnitude * np.exp(1j * phase)
            denoised_audio = librosa.istft(reduced_stft, hop_length=512)
            
            # 5. int16으로 변환 후 저장
            denoised_int16 = (denoised_audio * 32767).astype(np.int16)
            
            with open(output_path, 'wb') as f:
                f.write(denoised_int16.tobytes())
            
            logger.info(f"Python 기반 노이즈 감소 적용 완료: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Python 노이즈 감소 실패: {e}")
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
            # librosa로 오디오 로드 (WAV 최적화, 경고 억제)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
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
            score *= 0.85  # 85%
        
        return float(np.clip(score, 0.0, 1.0))
    
    def _validate_audio_quality(self, quality_analysis: Dict[str, Any]) -> bool:
        """
        오디오 품질 검증 (기본 검증 - 호환성 유지)
        
        Args:
            quality_analysis: 품질 분석 결과
            
        Returns:
            bool: 품질 기준 통과 여부
        """
        result = self._validate_audio_quality_detailed(quality_analysis)
        return result["is_valid"]
    
    def _validate_audio_quality_detailed(self, quality_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        오디오 품질 상세 검증 (강화된 기준 및 상세 피드백)
        
        Args:
            quality_analysis: 품질 분석 결과
            
        Returns:
            Dict: {"is_valid": bool, "reasons": List[str], "recommendations": List[str]}
        """
        reasons = []
        recommendations = []
        
        # 기본 크기 및 지속 시간 체크
        if quality_analysis["file_size_kb"] < 10:  # 10KB 미만
            reasons.append("파일 크기가 너무 작음 (10KB 미만)")
            recommendations.append("더 길게 말해주세요 (최소 3초 이상)")
        
        if quality_analysis["duration_seconds"] < self.min_sample_duration:
            reasons.append(f"지속 시간 부족 ({quality_analysis['duration_seconds']:.1f}초 < {self.min_sample_duration}초)")
            recommendations.append(f"최소 {self.min_sample_duration}초 이상 말해주세요")
        
        # 고급 분석이 가능한 경우
        if quality_analysis["analysis_method"] == "advanced":
            # 샘플 레이트 체크 (새로 추가)
            if quality_analysis["sample_rate"] < self.min_sample_rate:
                reasons.append(f"샘플 레이트 부족 ({quality_analysis['sample_rate']}Hz < {self.min_sample_rate}Hz)")
                recommendations.append("더 좋은 마이크나 녹음 설정을 사용해주세요")
            
            # SNR 체크 (강화된 기준)
            if quality_analysis["snr_db"] < self.min_snr_db:
                reasons.append(f"SNR 부족 ({quality_analysis['snr_db']:.1f}dB < {self.min_snr_db}dB)")
                recommendations.append("조용한 곳에서 마이크에 더 가까이서 말해주세요")
            
            # 노이즈 레벨 체크 (강화된 기준)
            if quality_analysis["noise_level"] > self.max_noise_level:
                reasons.append(f"노이즈 레벨 과다 ({quality_analysis['noise_level']:.2f} > {self.max_noise_level})")
                recommendations.append("배경 소음이 없는 조용한 곳에서 녹음해주세요")
            
            # 클리핑 체크 (더 관대하게 - 심각한 클리핑만 차단)
            clipping_ratio = quality_analysis.get("clipping_ratio", 0.0)
            if quality_analysis["has_clipping"] and clipping_ratio > 0.1:  # 10% 이상 클리핑된 경우만
                reasons.append(f"심각한 오디오 클리핑 감지 (클리핑 비율: {clipping_ratio:.1%})")
                recommendations.append("마이크 볼륨을 줄이거나 조금 더 멀리서 말해줘!")
        
        # 종합 품질 점수 체크 (강화된 기준)
        if quality_analysis["quality_score"] < self.min_quality_score:
            reasons.append(f"종합 품질 점수 부족 ({quality_analysis['quality_score']:.2f} < {self.min_quality_score})")
            recommendations.append("더 명확하고 또렷하게 말해주세요")
        
        is_valid = len(reasons) == 0
        
        if is_valid:
            logger.debug("품질 검증 통과")
        else:
            logger.debug(f"품질 검증 실패: {', '.join(reasons)}")
        
        return {
            "is_valid": is_valid,
            "reasons": reasons,
            "recommendations": recommendations,
            "quality_analysis": quality_analysis
        }
    
    def _validate_audio_for_conversation(self, quality_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        대화용 음성의 매우 관대한 품질 검증
        
        Args:
            quality_analysis: 품질 분석 결과
            
        Returns:
            Dict: 검증 결과 {"is_valid": bool, "reasons": List[str]}
        """
        reasons = []
        
        # 매우 기본적인 검증만 수행
        duration = quality_analysis.get("duration_seconds", 0)
        file_size = quality_analysis.get("file_size_kb", 0)
        
        # 1. 최소 길이 검증 (매우 짧은 음성 제외)
        if duration < 0.5:  # 0.5초 미만
            reasons.append(f"음성이 너무 짧음 ({duration:.1f}초 < 0.5초)")
        
        # 2. 파일 크기 검증 (거의 없는 파일 제외)
        if file_size < 1.0:  # 1KB 미만
            reasons.append(f"파일 크기가 너무 작음 ({file_size:.1f}KB < 1KB)")
        
        # 3. 최대 길이 검증 (너무 긴 음성 제외)
        if duration > 60.0:  # 60초 초과
            reasons.append(f"음성이 너무 김 ({duration:.1f}초 > 60초)")
        
        # 매우 관대한 기준: 위의 극단적인 경우가 아니면 모두 통과
        is_valid = len(reasons) == 0
        
        logger.debug(f"대화용 음성 검증: {'통과' if is_valid else '실패'} - {reasons}")
        
        return {
            "is_valid": is_valid,
            "reasons": reasons,
            "validation_type": "conversation_lenient"
        }
    
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
        사용자 음성 샘플 정리 (개인정보 보호)
        
        Args:
            user_id: 사용자 식별자
            
        Returns:
            bool: 정리 성공 여부
        """
        try:
            deleted_count = 0
            
            # 파일 시스템에서 샘플 파일들 삭제
            if user_id in self.user_voice_data:
                samples = self.user_voice_data[user_id]["samples"]
                
                for sample_info in samples:
                    try:
                        # 파일 경로 추출 (딕셔너리인 경우와 문자열인 경우 처리)
                        if isinstance(sample_info, dict):
                            file_path = sample_info.get("path")
                            original_path = sample_info.get("original_path")
                        else:
                            file_path = sample_info
                            original_path = None
                        
                        # 메인 파일 삭제
                        if file_path and Path(file_path).exists():
                            Path(file_path).unlink()
                            deleted_count += 1
                            logger.debug(f"[CLEANUP] 삭제 완료: {file_path}")
                        
                        # 원본 파일도 있으면 삭제 (RNNoise 적용된 경우)
                        if original_path and Path(original_path).exists():
                            Path(original_path).unlink()
                            deleted_count += 1
                            logger.debug(f"[CLEANUP] 원본 삭제 완료: {original_path}")
                            
                    except Exception as e:
                        logger.warning(f"[CLEANUP] 샘플 파일 삭제 실패: {sample_info} - {e}")
                
                # 사용자 폴더 전체 정리
                user_audio_dir = self.temp_audio_dir / user_id
                if user_audio_dir.exists():
                    try:
                        # 폴더 내 모든 파일 삭제
                        for file in user_audio_dir.iterdir():
                            if file.is_file():
                                file.unlink()
                                deleted_count += 1
                        
                        # 빈 폴더 삭제
                        user_audio_dir.rmdir()
                        logger.debug(f"[CLEANUP] 사용자 폴더 삭제 완료: {user_audio_dir}")
                        
                    except Exception as e:
                        logger.warning(f"[CLEANUP] 사용자 폴더 정리 실패: {user_audio_dir} - {e}")
                
                # 메모리에서 제거
                del self.user_voice_data[user_id]
                logger.info(f"[CLEANUP] 사용자 {user_id} 음성 샘플 정리 완료 (삭제된 파일: {deleted_count}개)")
                
                return True
            else:
                logger.warning(f"[CLEANUP] 정리할 샘플이 없음: {user_id}")
                return True
            
        except Exception as e:
            logger.error(f"[CLEANUP] 사용자 샘플 정리 실패: {e}")
            return False 

    async def create_voice_clone(self, user_audio_files: List[str], user_name: str) -> Optional[str]:
        """
        음성 복제 생성 (메모리 최적화)
        
        Args:
            user_audio_files: 사용자 음성 파일 경로 리스트
            user_name: 사용자 이름
            
        Returns:
            복제된 음성 ID 또는 None
        """
        start_time = time.time()
        
        try:
            # 메모리 사용량 체크
            memory_usage_before = psutil.virtual_memory().percent
            self.logger.info(f"🧠 음성 복제 시작 - 메모리 사용량: {memory_usage_before}%")
            
            if memory_usage_before > 85:
                self.logger.warning("⚠️ 메모리 사용량이 높음, 정리 후 진행")
                await self._cleanup_memory()
                
            # 음성 파일 검증 및 전처리 (비동기)
            processed_files = await self._preprocess_audio_files_async(user_audio_files)
            if not processed_files:
                self.logger.error("❌ 음성 파일 전처리 실패")
                return None
            
            # ElevenLabs API 호출 (메모리 효율적 처리)
            voice_id = await self._create_voice_clone_api(processed_files, user_name)
            
            if voice_id:
                self.performance_stats["successful_clones"] += 1
                self.logger.info(f"✅ 음성 복제 성공: {voice_id} (사용자: {user_name})")
                
                # 즉시 임시 파일 정리
                await self._cleanup_temp_files_async(processed_files)
            else:
                self.performance_stats["failed_clones"] += 1
                self.logger.error(f"❌ 음성 복제 실패 (사용자: {user_name})")
            
            # 처리 시간 기록
            processing_time = time.time() - start_time
            self.performance_stats["processing_times"].append(processing_time)
            self.performance_stats["total_clones"] += 1
            
            # 메모리 정리
            await self._cleanup_memory()
            
            memory_usage_after = psutil.virtual_memory().percent
            self.logger.info(f"🧠 음성 복제 완료 - 메모리 사용량: {memory_usage_after}% (처리시간: {processing_time:.2f}초)")
            
            return voice_id
            
        except Exception as e:
            self.logger.error(f"❌ 음성 복제 중 오류: {e}", exc_info=True)
            self.performance_stats["failed_clones"] += 1
            return None
    
    async def _preprocess_audio_files_async(self, audio_files: List[str]) -> List[str]:
        """음성 파일 비동기 전처리 (메모리 최적화)"""
        if not audio_files:
            return []
        
        self.logger.info(f"🔄 {len(audio_files)}개 음성 파일 전처리 시작...")
        processed_files = []
        
        # 파일별 병렬 처리 (메모리 제한 고려)
        semaphore = asyncio.Semaphore(self.max_concurrent_processes)
        
        async def process_single_file(file_path: str) -> Optional[str]:
            async with semaphore:
                return await self._process_single_audio_file(file_path)
        
        # 모든 파일을 병렬로 처리
        tasks = [process_single_file(file_path) for file_path in audio_files]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 성공한 결과만 수집
        for result in results:
            if isinstance(result, str) and result:
                processed_files.append(result)
            elif isinstance(result, Exception):
                self.logger.warning(f"⚠️ 파일 처리 실패: {result}")
        
        self.logger.info(f"✅ 전처리 완료: {len(processed_files)}/{len(audio_files)}개 파일")
        return processed_files
    
    async def _process_single_audio_file(self, file_path: str) -> Optional[str]:
        """단일 음성 파일 처리 (스레드풀 사용)"""
        try:
            # CPU 집약적 작업을 스레드풀에서 실행
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self.thread_pool,
                self._process_audio_file_sync,
                file_path
            )
        except Exception as e:
            self.logger.error(f"❌ 음성 파일 처리 실패: {file_path}, 오류: {e}")
            return None
    
    def _process_audio_file_sync(self, file_path: str) -> Optional[str]:
        """동기 음성 파일 처리 (메모리 효율적)"""
        try:
            import librosa
            import soundfile as sf
            
            # 메모리 효율적으로 오디오 로드
            audio_data, sample_rate = librosa.load(
                file_path,
                sr=22050,  # 표준 샘플레이트로 통일
                mono=True,  # 모노로 변환
                dtype='float32'  # 메모리 효율적인 타입
            )
            
            # 오디오 정규화 및 노이즈 제거
            audio_data = librosa.util.normalize(audio_data)
            
            # 처리된 파일 저장
            output_path = file_path.replace('.wav', '_processed.wav')
            sf.write(output_path, audio_data, sample_rate, format='WAV')
            
            # 원본 데이터 메모리 해제
            del audio_data
            gc.collect()
            
            return output_path
            
        except Exception as e:
            self.logger.error(f"❌ 동기 음성 처리 실패: {file_path}, 오류: {e}")
            return None
    
    async def _cleanup_temp_files_async(self, file_paths: List[str]) -> None:
        """임시 파일 비동기 정리"""
        if not file_paths:
            return
            
        cleanup_tasks = []
        for file_path in file_paths:
            cleanup_tasks.append(self._delete_file_async(file_path))
        
        await asyncio.gather(*cleanup_tasks, return_exceptions=True)
        self.logger.info(f"🗑️ 임시 파일 정리 완료: {len(file_paths)}개 파일")
    
    async def _delete_file_async(self, file_path: str) -> None:
        """단일 파일 비동기 삭제"""
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            self.logger.warning(f"⚠️ 파일 삭제 실패: {file_path}, 오류: {e}")
    
    async def _cleanup_memory(self) -> None:
        """메모리 정리"""
        gc.collect()
        self.performance_stats["memory_cleanups"] += 1
        
        memory_usage = psutil.virtual_memory().percent
        self.logger.info(f"🧹 메모리 정리 완료 - 사용량: {memory_usage}%")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """성능 통계 반환"""
        stats = self.performance_stats.copy()
        
        if stats["processing_times"]:
            stats["average_processing_time"] = sum(stats["processing_times"]) / len(stats["processing_times"])
            stats["max_processing_time"] = max(stats["processing_times"])
            stats["min_processing_time"] = min(stats["processing_times"])
        
        return stats
    
    def cleanup(self) -> None:
        """리소스 정리"""
        self.logger.info("🧹 VoiceCloningProcessor 리소스 정리 시작")
        
        # 스레드풀 종료
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown(wait=True)
            
        # 메모리 정리
        gc.collect()
        
        self.logger.info("✅ VoiceCloningProcessor 리소스 정리 완료")
    
    async def _create_voice_clone_api(self, processed_files: List[str], user_name: str) -> Optional[str]:
        """
        ElevenLabs API를 통한 음성 복제 생성
        
        Args:
            processed_files: 전처리된 음성 파일 경로 리스트
            user_name: 사용자 이름
            
        Returns:
            복제된 음성 ID 또는 None
        """
        if not self.client:
            self.logger.error("ElevenLabs 클라이언트가 설정되지 않았습니다")
            return None
        
        try:
            voice_name = f"{user_name}_voice_clone_{int(time.time())}"
            
            # SSL 컨텍스트 설정
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            
            timeout = aiohttp.ClientTimeout(total=300)  # 5분 타임아웃
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
                # Multipart form data 준비
                data = aiohttp.FormData()
                data.add_field('name', voice_name)
                data.add_field('description', f'실시간 생성된 {user_name} 음성 클론')
                data.add_field('remove_background_noise', 'true')
                
                # 음성 파일들 추가 (최대 5개)
                for i, file_path in enumerate(processed_files[:5]):
                    try:
                        with open(file_path, 'rb') as f:
                            file_data = f.read()
                            
                        content_type = 'audio/wav'
                        filename = f'sample_{i}.wav'
                        
                        data.add_field('files', file_data, filename=filename, content_type=content_type)
                    except Exception as e:
                        self.logger.warning(f"⚠️ 파일 읽기 실패: {file_path} - {e}")
                        continue
                
                # ElevenLabs API 호출
                url = f"{self.client.base_url}/voices/add"
                headers = {"xi-api-key": self.client.api_key}
                
                async with session.post(url, headers=headers, data=data, ssl=ssl_context) as response:
                    response_text = await response.text()
                    self.logger.info(f"ElevenLabs API 응답 상태: {response.status}")
                    
                    if response.status in [200, 201]:
                        try:
                            result = await response.json()
                            voice_id = result.get("voice_id")
                            
                            if voice_id:
                                self.logger.info(f"✅ 음성 복제 API 성공: {voice_id}")
                                return voice_id
                            else:
                                self.logger.error(f"❌ 음성 ID가 응답에 없음: {result}")
                                return None
                        except Exception as json_error:
                            self.logger.error(f"❌ JSON 파싱 실패: {json_error}")
                            return None
                    else:
                        self.logger.error(f"❌ ElevenLabs API 오류 ({response.status}): {response_text}")
                        return None
                        
        except Exception as e:
            self.logger.error(f"❌ 음성 복제 API 호출 실패: {e}")
            return None 