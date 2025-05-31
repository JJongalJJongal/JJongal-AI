"""
꼬기 (ChatBot B) - Enhanced 동화 생성 챗봇 통합 클래스

부기에서 수집한 이야기 요소를 바탕으로 완전한 멀티미디어 동화를 생성하는 메인 클래스
- 개선된 프롬프트 시스템 (v2.0) 적용
- 연령별 특화 생성 (4-7세, 8-9세)
- 성능 추적 및 최적화
"""

from shared.utils.logging_utils import get_module_logger
import os
import time
from typing import Dict, Any, Optional, Callable
from pathlib import Path

# 핵심 모듈
from .core import StoryGenerationEngine, ContentPipeline

# 생성자 모듈
from .generators import TextGenerator, ImageGenerator, VoiceGenerator

# 공유 유틸리티
from shared.utils.openai_utils import initialize_client

logger = get_module_logger(__name__)

class ChatBotB:
    """
    꼬기 - Enhanced 동화 생성 챗봇 메인 클래스
    
    부기에서 수집한 이야기 요소를 바탕으로:
    1. 상세 스토리 텍스트 생성 (연령별 특화)
    2. 챕터별 이미지 생성 (DALL-E 3, Enhanced)
    3. 등장인물별 음성 생성 (ElevenLabs)
    4. 완전한 멀티미디어 동화 제작
    
    Features:
    - 개선된 프롬프트 엔지니어링 (v2.0)
    - 연령별 맞춤 생성 (4-7세, 8-9세)
    - 체인 오브 소트 추론
    - 성능 추적 및 최적화
    """
    
    def __init__(self, 
                 output_dir: str = "output",
                 vector_db_path: str = None,
                 collection_name: str = "fairy_tales",
                 use_enhanced_generators: bool = True,
                 enable_performance_tracking: bool = True):
        """
        꼬기 챗봇 초기화
        
        Args:
            output_dir: 출력 디렉토리 경로
            vector_db_path: ChromaDB 벡터 데이터베이스 경로
            collection_name: ChromaDB 컬렉션 이름
            use_enhanced_generators: Enhanced 생성기 사용 여부
            enable_performance_tracking: 성능 추적 활성화
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.use_enhanced_generators = use_enhanced_generators
        self.enable_performance_tracking = enable_performance_tracking
        
        # 클라이언트 초기화
        self.openai_client = None
        self.elevenlabs_api_key = None
        
        # 스토리 설정
        self.target_age = None
        self.story_outline = None
        
        # 핵심 엔진들
        self.story_engine = None
        self.content_pipeline = None
        
        # 생성기들 (기본 + Enhanced)
        self.text_generator = None
        self.image_generator = None
        self.voice_generator = None
        self.enhanced_text_generator = None
        self.enhanced_image_generator = None
        
        # 성능 메트릭
        self.performance_metrics = {
            "total_stories_generated": 0,
            "successful_generations": 0,
            "average_generation_time": 0,
            "enhanced_mode_usage": 0,
            "age_group_statistics": {}
        }
        
        # 초기화
        self._initialize_clients()
        self._initialize_engines(vector_db_path, collection_name)
        
    def _initialize_clients(self):
        """API 클라이언트 초기화"""
        try:
            # OpenAI 클라이언트 초기화
            self.openai_client = initialize_client()
            
            # ElevenLabs API 키 로드
            self.elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")
            
            logger.info("API 클라이언트 초기화 완료")
            
        except Exception as e:
            logger.error(f"API 클라이언트 초기화 실패: {e}")
            
    def _initialize_engines(self, vector_db_path: str, collection_name: str):
        """엔진 및 생성기 초기화 (Enhanced 포함)"""
        try:
            # 1. 기본 생성기들 초기화
            self.text_generator = TextGenerator(
                openai_client=self.openai_client,
                vector_db_path=vector_db_path,
                collection_name=collection_name
            )
            
            self.image_generator = ImageGenerator( 
                openai_client=self.openai_client,
                model_name="dall-e-3",
                temp_storage_path=str(self.output_dir / "temp")
            )
            
            self.voice_generator = VoiceGenerator(
                elevenlabs_api_key=self.elevenlabs_api_key,
                temp_storage_path=str(self.output_dir / "temp")
            )
            
            # 2. Enhanced 생성기들 초기화 (선택적)
            if self.use_enhanced_generators:
                self.enhanced_text_generator = TextGenerator(
                    openai_client=self.openai_client,
                    vector_db_path=vector_db_path,
                    collection_name=collection_name,
                    enable_performance_tracking=self.enable_performance_tracking
                )
                
                self.enhanced_image_generator = ImageGenerator(
                    openai_client=self.openai_client,
                    model_name="dall-e-3",
                    temp_storage_path=str(self.output_dir / "temp"),
                    enable_performance_tracking=self.enable_performance_tracking
                )
                
                logger.info("생성자들 초기화 완료")
            
            # 3. 스토리 생성 엔진 초기화
            self.story_engine = StoryGenerationEngine(
                openai_client=self.openai_client,
                elevenlabs_client=None,
                output_dir=str(self.output_dir)
            )
            
            # 생성기들을 엔진에 주입
            self.story_engine.set_generators(
                text_generator=self.text_generator,
                image_generator=self.image_generator,
                voice_generator=self.voice_generator,
                rag_enhancer=None
            )
            
            # 4. 콘텐츠 파이프라인 초기화
            self.content_pipeline = ContentPipeline(
                openai_client=self.openai_client,
                vector_db_path=vector_db_path,
                collection_name=collection_name
            )
            
            logger.info("엔진 및 생성기 초기화 완료")
            
        except Exception as e:
            logger.error(f"엔진 초기화 실패: {e}")
            raise
    
    def set_target_age(self, age: int):
        """대상 연령 설정"""
        self.target_age = age
        logger.info(f"대상 연령 설정: {age}세 ({'Enhanced 모드' if self.use_enhanced_generators else '기본 모드'})")
    
    def set_cloned_voice_info(self, child_voice_id: str, main_character_name: str):
        """음성 클로닝 정보 설정"""
        self.child_voice_id = child_voice_id
        self.main_character_name = main_character_name
        
        # VoiceGenerator에 캐릭터 음성 매핑 설정
        if self.voice_generator:
            character_mapping = {
                main_character_name: child_voice_id
            }
            self.voice_generator.set_character_voice_mapping(character_mapping)
            logger.info(f"음성 클로닝 정보 설정 완료: {main_character_name} -> {child_voice_id}")
        else:
            logger.warning("VoiceGenerator가 초기화되지 않아 음성 매핑을 설정할 수 없습니다.")
     
    def set_story_outline(self, story_outline: Dict[str, Any]):
        """부기에서 수집한 스토리 개요 설정"""
        self.story_outline = story_outline
        logger.info("스토리 개요 설정 완료")
    
    async def generate_detailed_story(self, 
                                    use_enhanced: bool = None,
                                    progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        상세 동화 생성 (메인 메서드)
        
        Args:
            use_enhanced: Enhanced 생성기 사용 여부 (None이면 초기화 설정 사용)
            progress_callback: 진행 상황 콜백 함수
            
        Returns:
            Dict: 생성된 동화 데이터 with 성능 메트릭
        """
        if not self.story_outline:
            raise ValueError("스토리 개요가 설정되지 않았습니다. set_story_outline()을 먼저 호출하세요.")
        
        if not self.target_age:
            raise ValueError("대상 연령이 설정되지 않았습니다. set_target_age()를 먼저 호출하세요.")
        
        start_time = time.time()
        use_enhanced_mode = use_enhanced if use_enhanced is not None else self.use_enhanced_generators
        
        result = {} # Initialize result
        try:
            # 성능 추적 시작
            self.performance_metrics["total_stories_generated"] += 1
            if use_enhanced_mode:
                self.performance_metrics["enhanced_mode_usage"] += 1
            
            # 연령대별 통계 업데이트
            age_group_key = self._get_age_group_key(self.target_age)
            if age_group_key not in self.performance_metrics["age_group_statistics"]:
                self.performance_metrics["age_group_statistics"][age_group_key] = 0
            self.performance_metrics["age_group_statistics"][age_group_key] += 1
        
            # 스토리 개요에 연령 정보 추가
            enhanced_outline = {
                **self.story_outline,
                "age_group": self.target_age,
                "target_age": self.target_age,
                "enhanced_mode": use_enhanced_mode
            }
            
            if use_enhanced_mode and self.enhanced_text_generator:
                # Enhanced 모드로 생성
                result = await self._generate_with_enhanced_mode(enhanced_outline, progress_callback)
            else:
                # 기본 모드로 생성
                result = await self.story_engine.generate_complete_story(enhanced_outline)
        
            # 성능 메트릭 업데이트
            generation_time = time.time() - start_time
            self.performance_metrics["successful_generations"] += 1
            self._update_average_generation_time(generation_time)
            
            # 결과에 메타데이터 추가
            result["metadata"] = result.get("metadata", {})
            result["metadata"].update({
                "enhanced_mode": use_enhanced_mode,
                "generation_time": generation_time,
                "prompt_version": "2.0_enhanced" if use_enhanced_mode else "1.0_basic",
                "age_group": age_group_key
            })
            
            return result
            
        except Exception as e:
            logger.error(f"스토리 생성 실패: {e}")
            raise

    async def _generate_with_enhanced_mode(self, 
                                         enhanced_outline: Dict[str, Any],
                                         progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """생성자 모드로 완전한 스토리 생성"""
        
        # 1. Enhanced 텍스트 생성
        if progress_callback:
            await progress_callback({
                "step": "enhanced_story_planning",
                "status": "starting",
                "mode": "enhanced"
            })
        
        story_data = await self.enhanced_text_generator.generate(
            enhanced_outline, progress_callback
        )
        
        # 2. Enhanced 이미지 생성
        if progress_callback:
            await progress_callback({
                "step": "enhanced_image_planning",
                "status": "starting",
                "chapters": len(story_data.get("chapters", []))
            })
        
        image_data = await self.enhanced_image_generator.generate({
            "story_data": story_data,
            "story_id": story_data.get("story_id")
        }, progress_callback)
        
        # 3. 음성 생성 (기본 생성기 사용)
        if progress_callback:
            await progress_callback({
                "step": "voice_generation",
                "status": "starting"
            })
        
        voice_data = await self.voice_generator.generate({
            "story_data": story_data,
            "story_id": story_data.get("story_id")
        }, progress_callback)
        
        # 4. 결과 통합
        return {
            "story_data": story_data,
            "image_paths": [img.get("image_path") for img in image_data.get("images", [])],
            "audio_paths": voice_data.get("audio_paths", []),
            "story_id": story_data.get("story_id"),
            "status": "enhanced_complete",
            "enhanced_metadata": {
                "text_metrics": self.enhanced_text_generator.get_performance_metrics(),
                "image_metrics": self.enhanced_image_generator.get_performance_metrics()
            }
        }
    
    async def generate_text_only(self, 
                                use_enhanced: bool = None,
                                progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """텍스트만 생성 (생성자 지원)"""
        if not self.story_outline or not self.target_age:
            raise ValueError("스토리 개요와 대상 연령을 먼저 설정하세요.")
        
        use_enhanced_mode = use_enhanced if use_enhanced is not None else self.use_enhanced_generators
        
        enhanced_outline = {
            **self.story_outline,
            "age_group": self.target_age,
            "target_age": self.target_age
        }
        
        if use_enhanced_mode and self.enhanced_text_generator:
            story_data = await self.enhanced_text_generator.generate(enhanced_outline, progress_callback)
        else:
            story_data = await self.text_generator.generate(enhanced_outline, progress_callback)
        
        return {
            "story_data": story_data,
            "image_paths": [],
            "audio_paths": [],
            "story_id": story_data.get("story_id"),
            "status": "text_only_enhanced" if use_enhanced_mode else "text_only",
            "metadata": {
                "enhanced_mode": use_enhanced_mode,
                "prompt_version": "2.0_enhanced" if use_enhanced_mode else "1.0_basic"
            }
        } 
    
    async def generate_with_pipeline(self, 
                                   progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """콘텐츠 파이프라인을 통한 생성"""
        if not self.story_outline or not self.target_age:
            raise ValueError("스토리 개요와 대상 연령을 먼저 설정하세요.")
        
        enhanced_outline = {
            **self.story_outline,
            "age_group": self.target_age,
            "target_age": self.target_age
        }
        
        # 콘텐츠 파이프라인으로 생성
        result = await self.content_pipeline.execute_pipeline(enhanced_outline, progress_callback=progress_callback)
        
        return result
    
    def _get_age_group_key(self, age: int) -> str:
        """연령대 키 반환"""
        if 4 <= age <= 7:
            return "age_4_7"
        elif 8 <= age <= 9:
            return "age_8_9"
        else:
            return "age_other"
    
    def _update_average_generation_time(self, new_time: float):
        """평균 생성 시간 업데이트"""
        current_avg = self.performance_metrics["average_generation_time"]
        successful_count = self.performance_metrics["successful_generations"]
        
        if successful_count == 1:
            self.performance_metrics["average_generation_time"] = new_time
        else:
            # 증분 평균 계산
            self.performance_metrics["average_generation_time"] = (
                (current_avg * (successful_count - 1) + new_time) / successful_count
            )
    
    def get_generation_status(self) -> Dict[str, Any]:
        """생성자 생성 상태 조회"""
        status = {
            "story_engine_status": "ready" if self.story_engine else "not_initialized",
            "text_generator_status": "ready" if self.text_generator else "not_initialized",
            "image_generator_status": "ready" if self.image_generator else "not_initialized",
            "voice_generator_status": "ready" if self.voice_generator else "not_initialized",
            "enhanced_mode_available": bool(self.enhanced_text_generator and self.enhanced_image_generator),
            "target_age_set": self.target_age is not None,
            "story_outline_set": self.story_outline is not None,
            "performance_metrics": self.performance_metrics
        }
        
        # Enhanced 생성기 상태 추가
        if self.use_enhanced_generators:
            status.update({
                "enhanced_text_status": "ready" if self.enhanced_text_generator else "not_initialized",
                "enhanced_image_status": "ready" if self.enhanced_image_generator else "not_initialized"
            })
        
        return status
    
    async def health_check(self) -> Dict[str, bool]:
        """상태 확인"""
        health_status = {
            "openai_client": bool(self.openai_client),
            "elevenlabs_api_key": bool(self.elevenlabs_api_key),
            "text_generator": bool(self.text_generator),
            "image_generator": bool(self.image_generator),
            "voice_generator": bool(self.voice_generator),
            "story_engine": bool(self.story_engine),
            "content_pipeline": bool(self.content_pipeline)
        }
        
        # Enhanced 생성기 상태 확인
        if self.use_enhanced_generators:
            health_status.update({
                "enhanced_text_generator": bool(self.enhanced_text_generator),
                "enhanced_image_generator": bool(self.enhanced_image_generator)
            })
            
            # Enhanced 생성기 개별 헬스 체크
            if self.enhanced_text_generator:
                enhanced_text_health = await self.enhanced_text_generator.health_check()
                health_status["enhanced_text_detailed"] = enhanced_text_health
                
            if self.enhanced_image_generator:
                enhanced_image_health = await self.enhanced_image_generator.health_check()
                health_status["enhanced_image_detailed"] = enhanced_image_health
        
        # 전체 상태
        basic_health = all([health_status[key] for key in ["openai_client", "text_generator", "image_generator", "voice_generator"]])
        enhanced_health = True
        
        if self.use_enhanced_generators:
            enhanced_health = health_status.get("enhanced_text_generator", False) and health_status.get("enhanced_image_generator", False)
        
        health_status["overall_healthy"] = basic_health and (enhanced_health if self.use_enhanced_generators else True)
        
        return health_status
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """메트릭 조회"""
        metrics = {
            "chatbot_metrics": self.performance_metrics
        }
        
        # Enhanced 생성기 메트릭 추가
        if self.use_enhanced_generators:
            if self.enhanced_text_generator:
                metrics["text_metrics"] = self.enhanced_text_generator.get_performance_metrics()
            
            if self.enhanced_image_generator:
                metrics["image_metrics"] = self.enhanced_image_generator.get_performance_metrics()
        
        return metrics
    
    def cleanup(self):
        """리소스 정리"""
        logger.info("ChatBot B 리소스 정리 시작")
        
        # 생성기들 정리
        if self.story_engine:
            self.story_engine = None
        
        if self.content_pipeline:
            self.content_pipeline = None
        
        # Enhanced 생성기들 정리
        if self.enhanced_text_generator:
            self.enhanced_text_generator = None
        
        if self.enhanced_image_generator:
            self.enhanced_image_generator = None
        
        logger.info("ChatBot B 리소스 정리 완료")