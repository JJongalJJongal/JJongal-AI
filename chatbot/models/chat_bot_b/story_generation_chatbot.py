"""
꼬기 (ChatBot B) - 동화 생성 챗봇 통합 클래스

부기에서 수집한 이야기 요소를 바탕으로 완전한 멀티미디어 동화를 생성하는 메인 클래스
"""

import logging
import os
from typing import Dict, Any, Optional, Callable
from pathlib import Path

# 핵심 모듈
from .core import StoryGenerationEngine, ContentPipeline

# 생성자 모듈
from .generators import TextGenerator, ImageGenerator, VoiceGenerator

# 공유 유틸리티
from shared.utils.openai_utils import initialize_client # OpenAI 클라이언트 초기화

logger = logging.getLogger(__name__) # 로깅 설정

class StoryGenerationChatBot:
    """
    꼬기 - 동화 생성 챗봇 메인 클래스
    
    부기에서 수집한 이야기 요소를 바탕으로:
    1. 상세 스토리 텍스트 생성
    2. 챕터별 이미지 생성 (DALL-E 3)
    3. 등장인물별 음성 생성 (ElevenLabs)
    4. 완전한 멀티미디어 동화 제작
    """
    
    def __init__(self, 
                 output_dir: str = "output", # 출력 디렉토리 경로
                 vector_db_path: str = None, # ChromaDB 벡터 데이터베이스 경로
                 collection_name: str = "fairy_tales"): # ChromaDB 컬렉션 이름
        """
        꼬기 챗봇 초기화
        
        Args:
            output_dir: 출력 디렉토리 경로
            vector_db_path: ChromaDB 벡터 데이터베이스 경로
            collection_name: ChromaDB 컬렉션 이름
        """
        self.output_dir = Path(output_dir) # 출력 디렉토리 경로
        self.output_dir.mkdir(parents=True, exist_ok=True) # 출력 디렉토리 생성
        
        # 클라이언트 초기화
        self.openai_client = None # OpenAI 클라이언트
        self.elevenlabs_api_key = None # ElevenLabs API 키
        
        # 스토리 설정
        self.target_age = None # 대상 연령
        self.story_outline = None # 스토리 개요
        
        # 핵심 엔진들
        self.story_engine = None # 스토리 생성 엔진
        self.content_pipeline = None # 콘텐츠 파이프라인
        
        # 생성기들
        self.text_generator = None # 텍스트 생성기
        self.image_generator = None # 이미지 생성기
        self.voice_generator = None # 음성 생성기
        
        # 초기화
        self._initialize_clients() # API 클라이언트 초기화
        self._initialize_engines(vector_db_path, collection_name) # 엔진 및 생성기 초기화
        
    def _initialize_clients(self):
        """API 클라이언트 초기화"""
        try:
            # OpenAI 클라이언트 초기화
            self.openai_client = initialize_client() # OpenAI 클라이언트 초기화
            
            # ElevenLabs API 키 로드
            self.elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY") # ElevenLabs API 키 로드
            
            logger.info("API 클라이언트 초기화 완료") # 로깅
            
        except Exception as e:
            logger.error(f"API 클라이언트 초기화 실패: {e}") # 로깅
            
    def _initialize_engines(self, vector_db_path: str, collection_name: str):
        """엔진 및 생성기 초기화"""
        try:
            # 1. 생성기들 초기화
            self.text_generator = TextGenerator(
                openai_client=self.openai_client, # OpenAI 클라이언트
                vector_db_path=vector_db_path, # ChromaDB 벡터 데이터베이스 경로
                collection_name=collection_name # ChromaDB 컬렉션 이름
            )
            
            self.image_generator = ImageGenerator( 
                openai_client=self.openai_client, # OpenAI 클라이언트
                model_name="dall-e-3", # DALL-E 3 모델
                temp_storage_path=str(self.output_dir / "temp") # 임시 저장 경로
            )
            
            self.voice_generator = VoiceGenerator(
                elevenlabs_api_key=self.elevenlabs_api_key, # ElevenLabs API 키
                temp_storage_path=str(self.output_dir / "temp") # 임시 저장 경로
            )
            
            # 2. 스토리 생성 엔진 초기화
            self.story_engine = StoryGenerationEngine(
                openai_client=self.openai_client, # OpenAI 클라이언트
                elevenlabs_client=None,  # ElevenLabs 클라이언트
                output_dir=str(self.output_dir) # 출력 디렉토리 경로
            )
            
            # 생성기들을 엔진에 주입
            self.story_engine.set_generators(
                text_generator=self.text_generator, # 텍스트 생성기
                image_generator=self.image_generator, # 이미지 생성기
                voice_generator=self.voice_generator, # 음성 생성기
                rag_enhancer=None  # RAG 향상기
            )
            
            # 3. 콘텐츠 파이프라인 초기화
            self.content_pipeline = ContentPipeline(
                openai_client=self.openai_client, # OpenAI 클라이언트
                vector_db_path=vector_db_path, # ChromaDB 벡터 데이터베이스 경로
                collection_name=collection_name # ChromaDB 컬렉션 이름
            )
            
            logger.info("엔진 및 생성기 초기화 완료") # 로깅
            
        except Exception as e:
            logger.error(f"엔진 초기화 실패: {e}") # 로깅
            raise # 예외 발생
    
    def set_target_age(self, age: int):
        """대상 연령 설정"""
        self.target_age = age # 대상 연령 설정
        logger.info(f"대상 연령 설정: {age}세") # 로깅
     
    def set_story_outline(self, story_outline: Dict[str, Any]):
        """부기에서 수집한 스토리 개요 설정"""
        self.story_outline = story_outline # 스토리 개요 설정
        logger.info("스토리 개요 설정 완료") # 로깅
    
    async def generate_detailed_story(self) -> Dict[str, Any]:
        """
        상세 동화 생성 (메인 메서드)
        
        Args:
            progress_callback: 진행 상황 콜백 함수
            
        Returns:
            Dict: 생성된 동화 데이터
            {
                "story_data": 상세 스토리 데이터,
                "image_paths": 이미지 파일 경로들,
                "audio_paths": 음성 파일 경로들,
                "story_id": 스토리 ID,
                "status": 생성 상태
            }
        """
        if not self.story_outline: # 스토리 개요가 설정되지 않았으면 예외 발생
            raise ValueError("스토리 개요가 설정되지 않았습니다. set_story_outline()을 먼저 호출하세요.")
        
        if not self.target_age: # 대상 연령이 설정되지 않았으면 예외 발생
            raise ValueError("대상 연령이 설정되지 않았습니다. set_target_age()를 먼저 호출하세요.")
        
        # 스토리 개요에 연령 정보 추가
        enhanced_outline = {
            **self.story_outline, # 스토리 개요 복사
            "age_group": self.target_age, # 대상 연령 추가
            "target_age": self.target_age # 대상 연령 추가
        }
        
        # 스토리 생성 엔진으로 완전한 동화 생성
        result = await self.story_engine.generate_complete_story(enhanced_outline)
        
        return result
    
    async def generate_text_only(self, progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """텍스트만 생성 (빠른 생성)"""
        if not self.story_outline or not self.target_age: # 스토리 개요가 설정되지 않았으면 예외 발생
            raise ValueError("스토리 개요와 대상 연령을 먼저 설정하세요.")
        
        enhanced_outline = {
            **self.story_outline, # 스토리 개요 복사
            "age_group": self.target_age, # 대상 연령 추가
            "target_age": self.target_age # 대상 연령 추가
        }
        
        # 텍스트 생성기로 스토리만 생성
        story_data = await self.text_generator.generate(enhanced_outline, progress_callback)
        
        return {
            "story_data": story_data, # 스토리 데이터
            "image_paths": [], # 이미지 파일 경로들
            "audio_paths": [], # 음성 파일 경로들
            "story_id": None, # 스토리 ID
            "status": "text_only" # 생성 상태
        } 
    
    async def generate_with_pipeline(self, 
                                   execution_mode: str = "sequential",
                                   progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        콘텐츠 파이프라인을 사용한 생성
        
        Args:
            execution_mode: "sequential" 또는 "parallel"
            progress_callback: 진행 상황 콜백
        """
        if not self.story_outline or not self.target_age: # 스토리 개요가 설정되지 않았으면 예외 발생
            raise ValueError("스토리 개요와 대상 연령을 먼저 설정하세요.")
        
        enhanced_outline = {
            **self.story_outline, # 스토리 개요 복사
            "age_group": self.target_age, # 대상 연령 추가
            "target_age": self.target_age # 대상 연령 추가
        }
        
        # 콘텐츠 파이프라인으로 생성
        result = await self.content_pipeline.execute_pipeline(
            story_outline=enhanced_outline, # 스토리 개요
            execution_mode=execution_mode, # 실행 모드
            progress_callback=progress_callback # 진행 상황 콜백
        )
        
        return result # 생성 결과 반환
    
    def get_generation_status(self) -> Dict[str, Any]:
        """현재 생성 상태 반환"""
        return {
            "story_outline_set": self.story_outline is not None, # 스토리 개요 설정 여부
            "target_age_set": self.target_age is not None, # 대상 연령 설정 여부
            "engines_ready": all([
                self.story_engine is not None, # 스토리 생성 엔진 준비 여부
                self.text_generator is not None, # 텍스트 생성기 준비 여부
                self.image_generator is not None, # 이미지 생성기 준비 여부
                self.voice_generator is not None # 음성 생성기 준비 여부
            ]),
            "output_dir": str(self.output_dir) # 출력 디렉토리 경로
        }
    
    async def health_check(self) -> Dict[str, bool]:
        """시스템 상태 확인"""
        status = {
            "openai_client": self.openai_client is not None, # OpenAI 클라이언트 준비 여부
            "elevenlabs_api_key": self.elevenlabs_api_key is not None, # ElevenLabs API 키 준비 여부
            "text_generator": False, # 텍스트 생성기 준비 여부
            "image_generator": False, # 이미지 생성기 준비 여부
            "voice_generator": False # 음성 생성기 준비 여부
        }
        
        # 각 생성기 상태 확인
        if self.text_generator:
            try:
                status["text_generator"] = await self.text_generator.health_check() # 텍스트 생성기 상태 확인
            except:
                status["text_generator"] = False # 텍스트 생성기 상태 확인 실패
                
        if self.image_generator:
            try:
                status["image_generator"] = await self.image_generator.health_check() # 이미지 생성기 상태 확인
            except:
                status["image_generator"] = False # 이미지 생성기 상태 확인 실패
                
        if self.voice_generator:
            try:
                status["voice_generator"] = await self.voice_generator.health_check() # 음성 생성기 상태 확인
            except:
                status["voice_generator"] = False # 음성 생성기 상태
        
        return status
    
    def cleanup(self):
        """리소스 정리"""
        if self.story_engine:
            self.story_engine.cleanup() # 스토리 생성 엔진 리소스 정리
        
        logger.info("꼬기 챗봇 리소스 정리 완료") # 로깅