""" 텍스트 -> 이미지 -> 음성 순차적 파이프라인 관리하는 오케스트레이터 """

"""
단계별 실행 : 텍스트 완성 후, 이미지와 음성 병렬 생성    
"""

from shared.utils.logging_utils import get_module_logger
import asyncio
from typing import Dict, Any, List, Optional, Callable, Union
from pathlib import Path
from enum import Enum
import uuid
import os
import json
import base64

# Langchain imports
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper

# Project inner modules
from chatbot.data.vector_db.core import VectorDB
from chatbot.data.vector_db.query import query_vector_db, format_query_results

# logging 설정
logger = get_module_logger(__name__)

class PipelineStep(Enum):
    """파이프라인 단계 정의"""
    INITIALIZATION = "initialization"
    TEXT_GENERATION = "text_generation"
    IMAGE_GENERATION = "image_generation"
    VOICE_GENERATION = "voice_generation"
    FINALIZATION = "finalization"
    
class PipelineStatus(Enum):
    """파이프라인 상태 정의"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


class ContentPipeline():
    
    def __init__(self, 
                 openai_client=None, 
                 vector_db_path: str = None, 
                 collection_name: str = "fairy_tales",
                 max_retries: int = 3,
                ):
        """
        콘텐츠 파이프라인 초기화

        Args:
            openai_client: OpenAI Client 
            vector_db_path: ChromaDB 데이터베이스 경로
            collection_name: ChromaDB 컬렉션 이름
            max_retries: 실패 시 재시도 횟수
        """
        
        # LangChain chain
        self.text_chain = None # 텍스트 생성 체인
        self.image_prompt_chain = None # 이미지 프롬프트 생성 체인
        
        # ChromaDB 연결
        self.vector_store = None # 한국 동화 스토리 벡터 DB
        self.retriever = None # RAG 검색기
        
        # Client 설정
        self.openai_client = openai_client # OpenAI Client
        self.vector_db_path = vector_db_path # ChromaDB 데이터베이스 경로
        self.collection_name = collection_name # ChromaDB 컬렉션 이름
        self.max_retries = max_retries # 실패 시 재시도 횟수
        
        # 파이프라인 상태 관리
        self.current_pipeline_id = None # 현재 파이프라인 ID
        self.pipeline_status = PipelineStatus.PENDING # 파이프라인 상태
        self.current_step = PipelineStep.INITIALIZATION # 현재 단계
        
        self.pipeline_results = {} # 파이프라인 결과 저장
        self.progress = 0.0 # 진행 상태 추적
        self.completed_steps = [] # 완료된 단계 추적
        self.failed_steps = [] # 실패한 단계 추적
        self.chapter_progress = {"total_chapters": 0, "completed_chapters": 0} # 챕터별 진행 상태 추적
        
        # 파이프라인 초기화
        self._initialize_pipeline() # 파이프라인 초기화 메서드 호출
        
    def _initialize_pipeline(self):
        """ 파이프라인 구성 요소 초기화 """
        try:
            # 1. ChromaDB 초기화 (EC2 Instance)
            if self.vector_db_path:
                self.vector_store = VectorDB(
                    persist_directory=self.vector_db_path
                )
                try:
                    self.vector_store.get_collection(self.collection_name) # 컬렉션 존재 확인
                    logger.info(f"ChromaDB 컬렉션 '{self.collection_name}' 연결 완료")
                except Exception as e:
                    logger.warning(f"컬렉션 연결 실패 : {e}")
                    
            # 2. S3 Client 초기화
            import boto3
            self.s3_client = boto3.client('s3')
            self.s3_bucket = os.getenv("S3_BUCKET_NAME", 'fairy_tales')
            
            # 3. 임시 로컬 저장소 설정 (EC2 Disk)
            self.temp_storage = Path('/tmp/fairy_tales')
            self.temp_storage.mkdir(exist_ok=True)
            
            # 4. LangChain chain initialize
            self._setup_langchain_chains()
            
            logger.info("AWS EC2 + S3 환경 초기화 완료")
        
        except Exception as e:
            logger.error(f"AWS 환경 초기화 실패 : {e}")
            raise
    
    def _setup_langchain_chains(self):
        """LangChain 체인 설정"""
        
        try:
            if self.openai_client:
                # 1. Text 생성 체인
                
                with open('chatbot/data/prompts/chatbot_b_prompts.json', 'r', encoding='utf-8') as f:
                    prompts = json.load(f)
                
                text_prompt = ChatPromptTemplate.from_template(
                    prompts["story_generation_templates"]["detailed_story_system_message"]
                )
                
                model = ChatOpenAI(temperature=0.7, model="gpt-4o-mini")
                self.text_chain = text_prompt | model | StrOutputParser()
                
                # 2. Image prompt 체인
                image_prompt = ChatPromptTemplate.from_template(
                    prompts["story_generation_templates"]["image_prompt_system_message"]
                )
                self.image_prompt_chain = image_prompt | model | StrOutputParser()
                
                logger.info("LangChain 체인 설정 완료")
                
        except Exception as e:
            logger.error(f"LangChain 체인 설정 실패 : {e}")
            raise
    def _get_content_type(self, file_path: Path) -> str:
        """파일 확장자에 따른 Content-Type 변환"""
        suffix = file_path.suffix.lower()
        content_types = {
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.mp3': 'audio/mpeg',
            '.wav': 'audio/wav'
        }
        
        return content_types.get(suffix, 'application/octet-stream')
        
    async def execute_pipeline(self, story_outline, execution_mode="sequential", progress_callback=None):
        """
        AWS EC2 + S3 환경에서 파이프라인 실행

        흐름:
        1. 텍스트 생성 (ChromaDB 검색 + RAG System)
        2. 이미지 생성 -> EC2 임시 저장 -> S3 업로드
        3. 음성 생성 -> EC2 임시 저장 -> S3 업로드
        4. 로컬 임시 파일 정리
        5. S3 URL 반환
        """
        
        try:
            # 1. 텍스트 생성
            await self._execute_text_generation(story_outline, progress_callback)
            
            if execution_mode == "sequential": # 순차 모드
                # 2. 이미지 생성 + S3 업로드
                await self._execute_image_generation_with_s3(progress_callback)
                
                # 3. 음성 생성 + S3 업로드
                await self._execute_voice_generation_with_s3(progress_callback)
                
            else: 
                # 2. 이미지와 음성 병렬 생성
                await self._execute_parallel_media_with_s3(progress_callback)
                
            # 4. 임시 파일 정리
            await self._cleanup_temp_files()
            
            return self.pipeline_results
        
        except Exception as e:
            # 에러 발생 시에도 임시 파일은 정리
            await self._cleanup_temp_files()
            raise
        
    async def _execute_image_generation_with_s3(self, progress_callback):
        """ 이미지 생성 + S3 업로드 """
        
        try:
            story_data = self.pipeline_results.get("story_data", {})
            chapters = story_data.get("chapters", [])
            
            s3_image_urls = []
            
            for chapter in chapters:
                # 1. LangChain을 통한 향상된 이미지 프롬프트 생성
                if self.image_prompt_chain:
                    try:
                        prompt = await self.image_prompt_chain.ainvoke({
                            "chapter_title": chapter.get("chapter_title", ""),
                            "chapter_content": chapter.get("chapter_content", "")
                        })
                    except Exception as e:
                        logger.warning(f"LangChain 프롬프트 생성 실패, 기본 방식 사용: {e}")
                        prompt = self._create_basic_image_prompt(chapter)
                else:
                    prompt = self._create_basic_image_prompt(chapter)
                
                # 2. LangChain DALL-E 또는 직접 API로 이미지 생성
                temp_image_path = await self._generate_image_locally(prompt, chapter)
                
                # 3. S3 upload
                s3_url = await self._upload_to_s3(
                    local_path = temp_image_path,
                    s3_key=f"images/{self.current_pipeline_id}/chapter_{chapter['chapter_number']}.png"
                )       
                
                s3_image_urls.append({
                    "chapter_number": chapter.get("chapter_number", ""),
                    "s3_url": s3_url,
                    "local_temp_path": temp_image_path # 나중에 정리됨
                }) 
                
            self.pipeline_results["image_s3_urls"] = s3_image_urls
        
        except Exception as e:
            logger.error(f"이미지 생성 + S3 업로드 실패 : {e}")
            raise
    
    def _create_basic_image_prompt(self, chapter: Dict[str, Any]) -> str:
        """기본 이미지 프롬프트 생성"""
        chapter_title = chapter.get("chapter_title", "")
        chapter_content = chapter.get("chapter_content", "")
        
        return f"""
        Create a beautiful, child-friendly illustration for a Korean fairy tale chapter.
        
        Chapter Title: {chapter_title}
        Chapter Content: {chapter_content[:300]}...
        
        Style: Warm, colorful, cartoon-like, suitable for children's books.
        """
    
    async def _generate_image_locally(self, prompt: str, chapter: Dict[str, Any]) -> Path:
        """로컬에 이미지 생성"""
        try:
            # LangChain DALL-E Wrapper 사용 시도
            try:
                dalle_wrapper = DallEAPIWrapper(
                    model="dall-e-3",
                    size="1024x1024"
                )
                
                image_url = await asyncio.to_thread(dalle_wrapper.run, prompt)
                
                # URL에서 이미지 다운로드
                return await self._download_image_from_url(image_url, chapter)
                
            except Exception as e:
                logger.warning(f"LangChain DALL-E 실패, 직접 API 사용: {e}")
                
                # 직접 OpenAI API 호출
                response = await asyncio.to_thread(
                    self.openai_client.images.generate,
                    model="dall-e-3",
                    prompt=prompt,
                    size="1024x1024",
                    n=1,
                    response_format="b64_json"
                )
                
                # Base64 이미지 데이터 추출
                image_data = response.data[0].b64_json
                image_bytes = base64.b64decode(image_data)
                
                # 파일 저장
                chapter_number = chapter.get("chapter_number", 1)
                image_filename = f"chapter_{chapter_number}_{self.current_pipeline_id[:8]}.png"
                image_path = self.temp_storage / image_filename
                
                with open(image_path, 'wb') as f:
                    f.write(image_bytes)
                
                logger.info(f"이미지 생성 완료: {image_path}")
                return image_path
                
        except Exception as e:
            logger.error(f"이미지 생성 실패: {e}")
            raise
    
    async def _download_image_from_url(self, image_url: str, chapter: Dict[str, Any]) -> Path:
        """URL에서 이미지 다운로드"""
        import aiohttp
        
        chapter_number = chapter.get("chapter_number", 1)
        image_filename = f"chapter_{chapter_number}_{self.current_pipeline_id[:8]}.png"
        image_path = self.temp_storage / image_filename
        
        async with aiohttp.ClientSession() as session:
            async with session.get(image_url) as response:
                if response.status == 200:
                    image_bytes = await response.read()
                    with open(image_path, 'wb') as f:
                        f.write(image_bytes)
                    
                    logger.info(f"이미지 다운로드 완료: {image_path}")
                    return image_path
                else:
                    raise Exception(f"이미지 다운로드 실패: HTTP {response.status}")
    
    async def _upload_to_s3(self, local_path: Path, s3_key: str) -> str:
        """ 파일을 S3에 업로드 하고 URL 반환 """
        try:
            # 1. S3 file upload
            self.s3_client.upload_file(
                str(local_path),
                self.s3_bucket,
                s3_key,
                ExtraArgs={
                    'ContentType': self._get_content_type(local_path)
                }
            )
            
            # 2. S3 URL 생성
            s3_url = f"https://{self.s3_bucket}.s3.amazonaws.com/{s3_key}"
            
            logger.info(f"S3 업로드 완료 : {s3_url}")
            return s3_url
    
        except Exception as e:
            logger.error(f"S3 업로드 실패 : {e}")
            raise
    
    async def _cleanup_temp_files(self):
        """EC2 임시 파일 정리"""
        try:
            # 1. Image 임시 파일 삭제
            for image_data in self.pipeline_results.get("image_s3_urls", []):
                temp_path = Path(image_data.get("local_temp_path", ""))
                if temp_path.exists():
                    temp_path.unlink()
                    
            # 2. Audio 임시 파일 삭제
            for audio_data in self.pipeline_results.get("audio_s3_urls", []):
                temp_path = Path(audio_data.get("local_temp_path", ""))
                if temp_path.exists():
                    temp_path.unlink()
        
            logger.info("임시 파일 정리 완료")

        except Exception as e:
            logger.warning(f"임시 파일 정리 실패 : {e}")            
            
    async def _execute_text_generation(self, story_outline, progress_callback):
        """ 텍스트 생성 실행"""