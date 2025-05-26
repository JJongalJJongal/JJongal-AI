"""
이미지 생성기
LangChain DALL-E 기반 챕터별 이미지 생성
"""

import logging
import json
import uuid
import base64
from typing import Dict, Any, Optional, Callable, List
from pathlib import Path
import asyncio

# LangChain imports 

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper


# Project imports
from .base_generator import BaseGenerator

logger = logging.getLogger(__name__)

class ImageGenerator(BaseGenerator):
    """LangChain DALL-E 기반 이미지 생성기"""
    
    def __init__(self, 
                 openai_client=None,
                 prompts_file_path: str = "chatbot/data/prompts/chatbot_b_prompts.json",
                 model_name: str = "dall-e-3",
                 image_size: str = "1024x1024",
                 temp_storage_path: str = "/tmp/fairy_tales",
                 max_retries: int = 3):
        """
        Args:
            openai_client: OpenAI 클라이언트
            prompts_file_path: 프롬프트 JSON 파일 경로
            model_name: 이미지 생성 모델명 (dall-e-3)
            image_size: 이미지 크기 ("1024x1024", "1024x1536", "1536x1024")
            temp_storage_path: 임시 저장 경로
            max_retries: 최대 재시도 횟수
        """
        super().__init__(max_retries=max_retries, timeout=180.0)
        
        self.openai_client = openai_client
        self.prompts_file_path = prompts_file_path
        self.model_name = model_name
        self.image_size = image_size
        self.temp_storage_path = Path(temp_storage_path)
        
        # 프롬프트 템플릿
        self.prompts = None
        self.image_prompt_template = None
        
        # LangChain 구성 요소
        self.dalle_wrapper = None
        self.prompt_enhancer_chain = None
        self.llm = None
        
        # 초기화
        self._initialize_components()
    
    def _initialize_components(self):
        """구성 요소 초기화"""
        try:
            # 1. 임시 저장소 생성
            self.temp_storage_path.mkdir(parents=True, exist_ok=True)
            
            # 2. 프롬프트 로드
            self._load_prompts()
            
            # 3. LangChain 구성 요소 초기화
            self._setup_langchain_components()
            
            logger.info("ImageGenerator 초기화 완료")
            
        except Exception as e:
            logger.error(f"ImageGenerator 초기화 실패: {e}")
            raise
    
    def _load_prompts(self):
        """프롬프트 파일 로드"""
        try:
            with open(self.prompts_file_path, 'r', encoding='utf-8') as f:
                self.prompts = json.load(f)
            
            # 이미지 프롬프트 템플릿 추출
            story_templates = self.prompts.get("story_generation_templates", {})
            self.image_prompt_template = story_templates.get(
                "image_prompt_system_message", 
                "Create a beautiful illustration for: {chapter_title}\n\nContent: {chapter_content}"
            )
            
            logger.info(f"프롬프트 파일 로드 완료: {self.prompts_file_path}")
            
        except Exception as e:
            logger.error(f"프롬프트 파일 로드 실패: {e}")
            # 기본 템플릿 사용
            self.image_prompt_template = "Create a beautiful illustration for: {chapter_title}\n\nContent: {chapter_content}"
    
    def _setup_langchain_components(self):
        """LangChain 구성 요소 설정"""
        try:
            # 1. LLM 초기화
            self.llm = ChatOpenAI(
                temperature=0.7,
                model="gpt-4o-mini"
            )
            
            # 2. DALL-E Wrapper 초기화
            self.dalle_wrapper = DallEAPIWrapper(
                model=self.model_name,
                size=self.image_size
            )
            
            # 3. 프롬프트 개선 체인 설정 (JSON에서 템플릿 가져오기)
            enhancer_template = self._get_image_enhancer_template()
            prompt_template = ChatPromptTemplate.from_template(enhancer_template)
            self.prompt_enhancer_chain = prompt_template | self.llm | StrOutputParser()
            
            logger.info("LangChain 구성 요소 설정 완료")
            
        except Exception as e:
            logger.error(f"LangChain 구성 요소 설정 실패: {e}")
            raise
    
    def _get_image_enhancer_template(self) -> str:
        """JSON에서 이미지 프롬프트 개선 템플릿 가져오기"""
        try:
            # Korean fairy tale enhanced template 사용
            enhanced_templates = self.prompts.get("enhanced_image_templates", {})
            korean_template = enhanced_templates.get("korean_fairy_tale_enhanced", "")
            
            if korean_template:
                # 템플릿을 LangChain 체인용으로 수정
                enhancer_template = f"""
                다음 동화 챕터 정보를 바탕으로 한국 전래동화 스타일의 DALL-E 3 프롬프트를 작성해주세요.
                
                챕터 제목: {{chapter_title}}
                챕터 내용: {{chapter_content}}
                
                다음 템플릿을 참고하여 영어로 작성해주세요:
                {korean_template}
                
                scene_description을 챕터 내용으로, age_range를 적절한 연령대로 대체하여 완성된 DALL-E 3 프롬프트를 작성해주세요.
                """
                return enhancer_template
            
            # 기본 템플릿 사용
            return """
            다음 동화 챕터 정보를 바탕으로 DALL-E 3에 최적화된 이미지 생성 프롬프트를 작성해주세요.
            
            챕터 제목: {chapter_title}
            챕터 내용: {chapter_content}
            
            요구사항:
            - 동화적이고 따뜻한 분위기
            - 아이들이 좋아할 만한 귀여운 캐릭터
            - 밝고 화사한 색감
            - 상세하고 구체적인 묘사
            - 영어로 작성
            
            DALL-E 3 프롬프트:
            """
            
        except Exception as e:
            logger.warning(f"이미지 개선 템플릿 로드 실패, 기본 템플릿 사용: {e}")
            return """
            다음 동화 챕터 정보를 바탕으로 DALL-E 3에 최적화된 이미지 생성 프롬프트를 작성해주세요.
            
            챕터 제목: {chapter_title}
            챕터 내용: {chapter_content}
            
            요구사항:
            - 동화적이고 따뜻한 분위기
            - 아이들이 좋아할 만한 귀여운 캐릭터
            - 밝고 화사한 색감
            - 상세하고 구체적인 묘사
            - 영어로 작성
            
            DALL-E 3 프롬프트:
            """
    
    async def generate(self, 
                      input_data: Dict[str, Any], 
                      progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        챕터별 이미지 생성
        
        Args:
            input_data: {
                "story_data": {
                    "title": "동화 제목",
                    "chapters": [
                        {
                            "chapter_number": 1,
                            "chapter_title": "챕터 제목",
                            "chapter_content": "챕터 내용"
                        }
                    ]
                },
                "story_id": "스토리 ID"
            }
            progress_callback: 진행 상황 콜백
            
        Returns:
            {
                "images": [
                    {
                        "chapter_number": 1,
                        "image_path": "로컬 이미지 파일 경로",
                        "image_prompt": "사용된 이미지 프롬프트",
                        "generation_time": 생성 시간
                    }
                ],
                "metadata": {
                    "total_images": 생성된 이미지 수,
                    "model_used": "사용된 모델",
                    "total_generation_time": 총 생성 시간
                }
            }
        """
        
        task_id = str(uuid.uuid4())
        self.current_task_id = task_id
        
        try:
            story_data = input_data.get("story_data", {})
            story_id = input_data.get("story_id", task_id)
            chapters = story_data.get("chapters", [])
            
            if not chapters:
                raise ValueError("생성할 챕터가 없습니다")
            
            if progress_callback:
                await progress_callback({
                    "step": "image_generation",
                    "status": "starting",
                    "total_chapters": len(chapters),
                    "task_id": task_id
                })
            
            generated_images = []
            
            # 각 챕터별로 이미지 생성
            for i, chapter in enumerate(chapters):
                chapter_start_time = asyncio.get_event_loop().time()
                
                if progress_callback:
                    await progress_callback({
                        "step": "image_generation",
                        "status": "processing_chapter",
                        "current_chapter": i + 1,
                        "total_chapters": len(chapters),
                        "chapter_title": chapter.get("chapter_title", "")
                    })
                
                # 1. 이미지 프롬프트 생성 (LangChain 또는 기본)
                image_prompt = await self._create_enhanced_image_prompt(chapter, story_data)
                
                # 2. 이미지 생성 (LangChain 또는 직접 API)
                image_path = await self._generate_single_image(
                    prompt=image_prompt,
                    chapter_number=chapter.get("chapter_number", i + 1),
                    story_id=story_id
                )
                
                chapter_generation_time = asyncio.get_event_loop().time() - chapter_start_time
                
                generated_images.append({
                    "chapter_number": chapter.get("chapter_number", i + 1),
                    "image_path": str(image_path),
                    "image_prompt": image_prompt,
                    "generation_time": chapter_generation_time
                })
                
                if progress_callback:
                    await progress_callback({
                        "step": "image_generation",
                        "status": "chapter_completed",
                        "current_chapter": i + 1,
                        "total_chapters": len(chapters),
                        "generation_time": chapter_generation_time
                    })
            
            if progress_callback:
                await progress_callback({
                    "step": "image_generation",
                    "status": "completed",
                    "total_images": len(generated_images)
                })
            
            return {
                "images": generated_images,
                "metadata": {
                    "total_images": len(generated_images),
                    "model_used": self.model_name,
                    "total_generation_time": self.total_generation_time,
                    "story_id": story_id,
                    "task_id": task_id
                }
            }
            
        except Exception as e:
            logger.error(f"이미지 생성 실패 (task_id: {task_id}): {e}")
            raise
    
    async def _create_enhanced_image_prompt(self, chapter: Dict[str, Any], story_data: Dict[str, Any] = None) -> str:
        """LangChain을 사용한 향상된 이미지 프롬프트 생성"""
        
        chapter_title = chapter.get("chapter_title", "")
        chapter_content = chapter.get("content", chapter.get("chapter_content", ""))
        
        # 내용이 너무 길면 요약
        if len(chapter_content) > 500:
            chapter_content = chapter_content[:500] + "..."
        
        # LangChain 프롬프트 개선 체인 사용
        if self.prompt_enhancer_chain:
            try:
                enhanced_prompt = await self.prompt_enhancer_chain.ainvoke({
                    "chapter_title": chapter_title,
                    "chapter_content": chapter_content
                })
                return enhanced_prompt.strip()
            except Exception as e:
                logger.warning(f"LangChain 프롬프트 개선 실패, 기본 방식 사용: {e}")
        
        # 기본 프롬프트 생성 (연령대 정보 전달)
        age_group = None
        if story_data:
            age_group = story_data.get("age_group")
        
        return self._create_basic_image_prompt(chapter_title, chapter_content, age_group)
    
    def _create_basic_image_prompt(self, chapter_title: str, chapter_content: str, age_group: str = None) -> str:
        """기본 이미지 프롬프트 생성 (JSON 템플릿 사용)"""
        
        try:
            # JSON에서 이미지 생성 템플릿 가져오기
            image_templates = self.prompts.get("image_generation_templates", [])
            
            # 연령대별 템플릿 선택
            selected_template = self._select_age_appropriate_template(image_templates, age_group)
            
            if selected_template:
                # 템플릿에서 프롬프트 가져와서 데이터 삽입
                template_prompt = selected_template.get("prompt", "")
                
                # 기본 값들 설정
                scene_description = f"Chapter: {chapter_title}. {chapter_content[:200]}..."
                theme = "Korean fairy tale"
                characters = "kawaii characters"
                setting = "magical storybook world"
                
                prompt = template_prompt.format(
                    scene_description=scene_description,
                    theme=theme,
                    characters=characters,
                    setting=setting
                )
                
                return prompt
            
            # JSON 템플릿이 없으면 기본 템플릿 사용
            return self._get_fallback_image_prompt(chapter_title, chapter_content)
            
        except Exception as e:
            logger.warning(f"JSON 이미지 템플릿 사용 실패, 기본 템플릿 사용: {e}")
            return self._get_fallback_image_prompt(chapter_title, chapter_content)
    
    def _select_age_appropriate_template(self, templates: List[Dict], age_group: str = None) -> Dict[str, Any]:
        """연령대에 적합한 템플릿 선택"""
        
        if not templates:
            return None
        
        # 연령대가 지정된 경우 매칭되는 템플릿 찾기
        if age_group:
            # 정확한 매칭 시도
            for template in templates:
                if template.get("age_group") == age_group:
                    return template
            
            # 연령대 범위 매칭 시도 (예: "4-7"에서 "5" 찾기)
            try:
                target_age = int(age_group)
                for template in templates:
                    template_age = template.get("age_group", "")
                    if "-" in template_age:
                        min_age, max_age = map(int, template_age.split("-"))
                        if min_age <= target_age <= max_age:
                            return template
            except (ValueError, TypeError):
                pass
        
        # 기본 템플릿 선택 (4-7세용 우선, 없으면 첫 번째)
        for template in templates:
            if template.get("age_group") == "4-7":
                return template
        
        return templates[0] if templates else None
    
    def _get_fallback_image_prompt(self, chapter_title: str, chapter_content: str) -> str:
        """기본 폴백 이미지 프롬프트"""
        return f"""
        Create a heartwarming Japanese anime children's storybook illustration: {chapter_title}
        
        Scene: {chapter_content[:200]}...
        
        Style: Studio Ghibli watercolor style with incredibly detailed pastoral countryside elements
        Characters: Ultra kawaii characters with huge sparkling diamond eyes, baby-soft features, rosy button cheeks
        Colors: Warm pastel palette - gentle mint greens, cream whites, soft peach tones, sky blues
        Atmosphere: Golden hour lighting, peaceful countryside morning with soft warm sunbeams
        Mood: Incredibly warm, safe, nurturing atmosphere that feels like a gentle hug
        Art style: Studio Ghibli masterpiece quality with rich environmental storytelling
        """
    
    async def _generate_single_image(self, 
                                   prompt: str, 
                                   chapter_number: int, 
                                   story_id: str) -> Path:
        """단일 이미지 생성 (LangChain 또는 직접 API)"""
        
        try:
            # LangChain DALL-E Wrapper 사용
            if self.dalle_wrapper:
                try:
                    image_url = await asyncio.to_thread(
                        self.dalle_wrapper.run,
                        prompt
                    )
                    
                    # URL에서 이미지 다운로드
                    image_path = await self._download_image_from_url(
                        image_url, chapter_number, story_id
                    )
                    return image_path
                    
                except Exception as e:
                    logger.warning(f"LangChain DALL-E 실패, 직접 API 사용: {e}")
            
            # 직접 OpenAI API 호출
            return await self._generate_with_direct_api(prompt, chapter_number, story_id)
            
        except Exception as e:
            logger.error(f"이미지 생성 실패 (챕터 {chapter_number}): {e}")
            raise
    
    async def _generate_with_direct_api(self, 
                                      prompt: str, 
                                      chapter_number: int, 
                                      story_id: str) -> Path:
        """직접 OpenAI API를 사용한 이미지 생성"""
        
        response = await asyncio.to_thread(
            self.openai_client.images.generate,
            model=self.model_name,
            prompt=prompt,
            size=self.image_size,
            n=1,
            response_format="b64_json"
        )
        
        # Base64 이미지 데이터 추출
        image_data = response.data[0].b64_json
        image_bytes = base64.b64decode(image_data)
        
        # 파일 저장
        image_filename = f"chapter_{chapter_number}_{story_id[:8]}.png"
        image_path = self.temp_storage_path / image_filename
        
        with open(image_path, 'wb') as f:
            f.write(image_bytes)
        
        logger.info(f"이미지 생성 완료: {image_path}")
        return image_path
    
    async def _download_image_from_url(self, 
                                     image_url: str, 
                                     chapter_number: int, 
                                     story_id: str) -> Path:
        """URL에서 이미지 다운로드"""
        
        import aiohttp
        
        image_filename = f"chapter_{chapter_number}_{story_id[:8]}.png"
        image_path = self.temp_storage_path / image_filename
        
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
    
    
    async def generate_batch(self, 
                           chapters: List[Dict[str, Any]], 
                           story_id: str,
                           progress_callback: Optional[Callable] = None) -> List[Dict[str, Any]]:
        """여러 챕터의 이미지를 병렬로 생성"""
        
        if progress_callback:
            await progress_callback({
                "step": "image_generation",
                "status": "batch_starting",
                "total_chapters": len(chapters)
            })
        
        # 병렬 생성 태스크 생성
        tasks = []
        for i, chapter in enumerate(chapters):
            task = self._generate_chapter_image_task(chapter, story_id, i + 1)
            tasks.append(task)
        
        # 병렬 실행
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 결과 처리
        generated_images = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"챕터 {i + 1} 이미지 생성 실패: {result}")
                # 실패한 경우에도 빈 결과 추가
                generated_images.append({
                    "chapter_number": i + 1,
                    "image_path": None,
                    "error": str(result)
                })
            else:
                generated_images.append(result)
        
        if progress_callback:
            successful_count = sum(1 for img in generated_images if img.get("image_path"))
            await progress_callback({
                "step": "image_generation",
                "status": "batch_completed",
                "successful_images": successful_count,
                "total_chapters": len(chapters)
            })
        
        return generated_images
    
    async def _generate_chapter_image_task(self, 
                                         chapter: Dict[str, Any], 
                                         story_id: str, 
                                         chapter_number: int) -> Dict[str, Any]:
        """단일 챕터 이미지 생성 태스크"""
        
        try:
            start_time = asyncio.get_event_loop().time()
            
            # 이미지 프롬프트 생성 (향상된 버전)
            image_prompt = await self._create_enhanced_image_prompt(chapter)
            
            # 이미지 생성
            image_path = await self._generate_single_image(
                prompt=image_prompt,
                chapter_number=chapter_number,
                story_id=story_id
            )
            
            generation_time = asyncio.get_event_loop().time() - start_time
            
            return {
                "chapter_number": chapter_number,
                "image_path": str(image_path),
                "image_prompt": image_prompt,
                "generation_time": generation_time
            }
            
        except Exception as e:
            logger.error(f"챕터 {chapter_number} 이미지 생성 실패: {e}")
            raise
    
    # 임시 파일 정리는 ContentPipeline에서 담당
    # cleanup_temp_files 메서드 제거됨 - ContentPipeline._cleanup_temp_files() 사용
    
    async def health_check(self) -> bool:
        """ImageGenerator 상태 확인"""
        
        try:
            # 기본 상태 확인
            if not await super().health_check():
                return False
            
            # OpenAI 클라이언트 확인
            if not self.openai_client:
                logger.error("OpenAI 클라이언트가 설정되지 않음")
                return False
            
            # 임시 저장소 확인
            if not self.temp_storage_path.exists():
                logger.error(f"임시 저장소가 존재하지 않음: {self.temp_storage_path}")
                return False
            
            # 간단한 이미지 생성 테스트
            test_prompt = "A simple test image of a cute cartoon character"
            
            try:
                # 타임아웃을 짧게 설정하여 빠른 테스트
                original_timeout = self.timeout
                self.timeout = 30.0
                
                response = await asyncio.to_thread(
                    self.openai_client.images.generate,
                    model=self.model_name,
                    prompt=test_prompt,
                    size="1024x1024",
                    n=1,
                    response_format="b64_json"
                )
                
                # 응답 확인
                return len(response.data) > 0 and response.data[0].b64_json
                
            finally:
                self.timeout = original_timeout
                
        except Exception as e:
            logger.error(f"ImageGenerator health check 실패: {e}")
            return False
    
    def get_supported_sizes(self) -> List[str]:
        """지원되는 이미지 크기 목록 반환"""
        return ["1024x1024", "1024x1536", "1536x1024", "auto"]
    
    def estimate_generation_time(self, chapter_count: int) -> float:
        """예상 생성 시간 계산 (초)"""
        # gpt-image-1의 평균 생성 시간을 기반으로 추정
        avg_time_per_image = 15.0  # 초
        return chapter_count * avg_time_per_image
    
    