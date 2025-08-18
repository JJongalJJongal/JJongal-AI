"""
Modern Ari (ChatBot B) - API v2.0 Compliant Story Generation

Simplified story generation based on API_DOCUMENTATION.md v2.0:
- Accepts StoryRequest format input
- Returns Story format output with chapters
- Supports voice configuration and multimedia generation
- 2025 LangChain LCEL chains for efficiency
"""

import os
import time
import uuid
import asyncio
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime

# 2025 LangChain imports
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_openai import ChatOpenAI

from src.shared.utils.logging import get_module_logger
from src.shared.utils.openai import initialize_client

# Import existing generators  
from .generators.text_generator import TextGenerator
from .generators.image_generator import ImageGenerator
from .generators.voice_generator import VoiceGenerator

logger = get_module_logger(__name__)

class ModernAri:
    """
    Modern Ari (ChatBot B) - API v2.0 Compliant Story Generation
    
    Features:
    1. Accepts StoryRequest (API v2.0 format)
    2. Returns Story with chapters (API v2.0 format)
    3. LCEL-based generation chains
    4. Voice configuration support
    5. Multimedia generation (images + audio)
    """
    
    def __init__(self, 
                 model_name: str = "gpt-4o-mini",
                 output_dir: str = "output"):
        """
        Initialize Modern Ari
        
        Args:
            model_name: LLM model name
            output_dir: Output directory for generated files
        """
        self.model_name = model_name
        self.output_dir = output_dir
        
        # Modern LLM setup
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0.7,
            model_kwargs={"top_p": 0.9}
        )
        
        # Setup modern story generation chain
        self._setup_story_chain()
        
        # Initialize OpenAI client
        self.openai_client = initialize_client()
        
        # Initialize generators
        self.text_generator = TextGenerator(openai_client=self.openai_client)
        self.image_generator = ImageGenerator(openai_client=self.openai_client)
        self.voice_generator = VoiceGenerator()
        
        logger.info(f"Modern Ari initialized with {model_name}")
        
    def _setup_story_chain(self):
        """Setup modern LCEL story generation chain"""
        
        self.story_prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 '아리'입니다. 아이들을 위한 동화를 생성하는 전문가입니다.

아이 나이: {target_age}세
수집된 이야기 요소:
- 등장인물: {characters}
- 배경: {settings}  
- 문제: {problems}
- 해결방법: {resolutions}

지침:
1. 연령에 적합한 내용과 어휘 사용
2. 교육적 가치가 있는 스토리
3. 명확한 기승전결 구조
4. 3-5개 챕터로 구성
5. 각 챕터는 이미지화 가능하도록 묘사

JSON 형식으로 응답:
{{
  "title": "동화 제목",
  "chapters": [
    {{
      "chapter_number": 1,
      "title": "챕터 제목",
      "content": "챕터 내용",
      "image_prompt": "이미지 생성을 위한 프롬프트"
    }}
  ]
}}"""),
            ("human", "위 정보를 바탕으로 동화를 생성해주세요.")
        ])
        
        # Modern LCEL chain
        self.story_chain = (
            self.story_prompt
            | self.llm
            | JsonOutputParser()
        )
    async def generate_story_from_request(self, story_request: Dict[str, Any], progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Generate story from API v2.0 StoryRequest format
        
        Args:
            story_request: StoryRequest format from API v2.0
            progress_callback: Optional progress callback function
            
        Returns:
            Story format compliant with API v2.0
        """
        start_time = time.time()
        story_id = str(uuid.uuid4())
        
        try:
            # Extract data from StoryRequest
            child_name = story_request.get("child_name", "친구")
            age = story_request.get("age", 7)
            interests = story_request.get("interests", [])
            story_elements = story_request.get("story_elements", {})
            voice_config = story_request.get("voice_config", {})
            
            # Progress callback
            if progress_callback:
                await progress_callback({"step": "story_generation", "status": "starting"})
            
            # Generate story using LCEL chain
            story_data = await self.story_chain.ainvoke({
                "target_age": age,
                "characters": story_elements.get("main_character", ""),
                "settings": story_elements.get("setting", ""),
                "problems": story_elements.get("theme", ""),
                "resolutions": "용기와 지혜로 문제를 해결"
            })
            
            # Progress callback
            if progress_callback:
                await progress_callback({"step": "multimedia_generation", "status": "starting"})
            
            # Generate multimedia content
            multimedia_data = await self._generate_multimedia(
                story_data, story_id, voice_config, progress_callback
            )
            
            # Build API v2.0 compliant Story response
            chapters = []
            for i, chapter_data in enumerate(story_data.get("chapters", [])):
                chapter = {
                    "chapter_number": chapter_data.get("chapter_number", i + 1),
                    "title": chapter_data.get("title", f"Chapter {i + 1}"),
                    "content": chapter_data.get("content", "")
                }
                
                # Add multimedia URLs if available
                if multimedia_data["images"] and i < len(multimedia_data["images"]):
                    chapter["image_url"] = multimedia_data["images"][i].get("url", "")
                
                if multimedia_data["audio_files"] and i < len(multimedia_data["audio_files"]):
                    chapter["audio_url"] = multimedia_data["audio_files"][i].get("url", "")
                
                chapters.append(chapter)
            
            generation_time = time.time() - start_time
            
            # API v2.0 Story format
            result = {
                "story_id": story_id,
                "title": story_data.get("title", f"{child_name}의 모험"),
                "status": "completed",
                "chapters": chapters,
                "created_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
                "generation_time": round(generation_time, 1)
            }
            
            logger.info(f"Story generated successfully in {generation_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Story generation failed: {e}")
            # Return failed status instead of raising
            return {
                "story_id": story_id,
                "title": f"{story_request.get('child_name', '친구')}의 이야기",
                "status": "failed",
                "chapters": [],
                "created_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
                "generation_time": round(time.time() - start_time, 1)
            }
    
    async def _generate_multimedia(self, story_data: Dict, story_id: str, voice_config: Dict, progress_callback: Optional[Callable]) -> Dict[str, Any]:
        """Generate multimedia content (images and audio)"""
        multimedia_result = {
            "images": [],
            "audio_files": []
        }
        
        try:
            chapters = story_data.get("chapters", [])
            
            # Generate images for each chapter
            for chapter in chapters:
                if progress_callback:
                    await progress_callback({
                        "step": f"image_generation_chapter_{chapter.get('chapter_number', 0)}",
                        "status": "processing"
                    })
                
                image_prompt = chapter.get("image_prompt", "")
                if image_prompt:
                    # Use existing image generator
                    image_data = await self.image_generator.generate({
                        "prompt": image_prompt,
                        "story_id": story_id,
                        "chapter_number": chapter.get("chapter_number", 0)
                    })
                    
                    multimedia_result["images"].append(image_data)
            
            # Generate audio narration
            if progress_callback:
                await progress_callback({"step": "audio_generation", "status": "processing"})
            
            # Combine all chapter content for audio
            full_story_text = f"{story_data.get('title', '')}\n\n"
            for chapter in chapters:
                full_story_text += f"{chapter.get('title', '')}\n{chapter.get('content', '')}\n\n"
            
            # Use existing voice generator with voice config
            voice_id = voice_config.get("child_voice_id") or voice_config.get("parent_voice_id")
            audio_data = await self.voice_generator.generate({
                "text": full_story_text,
                "story_id": story_id,
                "voice_id": voice_id,
                "narrator_voice": voice_config.get("narrator_voice", "default")
            })
            
            multimedia_result["audio_files"] = audio_data.get("audio_files", [])
            
            return multimedia_result
            
        except Exception as e:
            logger.error(f"Multimedia generation failed: {e}")
            return multimedia_result
    
    # Backward compatibility methods
    async def generate_story(self, progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Legacy method - use generate_story_from_request for API v2.0"""
        logger.warning("Using legacy generate_story method. Use generate_story_from_request for API v2.0 compliance.")
        
        # Convert to StoryRequest format if possible
        story_request = {
            "child_name": "친구",
            "age": getattr(self, 'target_age', 7),
            "interests": [],
            "story_elements": getattr(self, 'story_outline', {}).get("story_elements", {}),
            "voice_config": {}
        }
        
        return await self.generate_story_from_request(story_request, progress_callback)
        
    async def generate_complete_story(self, **kwargs) -> Dict[str, Any]:
        """Legacy alias - use generate_story_from_request for API v2.0"""
        return await self.generate_story(**kwargs)
    
    def set_story_outline(self, outline: Dict[str, Any]):
        """Legacy method - use generate_story_from_request for API v2.0"""
        logger.warning("Using legacy set_story_outline. Use generate_story_from_request for API v2.0 compliance.")
        self.story_outline = outline
    
    def set_target_age(self, age: int):
        """Legacy method - use generate_story_from_request for API v2.0"""
        logger.warning("Using legacy set_target_age. Use generate_story_from_request for API v2.0 compliance.")
        self.target_age = age

# Backward compatibility aliases
ChatBotB = ModernAri
EnhancedAri = ModernAri

# Factory function  
def create_modern_ari(**kwargs) -> ModernAri:
    """Create ModernAri instance"""
    return ModernAri(**kwargs)
