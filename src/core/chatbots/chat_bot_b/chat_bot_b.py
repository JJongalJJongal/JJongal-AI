"""
Modern Ari (ChatBot B) - API v2.0 Compliant Story Generation

Simplified story generation based on API_DOCUMENTATION.md v2.0:
- Accepts StoryRequest format input
- Returns Story format output with chapters
- Supports voice configuration and multimedia generation
- 2025 LangChain LCEL chains for efficiency
"""

import time
import uuid
from typing import Dict, Any, Optional, Callable
from datetime import datetime

# 2025 LangChain imports
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
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
    Modern Ari (ChatBot B) - Story Generation
    
    Features:
    1. Accepts StoryRequest 
    2. Returns Story with chapters 
    3. LCEL-based generation chains
    4. Voice configuration support
    5. Multimedia generation (images + audio)
    """
    
    def __init__(self, 
                 model_name: str = "gpt-4o-mini",
                 output_dir: str = "output"):
        """
        Initialize Ari
        
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
        
        # Initialize generators with VectorDB support
        self.text_generator = TextGenerator(
            openai_client=self.openai_client,
            enable_rag=True  # Enable RAG for ChatBot B
        )
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

**중요**: 이미지 생성을 위한 image_prompt는 반드시 **영어로** 작성하세요. 
이미지 품질 향상을 위해 영어 묘사가 필요합니다.

JSON 형식으로 응답:
{{
  "title": "동화 제목",
  "chapters": [
    {{
      "chapter_number": 1,
      "title": "챕터 제목 (한글)",
      "content": "챕터 내용 (한글, 대사 포함)",
      "image_prompt": "English image description for DALL-E 3: characters, setting, mood, style details"
    }}
  ]
}}

**image_prompt 작성 예시:**
- "Cozy children's book illustration of a friendly rabbit and a young child in a magical garden with colorful flowers, hand-drawn watercolor style, soft pastels, warm lighting"
- "Beautiful storybook scene showing a gentle bear helping a lost princess in a peaceful forest, colored pencil sketch with watercolor washes, nostalgic fairy tale atmosphere"
"""),
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
            
            # Build API v2.0 compliant Story response with SAFE multimedia integration
            chapters = []
            
            # 🛡️ SECURITY: Safe multimedia mapping with validation
            image_map, audio_map = self._create_safe_multimedia_mapping(multimedia_data)
            logger.info(f"Safe mapping created: {len(image_map)} images, {len(audio_map)} audio files")
            
            for i, chapter_data in enumerate(story_data.get("chapters", [])):
                chapter_number = chapter_data.get("chapter_number", i + 1)
                
                chapter = {
                    "chapter_number": chapter_number,
                    "title": chapter_data.get("title", f"Chapter {chapter_number}"),
                    "content": chapter_data.get("content", "")
                }
                
                # 🎯 PERFORMANCE: O(1) lookup instead of linear search
                # 🛡️ SECURITY: Safe access with validation
                if chapter_number in image_map:
                    chapter["image_url"] = image_map[chapter_number]
                    logger.debug(f"Chapter {chapter_number}: Image URL attached")
                
                if chapter_number in audio_map:
                    chapter["audio_url"] = audio_map[chapter_number]  
                    logger.debug(f"Chapter {chapter_number}: Audio URL attached")
                
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
        """Generate multimedia content (images and audio) - IMPROVED VERSION"""
        multimedia_result = {
            "images": [],
            "audio_files": []
        }
        
        try:
            chapters = story_data.get("chapters", [])
            logger.info(f"Starting multimedia generation for {len(chapters)} chapters")
            
            # 🖼️ STEP 1: Generate images for each chapter
            if chapters:
                # Prepare image generation data in correct format
                image_chapters = []
                for chapter in chapters:
                    # Auto-generate image prompt if not present
                    image_prompt = chapter.get("image_prompt", "")
                    if not image_prompt:
                        # Create image prompt from chapter content
                        content = chapter.get("content", "")[:200]  # First 200 chars
                        title = chapter.get("title", "")
                        image_prompt = f"Children's book illustration: {title}. {content}. Colorful, friendly, child-appropriate style."
                    
                    image_chapters.append({
                        "title": chapter.get("title", ""),
                        "content": chapter.get("content", ""),
                        "image_prompt": image_prompt,
                        "chapter_number": chapter.get("chapter_number", len(image_chapters) + 1)
                    })
                
                if progress_callback:
                    await progress_callback({
                        "step": "image_generation_start",
                        "status": "processing",
                        "total_chapters": len(image_chapters)
                    })
                
                # Generate images using correct format
                image_result = await self.image_generator.generate({
                    "story_data": {"chapters": image_chapters},
                    "story_id": story_id,
                    "target_age": 7
                }, progress_callback)
                
                # Extract image results
                if image_result.get("images"):
                    multimedia_result["images"] = image_result["images"]
                    logger.info(f"Generated {len(multimedia_result['images'])} images")
                else:
                    logger.warning("No images were generated")
            
            # 🔊 STEP 2: Generate audio narration
            if chapters and progress_callback:
                await progress_callback({"step": "audio_generation", "status": "processing"})
            
            # Generate audio using correct VoiceGenerator format
            if chapters:
                audio_result = await self.voice_generator.generate({
                    "story_data": {"chapters": chapters},
                    "voice_config": voice_config,
                    "story_id": story_id,
                    "target_age": 7
                }, progress_callback)
                
                # Extract audio results
                if audio_result.get("audio_files"):
                    multimedia_result["audio_files"] = audio_result["audio_files"]
                    logger.info(f"Generated {len(multimedia_result['audio_files'])} audio files")
                else:
                    logger.warning("No audio files were generated")
            
            logger.info(f"Multimedia generation completed: {len(multimedia_result['images'])} images, {len(multimedia_result['audio_files'])} audio files")
            return multimedia_result
            
        except Exception as e:
            logger.error(f"Multimedia generation failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return multimedia_result
    
    def _create_safe_multimedia_mapping(self, multimedia_data: Dict[str, Any]) -> tuple[Dict[int, str], Dict[int, str]]:
        """
        🛡️ SECURITY-FIRST: Create safe multimedia-to-chapter mapping
        🎯 PERFORMANCE-OPTIMIZED: O(1) lookup with hashmaps
        📚 LEARNING: Demonstrates defensive programming principles
        
        Args:
            multimedia_data: Raw multimedia generation results
            
        Returns:
            tuple: (image_map, audio_map) - Chapter number to URL mapping
            
        Security Features:
        - Input validation and type checking
        - Status verification for each item
        - Safe dictionary access with .get()
        - Exception isolation for individual items
        
        Performance Features:  
        - Single-pass iteration O(n)
        - HashMap lookup O(1)
        - Memory-efficient processing
        """
        image_map = {}
        audio_map = {}
        
        try:
            # 🛡️ SECURITY: Validate input structure
            if not isinstance(multimedia_data, dict):
                logger.warning("Multimedia data is not a dictionary")
                return image_map, audio_map
            
            # 🖼️ PHASE 1: Process Images with Security Validation
            images = multimedia_data.get("images", [])
            if isinstance(images, list):
                for i, img in enumerate(images):
                    try:
                        # 🛡️ Multi-layer validation
                        if (isinstance(img, dict) and 
                            img.get("status") == "success" and 
                            isinstance(img.get("chapter_number"), int) and
                            isinstance(img.get("url"), str) and
                            len(img.get("url", "")) > 0):
                            
                            chapter_num = img["chapter_number"]
                            image_url = img["url"]
                            
                            # 🛡️ URL validation (basic)
                            if image_url.startswith(("http://", "https://")):
                                image_map[chapter_num] = image_url
                                logger.debug(f"✅ Image mapped: Chapter {chapter_num} -> URL")
                            else:
                                logger.warning(f"⚠️ Invalid image URL format for chapter {chapter_num}")
                        else:
                            logger.debug(f"⚠️ Image {i} validation failed or status != success")
                    
                    except Exception as e:
                        logger.warning(f"⚠️ Error processing image {i}: {str(e)[:100]}")
                        continue
            
            # 🔊 PHASE 2: Process Audio with Security Validation  
            audio_files = multimedia_data.get("audio_files", [])
            if isinstance(audio_files, list):
                for i, audio in enumerate(audio_files):
                    try:
                        # 🛡️ Multi-layer validation
                        if (isinstance(audio, dict) and 
                            audio.get("status") == "success" and 
                            isinstance(audio.get("chapter_number"), int) and
                            audio.get("audio_path") is not None):
                            
                            chapter_num = audio["chapter_number"]
                            audio_path = audio["audio_path"]
                            
                            # 🔄 FUTURE: Convert local path to URL
                            # For now, use local path directly
                            audio_url = self._convert_audio_path_to_url(audio_path)
                            
                            if audio_url:
                                audio_map[chapter_num] = audio_url
                                logger.debug(f"✅ Audio mapped: Chapter {chapter_num} -> {audio_url}")
                            else:
                                logger.warning(f"⚠️ Failed to convert audio path for chapter {chapter_num}")
                        else:
                            logger.debug(f"⚠️ Audio {i} validation failed or status != success")
                    
                    except Exception as e:
                        logger.warning(f"⚠️ Error processing audio {i}: {str(e)[:100]}")
                        continue
            
            # 📊 SUMMARY: Log mapping results
            logger.info(f"🎯 Safe mapping complete: {len(image_map)} images, {len(audio_map)} audio files successfully mapped")
            
            return image_map, audio_map
            
        except Exception as e:
            logger.error(f"❌ Critical error in multimedia mapping: {e}")
            return {}, {}
    
    def _convert_audio_path_to_url(self, audio_path: str) -> str:
        """
        🔄 Convert local audio file path to accessible URL
        📚 LEARNING: Placeholder for future URL conversion logic
        
        Args:
            audio_path: Local file system path
            
        Returns:
            str: Accessible URL or path
            
        TODO: Implement proper URL conversion for production
        - Static file serving
        - CDN integration  
        - Signed URLs for security
        """
        if not audio_path or not isinstance(audio_path, str):
            return ""
        
        # 🔄 TEMPORARY: Return local path for development
        # In production, this should convert to a proper URL
        return audio_path
    
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

# Factory function  
def create_modern_ari(**kwargs) -> ModernAri:
    """Create ModernAri instance"""
    return ModernAri(**kwargs)
