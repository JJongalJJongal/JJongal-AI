import asyncio
import uuid
from typing import Dict, Any, Optional, Callable, List
from pathlib import Path
import time

from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper
from langchain_community.tools.openai_dalle_image_generation import OpenAIDALLEImageGenerationTool

from .base_generator import BaseGenerator
from src.shared.configs.prompts import load_prompts_config
from src.shared.utils.logging import get_module_logger

logger = get_module_logger(__name__)

class ImageGenerator(BaseGenerator):
    def __init__(self, openai_client=None, output_dir: str = "output/temp/images", max_retries: int = 3):
        super().__init__(max_retries=max_retries, timeout=120.0)

        self.openai_client = openai_client
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.prompts_config = load_prompts_config("chatbot_b")
        self.dalle_tool: Optional[OpenAIDALLEImageGenerationTool] = None
        self._setup_dalle_tool() 

        logger.info(f"ImageGenerator initialize : {output_dir}")
    
    def _setup_dalle_tool(self) -> None:
        try:
            self.dalle_wrapper = DallEAPIWrapper(
                model="dall-e-3",
                size="1024x1024",
                quality="hd",
                n=1
            )

            self.dalle_tool = OpenAIDALLEImageGenerationTool(
                api_wrapper=self.dalle_wrapper
            )

            logger.info("DALL-E Tool initialize")

        except Exception as e:
            logger.error(f"DALL-E Tool initialization failed : {e}")
            self.dalle_tool = None
            self.dalle_wrapper = None
    
    async def generate(self, input_data: Dict[str, Any], progress_callback: Optional[Callable] = None) -> Dict[str, Any]:

        start_time = time.time()
        
        try:
            # Input data
            story_data = input_data.get("story_data", {})
            story_id = input_data.get("story_id", str(uuid.uuid4()))
            target_age = input_data.get("target_age", 7)
            chapters = story_data.get("chapters", [])

            if not chapters:
                return self._empty_result()
            
            # Progress update
            if progress_callback:
                await progress_callback({
                    "step": "image_generation_start",
                    "total_chapters": len(chapters)
                })

            # Image generation for Chapter
            image_results = []
            for i, chapter in enumerate(chapters):
                try:
                    if progress_callback:
                        await progress_callback({
                            "step": "generating_chapter_image",
                            "chapter": i + 1,
                            "total": len(chapters)
                        })

                    image_result = await self._generate_chapter_image(
                        chapter=chapter,
                        story_id=story_id,
                        target_age=target_age,
                        chapter_index=i
                    )

                    image_results.append(image_result)

                    if i < len(chapters) - 1:
                        await asyncio.sleep(2)
                except Exception as e:
                    logger.error(f"Chapter {i+1} Image generation failed : {e}")
                    image_results.append(self._error_result(i+1, str(e)))
        
            generation_time = time.time() - start_time

            return {
                "images": image_results,
                "metadata": {
                    "total_images": len(image_results),
                    "successful_images": len([r for r in image_results if r.get("url")]),
                    "generation_time": round(generation_time, 2),
                    "model_used": "dall-e-3"
                }
            }
        except Exception as e:
            logger.error(f"Image generation failed : {e}")
            raise
    
    async def _generate_chapter_image(self, chapter: Dict[str, Any], story_id: str, target_age: int, chapter_index: int) -> Dict[str, Any]:
        try:
            prompt = self._create_dalle_prompt(chapter, target_age)

            logger.info(f"Chapter {chapter_index + 1} Image generation ...")

            if not self.dalle_tool:
                raise Exception("DALL-E Tool not initialized. Please check OpenAI API Key")
            
            image_url = await asyncio.to_thread(self.dalle_tool.run, prompt)

            image_path = await self._save_image_from_url(
                url=image_url,
                story_id=story_id,
                chapter_number = chapter.get("chapter_number", chapter_index + 1)
            )

            return {
                "chapter_number": chapter.get("chapter_number", chapter_index + 1),
                "url": image_url,
                "image_path": str(image_path),
                "image_prompt": prompt,
                "status": "success"
            }

        except Exception as e:
            logger.error(f"Chapter Image generation failed : {e}")
            return self._error_result(
                chapter.get("chapter_number", chapter_index + 1),
                str(e)
            )
    
    def _create_dalle_prompt(self, chapter: Dict[str, Any], target_age: int) -> str:

        try:
            age_group = "age_4_7" if target_age <= 7 else "age_8_9"

            image_config = self.prompts_config.get("image_generation", {})
            optimization_config = image_config.get("dall_e_3_optimization", {})
            age_template = optimization_config.get(f"{age_group}_template", {})

            if not age_template:
                return self._create_fallback_prompt()
            
            prompt_template = age_template.get("prompt_template", "")
            if not prompt_template:
                return self._create_fallback_prompt()
            
            characters = self._extract_characters(chapter)
            setting = self._extract_setting(chapter)

            final_prompt = prompt_template.replace("{{characters}}", characters)
            final_prompt = final_prompt.replace("{{setting}}", setting)

            safe_prompt = self._apply_safety_filters(final_prompt, optimization_config)

            return safe_prompt
        except Exception as e:
            logger.error(f"Prompt generation failed: {e}")
            return self._create_fallback_prompt()
    
    def _extract_characters(self, chapter: Dict[str, Any]) -> str:
        title = chapter.get("title", "")
        content = chapter.get("content", "")
        text = f"{title} {content}".lower()

        characters = []
        char_keywords = ["토끼", "곰", "새", "고양이", "강아지", "공주", "왕자", "아이", "친구"]

        for keyword in char_keywords:
            if keyword in text:
                characters.append(keyword)
                if len(characters) >= 3:
                    break
        
        return ", ".join(characters) if characters else "cute friendly characters"
    
    def _extract_setting(self, chapter: Dict[str, Any]) -> str:
        title = chapter.get("title", "")
        content = chapter.get("content", "")
        text = f"{title} {content}".lower()

        settings = ["숲", "집", "학교", "공원", "바다", "하늘", "정원", "마을", "유치원"]

        for setting in settings:
            if setting in text:
                return f"in a peaceful {setting}"
        return "in a beautiful peaceful place"
    
    def _apply_safety_filters(self, prompt: str, config: Dict[str, Any]) -> str:
        try:
            safety_config = config.get("safety_filters", {})

            unsafe_keywords = safety_config.get("unsafe_keywords", {})
            for keyword in unsafe_keywords:
                prompt = prompt.replace(keyword, "gentle")
            
            replacements = safety_config.get("positive_replacements", {})
            for old, new in replacements.items():
                prompt = prompt.replace(old, new)
            
            max_length = safety_config.get("max_prompt_length", 400)
            if len(prompt) > max_length:
                prompt = prompt[:max_length-3] + "..."
            
            if "NO text visible" not in prompt:
                prompt += " NO Korean text, NO text visible, child-friendly"
            
            return prompt.strip()
    
        except Exception as e:
            logger.warning(f"safety filter application failed : {e}")
            return prompt + " Child-friendly and safe, NO text visible."
    
    def _create_fallback_prompt(self) -> str:
        return (
            "A gentle, child-friendly watercolor illustration with soft colors "
            "and cute characters in a peaceful setting. Hand-drawn style, "
            "pastel colors, no text visible. Safe for children."
        )

    async def _save_image_from_url(self, url: str, story_id: str, chapter_number: int) -> Path:
        import aiohttp
        import ssl

        story_dir = self.output_dir / story_id[:8]
        story_dir.mkdir(parents=True, exist_ok=True)

        image_path = story_dir / f"chapter_{chapter_number}_{story_id[:8]}.JPEG"

        try:
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE

            timeout = aiohttp.ClientTimeout(total=30)

            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, ssl=ssl_context) as response:
                    if response.status == 200:
                        image_data = await response.read()
                        with open(image_path, 'wb') as f:
                            f.write(image_data)
                        
                        logger.info(f"Image save complete : {image_path}")
                        return image_path
                    else:
                        raise Exception(f"HTTP {response.status}")
        except Exception as e:
            logger.error(f"Image save failed : {e}")
            raise
    
    def _empty_result(self) -> Dict[str, Any]:
        return {
            "images": [],
            "metadata": {
                "total_images": 0,
                "successful_images": 0,
                "generation_time": 0,
                "model_used": "dall-e-3"
            }
        }
    
    def _error_result(self, chapter_number: int, error: str) -> Dict[str, Any]:
        return {
            "chapter_number": chapter_number,
            "url": None,
            "image_path": None,
            "image_prompt": None,
            "status": "error",
            "error": error
        }
    
    async def health_check(self) -> bool:
        try:
            checks = [
                bool(self.prompts_config),
                bool(self.dalle_tool),
                self.output_dir.exists()
            ]
            return all(checks)
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
        



