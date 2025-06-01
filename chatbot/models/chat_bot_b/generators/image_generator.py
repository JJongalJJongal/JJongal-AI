"""
이미지 생성기
LangChain DALL-E 기반 챕터별 이미지 생성
- 연령별 특화 Image Prompt 생성 (4-7세, 8-9세)
- 구조화된 프롬프트 접근법
- 성능 최적화 및 추적
"""

from shared.utils.logging_utils import get_module_logger
import json
import uuid
import base64
from typing import Dict, Any, Optional, Callable, List, Union
from pathlib import Path
import asyncio
import aiohttp
import ssl
import time

# LangChain imports 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper


# Project imports
from .base_generator import BaseGenerator

logger = get_module_logger(__name__)

class ImageGenerator(BaseGenerator):
    """
    LangChain DALL-E 기반 이미지 생성기
    
    Features:
        - 연령별 특화 Image Prompt (4-7세, 8-9세)
        - structured prompt engineering
        - 안전성 검사
        - 성능 추적 및 최적화
        - A/B Test
    """
    
    def __init__(self, 
                 openai_client=None,
                 prompts_file_path: str = "chatbot/data/prompts/chatbot_b_prompts.json",
                 model_name: str = "dall-e-3",
                 image_size: str = "1024x1024",
                 temp_storage_path: str = "/tmp/fairy_tales",
                 max_retries: int = 3,
                 enable_performance_tracking: bool = True):
        """
        Args:
            openai_client: OpenAI 클라이언트
            prompts_file_path: 프롬프트 JSON 파일 경로
            model_name: 이미지 생성 모델명 (dall-e-3)
            image_size: 이미지 크기 ("1024x1024", "1024x1536", "1536x1024")
            temp_storage_path: 임시 저장 경로
            max_retries: 최대 재시도 횟수
            enable_performance_tracking: 성능 추적 활성화
        """
        super().__init__(max_retries=max_retries, timeout=240.0)
        
        # Ensure performance_metrics is initialized, in case BaseGenerator doesn't or it gets overwritten.
        self.performance_metrics: Dict[str, Any] = {
            "generation_times": [],
            "error_count": 0,
            "success_rate": 0.0,
            "age_group_usage": {} 
        }
        
        self.openai_client = openai_client # OpenAI 클라이언트
        self.prompts_file_path = prompts_file_path # 프롬프트 JSON 파일 경로
        self.model_name = model_name # 이미지 생성 모델명 (dall-e-3)
        self.image_size = image_size # 이미지 크기 ("1024x1024", "1024x1536", "1536x1024")
        self.temp_storage_path = Path(temp_storage_path) # 임시 저장 경로
        self.enable_performance_tracking = enable_performance_tracking # 성능 추적 활성화
        
        # 프롬프트 템플릿
        self.prompts = None # 프롬프트 템플릿
        self.image_prompt_template = None # 이미지 프롬프트 템플릿
        
        # LangChain 구성 요소
        self.dalle_wrapper = None # DALL-E Wrapper
        self.prompt_enhancer_chain: Dict[str, Any] = {} # 프롬프트 개선 체인 (연령별)
        self.llm = None # LLM
        
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
            image_generation_templates = self.prompts.get("image_generation", {}) # 이미지 프롬프트 템플릿 추출
            dall_e_optimization = image_generation_templates.get("dall_e_3_optimization", {}) # 이미지 프롬프트 템플릿 추출
            
            # 연령별 템플릿 설정
            self.age_specific_templates = {
                "age_4_7": dall_e_optimization.get("age_4_7_template", {}),
                "age_8_9": dall_e_optimization.get("age_8_9_template", {})
            }
            
            logger.info(f"프롬프트 파일 로드 완료: {self.prompts_file_path}") # 프롬프트 파일 로드 완료 로깅
            
        except Exception as e:
            logger.error(f"프롬프트 파일 로드 실패: {e}") # 프롬프트 파일 로드 실패 로깅
            
            # 기본 Template 설정
            self._set_fallback_templates()
    
    def _set_fallback_templates(self):
        """기본 템플릿 설정 (Prompt Load 실패할 경우)"""
        self.age_specific_templates = {
            "age_4_7": {
                "style_specifications": {
                    "art_style": "Adorable kawaii-style illustration with extremely soft, rounded shapes and cute characters perfect for young children",
                    "color_palette": "Soft pastel tones: gentle mint green, cream white, warm peach, soft sky blue, light lavender, warm yellow",
                    "composition": "Simple, clean layouts with cute focal points and large, friendly round character designs",
                    "character_design": "Round, chubby characters with large sparkly eyes, soft features, and gentle expressions",
                    "safety_elements": "No sharp edges, dark shadows, or potentially frightening elements - only cute and comforting imagery"
                },
                "prompt_template": "Create an adorable kawaii-style illustration for a Korean children's storybook. Scene: {scene_description}. Style: Soft, cute cartoon illustration with rounded shapes and gentle gradients. Character design: Extremely cute animals or characters with big round eyes, chubby cheeks, soft pastel colors. Environment: {safe_environment_description} with rolling hills, soft trees, and peaceful nature. Mood: {mood}. Technical: Smooth gradients, no harsh lines, child-friendly and heartwarming atmosphere like a gentle storybook illustration."
            },
            "age_8_9": {
                "style_specifications": {
                    "art_style": "Charming kawaii illustration with more detailed but still soft, appealing character designs",
                    "color_palette": "Rich but gentle pastels with harmonious color relationships and warm undertones",
                    "composition": "More sophisticated layouts with multiple cute elements and gentle depth layers",
                    "character_design": "Detailed kawaii characters with expressive large eyes, soft clothing, and endearing poses",
                    "symbolic_elements": "Age-appropriate cute symbols and gentle metaphorical visual elements"
                },
                "prompt_template": "Create a charming kawaii-style illustration for a Korean children's storybook (ages 8-9). Scene: {scene_description}. Style: Professional cute cartoon illustration with soft gradients and detailed kawaii elements. Character details: Adorable characters with distinctive cute clothing, gentle emotions, and round features. Environment: Rich textures with cute details, peaceful backgrounds with soft hills and friendly nature elements. Mood: {complex_mood}. Technical quality: High-quality cute illustration with soft lighting and warm, comforting atmosphere."
            }
        }
                
    def _setup_langchain_components(self):
        """LangChain 구성 요소 설정"""
        try:
            # 1. LLM 초기화
            self.llm = ChatOpenAI(
                temperature=1.0, # 창의성 조절 (0.0 ~ 1.0)
                model="gpt-4o",
                api_key=self.openai_client.api_key # OpenAI API Key
            )
            
            # 2. DALL-E Wrapper 초기화
            self.dalle_wrapper = DallEAPIWrapper(
                model=self.model_name, # dall-e-3
                size=self.image_size, # 1024x1024, 1024x1536, 1536x1024
                quality="hd", # hd, standard
                style="natural", # vivid, natural
                response_format="url", # url, b64_json
                n=1 # 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
            )
            
            # 3. 연령별 프롬프트 개선 체인 설정
            self._setup_age_specific_chains()
            
            logger.info("LangChain 구성 요소 설정 완료")
            
        except Exception as e:
            logger.error(f"LangChain 구성 요소 설정 실패: {e}")
            raise
    
    def _setup_age_specific_chains(self):
        """연령별 프롬프트 개선 체인 설정"""
        
        for age_group, template_config in self.age_specific_templates.items():
            try:
                # Structured Prompt
                system_template = self._build_prompt_template(age_group, template_config)
                
                prompt_template = ChatPromptTemplate.from_template(system_template)
                self.prompt_enhancer_chain[age_group] = prompt_template | self.llm | StrOutputParser()
                
                logger.info(f"연령별 이미지 체인 생성 완료 : {age_group}")
                
            except Exception as e:
                logger.error(f"연령별 이미지 체인 생성 실패 ({age_group}): {e}")
                raise
    
    def _build_prompt_template(self, age_group: str, template_config: Dict[str, Any]) -> str:
        """구조화된 프롬프트 템플릿 생성"""
        
        style_specs = template_config.get("style_specifications", {})
        
        prompt_parts = [
            "## ROLE",
            f"You are a professional children's book illustrator specializing in {age_group.replace('_', '-')} year old content.",
            "",
            "## OBJECTIVE", 
            "Create a detailed DALL-E 3 prompt that will generate beautiful, age-appropriate watercolor illustrations for a Korean children's storybook.",
            "",
            "## STYLE REQUIREMENTS"
        ]

        # 스타일 명세 추가
        for spec_key, spec_value in style_specs.items():
            prompt_parts.append(f"- **{spec_key.replace('_', ' ').title()}**: {spec_value}")
        
        prompt_parts.extend([
            "",
            "## SAFETY GUIDELINES",
            "- Ensure all content is 100% child-safe and positive",
            "- No violent, scary, or inappropriate elements",
            "- Use warm, comforting colors and expressions",
            "- Focus on friendship, learning, and positive emotions",
            "",
            "## INPUT PROCESSING",
            "Based on the following information, create an optimized DALL-E 3 prompt:",
            "",
            "Chapter Title: {chapter_title}",
            "Chapter Content: {chapter_content}",
            "Story Theme: {theme}",
            "Main Characters: {characters}",
            "Setting: {setting}",
            "Target Age: {target_age}",
            "Mood: {mood}",
            "",
            "## OUTPUT FORMAT",
            "Provide a single, well-crafted DALL-E 3 prompt (maximum 400 characters) that incorporates all style requirements and safety guidelines."
        ])
        
        return "\n".join(prompt_parts)
    
    def _determine_age_group(self, target_age: Union[int, str]) -> str:
        """연령대에 따른 체인 선택"""
        
        current_age = target_age
        if isinstance(target_age, str):
            # "age_4_7" 형태의 문자열에서 나이 추출 시도 (가운데 숫자 사용 또는 기본값)
            try:
                # "4-7" 부분 추출 후 첫번째 숫자 사용
                age_range_str = target_age.split('_')[-1]
                current_age = int(age_range_str.split('-')[0]) 
            except:
                logger.warning(f"target_age 문자열 '{target_age}'에서 나이 추출 실패, 기본값 7 사용")
                current_age = 7 # 기본값

        if 4 <= current_age <= 7:
            return "age_4_7"
        elif 8 <= current_age <= 9:
            return "age_8_9"
        else:
            # 범위를 벗어날 경우, 더 어린 연령대 또는 더 높은 연령대 중 가까운 쪽으로
            logger.warning(f"대상 연령({current_age})이 표준 범위를 벗어났습니다.")
            return "age_4_7" if current_age < 8 else "age_8_9"
        
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
                    ],
                    "metadata": {"age_group": "연령대"}
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
        
        start_time = time.time()
        task_id = str(uuid.uuid4()) # 임시 Task ID
        self.current_task_id = task_id # 현재 작업 ID 저장
        
        try:
            story_data = input_data.get("story_data", {}) # 스토리 데이터
            story_id = input_data.get("story_id", task_id) # 스토리 ID
            chapters = story_data.get("chapters", []) # 챕터 목록
            
            # 연령대 결정
            metadata = story_data.get("metadata", {})
            # target_age를 우선적으로 사용하고, 없으면 age_group에서 추출 시도, 최종적으로 기본값 7 setting
            raw_target_age = metadata.get("target_age", metadata.get("age_group"))
            
            if raw_target_age is None:
                logger.warning("target_age 및 age_group 정보 없음, 기본값 7세 사용")
                final_target_age = 7
            elif isinstance(raw_target_age, int):
                final_target_age = raw_target_age
            elif isinstance(raw_target_age, str):
                # age_group 문자열 (e.g., "age_4_7") 또는 숫자 문자열 처리
                try:
                    final_target_age = int(raw_target_age) # "7" 같은 경우
                except ValueError:
                    # age_4_7 같은 경우
                    try:
                        age_range_str = raw_target_age.split('_')[-1] # "4-7" 부분 추출
                        final_target_age = int(age_range_str.split('-')[0]) # 첫번째 숫자 사용
                        logger.info(f"age_group '{raw_target_age}'에서 target_age: {final_target_age} 추출")
                    except:
                        logger.warning(f"age_group 문자열 '{raw_target_age}'에서 나이 추출 실패, 기본값 7 사용")
                        final_target_age = 7
            else:
                logger.warning(f"알 수 없는 target_age 형식: {raw_target_age}, 기본값 7세 사용")
                final_target_age = 7
                
            age_group_key = self._determine_age_group(final_target_age)
            
            # 진행 상황 업데이트
            if progress_callback:
                await progress_callback({
                    "step": "image_generation",
                    "status": "starting",
                    "total_chapters": len(chapters),
                    "age_group": age_group_key
                })
            
            # Image Batch Generation
            image_results = await self._generate_batch(
                chapters=chapters,
                story_data=story_data,
                story_id=story_id,
                age_group=age_group_key,
                progress_callback=progress_callback
            )
            
            # 성능 Metric 수집
            generation_time = time.time() - start_time
            self._update_performance_metrics(generation_time, True, age_group_key, len(image_results))
            
            # 최종 결과 구성
            result = {
                "images": image_results, # 생성된 이미지 목록
                "metadata": {
                    "total_images": len(image_results), # 생성된 이미지 수
                    "successful_images": len([img for img in image_results if img.get("image_path")]), # 성공한 이미지 수
                    "model_used": self.model_name, # 사용된 모델
                    "total_generation_time": generation_time, # 총 생성 시간
                    "age_group": age_group_key, # 연령대
                    "average_time_per_image": generation_time / len(chapters) if chapters else 0 # 평균 생성 시간
                }
            }
            
            # 진행 상황 업데이트
            if progress_callback:
                await progress_callback({
                    "step": "image_generation", # 작업 단계
                    "status": "completed", # 상태
                    "total_images": len(image_results), # 생성된 이미지 수
                    "generation_time": generation_time # 생성 시간
                })
            
            return result
            
        except Exception as e:
            self._update_performance_metrics(0, False, "UnKnown", 0) # 성공 여부, 연령대, 이미지 수
            logger.error(f"이미지 생성 실패: {e}")
            raise
            
    async def _generate_batch(self,
                              chapters: List[Dict[str, Any]],
                              story_data: Dict[str, Any],
                              story_id: str,
                              age_group: str,
                              progress_callback: Optional[Callable] = None) -> List[Dict[str, Any]]:
        """이미지 배치 생성"""
        
        image_results = [] # 생성된 이미지 목록
        chain = self.prompt_enhancer_chain.get(age_group) # 연령별 특화 체인
        
        if not chain:
            logger.error(f"연령별 체인을 찾을 수 없음: {age_group}")
            return []
        for i, chapter in enumerate(chapters):
            try:
                # 진행 상황 업데이트
                if progress_callback:
                    await progress_callback({
                        "step": "chapter_image_generation",
                        "status": "processing",
                        "current_chapter": i + 1,
                        "total_chapters": len(chapters),
                        "chapter_title": chapter.get("chapter_title", f"Chapter {i+1}")
                    })
                
                # Enhanced 프롬프트 생성
                enhanced_prompt = await self._create_enhanced_image_prompt(
                    chapter=chapter,
                    story_data=story_data,
                    age_group=age_group,
                    chain=chain
                )
                
                # 이미지 생성
                image_result = await self._generate_single_enhanced_image(
                    prompt=enhanced_prompt,
                    chapter_number=chapter.get("chapter_number", i + 1),
                    story_id=story_id,
                    age_group=age_group
                )
                
                image_results.append(image_result)
                
                # API 레이트 리밋 고려 (15초 대기)
                if i < len(chapters) - 1:  # 마지막 이미지가 아니면
                    logger.info(f"다음 이미지 생성을 위해 15초 대기...")
                    await asyncio.sleep(10) # 10초 대기
                
            except Exception as e:
                logger.error(f"챕터 {i+1} 이미지 생성 실패: {e}")
                image_results.append({
                    "chapter_number": chapter.get("chapter_number", i + 1),
                    "image_path": None,
                    "image_prompt": None,
                    "status": "error",
                    "error": str(e),
                    "age_group": age_group
                })
        
        return image_results
                
    
    async def _create_enhanced_image_prompt(self,
                                            chapter: Dict[str, Any],
                                            story_data: Dict[str, Any],
                                            age_group: str,
                                            chain) -> str:
        
        """LangChain을 사용한 향상된 이미지 프롬프트 생성"""
        
        try:
            # 연령별 특화 프롬프트 템플릿 추출
            age_specific_config = self.age_specific_templates.get(age_group, {}) 
            user_defined_dalle_template = age_specific_config.get("prompt_template") # 연령별 특화 프롬프트 템플릿

            if not user_defined_dalle_template: # 만약 연령별 특화 프롬프트 템플릿이 없으면
                logger.error(f"User-defined DALL-E template not found for age group {age_group}. Falling back to basic prompt.") # 기본 프롬프트 생성
                return self._create_fallback_prompt(chapter, age_group) # 기본 프롬프트 생성

            # 프롬프트 템플릿 데이터 준비
            characters_value = self._extract_characters(story_data)
            setting_value = self._extract_setting(chapter)
            
            # 프롬프트 템플릿 데이터 치환
            final_dalle_prompt = user_defined_dalle_template.replace("{{characters}}", characters_value)
            final_dalle_prompt = final_dalle_prompt.replace("{{setting}}", setting_value)
            
            # 안전성 필터 적용 및 길이 제한
            safe_prompt = self._apply_safety_filters(final_dalle_prompt)
            
            logger.info(f"Using user-defined DALL-E template for age group {age_group}.") # 연령별 특화 프롬프트 사용
            logger.debug(f"Formatted DALL-E prompt: {safe_prompt}") # 포맷팅된 DALL-E 프롬프트
            
            return safe_prompt
            
        except Exception as e:
            logger.error(f"Failed to create image prompt from user-defined template: {e}") # 프롬프트 생성 실패
            return self._create_fallback_prompt(chapter, age_group) # 기본 프롬프트 생성
    
    def _extract_characters(self, story_data: Dict[str, Any]) -> str:
        """스토리에서 주요 캐릭터 추출"""
        characters = []
        
        # 챕터들에서 캐릭터 이름 추출
        chapters = story_data.get("chapters", [])
        for chapter in chapters[:2]:  # 첫 2개 챕터만 확인
            content = chapter.get("chapter_content", "")
            # 한국어 이름 패턴 추출 (간단한 패턴)
            import re
            names = re.findall(r'[가-힣]{2,4}(?=은|는|이|가|을|를|와|과|에게|한테)', content)
            characters.extend(names[:2])  # 최대 2개
        
        return ", ".join(list(set(characters))[:3]) if characters else "friendly characters" # 최대 3개의 캐릭터 반환
    
    def _extract_setting(self, chapter: Dict[str, Any]) -> str:
        """챕터에서 배경 추출"""
        content = chapter.get("chapter_content", "") # 챕터 내용
        
        # 장소 관련 키워드 검색
        location_keywords = ["숲", "집", "학교", "공원", "바다", "산", "마을", "도시", "방", "정원"]
        for keyword in location_keywords:
            if keyword in content:
                return f"{keyword}에서" # 키워드가 있으면 키워드 반환
        
        return "평화로운 장소에서" # 키워드가 없으면 기본값 반환
    
    def _determine_mood(self, chapter: Dict[str, Any]) -> str:
        """챕터의 분위기 결정"""
        content = chapter.get("chapter_content", "").lower()
        
        # 감정 키워드 매핑
        mood_mapping = {
            "기쁨": ["기쁘", "즐거", "행복", "웃", "놀이"],
            "모험": ["모험", "탐험", "여행", "발견"],
            "평화": ["평화", "조용", "안전", "편안"],
            "우정": ["친구", "도움", "함께", "협력"],
            "학습": ["배우", "공부", "알아", "깨달"]
        }
        
        for mood, keywords in mood_mapping.items():
            if any(keyword in content for keyword in keywords):
                return mood # 키워드가 있으면 분위기 반환
        
        return "따뜻하고 평화로운" # 키워드가 없으면 기본값 반환
    
    def _create_fallback_prompt(self, chapter: Dict[str, Any], age_group: str) -> str:
        """기본 프롬프트 생성 (오류 시 사용)"""
        template = self.age_specific_templates.get(age_group, {})
        basic_template = template.get("prompt_template", "A gentle watercolor illustration for children showing {scene}. Child-friendly, warm colors, safe environment.")
        
        scene = chapter.get("chapter_title", "a peaceful scene")
        return basic_template.format(scene_description=scene, mood="peaceful", safe_environment_description="safe and welcoming")

    def _apply_safety_filters(self, prompt: str) -> str:
        """안전성 필터 적용 및 프롬프트 정리"""
        try:
            # 1. 기본적인 정리
            safe_prompt = prompt.strip()
            
            # 2. 부적절한 키워드 필터링 (아동용 안전성)
            unsafe_keywords = [
                "violence", "violent", "scary", "frightening", "dark", "evil", 
                "weapon", "gun", "knife", "blood", "death", "kill", "hurt",
                "폭력", "무서운", "어두운", "악한", "무기", "총", "칼", "피", "죽음", "죽이"
            ]
            
            for keyword in unsafe_keywords:
                safe_prompt = safe_prompt.replace(keyword, "gentle")
            
            # 3. 긍정적 키워드로 대체
            replacements = {
                "fight": "play together",
                "attack": "approach kindly", 
                "destroy": "transform",
                "싸우": "함께 놀다",
                "공격": "친근하게 다가가다",
                "파괴": "변화시키다"
            }
            
            for old, new in replacements.items():
                safe_prompt = safe_prompt.replace(old, new)
            
            # 4. 길이 제한 (DALL-E 3 권장사항: 400자 이하)
            if len(safe_prompt) > 400:
                safe_prompt = safe_prompt[:397] + "..."
            
            # 5. 아동 안전성 강조
            if "child-friendly" not in safe_prompt.lower() and "kawaii" not in safe_prompt.lower():
                safe_prompt += " Child-friendly and safe."
            
            return safe_prompt
            
        except Exception as e:
            logger.warning(f"안전성 필터 적용 중 오류: {e}")
            # 오류 시 기본 안전 프롬프트 반환
            return "A gentle, child-friendly kawaii illustration with soft colors and cute characters."

    async def _generate_single_enhanced_image(self,
                                            prompt: str,
                                            chapter_number: int,
                                            story_id: str,
                                            age_group: str) -> Dict[str, Any]:
        """Enhanced 단일 이미지 생성"""
        
        start_time = time.time()
        
        try:
            logger.info(f"[Enhanced 이미지 생성 시작] 챕터 {chapter_number}")
            logger.info(f"Age Group: {age_group}")
            logger.info(f"프롬프트: {prompt[:100]}...")
            
            # LangChain DALL-E Wrapper 사용 시도
            try:
                image_url = self.dalle_wrapper.run(prompt) # 이미지 URL 생성
                logger.info(f"LangChain 이미지 URL 생성 성공: {image_url[:100]}...")
                
                # 이미지 다운로드
                image_path = await self._download_image_from_url( 
                    image_url=image_url, # 이미지 URL
                    chapter_number=chapter_number, # 챕터 번호
                    story_id=story_id # 스토리 ID
                )
                
                generation_time = time.time() - start_time # 생성 시간
                
                result = {
                    "chapter_number": chapter_number, # 챕터 번호
                    "image_path": str(image_path), # 이미지 경로
                    "image_prompt": prompt, # 이미지 프롬프트
                    "generation_time": generation_time, # 생성 시간
                    "age_group": age_group, # 연령대
                    "method": "langchain_dalle_wrapper", # 메서드
                    "status": "success" # 상태
                }
                
                logger.info(f"이미지 생성 완료: {image_path}") # 이미지 생성 완료
                return result # 결과 반환
                
            except Exception as e:
                logger.warning(f"LangChain DALL-E 실패, 직접 API 시도: {e}") # 직접 API 시도
                return await self._generate_with_direct_api_enhanced(prompt, chapter_number, story_id, age_group) # 직접 API 시도
                
        except Exception as e:
            logger.error(f"Enhanced 이미지 생성 완전 실패 (챕터 {chapter_number}): {e}")
            return {
                "chapter_number": chapter_number, # 챕터 번호
                "image_path": None, # 이미지 경로
                "image_prompt": prompt, # 이미지 프롬프트
                "status": "error", # 상태
                "error": str(e), # 에러
                "age_group": age_group # 연령대
            }

    async def _generate_with_direct_api_enhanced(self,
                                               prompt: str,
                                               chapter_number: int,
                                               story_id: str,
                                               age_group: str) -> Dict[str, Any]:
        """Enhanced 직접 API 이미지 생성"""
        
        try:
            logger.info(f"Enhanced 직접 API 방식 사용 (챕터 {chapter_number})") # 직접 API 방식 사용
            
            # OpenAI API 직접 호출
            response = self.openai_client.images.generate(
                model=self.model_name, # 모델 이름
                prompt=prompt, # 이미지 프롬프트
                size=self.image_size, # 이미지 크기
                n=1, # 이미지 수
                response_format="b64_json"
            )
            
            # Base64 이미지 데이터 처리
            image_data = response.data[0].b64_json
            image_path = self.temp_storage_path / f"enhanced_chapter_{chapter_number}_{story_id[:8]}.png" # 이미지 경로
            
            # 이미지 저장
            with open(image_path, "wb") as f:
                f.write(base64.b64decode(image_data)) # 이미지 데이터 저장
            
            logger.info(f"직접 API 이미지 생성 완료: {image_path}") # 이미지 생성 완료
            
            return {
                "chapter_number": chapter_number, # 챕터 번호
                "image_path": str(image_path), # 이미지 경로
                "image_prompt": prompt, # 이미지 프롬프트
                "method": "direct_api_enhanced", # 메서드
                "status": "success", # 상태
                "age_group": age_group # 연령대
            }
            
        except Exception as e:
            logger.error(f"직접 API 실패 (챕터 {chapter_number}): {e}") # 직접 API 실패
            raise

    async def _download_image_from_url(self,
                                     image_url: str,
                                     chapter_number: int,
                                     story_id: str) -> Path:
        """URL에서 이미지 다운로드 (Enhanced)"""
        
        image_path = self.temp_storage_path / f"enhanced_chapter_{chapter_number}_{story_id[:8]}.png" # 이미지 경로
        
        try:
            # SSL 컨텍스트 설정
            ssl_context = ssl.create_default_context() # SSL 컨텍스트 생성
            ssl_context.check_hostname = False # 호스트 이름 검증 비활성화
            ssl_context.verify_mode = ssl.CERT_NONE # 인증서 검증 비활성화
            
            timeout = aiohttp.ClientTimeout(total=60) # 타임아웃 설정
            
            async with aiohttp.ClientSession(timeout=timeout) as session: # 클라이언트 세션 생성
                async with session.get(image_url, ssl=ssl_context) as response: # 이미지 URL 요청
                    if response.status == 200:
                        image_data = await response.read() # 이미지 데이터 읽기
                        with open(image_path, 'wb') as f: # 이미지 경로에 저장
                            f.write(image_data) # 이미지 데이터 저장
                        
                        logger.info(f"이미지 다운로드 완료: {image_path}") # 이미지 다운로드 완료
                        return image_path # 이미지 경로 반환
                    else:
                        raise Exception(f"HTTP {response.status}: {await response.text()}") # 에러 발생
                        
        except Exception as e:
            logger.error(f"이미지 다운로드 실패: {e}") # 이미지 다운로드 실패
            raise

    def _update_performance_metrics(self, generation_time: float, success: bool, age_group: str, image_count: int):
        """성능 메트릭 업데이트"""
        if not self.enable_performance_tracking: # 성능 추적 비활성화 시
            return # 성능 추적 비활성화 시
            
        if success:
            self.performance_metrics["generation_times"].append(generation_time) # 생성 시간 추가
            
            # 연령대별 사용량 추적
            if age_group not in self.performance_metrics["age_group_usage"]: # 연령대별 사용량 추적
                self.performance_metrics["age_group_usage"][age_group] = 0 # 연령대별 사용량 초기화
            self.performance_metrics["age_group_usage"][age_group] += image_count # 연령대별 사용량 추가
            
            # 성공률 계산
            total_attempts = len(self.performance_metrics["generation_times"]) + self.performance_metrics["error_count"]
            self.performance_metrics["success_rate"] = len(self.performance_metrics["generation_times"]) / total_attempts
        else:
            self.performance_metrics["error_count"] += 1 # 에러 발생 시 에러 카운트 증가

    def get_performance_metrics(self) -> Dict[str, Any]: # 성능 메트릭 조회
        """성능 메트릭 조회 (Enhanced)"""
        if not self.performance_metrics["generation_times"]: # 생성 시간이 없으면
            return self.performance_metrics # 성능 메트릭 반환
            
        times = self.performance_metrics["generation_times"] # 생성 시간
        return {
            **self.performance_metrics,
            "avg_generation_time": sum(times) / len(times), # 평균 생성 시간
            "min_generation_time": min(times), # 최소 생성 시간
            "max_generation_time": max(times), # 최대 생성 시간
            "total_generations": len(times), # 총 생성 횟수
            "most_used_age_group": max(self.performance_metrics["age_group_usage"], 
                                     key=self.performance_metrics["age_group_usage"].get, 
                                     default="unknown") if self.performance_metrics["age_group_usage"] else "unknown"
        }

    async def health_check(self) -> Dict[str, bool]:
        """Enhanced 상태 확인"""
        health_status = {
            "enhanced_prompts_loaded": bool(self.prompts),
            "dalle_wrapper_ready": bool(self.dalle_wrapper), # DALL-E 랩퍼 준비 여부
            "age_4_7_chain_ready": "age_4_7" in self.prompt_enhancer_chain, # 4-7세 체인 준비 여부
            "age_8_9_chain_ready": "age_8_9" in self.prompt_enhancer_chain, # 8-9세 체인 준비 여부
            "temp_storage_accessible": self.temp_storage_path.exists(), # 임시 저장소 접근 가능 여부
            "performance_tracking": self.enable_performance_tracking # 성능 추적 여부
        }
        
        # 전체 상태
        health_status["overall_healthy"] = all(health_status.values())
        
        return health_status

    def get_supported_sizes(self) -> List[str]:
        """지원되는 이미지 크기 목록"""
        return ["1024x1024", "1024x1536", "1536x1024"]

    def estimate_generation_time(self, chapter_count: int) -> float:
        """예상 생성 시간 계산 (Enhanced)"""
        base_time_per_image = 20.0  # 기본 20초
        wait_time_between = 15.0    # 이미지 간 대기 시간
        
        if self.performance_metrics["generation_times"]:
            avg_time = sum(self.performance_metrics["generation_times"]) / len(self.performance_metrics["generation_times"])
            base_time_per_image = avg_time / max(1, len(self.performance_metrics["generation_times"]))
        
        total_time = (base_time_per_image * chapter_count) + (wait_time_between * max(0, chapter_count - 1))
        return total_time 
    
    