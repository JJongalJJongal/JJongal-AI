"""
동화 줄거리를 바탕으로 일러스트와 내레이션을 생성하는 AI 챗봇 모듈
"""
from typing import List, Dict, Optional, Tuple, Any, Union
from pathlib import Path

# 자체 개발 모듈 임포트
from ..rag_system import RAGSystem
from .content_generator import ContentGenerator
from .story_parser import StoryParser
from .media_manager import MediaManager
from .data_persistence import DataPersistence

# 공통 유틸리티 모듈 임포트
from shared.utils.logging_utils import get_module_logger
from shared.utils.openai_utils import initialize_client
from shared.utils.file_utils import ensure_directory
from shared.utils.audio_utils import initialize_elevenlabs
from shared.configs.prompts_config import load_chatbot_b_prompts
from shared.configs.app_config import get_env_vars, get_project_root

# 로거 설정
logger = get_module_logger(__name__)

class StoryGenerationChatBot:
    """
    동화 줄거리를 바탕으로 일러스트와 내레이션을 생성하는 AI 챗봇 클래스
    
    Attributes:
        story_outline (Dict[str, str]): 동화 줄거리 정보
        generated_images (List[str]): 생성된 이미지 파일 경로 목록
        narration_audio (str): 생성된 내레이션 오디오 파일 경로
        prompts (Dict): JSON 파일에서 로드한 프롬프트
        output_dir (Path): 생성된 파일들을 저장할 디렉토리
        voice_id (str): ElevenLabs 음성 ID
        rag_system (RAGSystem): LangChain 기반 RAG 시스템
    """
    
    def __init__(self, output_dir: str = "output"):
        """
        챗봇 초기화 및 기본 속성 설정
        
        Args:
            output_dir (str): 생성된 파일들을 저장할 디렉토리 경로
        """
        # 환경 변수 로드
        env_vars = get_env_vars()
        output_path = env_vars.get("output_dir", output_dir)
        
        self.story_outline = None
        self.generated_images = []
        self.narration_audio = None
        self.output_dir = Path(output_path)
        self.voice_id = None
        self.target_age = None
        self.detailed_story = None
        
        # 음성 클론 관련 속성
        self.child_voice_id = None  # 아이의 클론된 음성 ID
        self.main_character_name = None  # 주인공 캐릭터 이름
        self.has_cloned_voice = False  # 음성 클론 존재 여부
        
        # 출력 디렉토리 생성
        ensure_directory(self.output_dir)
        ensure_directory(self.output_dir / "images")
        ensure_directory(self.output_dir / "audio")
        
        # 프롬프트 로드
        self.prompts = load_chatbot_b_prompts()
        
        # OpenAI 클라이언트 초기화
        try:
            self.openai_client = initialize_client()
        except Exception as e:
            logger.error(f"OpenAI 클라이언트 초기화 실패: {e}")
            self.openai_client = None
            
        # ElevenLabs 클라이언트 초기화
        self.elevenlabs_client = initialize_elevenlabs()
        
        # RAG 시스템 초기화
        try:
            self.rag_system = RAGSystem()
        except Exception as e:
            logger.error(f"RAG 시스템 초기화 중 오류 발생: {e}")
            self.rag_system = None
            
        # 모듈화된 컴포넌트 초기화
        self.content_generator = ContentGenerator(
            self.openai_client, 
            self.elevenlabs_client,
            self.output_dir / "images",
            self.output_dir / "audio",
            child_voice_id=self.child_voice_id,
            main_character_name=self.main_character_name
        )
        self.story_parser = StoryParser()
        self.media_manager = MediaManager(self.output_dir)
        self.data_persistence = DataPersistence(self.output_dir)
    
    def _update_content_generator_voice_info(self):
        """Helper to update voice info in ContentGenerator."""
        if hasattr(self.content_generator, 'child_voice_id'):
            self.content_generator.child_voice_id = self.child_voice_id
        if hasattr(self.content_generator, 'main_character_name'):
            self.content_generator.main_character_name = self.main_character_name

    def set_cloned_voice_info(self, child_voice_id: str, main_character_name: Optional[str] = None):
        """
        클론된 음성 정보 및 주인공 이름을 설정합니다.

        Args:
            child_voice_id (str): ElevenLabs에서 클론된 아이의 음성 ID.
            main_character_name (Optional[str]): 주인공의 이름. None이면 story_outline에서 첫 번째 캐릭터로 추론.
        """
        self.child_voice_id = child_voice_id
        self.has_cloned_voice = bool(child_voice_id)
        logger.info(f"아이 클론 음성 ID 설정: {self.child_voice_id}")

        if main_character_name:
            self.main_character_name = main_character_name
            logger.info(f"주인공 이름 명시적 설정: {self.main_character_name}")
        elif not self.main_character_name and self.story_outline and self.story_outline.get("characters"):
            # story_outline이 있고, main_character_name이 아직 설정되지 않았다면 첫번째 캐릭터로 설정
            self.main_character_name = self.story_outline["characters"][0]
            logger.info(f"주인공 이름 추론 설정 (story_outline): {self.main_character_name}")
        
        self._update_content_generator_voice_info()

    def set_story_outline(self, story_outline: Dict[str, str]):
        """
        동화 줄거리 정보를 설정하는 함수
        
        Args:
            story_outline (Dict[str, str]): 동화 줄거리 정보
                - theme: 주제
                - characters: 주요 캐릭터 (첫 번째 캐릭터가 주인공으로 간주될 수 있음)
                - setting: 배경 설정
                - plot_summary: 간략한 줄거리
                - educational_value: 교육적 가치
                - target_age: 적합한 연령대
        """
        self.story_outline = story_outline
        logger.info(f"동화 줄거리 설정 완료: {self.story_outline.get('theme', '미지정')}")
        
        # 주인공 이름 설정 (만약 명시적으로 설정되지 않았거나, set_cloned_voice_info에서 설정되지 않았다면)
        if not self.main_character_name and self.story_outline and "characters" in self.story_outline and self.story_outline["characters"]:
            self.main_character_name = self.story_outline["characters"][0]
            logger.info(f"주인공 이름 설정 (set_story_outline): {self.main_character_name}")
            self._update_content_generator_voice_info() # Update content_generator
            
    def set_target_age(self, age: int):
        """
        대상 연령을 설정하는 함수
        
        Args:
            age (int): 아이의 연령
        """
        self.target_age = age
        
        # 스토리 아웃라인에도 추가
        if self.story_outline:
            self.story_outline["target_age"] = age
            
        logger.info(f"대상 연령 설정: {age}세")
    
    def generate_image(self, scene_description: str, style: str = "아기자기한 동화 스타일") -> str:
        """
        장면 설명을 바탕으로 일러스트를 생성하는 함수
        
        Args:
            scene_description (str): 장면 설명
            style (str): 일러스트 스타일
            
        Returns:
            str: 생성된 이미지 파일 경로
        """
        if not self.openai_client:
            logger.error("OpenAI 클라이언트가 초기화되지 않았습니다.")
            return None
            
        try:
            # 이미지 생성 프롬프트 템플릿 가져오기
            prompt_template = self.prompts.get(
                "image_generation_prompt_template", 
                "아이들을 위한 동화책 일러스트레이션. 테마: {theme}, 등장인물: {characters}, 배경: {setting}, 스타일: {style}."
            )
            
            # 프롬프트 포맷팅
            theme = self.story_outline.get("theme", "동화")
            characters = self.story_outline.get("characters", "아이")
            setting = self.story_outline.get("setting", "숲속")
            
            prompt = prompt_template.format(
                theme=theme,
                characters=characters,
                setting=setting,
                style=style
            )
            
            # 장면 설명 추가
            prompt = f"{prompt}\n\n장면 설명: {scene_description}"
            
            # 스토리 ID 설정 (첫 이미지 생성 시)
            if not self.content_generator.current_story_id:
                story_id = self.data_persistence.generate_story_id()
                self.content_generator.current_story_id = story_id
                self.media_manager.set_current_story_id(story_id)
            
            # ContentGenerator를 사용하여 이미지 생성
            chapter_data = {
                "chapter_number": len(self.generated_images) + 1,
                "title": f"Scene {len(self.generated_images) + 1}",
                "narration": scene_description
            }
            
            story_info = {
                "characters": self.story_outline.get("characters", []),
                "setting": self.story_outline.get("setting", ""),
                "age_group": self.target_age or 5
            }
            
            image_info = self.content_generator.generate_image_for_chapter(chapter_data, story_info)
            
            if image_info and "file_path" in image_info:
                file_path = image_info["file_path"]
                self.generated_images.append(file_path)
                logger.info(f"이미지 생성 완료: {file_path}")
                return file_path
            else:
                logger.error("이미지 생성 실패")
                return None
            
        except Exception as e:
            logger.error(f"이미지 생성 중 오류 발생: {e}")
            return None
    
    async def generate_narration(self, text: str, file_name: str = None) -> str:
        """
        텍스트를 바탕으로 내레이션 오디오를 생성하는 함수
        
        Args:
            text (str): 내레이션 텍스트
            file_name (str, optional): 저장할 파일 이름 (없으면 자동 생성)
            
        Returns:
            str: 생성된 내레이션 오디오 파일 경로
        """
        if not self.elevenlabs_client:
            logger.error("ElevenLabs 클라이언트가 초기화되지 않았습니다.")
            return None
            
        try:
            # 스토리 ID 설정 (첫 오디오 생성 시)
            if not self.content_generator.current_story_id:
                story_id = self.data_persistence.generate_story_id()
                self.content_generator.current_story_id = story_id
                self.media_manager.set_current_story_id(story_id)
            
            # ContentGenerator를 사용하여 오디오 생성
            audio_path = self.content_generator.generate_audio_for_text(
                text, 
                "narrator", 
                file_name
            )
            
            if audio_path:
                self.narration_audio = audio_path
                logger.info(f"내레이션 생성 완료: {audio_path}")
                return audio_path
            else:
                logger.error("내레이션 생성 실패")
                return None
                
        except Exception as e:
            logger.error(f"내레이션 생성 중 오류 발생: {e}")
            return None
    
    async def generate_voice(self) -> Dict[str, str]:
        """
        상세 스토리를 바탕으로 내레이션 및 캐릭터 대사 오디오를 생성하는 함수
        
        Returns:
            Dict[str, str]: 생성된 오디오 파일 경로 정보
        """
        if not self.elevenlabs_client or not self.detailed_story:
            logger.error("ElevenLabs 클라이언트 또는 상세 스토리가 없습니다.")
            return {}
            
        try:
            # 결과 저장용 딕셔너리
            audio_paths = {}
            
            # 챕터별로 오디오 생성
            for chapter in self.detailed_story.get("chapters", []):
                # ContentGenerator를 사용하여 챕터 오디오 생성
                chapter_audio = self.content_generator.generate_chapter_audio(chapter)
                
                # 결과 딕셔너리에 통합
                audio_paths.update(chapter_audio)
            
            # 미디어 메타데이터 저장
            self.media_manager.save_metadata()
            
            logger.info(f"음성 생성 완료: {len(audio_paths)} 개 파일")
            return audio_paths
            
        except Exception as e:
            logger.error(f"음성 생성 중 오류 발생: {e}")
            return {}
    
    def save_story_data(self, file_path: str) -> bool:
        """
        생성된 스토리 데이터를 JSON 파일로 저장하는 함수
        
        Args:
            file_path (str): 저장할 파일 경로
            
        Returns:
            bool: 성공 여부
        """
        try:
            # DataPersistence를 사용하여 저장
            data = {
                "story_outline": self.story_outline,
                "detailed_story": self.detailed_story,
                "generated_images": self.generated_images,
                "narration_audio": self.narration_audio,
                "target_age": self.target_age
            }
            
            story_id = self.data_persistence.save_story_data(data)
            
            if story_id:
                logger.info(f"스토리 데이터 저장 완료: {file_path} (ID: {story_id})")
                return True
            else:
                logger.error(f"스토리 데이터 저장 실패: {file_path}")
                return False
            
        except Exception as e:
            logger.error(f"스토리 데이터 저장 중 오류 발생: {e}")
            return False
    
    def load_story_data(self, file_path: str) -> bool:
        """
        스토리 데이터를 JSON 파일에서 로드하는 함수
        
        Args:
            file_path (str): 로드할 파일 경로
            
        Returns:
            bool: 성공 여부
        """
        try:
            # DataPersistence를 사용하여 로드
            data = self.data_persistence.load_story_data(file_path)
            
            if not data:
                logger.error(f"스토리 데이터 로드 실패: 파일이 없거나 빈 파일 - {file_path}")
                return False
            
            # 데이터 복원
            self.story_outline = data.get("story_outline")
            self.detailed_story = data.get("detailed_story")
            self.generated_images = data.get("generated_images", [])
            self.narration_audio = data.get("narration_audio")
            self.target_age = data.get("target_age")
            
            # 미디어 관리자 설정
            if "story_id" in data:
                self.media_manager.set_current_story_id(data["story_id"])
                self.content_generator.current_story_id = data["story_id"]
            
            logger.info(f"스토리 데이터 로드 완료: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"스토리 데이터 로드 중 오류 발생: {e}")
            return False
    
    def get_story_preview(self) -> Dict[str, str]:
        """
        생성된 스토리의 간략한 미리보기를 반환하는 함수
        
        Returns:
            Dict[str, str]: 스토리 미리보기 정보
        """
        if not self.detailed_story:
            return {
                "status": "not_generated",
                "message": "스토리가 아직 생성되지 않았습니다."
            }
            
        return {
            "status": "generated",
            "title": self.detailed_story.get("title", "제목 없음"),
            "scenes_count": len(self.detailed_story.get("chapters", [])),
            "images_count": len(self.generated_images),
            "has_audio": bool(self.narration_audio),
            "moral": self.detailed_story.get("educational_value", "")
        }
    
    async def generate_story(self) -> Tuple[List[str], str]:
        """
        동화 줄거리를 바탕으로 상세 스토리, 일러스트, 내레이션을 생성하는 함수
        
        Returns:
            Tuple[List[str], str]: 생성된 일러스트 파일 경로 목록, 내레이션 오디오 파일 경로
        """
        try:
            # 상세 스토리 생성
            detailed_story = self.generate_detailed_story()
            
            if not detailed_story:
                logger.error("상세 스토리 생성 실패")
                return [], ""
                
            # 일러스트 생성
            illustrations = self.generate_illustrations()
            
            # 내레이션 생성
            narration_info = await self.generate_voice()
            
            return illustrations, narration_info.get("narrator_audio", "")
            
        except Exception as e:
            logger.error(f"스토리 생성 중 오류 발생: {e}")
            return [], ""
    
    def _split_plot_into_scenes(self) -> List[str]:
        """
        줄거리를 여러 장면으로 분할하는 함수
        
        Returns:
            List[str]: 분할된 장면 설명 목록
        """
        if not self.story_outline or not self.story_outline.get("plot_summary"):
            logger.error("줄거리 정보가 없습니다.")
            return []
            
        try:
            plot_summary = self.story_outline.get("plot_summary", "")
            
            if self.openai_client:
                # OpenAI API를 사용하여 장면 분할
                messages = [
                    {"role": "system", "content": "다음 동화 줄거리를 5개의 핵심 장면으로 분할해주세요. 각 장면은 번호와 함께 간략한 설명으로 구성해주세요."},
                    {"role": "user", "content": f"동화 줄거리: {plot_summary}"}
                ]
                
                content, _ = self.openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    temperature=0.3
                )
                
                if content:
                    # 응답에서 장면 추출
                    scenes = []
                    for line in content.choices[0].message.content.strip().split('\n'):
                        if line.strip() and (':' in line or '.' in line[:3]):
                            scenes.append(line.strip())
                    
                    if scenes:
                        logger.info(f"줄거리를 {len(scenes)}개 장면으로 분할 완료")
                        return scenes
            
            # 기본적인 처리: 문단 단위로 분할
            paragraphs = [p.strip() for p in plot_summary.split('\n') if p.strip()]
            if len(paragraphs) >= 3:
                logger.info(f"줄거리를 {len(paragraphs)}개 장면으로 분할 완료")
                return paragraphs
            else:
                # 문장 단위로 분할
                sentences = [s.strip() for s in plot_summary.split('.') if s.strip()]
                logger.info(f"줄거리를 {len(sentences)}개 장면으로 분할 완료")
                return sentences[:5]  # 최대 5개 장면으로 제한
                
        except Exception as e:
            logger.error(f"줄거리 분할 중 오류 발생: {e}")
            return []
    
    def generate_detailed_story(self) -> Dict:
        """
        간략한 줄거리를 바탕으로 상세 스토리를 생성하는 함수
        
        Returns:
            Dict: 생성된 상세 스토리 데이터
        """
        if not self.openai_client or not self.story_outline:
            logger.error("OpenAI 클라이언트 또는 줄거리 정보가 없습니다.")
            return None
            
        try:
            # ContentGenerator를 사용하여 상세 스토리 생성
            story_data = self.content_generator.generate_detailed_story(
                self.story_outline,
                self.target_age or 5
            )
            
            if not story_data or "error" in story_data:
                logger.error(f"상세 스토리 생성 실패: {story_data.get('error', '알 수 없는 오류')}")
                return None
            
            # StoryParser를 사용하여 구조 검증
            if not self.story_parser.validate_story_structure(story_data):
                logger.error("상세 스토리 구조 검증 실패")
                return None
            
            # 상세 스토리 저장
            self.detailed_story = story_data
            
            # DataPersistence를 사용하여 스토리 저장
            story_id = self.data_persistence.save_story_data(story_data)
            
            if story_id:
                # 미디어 관리자에 스토리 ID 설정
                self.media_manager.set_current_story_id(story_id)
                self.content_generator.current_story_id = story_id
                logger.info(f"상세 스토리 생성 및 저장 완료: {story_data.get('title', '제목 없음')} (ID: {story_id})")
            
            return story_data
            
        except Exception as e:
            logger.error(f"상세 스토리 생성 중 오류 발생: {e}")
            return None
    
    def generate_illustrations(self) -> List[str]:
        """
        상세 스토리를 바탕으로 일러스트를 생성하는 함수
        
        Returns:
            List[str]: 생성된 일러스트 파일 경로 목록
        """
        if not self.detailed_story or not self.detailed_story.get("chapters"):
            logger.error("상세 스토리가 없습니다.")
            return []
            
        illustrations = []
        
        # 챕터별 일러스트 생성
        for chapter in self.detailed_story.get("chapters", []):
            chapter_number = chapter.get("chapter_number", 0)
            chapter_title = chapter.get("title", "")
            narration = chapter.get("narration", "")
            
            # 이미지 생성 진행상황 로깅
            logger.info(f"일러스트 생성 중: 챕터 {chapter_number}/{len(self.detailed_story.get('chapters', []))}")
            
            # ContentGenerator를 사용하여 이미지 생성
            image_info = self.content_generator.generate_image_for_chapter(
                chapter, 
                self.detailed_story
            )
            
            if image_info and "file_path" in image_info:
                file_path = image_info["file_path"]
                illustrations.append(file_path)
                chapter["image"] = file_path
            
        return illustrations 