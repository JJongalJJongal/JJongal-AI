"""
CCB_AI Multimedia Coordinator

이야기에 대한 멀티미디어 자산(이미지, 오디오) 생성을 조정하는 모듈.
"""

import asyncio
import logging
import os
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
from pathlib import Path

from .story_schema import StoryDataSchema, MultimediaAssets

# 로깅 설정
logger = logging.getLogger(__name__)

# 외부 라이브러리 임포트 (선택적)
try:
    import openai
    OPENAI_AVAILABLE = True
    logger.info("OpenAI 라이브러리 이용 가능")
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI 라이브러리 이용 불가")

try:
    from elevenlabs import save
    ELEVENLABS_AVAILABLE = True
    logger.info("ElevenLabs 라이브러리 이용 가능")
except ImportError:
    ELEVENLABS_AVAILABLE = False
    logger.warning("ElevenLabs 라이브러리 이용 불가")

# ChatBotB Import 추가
try:
    from chatbot.models.chat_bot_b import ChatBotB
    from chatbot.models.chat_bot_b.generators.voice_generator import VoiceGenerator
    CHATBOT_B_AVAILABLE = True
    logger.info("ChatBotB 모듈 이용 가능")
except ImportError as e:
    CHATBOT_B_AVAILABLE = False
    logger.warning(f"ChatBotB 모듈 이용 불가: {e}")

class MultimediaCoordinator:
    """
    멀티미디어 조정자
    
    이야기에 대한 이미지와 오디오 생성을 관리합니다.
    """
    
    def __init__(self, output_dir: str = "output"):
        """
        멀티미디어 조정자 초기화
        
        Args:
            output_dir: 출력 디렉토리 경로
        """
        # 출력 디렉토리 설정
        self.output_dir = os.getenv("MULTIMEDIA_OUTPUT_DIR", output_dir)
        self.logger = logging.getLogger(__name__)
        
        # 출력 디렉토리 생성 (절대 경로로 설정)
        if not os.path.isabs(self.output_dir):
            self.output_dir = os.path.abspath(self.output_dir)
        
        # 통일된 temp 경로 구조 사용 - 중복 제거
        base_output_path = Path(self.output_dir)
        temp_path = base_output_path / "temp"
        
        self.images_dir = str(temp_path / "images")  # output/temp/images
        self.audio_dir = str(temp_path / "audio")    # output/temp/audio
        
        try:
            os.makedirs(self.images_dir, exist_ok=True) # 이미지 디렉토리 생성
            os.makedirs(self.audio_dir, exist_ok=True) # 오디오 디렉토리 생성
            self.logger.info(f"멀티미디어 디렉토리 생성됨: {self.output_dir} (temp: {temp_path})") # 멀티미디어 디렉토리 생성됨
        except PermissionError as e:
            self.logger.error(f"멀티미디어 디렉토리 생성 권한 오류: {e}") # 멀티미디어 디렉토리 생성 권한 오류
            # 권한 오류 시 다시 시도
            os.makedirs(self.images_dir, exist_ok=True) # 이미지 디렉토리 생성
            os.makedirs(self.audio_dir, exist_ok=True) # 오디오 디렉토리 생성
            
        
        # 설정
        self.image_config = {
            "style": "children_book",
            "size": "1024x1024",
            "quality": "standard",
            "max_images": 5
        }
        
        self.audio_config = {
            "voice": "child_friendly",
            "speed": 1.0,
            "format": "mp3"
        }
        
        # API 키 확인
        self.openai_api_key = os.getenv("OPENAI_API_KEY") # OPENAI_API_KEY 환경변수 가져오기
        self.elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY") # ELEVENLABS_API_KEY 환경변수 가져오기
        
        # ChatBotB 인스턴스 초기화 (실제 이미지 생성용)
        self.chat_bot_b = None
        # VoiceGenerator 직접 초기화 (등장인물별 음성 생성용)
        self.voice_generator = None
        
        if CHATBOT_B_AVAILABLE:
            try:
                # 환경변수에서 vector_db_path 가져오기
                vector_db_path = os.getenv("VECTOR_DB_PATH", "/app/chatbot/data/vector_db/main")
                self.chat_bot_b = ChatBotB(
                    output_dir=self.output_dir,  # temp 제외한 기본 경로만 전달
                    vector_db_path=vector_db_path,
                    collection_name="fairy_tales",
                    use_enhanced_generators=True,
                    enable_performance_tracking=True
                )
                self.logger.info(f"ChatBotB 초기화 완료 (vector_db_path: {vector_db_path}) - 실제 스토리 생성 사용")
                
                # VoiceGenerator 직접 초기화 - 통일된 temp 경로 사용
                self.voice_generator = VoiceGenerator(
                    elevenlabs_api_key=self.elevenlabs_api_key,
                    temp_storage_path=self.audio_dir  # 이미 설정된 temp/audio 경로 사용
                )
                self.logger.info("VoiceGenerator 직접 초기화 완료")
                
            except Exception as e:
                self.logger.error(f"ChatBotB 초기화 실패: {e}")
                self.chat_bot_b = None
                self.voice_generator = None
        
        # 초기화 상태 로깅
        self._log_initialization_status() # 초기화 상태 로깅
        
        self.logger.info("멀티미디어 조정자 초기화 완료")
    
    def _log_initialization_status(self):
        """초기화 상태 로깅"""
        status = {
            "output_dir": self.output_dir,
            "images_dir": self.images_dir,
            "audio_dir": self.audio_dir,
            "openai_available": OPENAI_AVAILABLE,
            "elevenlabs_available": ELEVENLABS_AVAILABLE,
            "chatbot_b_available": CHATBOT_B_AVAILABLE,
            "openai_api_key_set": bool(self.openai_api_key),
            "elevenlabs_api_key_set": bool(self.elevenlabs_api_key),
            "chatbot_b_initialized": self.chat_bot_b is not None
        }
        
        self.logger.info("멀티미디어 조정자 초기화 상태:")
        for key, value in status.items():
            self.logger.info(f"  {key}: {value}")
        
        # 경고 메시지
        if not status["openai_api_key_set"] and status["openai_available"]:
            self.logger.warning("OpenAI API 키가 설정되지 않음 - 이미지 생성이 제한됩니다")
        if not status["elevenlabs_api_key_set"] and status["elevenlabs_available"]:
            self.logger.warning("ElevenLabs API 키가 설정되지 않음 - 오디오 생성이 제한됩니다")
    
    async def generate_images(self, story_schema: StoryDataSchema) -> List[Dict[str, str]]:
        """
        이야기에 대한 이미지 생성
        
        Args:
            story_schema: 이야기 스키마
            
        Returns:
            생성된 이미지 정보 목록
        """
        try:
            self.logger.info(f"이미지 생성 시작: {story_schema.metadata.story_id}")
            
            images = []
            story_id = story_schema.metadata.story_id
            
            # 이미지 저장 디렉토리 생성 (짧은 story_id 사용 - 오디오와 통일)
            story_id_short = story_id[:8] if len(story_id) > 8 else story_id
            story_images_dir = os.path.join(self.images_dir, story_id_short)
            os.makedirs(story_images_dir, exist_ok=True)
            
            # 이야기 내용에서 장면 추출
            scenes = self._extract_scenes(story_schema)
            
            for i, scene in enumerate(scenes[:self.image_config["max_images"]]):
                try:
                    # 이미지 생성 - 짧은 story_id 전달 (오디오와 통일)
                    image_info = await self._generate_single_image(scene, i, story_images_dir, story_id_short)
                    if image_info:
                        images.append(image_info)
                        
                except Exception as e:
                    self.logger.error(f"이미지 생성 실패 (장면 {i}): {e}")
                    # 기본 이미지 정보 추가
                    images.append({
                        "url": "",
                        "description": scene["description"],
                        "scene": f"scene_{i}",
                        "error": str(e)
                    })
            
            self.logger.info(f"이미지 생성 완료: {len(images)}개")
            return images
            
        except Exception as e:
            self.logger.error(f"이미지 생성 실패: {e}")
            return []
    
    async def generate_audio(self, story_schema: StoryDataSchema) -> List[Dict[str, str]]:
        """
        이야기에 대한 오디오 생성
        
        Args:
            story_schema: 이야기 스키마
            
        Returns:
            생성된 오디오 정보 목록
        """
        try:
            self.logger.info(f"오디오 생성 시작: {story_schema.metadata.story_id}")
            
            if not self.chat_bot_b or not self.chat_bot_b.voice_generator:
                self.logger.warning("ChatBotB 또는 VoiceGenerator가 초기화되지 않아 오디오를 생성할 수 없습니다.")
                return []
            
            # VoiceGenerator에 필요한 입력 데이터 구성
            input_data = {
                "story_data": story_schema.generated_story.to_dict(),
                "story_id": story_schema.metadata.story_id,
            }
            
            # 꼬기의 VoiceGenerator 사용
            audio_result = await self.chat_bot_b.voice_generator.generate(input_data)

            audio_files = audio_result.get("audio_files", [])
            
            final_audio_list = []
            
            # 생성된 오디오 파일 목록을 순회하며 결과 리스트 반환
            for chapter_audio in audio_files:
                if chapter_audio.get("combined_audio"):
                    final_audio_list.append({
                        "path": chapter_audio["combined_audio"],
                        "chapter": chapter_audio["chapter_number"]
                    })
                # 합쳐진 파일이 없고 내레이션 파일만 있는 경우
                elif chapter_audio.get("narration_audio"):
                    final_audio_list.append({
                        "path": chapter_audio["narration_audio"],
                        "chapter": chapter_audio["chapter_number"]
                    })
                # 합쳐진 파일이 없고 내레이션 파일도 없는 경우
                else:
                    self.logger.warning(f"오디오 파일이 없음: {chapter_audio}")
                    return []
            
            self.logger.info(f"오디오 생성 완료: {len(final_audio_list)}개")
            return final_audio_list
            
        except Exception as e:
            self.logger.error(f"오디오 생성 실패: {e}")
            return []
    
    def _extract_scenes(self, story_schema: StoryDataSchema) -> List[Dict[str, str]]:
        """이야기에서 장면 추출"""
        scenes = []
        
        try:
            # 이야기 요소에서 장면 생성
            from .story_schema import ElementType
            characters = story_schema.get_elements_by_type(ElementType.CHARACTER)
            settings = story_schema.get_elements_by_type(ElementType.SETTING)
            problems = story_schema.get_elements_by_type(ElementType.PROBLEM)
            
            # 기본 장면들
            if characters and settings:
                scenes.append({
                    "description": f"{characters[0].content}이(가) {settings[0].content}에 있는 모습",
                    "type": "character_introduction"
                })
            
            if problems:
                scenes.append({
                    "description": f"{problems[0].content} 상황",
                    "type": "problem"
                })
            
            # 생성된 이야기에서 추가 장면 추출
            if story_schema.generated_story and story_schema.generated_story.chapters:
                for i, chapter in enumerate(story_schema.generated_story.chapters):
                    scenes.append({
                        "description": chapter.get("title", f"챕터 {i+1}"),
                        "type": "chapter",
                        "content": chapter.get("content", "")[:200]  # 처음 200자
                    })
            
            # 최대 5개 장면으로 제한
            return scenes[:5]
            
        except Exception as e:
            self.logger.error(f"장면 추출 실패: {e}")
            return [{"description": "동화 장면", "type": "default"}]
    
    async def _generate_single_image(
        self,
        scene: Dict[str, str],
        index: int,
        output_dir: str,
        actual_story_id: str = None
    ) -> Optional[Dict[str, str]]:
        """단일 이미지 생성 - ChatBotB ImageGenerator 사용"""
        try:
            # ChatBotB가 있으면 실제 이미지 생성 사용
            if self.chat_bot_b and CHATBOT_B_AVAILABLE:
                self.logger.info(f"ChatBotB ImageGenerator로 실제 이미지 생성 시작 (장면 {index})")
                
                # 이미지 생성을 위한 데이터 구성 (ChatBotB 형식에 맞춤)
                image_gen_input = {
                    "story_data": {
                        "title": "Generated Scene",
                        "chapters": [{
                            "chapter_number": index + 1,
                            "chapter_title": f"Scene {index}",
                            "chapter_content": scene.get("description", "")
                        }],
                        "metadata": {"target_age": "4-7세"}  # 기본 연령대
                    },
                    "story_id": actual_story_id or f"scene_{index}"
                }
                
                # ChatBotB ImageGenerator 사용 
                result = await self.chat_bot_b.image_generator.generate(image_gen_input)
                
                if result and result.get("images") and len(result["images"]) > 0:
                    image_info = result["images"][0]
                    image_path = image_info.get("image_path")
                    
                    if image_path and os.path.exists(image_path):
                        # 이미지 파일을 output 디렉토리로 복사
                        image_filename = f"scene_{index}.png"
                        dest_path = os.path.join(output_dir, image_filename)
                        
                        import shutil
                        shutil.copy2(image_path, dest_path)
                        
                        self.logger.info(f"실제 이미지 생성 완료: {dest_path}")
                        
                        return {
                            "url": dest_path,
                            "description": scene["description"],
                            "scene": f"scene_{index}",
                            "prompt": image_info.get("image_prompt", ""),
                            "generated": True,
                            "method": "chatbot_b_dalle"
                        }
            
    
            # 이미지 생성 프롬프트 구성
            prompt = self._create_image_prompt(scene)
            
            # 시뮬레이션 이미지 생성
            image_filename = f"scene_{index}.png"
            image_path = os.path.join(output_dir, image_filename)
            
            await self._simulate_image_generation(image_path)
            
            return {
                "url": image_path,
                "description": scene["description"],
                "scene": f"scene_{index}",
                "prompt": prompt,
                "generated": True,
                "method": "simulation"
            }
            
        except Exception as e:
            self.logger.error(f"단일 이미지 생성 실패 (장면 {index}): {e}")
            return None
    
    async def _generate_story_audio(
        self,
        text: str,
        output_dir: str,
        filename: str
    ) -> Optional[Dict[str, str]]:
        """이야기 오디오 생성"""
        try:
            if not text.strip():
                return None
            
            audio_filename = f"{filename}.{self.audio_config['format']}"
            audio_path = os.path.join(output_dir, audio_filename)
            
            # VoiceGenerator 직접 사용 (등장인물별 음성 생성)
            if self.voice_generator:
                try:
                    # VoiceGenerator에 맞는 input 구성
                    voice_input = {
                        "story_data": {
                            "title": "Character Voice Audio Generation",
                            "chapters": [{
                                "chapter_number": 1,
                                "chapter_content": text,  # 전체 텍스트
                                "narration": text,        # VoiceGenerator가 내부적으로 분리함
                                "dialogues": []           # VoiceGenerator가 내부적으로 추출함
                            }],
                            "characters": [            # 기본 캐릭터 설정
                                {"name": "내레이터", "type": "narrator"},
                                {"name": "주인공", "type": "child"}
                            ]
                        },
                        "story_id": filename
                    }
                    
                    # VoiceGenerator의 generate 메서드 직접 호출
                    result = await self.voice_generator.generate(voice_input, use_websocket=False)
                    
                    if result and result.get("audio_files") and len(result["audio_files"]) > 0:
                        # 첫 번째 챕터의 오디오 파일 사용
                        chapter_audio = result["audio_files"][0]
                        generated_audio_path = chapter_audio.get("narration_audio") or chapter_audio.get("audio_path")
                        
                        if generated_audio_path and os.path.exists(generated_audio_path):
                            # 생성된 파일을 지정된 경로로 복사
                            import shutil
                            shutil.copy2(generated_audio_path, audio_path)
                            self.logger.info(f"VoiceGenerator로 등장인물별 오디오 생성 완료: {audio_path}")
                        else:
                            self.logger.error("VoiceGenerator에서 오디오 파일 생성 실패")
                            return None
                    else:
                        self.logger.error("VoiceGenerator 결과가 비어있음")
                        return None
                        
                except Exception as e:
                    self.logger.error(f"VoiceGenerator 오디오 생성 실패: {e}")
                    return None
            else:
                self.logger.error("VoiceGenerator를 사용할 수 없음 - ElevenLabs API 키 확인 필요")
                
            
            # 오디오 길이 계산 (대략적)
            duration = len(text.split()) * 0.5  # 단어당 0.5초 가정하였음.
            
            return {
                "url": audio_path,
                "type": "narration",
                "duration": f"{duration:.1f}s",
                "filename": audio_filename,
                "generated": True
            }
            
        except Exception as e:
            self.logger.error(f"오디오 생성 실패: {e}")
            return None
    
    def _create_image_prompt(self, scene: Dict[str, str]) -> str:
        """이미지 생성 프롬프트 생성"""
        scene_desc = scene["description"]
        
        # 자연스럽고 손으로 그린 느낌의 프롬프트
        prompt = f"Gentle children's storybook illustration: {scene_desc}. "
        prompt += "Hand-sketched with colored pencils and soft watercolor, like a beloved worn picture book. "
        prompt += "Visible pencil lines, slightly uneven coloring, warm paper texture. "
        prompt += "Muted pastels - faded pinks, sage greens, cream yellows, dusty blues. "
        prompt += "Cozy and nostalgic, imperfect but charming."
        
        return prompt
    
    async def _simulate_image_generation(self, output_path: str):
        """이미지 생성 시뮬레이션"""
        # 실제 구현에서는 OpenAI DALL-E API 호출
        await asyncio.sleep(1)  # 생성 시간 시뮬레이션
        
        # 빈 파일 생성 (실제로는 이미지 데이터)
        with open(output_path, 'w') as f:
            f.write("# Generated image placeholder")
    
    async def generate_multimedia_package(self, story_schema: StoryDataSchema) -> MultimediaAssets:
        """완전한 멀티미디어 패키지 생성"""
        try:
            self.logger.info(f"멀티미디어 패키지 생성 시작: {story_schema.metadata.story_id}")
            
            # 병렬로 이미지와 오디오 생성
            images_task = self.generate_images(story_schema)
            audio_task = self.generate_audio(story_schema)
            
            images, audio_files = await asyncio.gather(images_task, audio_task)
            
            # 멀티미디어 자산 객체 생성
            multimedia_assets = MultimediaAssets(
                images=images,
                audio_files=audio_files,
                video_files=[],  # 비디오는 향후 구현
                generated_at=datetime.now()
            )
            
            # 메타데이터 저장
            await self._save_multimedia_metadata(story_schema.metadata.story_id, multimedia_assets)
            
            self.logger.info(f"멀티미디어 패키지 생성 완료: {story_schema.metadata.story_id}")
            return multimedia_assets
            
        except Exception as e:
            self.logger.error(f"멀티미디어 패키지 생성 실패: {e}")
            return MultimediaAssets()
    
    async def _save_multimedia_metadata(self, story_id: str, assets: MultimediaAssets):
        """멀티미디어 메타데이터 저장"""
        try:
            metadata_file = os.path.join(self.output_dir, "metadata", f"{story_id}_multimedia.json")
            os.makedirs(os.path.dirname(metadata_file), exist_ok=True)
            
            metadata = {
                "story_id": story_id,
                "generated_at": assets.generated_at.isoformat(),
                "images_count": len(assets.images),
                "audio_count": len(assets.audio_files),
                "video_count": len(assets.video_files),
                "assets": assets.to_dict()
            }
            
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            self.logger.error(f"멀티미디어 메타데이터 저장 실패: {e}")
    
    def get_multimedia_status(self, story_id: str) -> Dict[str, Any]:
        """멀티미디어 생성 상태 조회"""
        try:
            # 짧은 story_id 사용 (저장 시와 동일)
            story_id_short = story_id[:8] if len(story_id) > 8 else story_id
            
            # 이미지 디렉토리 확인
            story_images_dir = os.path.join(self.images_dir, story_id_short)
            images_exist = os.path.exists(story_images_dir)
            image_count = len(os.listdir(story_images_dir)) if images_exist else 0
            
            # 오디오 디렉토리 확인
            story_audio_dir = os.path.join(self.audio_dir, story_id_short)
            audio_exist = os.path.exists(story_audio_dir)
            audio_count = len(os.listdir(story_audio_dir)) if audio_exist else 0
            
            return {
                "story_id": story_id,
                "images_generated": images_exist,
                "image_count": image_count,
                "audio_generated": audio_exist,
                "audio_count": audio_count,
                "multimedia_complete": images_exist and audio_exist
            }
            
        except Exception as e:
            self.logger.error(f"멀티미디어 상태 조회 실패: {e}")
            return {"error": str(e)}
    
    def cleanup_multimedia_files(self, story_id: str) -> bool:
        """멀티미디어 파일 정리"""
        try:
            import shutil
            
            # 이미지 디렉토리 삭제
            story_images_dir = os.path.join(self.images_dir, story_id)
            if os.path.exists(story_images_dir):
                shutil.rmtree(story_images_dir)
            
            # 오디오 디렉토리 삭제
            story_audio_dir = os.path.join(self.audio_dir, story_id)
            if os.path.exists(story_audio_dir):
                shutil.rmtree(story_audio_dir)
            
            # 메타데이터 파일 삭제
            metadata_file = os.path.join(self.output_dir, "metadata", f"{story_id}_multimedia.json")
            if os.path.exists(metadata_file):
                os.remove(metadata_file)
            
            self.logger.info(f"멀티미디어 파일 정리 완료: {story_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"멀티미디어 파일 정리 실패: {e}")
            return False 
