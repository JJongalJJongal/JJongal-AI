""" 동화 상세 스토리 생성 엔진 """

""" 부기 (chatbot_a) 에서 수집한 이야기 요소 (요약된 스토리, 아이의 관심사, 나이, 이름)를 바탕으로
    완전한 동화를 생성하는 메인 엔진
"""

from src.shared.utils.logging import get_module_logger
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import asyncio

# Logging 설정
logger = get_module_logger(__name__) # 현재 모듈 이름으로 로거 생성

class StoryGenerationEngine:
    
    """
    동화 생성 핵심 엔진
    
    부기에서 수집한 이야기 요소들을 바탕으로
    1. 상세 스토리 Text 생성
    2. 챕터별 이미지 생성
    3. 챕터별 오디오 생성
    4. 완전한 멀티미디어 동화 생성
    """
    
    def __init__(self, openai_client = None, elevenlabs_client = None, rag_system = None, output_dir: str = "output"):
        """
        스토리 생성 엔진 초기화
        
        Args:
            openai_client: OpenAI API (GPT-4o, DALL-E 3)
            elevenlabs_client: ElevenLabs API (음성 생성, 음성 클로닝)
            rag_system: RAG System (한국 동화 스토리)
            output_dir: 출력 directory 경로
        """ 
        
        self.openai_client = openai_client
        self.elevenlabs_client = elevenlabs_client
        self.rag_system = rag_system
        self.output_dir = Path(output_dir)
        
        # 의존성 주입으로 설정함.
        self.text_generator = None
        self.image_generator = None
        self.voice_generator = None
        self.rag_enhancer = None
        
        # 현재 생성중인 스토리 정보
        self.current_story_id = None
        self.current_story_data = None
        
        logger.info("스토리 생성 엔진 초기화 완료")
            
    def set_generators(self, text_generator, image_generator, voice_generator, rag_enhancer):
        """
        생성기 설정 (의존성 주입)
        
        Args:
            text_generator: 텍스트 생성기
            image_generator: 이미지 생성기
            voice_generator: 음성 생성기
            rag_enhancer: RAG 향상기
        """
        
        self.text_generator = text_generator
        self.image_generator = image_generator
        self.voice_generator = voice_generator
        self.rag_enhancer = rag_enhancer
        
        logger.info("모든 생성기 설정 완료")
        
    async def generate_complete_story(self, story_outline: Dict) -> Dict[str, Any]:
        """
        완전한 동화 생성 (main method)

        Args:
            story_outline: 부기에서 수집한 이야기 개요
            {
                "theme": 이야기 주제,
                "age_group": 나이 그룹,
                "interests": 아이의 관심사,
                "name": 아이 이름
                "summary": 요약된 스토리 (부기가 꼬기에게 이러이러한 이야기 만들어줘.)
            }

        Returns:
            Dict: 완성된 동화 데이터
            {
                "story_data": 상세한 스토리 데이터,
                "image_paths": 챕터별 이미지 파일 경로들,
                "audio_paths": 챕터별 오디오 파일 경로들,
                "story_id": 생성된 고유 스토리 ID
                "status": 생성 상태 (partial, complete, failed)
                "error_message": 오류 메시지 (생성 실패 시)
                
            }
        """
        try:
            logger.info(f"동화 생성 시작 : {story_outline.get('title', '제목 없음')}")
            
            # 1. 스토리 ID 생성 및 설정
            import uuid
            self.current_story_id = str(uuid.uuid4())
            
            # 2. RAG System 으로 outline 향상
            if self.rag_enhancer:
                logger.info("RAG System 활용하여 스토리 개요 향상 시작")
                enhanced_outline = await self.rag_enhancer.enhance_story_outline(story_outline)
                
            else:
                enhanced_outline = story_outline
                logger.warning("RAG System 오류 발생. 스토리 개요 향상 불가")
                
            # 3. 상세 스토리 텍스트 생성
            logger.info("상세 스토리 텍스트 생성 시작")
            if not self.text_generator:
                raise ValueError("텍스트 생성기 설정 오류")

            detailed_story = await self.text_generator.generate(enhanced_outline)
            self.current_story_data = detailed_story
            
            
            # 4. 병렬로 이미지와 음성 생성
            logger.info("챕터별 이미지와 음성 병렬로 생성 시작")
            
            # Image 생성 Task
            image_task = None
            if self.image_generator:
                image_task = asyncio.create_task(
                    self.image_generator.generate_batch(detailed_story.get("chapters", []), self.current_story_id)
                )
                
            # 음성 생성 Task
            audio_task = None
            if self.voice_generator:
                audio_task = asyncio.create_task(
                    self.voice_generator.generate({
                        "story_data": detailed_story,
                        "story_id": self.current_story_id
                    })
                )
            
            # 병렬 실행 결과 수집
            image_paths = []
            audio_paths = []
            
            if image_task: # 이미지 생성 완료 시 결과 수집
                try:
                    image_paths = await image_task # 이미지 생성 완료 시 결과 수집
                    logger.info(f"챕터별 이미지 생성 완료 : {len(image_paths)} 개")
                except Exception as e:
                    logger.error(f"이미지 생성 오류 발생 : {e}")
            
            if audio_task: # 음성 생성 완료 시 결과 수집
                try:
                    audio_paths = await audio_task # 음성 생성 완료 시 결과 수집
                    logger.info(f"챕터별 음성 생성 완료 : {len(audio_paths)} 개")
                except Exception as e:
                    logger.error(f"음성 생성 오류 발생 : {e}")
                    
            # 5. 결과 조합 및 상태 결정
            status = "complete"
            if not image_paths and not audio_paths:
                status = "partial" # 텍스트만 성공
            elif not image_paths or not audio_paths:
                status = "partial" # 일부만 성공 (이미지만 or 음성만)
            
            results = {
                "story_data": detailed_story,
                "image_paths": image_paths,
                "audio_paths": audio_paths,
                "story_id": self.current_story_id,
                "status": status,
                "error_message": None
            }
            
            logger.info(f"동화 생성 완료: {self.current_story_id} (상태 : {status})")
            return results

        except Exception as e:
            logger.error(f"동화 생성 오류 발생 : {e}")
            return {
                "story_data": None,
                "image_paths": [],
                "audio_paths": [],
                "story_id": self.current_story_id,
                "status": "failed",
                "error_message": str(e)
            }
            
    def reset_engine(self):
        """ 엔진 상태 완전 초기화 """
        
        try:
            # 생성 완료 후 초기화
            self.cleanup()
            
            # 생성기들 초기화 (의존성 주입 대기 상태)
            self.text_generator = None
            self.image_generator = None
            self.voice_generator = None
            self.rag_enhancer = None
            
            logger.info("스토리 생성 엔진 완전 초기화 완료")
            
        except Exception as e:
            logger.error(f"엔진 초기화 오류 발생 : {e}")
            
        
    def cleanup(self):
        """ 리소스 정리 """
        
        try:
            # 생성하고 있는 스토리 정보 초기화
            self.current_story_id = None
            self.current_story_data = None
            
            # 생성기들의 리소스 정리 (있는 경우)
            if hasattr(self.text_generator, "cleanup"):
                self.text_generator.cleanup()
            if hasattr(self.image_generator, "cleanup"):
                self.image_generator.cleanup()
            if hasattr(self.voice_generator, "cleanup"):
                self.voice_generator.cleanup()
            if hasattr(self.rag_enhancer, "cleanup"):
                self.rag_enhancer.cleanup()
                
            logger.info("엔진 리소스 정리 완료")
        except Exception as e:
            logger.error(f"리소스 정리 오류 발생 : {e}")
            