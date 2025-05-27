from shared.utils.logging_utils import get_module_logger
from typing import Dict, Any, Optional, Callable

logger = get_module_logger(__name__)

class StoryGenerationChatBot:
    def __init__(self):
        self.story_engine = None
        self.target_age = None
        self.story_outline = None

    def set_target_age(self, age: int):
        """대상 연령 설정"""
        self.target_age = age # 대상 연령 설정
        logger.info(f"대상 연령 설정: {age}세") # 로깅
     
    def set_story_outline(self, story_outline: Dict[str, Any]):
        """부기에서 수집한 스토리 개요 설정"""
        self.story_outline = story_outline # 스토리 개요 설정
        logger.info("스토리 개요 설정 완료") # 로깅

    def set_cloned_voice_info(self, child_voice_id: str, main_character_name: str):
        """
        아이의 클론된 음성 ID와 메인 캐릭터 이름을 설정합니다.
        이는 VoiceGenerator의 캐릭터 음성 매핑을 업데이트합니다.
        """
        # story_engine과 voice_generator가 초기화되었는지 확인
        if not hasattr(self, 'story_engine') or not self.story_engine:
            logger.warning(
                "StoryEngine이 아직 초기화되지 않았습니다. "
                "클론된 음성 정보 설정이 적용되지 않을 수 있습니다."
            )
            # TODO: 이 경우를 대비하여 정보를 임시 저장했다가 story_engine 초기화 후 적용하는 로직 고려
            return

        if not hasattr(self.story_engine, 'voice_generator') or not self.story_engine.voice_generator:
            logger.warning(
                "StoryEngine 내의 VoiceGenerator가 아직 초기화되지 않았습니다. "
                "클론된 음성 정보 설정이 적용되지 않을 수 있습니다."
            )
            # TODO: 마찬가지로 임시 저장 로직 고려
            return

        character_mapping = {main_character_name: child_voice_id}
        
        try:
            # VoiceGenerator의 set_character_voice_mapping 호출
            self.story_engine.voice_generator.set_character_voice_mapping(character_mapping)
            logger.info(f"클론된 음성 정보 설정 완료: 캐릭터 '{main_character_name}'에 음성 ID '{child_voice_id}'가 매핑되었습니다.")
        except AttributeError:
            logger.error(
                "VoiceGenerator에 'set_character_voice_mapping' 메서드가 없거나, "
                "story_engine.voice_generator가 올바르게 설정되지 않았습니다."
            )
        except Exception as e:
            logger.error(f"클론된 음성 정보 설정 중 예상치 못한 오류 발생: {e}")

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
        
        if self.text_generator and hasattr(self.text_generator, 'health_check'):
            status["text_generator"] = await self.text_generator.health_check()
            
        if self.image_generator and hasattr(self.image_generator, 'health_check'):
            status["image_generator"] = await self.image_generator.health_check()
            
        if self.voice_generator and hasattr(self.voice_generator, 'health_check'):
            status["voice_generator"] = await self.voice_generator.health_check()
            
        return status
    
    def cleanup(self):
        """임시 파일 및 리소스 정리"""
        if self.story_engine:
            self.story_engine.cleanup()
        # 추가적인 정리 로직 (예: self.output_dir 내 임시 파일)
        logger.info("StoryGenerationChatBot 리소스 정리 완료.")