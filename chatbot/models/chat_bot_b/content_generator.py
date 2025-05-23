"""
이야기, 이미지, 음성 생성을 담당하는 모듈
"""
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import os
import base64
import json
import time
import random

from shared.utils.logging_utils import get_module_logger
from shared.utils.openai_utils import generate_chat_completion
from shared.utils.file_utils import ensure_directory, save_image_from_base64, save_audio

logger = get_module_logger(__name__)

class ContentGenerator:
    """
    이야기, 이미지, 음성 생성을 담당하는 클래스
    """
    
    def __init__(self, openai_client=None, elevenlabs_client=None, 
                 images_dir: Union[str, Path] = None, audio_dir: Union[str, Path] = None,
                 current_story_id: str = None,
                 child_voice_id: Optional[str] = None,  # 아이의 클론된 음성 ID
                 main_character_name: Optional[str] = None,  # 주인공 캐릭터 이름
                 voice_settings: Dict = None,
                 voice_model: str = "eleven_multilingual_v2"
                 ):
        """
        컨텐츠 생성기 초기화
        
        Args:
            openai_client: OpenAI API 클라이언트
            elevenlabs_client: ElevenLabs API 클라이언트
            images_dir: 이미지 저장 디렉토리
            audio_dir: 오디오 저장 디렉토리
            current_story_id: 현재 스토리 ID
            child_voice_id: 아이의 클론된 음성 ID
            main_character_name: 주인공 캐릭터 이름
            voice_settings: ElevenLabs 음성 설정
            voice_model: ElevenLabs 음성 모델
        """
        # API 클라이언트 설정
        self.openai_client = openai_client
        self.elevenlabs_client = elevenlabs_client
        
        # 저장 디렉토리 설정
        self.images_dir = Path(images_dir) if images_dir else None
        self.audio_dir = Path(audio_dir) if audio_dir else None
        
        # 현재 스토리 ID
        self.current_story_id = current_story_id
        
        # 클론된 음성 정보
        self.child_voice_id = child_voice_id
        self.main_character_name = main_character_name
        
        # 음성 설정
        self.voice_settings = voice_settings if voice_settings else {
            "stability": 0.75,
            "similarity_boost": 0.75
        }
        
        # 캐릭터별 음성 ID 매핑
        self.character_voices = {
            "narrator": "XrExE9yKIg1WjnnlVkGX",   # 여성 내레이터
            "male_adult": "pNInz6obpgDQGcFmaJgB", # 성인 남성
            "female_adult": "EXAVITQu4vr4xnSDxMaL", # 성인 여성
            "male_child": "jsCqWAovK2LkecY7zXl4", # 아동 남성
            "female_child": "z9fAnlkpzviPz146aGWa" # 아동 여성
        }
        
        # 음성 모델
        self.voice_model = voice_model
    
    def generate_detailed_story(self, story_outline: Dict, target_age: int = 5) -> Dict:
        """
        개요를 바탕으로 상세 이야기 생성
        
        Args:
            story_outline: 이야기 개요 (주제, 캐릭터, 배경, 줄거리 요약, 교육적 가치 등)
            target_age: 대상 연령대
            
        Returns:
            Dict: 상세 이야기 (챕터, 대사, 설명 등)
        """
        if not self.openai_client:
            logger.error("OpenAI 클라이언트가 초기화되지 않았습니다.")
            return {"error": "OpenAI 클라이언트가 초기화되지 않았습니다."}
        
        # 주요 요소 추출
        theme = story_outline.get("theme", "모험과 우정")
        characters = story_outline.get("characters", ["주인공"])
        setting = story_outline.get("setting", "판타지 세계")
        plot_summary = story_outline.get("plot_summary", "")
        educational_value = story_outline.get("educational_value", "")
        
        # 시스템 메시지 구성
        system_message = f"""
        당신은 아이들을 위한 동화를 작성하는 전문 동화 작가입니다.
        다음 요소를 활용하여 {target_age}세 어린이를 위한 자세한 챕터별 동화를 작성해주세요.
        작품은 5개의 챕터로 구성되어야 합니다.

        1. 각 챕터는 다음 요소를 포함해야 합니다:
           - 챕터 제목
           - 내레이션(설명 텍스트)
           - 캐릭터 대사(있을 경우)
           - 챕터 분위기와 교훈

        2. 출력 형식은 다음과 같은 JSON 형식이어야 합니다:
        {{
          "title": "동화 제목",
          "age_group": {target_age},
          "theme": "{theme}",
          "educational_value": "{educational_value}",
          "characters": [{", ".join([f'"{c}"' for c in characters])}],
          "setting": "{setting}",
          "chapters": [
            {{
              "chapter_number": 1,
              "title": "챕터 제목",
              "narration": "내레이션 텍스트...",
              "dialogues": [
                {{"speaker": "캐릭터명", "text": "대사 내용..."}},
                {{"speaker": "다른 캐릭터", "text": "대사 내용..."}}
              ],
              "mood": "밝음/긴장감/신비로움 등",
              "moral_lesson": "이 챕터의 교훈"
            }},
            ...나머지 챕터들...
          ]
        }}
        """
        
        # 사용자 메시지 구성
        user_message = f"""
        다음 동화 개요를 바탕으로 {target_age}세 어린이를 위한 자세한 챕터별 동화를 작성해 주세요:
        
        주제: {theme}
        등장인물: {", ".join(characters)}
        배경: {setting}
        줄거리 요약: {plot_summary}
        교육적 가치: {educational_value}
        
        대상 연령대에 맞는 적절한 어휘와 문장 길이를 사용해 주세요.
        각 챕터에 내레이션과 대사를 적절히 혼합해 주세요.
        명확한 JSON 형식으로 출력해 주세요.
        """
        
        try:
            # GPT-4를 사용하여 상세 이야기 생성
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",  # 더 상세한 이야기를 위해 GPT-4 사용
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.7,
                max_tokens=3000
            )
            
            # 응답 추출
            response_text = response.choices[0].message.content
            
            # JSON 형식의 응답 파싱
            try:
                # JSON 응답 파싱
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                
                if json_start >= 0 and json_end > json_start:
                    json_str = response_text[json_start:json_end]
                    story_data = json.loads(json_str)
                    
                    # 결과 로깅
                    logger.info(f"상세 이야기 생성 완료: {story_data.get('title', '제목 없음')} ({len(story_data.get('chapters', []))} 챕터)")
                    
                    return story_data
                else:
                    logger.error("JSON 형식의 응답을 찾을 수 없습니다.")
                    return {"error": "JSON 형식의 응답을 찾을 수 없습니다.", "raw_text": response_text}
            
            except json.JSONDecodeError as e:
                logger.error(f"JSON 파싱 오류: {e}")
                return {"error": f"JSON 파싱 오류: {e}", "raw_text": response_text}
            
        except Exception as e:
            logger.error(f"상세 이야기 생성 중 오류 발생: {e}")
            return {"error": f"상세 이야기 생성 중 오류 발생: {e}"}
    
    def generate_image_for_chapter(self, chapter_data: Dict, story_info: Dict) -> Optional[Dict]:
        """
        챕터 내용을 바탕으로 이미지 생성
        
        Args:
            chapter_data: 챕터 데이터
            story_info: 이야기 기본 정보
            
        Returns:
            Optional[Dict]: 이미지 정보 (URL, 파일 경로, 프롬프트 등) 또는 None
        """
        if not self.openai_client:
            logger.error("OpenAI 클라이언트가 초기화되지 않았습니다.")
            return None
        
        if not self.images_dir or not self.current_story_id:
            logger.error("이미지 저장 디렉토리 또는 스토리 ID가 설정되지 않았습니다.")
            return None
        
        # 챕터 정보 추출
        chapter_number = chapter_data.get("chapter_number", 0)
        chapter_title = chapter_data.get("title", f"Chapter {chapter_number}")
        narration = chapter_data.get("narration", "")
        mood = chapter_data.get("mood", "")
        
        # 이야기 기본 정보 추출
        characters = story_info.get("characters", [])
        setting = story_info.get("setting", "")
        age_group = story_info.get("age_group", 5)
        
        # 이미지 프롬프트 생성
        image_prompt = f"""
        Create a detailed, colorful illustration for a children's story with the following elements:
        
        Title: {chapter_title}
        Setting: {setting}
        Characters: {', '.join(characters)}
        Mood: {mood}
        
        Scene Description: {narration[:300]}...
        
        Style: Child-friendly, colorful, detailed, suitable for {age_group}-year-old children.
        Avoid scary or disturbing elements. Create a warm, engaging scene that captures the essence of the chapter.
        """
        
        try:
            # GPT-4o 모델을 사용하여 이미지 생성
            response = self.openai_client.images.generate(
                model="dall-e-3",
                prompt=image_prompt,
                size="1024x1024",
                quality="standard",
                n=1,
                response_format="b64_json"
            )
            
            # 이미지 정보 추출
            image_data = response.data[0]
            image_b64 = image_data.b64_json
            
            # 이미지 저장을 위한 디렉토리 생성
            story_images_dir = self.images_dir / self.current_story_id
            ensure_directory(story_images_dir)
            
            # 이미지 파일 경로 생성 (chapter_1.jpg, chapter_2.jpg, ...)
            image_filename = f"chapter_{chapter_number}.jpg"
            image_path = story_images_dir / image_filename
            
            # Base64 이미지 저장
            success = save_image_from_base64(image_b64, image_path)
            
            if success:
                # 이미지 정보 반환
                image_info = {
                    "chapter_number": chapter_number,
                    "chapter_title": chapter_title,
                    "file_path": str(image_path),
                    "prompt": image_prompt,
                    "timestamp": time.time()
                }
                
                logger.info(f"챕터 {chapter_number} 이미지 생성 및 저장 완료: {image_path}")
                return image_info
            else:
                logger.error(f"챕터 {chapter_number} 이미지 저장 실패")
                return None
                
        except Exception as e:
            logger.error(f"챕터 {chapter_number} 이미지 생성 중 오류 발생: {e}")
            return None
    
    def generate_audio_for_text(self, text: str, speaker_type: str = "narrator", 
                                filename: str = None) -> Optional[str]:
        """
        텍스트를 바탕으로 음성 오디오 생성
        
        Args:
            text: 변환할 텍스트
            speaker_type: 화자 유형 (narrator, male_adult, female_adult, male_child, female_child)
            filename: 저장할 파일 이름 (기본값: 자동 생성)
            
        Returns:
            Optional[str]: 생성된 오디오 파일의 경로 또는 None
        """
        if not self.elevenlabs_client:
            logger.error("ElevenLabs 클라이언트가 초기화되지 않았습니다.")
            return None
        
        if not self.audio_dir or not self.current_story_id:
            logger.error("오디오 저장 디렉토리 또는 스토리 ID가 설정되지 않았습니다.")
            return None
        
        # 음성 ID 결정 로직 수정
        voice_id_to_use = None
        speaker_name_lower = speaker_type.lower()

        # 1. 주인공 캐릭터이고 클론된 음성이 있는 경우
        if self.main_character_name and speaker_name_lower == self.main_character_name.lower() and self.child_voice_id:
            voice_id_to_use = self.child_voice_id
            logger.info(f"주인공 ({self.main_character_name}) 음성으로 클론된 ID ({self.child_voice_id}) 사용")
        
        # 2. 일반 캐릭터 음성 매핑 시도
        if not voice_id_to_use:
            voice_id_to_use = self.character_voices.get(speaker_name_lower)
            if voice_id_to_use:
                logger.info(f"캐릭터 '{speaker_type}'에 대해 매핑된 음성 ID ({voice_id_to_use}) 사용")

        # 3. 캐릭터 매핑 실패 시, 역할 기반 음성 매핑 (예: 'boy', 'girl', 'man', 'woman', 'narrator')
        if not voice_id_to_use:
            # speaker_type을 분석하여 일반적인 역할 추론 (간단한 예시)
            if "boy" in speaker_name_lower or "male child" in speaker_name_lower:
                voice_id_to_use = self.character_voices.get("male_child")
            elif "girl" in speaker_name_lower or "female child" in speaker_name_lower:
                voice_id_to_use = self.character_voices.get("female_child")
            elif "man" in speaker_name_lower or "male adult" in speaker_name_lower:
                voice_id_to_use = self.character_voices.get("male_adult")
            elif "woman" in speaker_name_lower or "female adult" in speaker_name_lower:
                voice_id_to_use = self.character_voices.get("female_adult")
            
            if voice_id_to_use:
                logger.info(f"캐릭터 '{speaker_type}'에 대해 역할 기반 음성 ID ({voice_id_to_use}) 사용")

        # 4. 모든 매핑 실패 시 내레이터 음성 또는 기본 음성 사용
        if not voice_id_to_use:
            voice_id_to_use = self.character_voices.get("narrator") # 기본 내레이터 음성
            logger.info(f"캐릭터 '{speaker_type}'에 대한 특정 음성 ID 없음. 기본 내레이터 음성 ({voice_id_to_use}) 사용")

        if not voice_id_to_use: # 내레이터 음성도 없는 극단적인 경우
            logger.error("사용 가능한 음성 ID가 없습니다. ElevenLabs 설정을 확인하세요.")
            return None

        # 오디오 저장을 위한 디렉토리 생성
        story_audio_dir = self.audio_dir / self.current_story_id
        ensure_directory(story_audio_dir)
        
        # 파일 이름 생성 (제공되지 않은 경우)
        if not filename:
            timestamp = int(time.time())
            filename = f"{speaker_type}_{timestamp}.mp3"
        elif not filename.endswith(".mp3"):
            filename = f"{filename}.mp3"
        
        # 출력 파일 경로
        output_path = story_audio_dir / filename
        
        try:
            # ElevenLabs API를 사용하여 음성 생성
            audio_stream = self.elevenlabs_client.generate(
                text=text,
                voice=self.voice_settings.get(voice_id_to_use, self.voice_settings["narrator"]),
                model=self.voice_model
            )
            
            # 스트리밍 데이터를 바이트로 변환
            audio_bytes = b"".join([chunk for chunk in audio_stream])

            # 파일 저장
            ensure_directory(output_path.parent)
            with open(output_path, "wb") as f:
                f.write(audio_bytes) # 바이트 데이터 직접 저장

            logger.info(f"음성 파일 저장 완료: {output_path} (Speaker: {speaker_type}, Voice ID: {voice_id_to_use})")
            return str(output_path)
                
        except Exception as e:
            logger.error(f"음성 생성 중 오류 발생: {e}")
            return None
    
    def generate_chapter_audio(self, chapter_data: Dict) -> Dict[str, str]:
        """
        챕터의 내레이션과 대화를 오디오로 변환
        
        Args:
            chapter_data: 챕터 데이터
            
        Returns:
            Dict[str, str]: 생성된 오디오 파일 경로 매핑
        """
        # 결과 저장 딕셔너리
        audio_files = {}
        
        # 챕터 정보 추출
        chapter_number = chapter_data.get("chapter_number", 0)
        narration = chapter_data.get("narration", "")
        dialogues = chapter_data.get("dialogues", [])
        
        # 내레이션 오디오 생성
        if narration:
            narration_filename = f"chapter_{chapter_number}_narration.mp3"
            narration_path = self.generate_audio_for_text(narration, "narrator", narration_filename)
            if narration_path:
                audio_files["narration"] = narration_path
        
        # 대화 오디오 생성
        for idx, dialogue in enumerate(dialogues):
            speaker = dialogue.get("speaker", "unknown")
            text = dialogue.get("text", "")
            
            if not text:
                continue
                
            # 화자 유형 결정 (기본값: 성인 남성)
            speaker_type = "male_adult"
            
            # 화자 이름에 따른 간단한 음성 할당 (더 복잡한 로직으로 확장 가능)
            speaker_lower = speaker.lower()
            if "아빠" in speaker_lower or "아버지" in speaker_lower or "할아버지" in speaker_lower:
                speaker_type = "male_adult"
            elif "엄마" in speaker_lower or "어머니" in speaker_lower or "할머니" in speaker_lower:
                speaker_type = "female_adult"
            elif "소년" in speaker_lower or "남자아이" in speaker_lower:
                speaker_type = "male_child"
            elif "소녀" in speaker_lower or "여자아이" in speaker_lower:
                speaker_type = "female_child"
            
            # 오디오 파일 이름 생성
            dialogue_filename = f"chapter_{chapter_number}_dialogue_{idx+1}_{speaker}.mp3"
            
            # 오디오 생성
            dialogue_path = self.generate_audio_for_text(text, speaker_type, dialogue_filename)
            if dialogue_path:
                audio_files[f"dialogue_{idx+1}_{speaker}"] = dialogue_path
        
        return audio_files 