import openai
from openai import OpenAI
from typing import List, Dict, Optional, Tuple
from dotenv import load_dotenv
import os
import json
import base64
from PIL import Image
import io
import requests
import numpy as np
from pathlib import Path
import elevenlabs
from elevenlabs import generate, save, set_api_key, voices

# 환경 변수 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
dotenv_path = os.path.join(project_root, '.env')
load_dotenv(dotenv_path=dotenv_path)
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
elevenlabs_api_key = os.getenv('ELEVENLABS_API_KEY')
if not os.getenv('OPENAI_API_KEY'):
    print(f"Warning: OPENAI_API_KEY environment variable not found. Looking for .env file at: {dotenv_path}")
if elevenlabs_api_key:
    elevenlabs.set_api_key(elevenlabs_api_key)
else:
    print(f"Warning: ELEVENLABS_API_KEY environment variable not found. Looking for .env file at: {dotenv_path}")

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
    """
    
    def __init__(self, output_dir: str = "output"):
        """
        챗봇 초기화 및 기본 속성 설정
        
        Args:
            output_dir (str): 생성된 파일들을 저장할 디렉토리 경로
        """
        self.story_outline = None
        self.generated_images = []
        self.narration_audio = None
        self.output_dir = Path(output_dir)
        self.voice_id = None
        self.target_age = None
        self.detailed_story = None
        
        # 음성 클론 관련 속성
        self.child_voice_id = None  # 아이의 클론된 음성 ID
        self.main_character_name = None  # 주인공 캐릭터 이름
        self.has_cloned_voice = False  # 음성 클론 존재 여부
        
        # 출력 디렉토리 생성
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "images").mkdir(exist_ok=True)
        (self.output_dir / "audio").mkdir(exist_ok=True)
        
        # 프롬프트 로드
        self.load_prompts()
    
    def load_prompts(self):
        """프롬프트 파일을 로드하는 함수"""
        try:
            # 먼저 전용 프롬프트 파일 시도
            prompts_path = os.path.join(project_root, 'data', 'prompts', 'chatbot_b_prompts.json')
            if os.path.exists(prompts_path):
                with open(prompts_path, 'r', encoding='utf-8') as f:
                    self.prompts = json.load(f)
                print(f"꼬기 전용 프롬프트 파일 로드 성공: {prompts_path}")
                return
                
            # 전용 프롬프트 파일이 없으면 공용 프롬프트 파일에서 로드
            with open(os.path.join(project_root, 'data', 'prompts', 'chatbot_prompts.json'), 'r', encoding='utf-8') as f:
                prompts = json.load(f)
                self.prompts = prompts['chatbot_b']
                print("공용 프롬프트 파일에서 챗봇 B 프롬프트 로드")
        except Exception as e:
            print(f"프롬프트 로드 중 오류 발생: {str(e)}")
            # 기본 프롬프트 설정
            self.prompts = {
                "system_message_template": "당신은 대략적인 이야기를 바탕으로 상세한 이야기를 만들어내고, 그 상세한 이야기를 통해 동화 일러스트와 내레이션을 생성하는 전문가입니다.",
                "image_generation_template": "주제: {theme}\n캐릭터: {characters}\n배경: {setting}\n스타일: {style}",
                "narration_template": "줄거리: {plot_summary}\n교육적 가치: {educational_value}\n대상 연령: {target_age}"
            }
    
    def set_story_outline(self, story_outline: Dict[str, str]):
        """
        동화 줄거리 정보를 설정하는 함수
        
        Args:
            story_outline (Dict[str, str]): 동화 줄거리 정보
                - theme: 주제
                - characters: 주요 캐릭터
                - setting: 배경 설정
                - plot_summary: 간략한 줄거리
                - educational_value: 교육적 가치
                - target_age: 적합한 연령대
        """
        self.story_outline = story_outline
        
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
    
    def generate_image(self, scene_description: str, style: str = "아기자기한 동화 스타일") -> str:
        """
        장면 설명을 바탕으로 일러스트를 생성하는 함수
        
        Args:
            scene_description (str): 장면 설명
            style (str): 일러스트 스타일
            
        Returns:
            str: 생성된 이미지 파일 경로
            
        Note:
            - DALL-E 3를 사용하여 고품질 일러스트 생성
            - 아이의 연령대에 맞는 스타일로 생성
            - 안전하고 교육적인 이미지 생성 보장
        """
        try:
            # 이미지 생성 프롬프트 생성
            prompt = self.prompts["image_generation_template"].format(
                theme=self.story_outline["theme"],
                characters=self.story_outline["characters"],
                setting=self.story_outline["setting"],
                style=style
            )
            
            # DALL-E 3를 사용하여 이미지 생성
            response = client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                size="1024x1024",
                quality="standard",
                n=1,
                response_format="b64_json"
            )
            
            # Base64 이미지 데이터를 파일로 저장
            image_data = base64.b64decode(response.data[0].b64_json)
            image = Image.open(io.BytesIO(image_data))
            
            # 파일명 생성 및 저장
            file_name = f"scene_{len(self.generated_images) + 1}.png"
            file_path = self.output_dir / "images" / file_name
            image.save(file_path)
            
            # 생성된 이미지 경로 저장
            self.generated_images.append(str(file_path))
            
            return str(file_path)
            
        except Exception as e:
            print(f"이미지 생성 중 오류 발생: {str(e)}")
            return None
    
    def generate_narration(self) -> str:
        """
        동화 줄거리를 바탕으로 내레이션을 생성하는 함수
        
        Returns:
            str: 생성된 내레이션 오디오 파일 경로
            
        Note:
            - GPT-4o를 사용하여 자연스러운 내레이션 텍스트 생성
            - 아이의 연령대에 맞는 언어와 톤 사용
            - 교육적 가치를 담은 내레이션 생성
        """
        try:
            # 내레이션 생성 프롬프트 생성
            prompt = self.prompts["narration_template"].format(
                plot_summary=self.story_outline["plot_summary"],
                educational_value=self.story_outline["educational_value"],
                target_age=self.story_outline["target_age"]
            )
            
            # GPT-4o를 사용하여 내레이션 텍스트 생성
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": self.prompts["system_message_template"]},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            narration_text = response.choices[0].message.content
            
            # 내레이션 텍스트를 오디오로 변환
            audio_response = client.audio.speech.create(
                model="tts-1",
                voice="alloy",
                input=narration_text
            )
            
            # 오디오 파일 저장
            file_name = "narration.mp3"
            file_path = self.output_dir / "audio" / file_name
            
            with open(file_path, "wb") as f:
                f.write(audio_response.content)
            
            self.narration_audio = str(file_path)
            return str(file_path)
            
        except Exception as e:
            print(f"내레이션 생성 중 오류 발생: {str(e)}")
            return None
    
    def generate_story(self) -> Tuple[List[str], str]:
        """
        동화 줄거리를 바탕으로 일러스트와 내레이션을 생성하는 함수
        
        Returns:
            Tuple[List[str], str]: (생성된 이미지 파일 경로 목록, 내레이션 오디오 파일 경로)
            
        Note:
            - 줄거리를 여러 장면으로 나누어 일러스트 생성
            - 각 장면에 맞는 내레이션 생성
            - 모든 생성물을 지정된 디렉토리에 저장
        """
        if not self.story_outline:
            raise ValueError("동화 줄거리가 설정되지 않았습니다.")
        
        # 줄거리를 장면으로 나누기
        scenes = self._split_plot_into_scenes()
        
        # 각 장면에 대한 일러스트 생성
        for scene in scenes:
            self.generate_image(scene)
        
        # 사용자 음성을 주인공의 목소리로 사용
        user_voice_path = os.path.join(self.output_dir, "user_voice.wav")
        if os.path.exists(user_voice_path):
            print(f"사용자 음성을 주인공의 목소리로 사용합니다: {user_voice_path}")
            # 주인공의 대사에 사용자 음성 사용 로직 추가
            # ...
        else:
            print("사용자 음성을 찾을 수 없습니다. 기본 음성을 사용합니다.")

        # 내레이션 생성
        narration_path = self.generate_narration()
        
        return self.generated_images, narration_path
    
    def _split_plot_into_scenes(self) -> List[str]:
        """
        줄거리를 여러 장면으로 나누는 함수
        
        Returns:
            List[str]: 장면 설명 목록
            
        Note:
            - 줄거리의 주요 순간들을 장면으로 분리
            - 각 장면은 일러스트 생성에 적합한 설명을 포함
        """
        try:
            # 장면 분리 템플릿 준비
            scene_division_prompt = self.prompts.get("scene_division_template", """
                다음 줄거리를 3-5개의 장면으로 나누어주세요:
                {plot_summary}
                
                각 장면은 일러스트 생성에 적합한 구체적인 설명을 포함해야 합니다.
            """).format(
                plot_summary=self.story_outline['plot_summary']
            )
            
            # GPT-4o를 사용하여 줄거리를 장면으로 분리
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "당신은 동화 줄거리를 장면으로 나누는 전문가입니다."},
                    {"role": "user", "content": scene_division_prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            # 응답을 장면 목록으로 변환
            scenes = response.choices[0].message.content.split("\n\n")
            return [scene.strip() for scene in scenes if scene.strip()]
            
        except Exception as e:
            print(f"장면 분리 중 오류 발생: {str(e)}")
            # 기본 장면 반환
            return [self.story_outline["plot_summary"]]
    
    def generate_detailed_story(self) -> Dict:
        """
        동화 줄거리를 바탕으로 상세 이야기를 생성하는 함수
        
        Returns:
            Dict: 상세 이야기 정보
                - title: 제목
                - target_age: 적합한 연령대
                - characters: 등장인물 정보 (is_main_character 태그 추가)
                - scenes: 장면 정보
                - educational_value: 교육적 가치
                - theme: 주제
                
        Note:
            - 줄거리를 바탕으로 전체 동화 구성
            - 각 장면에 대한 상세 설명 제공
            - 교육적 가치와 주제 강화
            - 주인공 캐릭터 표시 (아이의 음성 사용)
        """
        if not self.story_outline:
            raise ValueError("동화 줄거리가 설정되지 않았습니다.")
            
        try:
            # 시스템 메시지 템플릿 준비
            system_message = self.prompts["system_message_template"].format(
                age=self.target_age or self.story_outline.get("target_age", 5),
                chatbot_name="꼬기"
            )
            
            # 상세 스토리 템플릿 준비
            detailed_story_prompt = self.prompts.get("detailed_story_template", """
                다음 줄거리를 바탕으로 {target_age}세 아이에게 적합한 동화를 작성해주세요:
                
                줄거리: {plot_summary}
                주제: {theme}
                교육적 가치: {educational_value}
                
                다음 형식으로 JSON 응답을 제공해주세요:
                {{
                    "title": "동화 제목",
                    "target_age": {target_age},
                    "characters": [
                        {{"name": "캐릭터 이름", "description": "캐릭터 설명", "role": "역할"}}
                    ],
                    "scenes": [
                        {{
                            "title": "장면 제목",
                            "description": "장면 설명",
                            "narration": "내레이션 텍스트",
                            "dialogues": [
                                {{"character": "캐릭터 이름", "text": "대사 내용"}}
                            ]
                        }}
                    ],
                    "educational_value": "교육적 가치 상세 설명",
                    "theme": "주제 상세 설명"
                }}
                
                주의사항:
                1. 가장 중요한 주인공 캐릭터를 첫 번째로 배치해주세요. 이 캐릭터의 목소리가 아이의 목소리로 처리됩니다.
                2. 내레이션과 대사를 명확히 구분해주세요.
                3. 각 장면에서 주인공의 대사가 적절히 포함되도록 해주세요.
            """).format(
                plot_summary=self.story_outline.get('plot_summary', ''),
                theme=self.story_outline.get('theme', '모험과 우정'),
                educational_value=self.story_outline.get('educational_value', '협동과 문제 해결 능력'),
                target_age=self.target_age or self.story_outline.get('target_age', 5)
            )
            
            # 이야기 생성 요청
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": detailed_story_prompt}
                ],
                temperature=0.7,
                max_tokens=2000,
                response_format={"type": "json_object"}
            )
            
            # JSON 응답 파싱
            detailed_story = json.loads(response.choices[0].message.content)
            
            # 주인공 캐릭터 표시 (첫 번째 캐릭터를 주인공으로 가정)
            if detailed_story.get("characters") and len(detailed_story["characters"]) > 0:
                # 첫 번째 캐릭터에 is_main_character = True 추가
                detailed_story["characters"][0]["is_main_character"] = True
                
                # 나머지 캐릭터는 is_main_character = False
                for i in range(1, len(detailed_story["characters"])):
                    detailed_story["characters"][i]["is_main_character"] = False
                
                # 주인공 캐릭터 이름 저장
                self.main_character_name = detailed_story["characters"][0]["name"]
                print(f"주인공 캐릭터: {self.main_character_name}")
            
            self.detailed_story = detailed_story
            
            return detailed_story
            
        except Exception as e:
            print(f"상세 이야기 생성 중 오류 발생: {str(e)}")
            # 기본 응답 생성
            default_story = {
                "title": self.story_outline.get("theme", "이야기"),
                "target_age": self.target_age or self.story_outline.get("target_age", 5),
                "characters": [{"name": "주인공", "description": "이야기의 주인공", "role": "주인공", "is_main_character": True}],
                "scenes": [{"title": "장면 1", "description": self.story_outline.get("plot_summary", ""), "narration": self.story_outline.get("plot_summary", ""), "dialogues": []}],
                "educational_value": self.story_outline.get("educational_value", ""),
                "theme": self.story_outline.get("theme", "")
            }
            self.detailed_story = default_story
            self.main_character_name = "주인공"
            return default_story
    
    def generate_illustrations(self) -> List[str]:
        """
        상세 이야기를 바탕으로 장면별 일러스트를 생성하는 함수
        
        Returns:
            List[str]: 생성된 이미지 파일 경로 목록
            
        Note:
            - 각 장면에 대한 DALL-E 3 일러스트 생성
            - 아이의 연령에 맞는 스타일로 생성
            - 모든 일러스트를 일관된 스타일로 생성
        """
        if not self.detailed_story:
            raise ValueError("상세 이야기가 먼저 생성되어야 합니다.")
            
        # 이미지를 저장할 디렉토리 확인
        image_dir = self.output_dir / "images"
        image_dir.mkdir(exist_ok=True)
        
        # 이미지 생성 결과 저장
        generated_images = []
        
        try:
            # 각 장면에 대한 이미지 생성
            for i, scene in enumerate(self.detailed_story["scenes"]):
                # 이미지 생성 프롬프트 준비
                prompt = f"""
                장면: {scene['description']}
                
                {self.detailed_story['target_age']}세 아이를 위한 동화책 삽화를 생성해주세요.
                제목: {self.detailed_story['title']}
                캐릭터: {', '.join([char['name'] for char in self.detailed_story['characters']])}
                
                스타일: 아기자기하고 밝은 색감의 동화책 일러스트, 선명하고 단순한 형태, 아이가 이해하기 쉬운 구도
                """
                
                # DALL-E 3를 사용하여 이미지 생성
                response = client.images.generate(
                    model="dall-e-3",
                    prompt=prompt,
                    size="1024x1024",
                    quality="standard",
                    n=1,
                    response_format="b64_json"
                )
                
                # 응답에서 이미지 데이터 추출
                image_data = base64.b64decode(response.data[0].b64_json)
                
                # 이미지 저장
                image_path = image_dir / f"scene_{i+1}.png"
                with open(image_path, "wb") as f:
                    f.write(image_data)
                
                # 생성된 이미지 경로 저장
                generated_images.append(str(image_path))
                print(f"장면 {i+1}/{len(self.detailed_story['scenes'])} 일러스트 생성 완료")
            
            # 클래스 변수 업데이트
            self.generated_images = generated_images
            
            return generated_images
            
        except Exception as e:
            print(f"일러스트 생성 중 오류 발생: {str(e)}")
            return []
    
    def generate_voice(self) -> Dict[str, str]:
        """
        상세 이야기를 바탕으로 내레이션과 대사를 음성으로 생성하는 함수
        
        Returns:
            Dict[str, str]: 생성된 오디오 파일 경로 딕셔너리
                - narration: 내레이션 오디오 파일 경로
                - characters: 캐릭터별 대사 오디오 파일 경로
                
        Note:
            - 주인공 캐릭터는 아이의 음성 클로닝 사용 (가능한 경우)
            - 조연 캐릭터는 일반 TTS 음성 사용
            - 내레이션은 별도의 내레이터 음성 사용
        """
        if not self.detailed_story:
            raise ValueError("상세 이야기가 먼저 생성되어야 합니다.")
            
        # 오디오를 저장할 디렉토리 확인
        audio_dir = self.output_dir / "audio"
        audio_dir.mkdir(exist_ok=True)
        
        # 오디오 생성 결과 저장
        generated_audio = {
            "narration": {},
            "characters": {}
        }
        
        try:
            # ElevenLabs 사용 가능 여부 확인
            use_elevenlabs = elevenlabs_api_key is not None
            
            # 클론된 음성 사용 가능 여부 확인
            has_cloned_voice = self.check_cloned_voice()
            
            # 내레이터 음성 설정
            narrator_voice = "bella" if use_elevenlabs else "nova"
            
            # 각 장면에 대한 내레이션 생성
            for i, scene in enumerate(self.detailed_story["scenes"]):
                print(f"장면 {i+1}/{len(self.detailed_story['scenes'])} 음성 생성 중...")
                
                # 내레이션 텍스트 준비
                narration_text = scene["narration"]
                
                # 내레이션 오디오 생성
                if use_elevenlabs:
                    # ElevenLabs로 음성 생성
                    narration_audio = elevenlabs.generate(
                        text=narration_text,
                        voice=narrator_voice,
                        model="eleven_multilingual_v2"  # 다국어 모델 (한국어 지원)
                    )
                else:
                    # OpenAI TTS 사용
                    narration_response = client.audio.speech.create(
                        model="tts-1",
                        voice="nova",
                        input=narration_text
                    )
                    narration_audio = narration_response.content
                
                # 내레이션 오디오 저장
                narration_path = audio_dir / f"scene_{i+1}_narration.mp3"
                with open(narration_path, "wb") as f:
                    f.write(narration_audio)
                
                generated_audio["narration"][f"scene_{i+1}"] = str(narration_path)
                
                # 각 대사에 대한 오디오 생성
                for j, dialogue in enumerate(scene.get("dialogues", [])):
                    character_name = dialogue["character"]
                    dialogue_text = dialogue["text"]
                    
                    # 주인공 캐릭터 여부 확인
                    is_main_character = character_name == self.main_character_name
                    
                    # 캐릭터별 음성 설정
                    if is_main_character and has_cloned_voice and use_elevenlabs:
                        # 주인공은 클론된 음성 사용
                        voice = self.child_voice_id or "Josh"  # 클론된 음성 ID 또는 기본값
                        print(f"주인공 '{character_name}'에 아이의 클론된 음성을 적용합니다.")
                    else:
                        # 조연은 일반 음성 사용
                        if character_name not in generated_audio["characters"]:
                            # 캐릭터에 따라 다른 목소리 할당
                            available_voices = ["Antoni", "Josh", "Rachel", "Domi", "Bella", "Arnold"]
                            voice = available_voices[hash(character_name) % len(available_voices)]
                            generated_audio["characters"][character_name] = {"voice": voice}
                        else:
                            voice = generated_audio["characters"][character_name]["voice"]
                    
                    # 대사 오디오 생성
                    if use_elevenlabs:
                        # ElevenLabs로 음성 생성
                        dialogue_audio = elevenlabs.generate(
                            text=dialogue_text,
                            voice=voice,
                            model="eleven_multilingual_v2"
                        )
                    else:
                        # OpenAI TTS 사용 (캐릭터별 다른 음성 선택)
                        voice_name = "alloy"  # 기본값
                        if is_main_character:
                            voice_name = "shimmer"  # 주인공 음성
                        elif hash(character_name) % 3 == 0:
                            voice_name = "echo"
                        elif hash(character_name) % 3 == 1:
                            voice_name = "fable"
                        
                        dialogue_response = client.audio.speech.create(
                            model="tts-1",
                            voice=voice_name,
                            input=dialogue_text
                        )
                        dialogue_audio = dialogue_response.content
                    
                    # 대사 오디오 저장
                    dialogue_path = audio_dir / f"scene_{i+1}_{character_name}_{j+1}.mp3"
                    with open(dialogue_path, "wb") as f:
                        f.write(dialogue_audio)
                    
                    if "dialogues" not in generated_audio["characters"]:
                        if character_name not in generated_audio["characters"]:
                            generated_audio["characters"][character_name] = {}
                        
                        if "dialogues" not in generated_audio["characters"][character_name]:
                            generated_audio["characters"][character_name]["dialogues"] = {}
                    
                    generated_audio["characters"][character_name]["dialogues"][f"scene_{i+1}_dialogue_{j+1}"] = str(dialogue_path)
            
            # 최종 내레이션 경로 저장
            self.narration_audio = generated_audio
            
            print("모든 음성 생성이 완료되었습니다.")
            return generated_audio
            
        except Exception as e:
            print(f"음성 생성 중 오류 발생: {str(e)}")
            return generated_audio
    
    def save_story_data(self, file_path: str):
        """
        생성된 스토리 데이터를 파일로 저장하는 함수
        
        Args:
            file_path (str): 저장할 파일 경로
            
        Note:
            - 줄거리, 이미지 경로, 내레이션 경로 등을 JSON 형식으로 저장
            - 나중에 스토리 데이터를 불러와서 재사용 가능
        """
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "story_outline": self.story_outline,
                    "detailed_story": self.detailed_story,
                    "generated_images": self.generated_images,
                    "narration_audio": self.narration_audio
                }, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"스토리 데이터 저장 중 오류 발생: {str(e)}")
    
    def load_story_data(self, file_path: str):
        """
        저장된 스토리 데이터를 불러오는 함수
        
        Args:
            file_path (str): 불러올 파일 경로
            
        Note:
            - 저장된 줄거리, 이미지 경로, 내레이션 경로 등을 복원
            - 이전에 생성한 스토리를 이어서 작업 가능
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.story_outline = data["story_outline"]
                self.detailed_story = data.get("detailed_story")
                self.generated_images = data["generated_images"]
                self.narration_audio = data["narration_audio"]
        except Exception as e:
            print(f"스토리 데이터 불러오기 중 오류 발생: {str(e)}")
    
    def get_story_preview(self) -> Dict[str, str]:
        """
        생성된 스토리의 미리보기 정보를 반환하는 함수
        
        Returns:
            Dict[str, str]: 스토리 미리보기 정보
                - title: 제목
                - summary: 요약
                - image_count: 이미지 수
                - duration: 예상 재생 시간
                
        Note:
            - 스토리의 주요 정보를 간단히 보여줌
            - 사용자에게 스토리 개요 제공
        """
        try:
            # GPT-4o를 사용하여 스토리 미리보기 생성
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "당신은 동화 미리보기를 생성하는 전문가입니다."},
                    {"role": "user", "content": f"""
                    다음 동화 정보를 바탕으로 미리보기를 생성해주세요:
                    주제: {self.story_outline['theme']}
                    줄거리: {self.story_outline['plot_summary']}
                    교육적 가치: {self.story_outline['educational_value']}
                    """}
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            preview_text = response.choices[0].message.content
            
            return {
                "title": self.detailed_story["title"] if self.detailed_story else self.story_outline["theme"],
                "summary": preview_text,
                "image_count": len(self.generated_images),
                "duration": f"{len(self.generated_images) * 30}초"  # 각 이미지당 30초 가정
            }
            
        except Exception as e:
            print(f"스토리 미리보기 생성 중 오류 발생: {str(e)}")
            return {
                "title": self.detailed_story["title"] if self.detailed_story else self.story_outline["theme"],
                "summary": self.story_outline["plot_summary"],
                "image_count": len(self.generated_images),
                "duration": "알 수 없음"
            }
    
    def set_cloned_voice(self, voice_id: str, main_character_name: Optional[str] = None):
        """
        아이의 클론된 음성 ID 설정
        
        Args:
            voice_id (str): ElevenLabs 음성 ID
            main_character_name (str, optional): 주인공 캐릭터 이름. 없으면 자동 추출
        """
        self.child_voice_id = voice_id
        self.has_cloned_voice = True
        
        # 주인공 캐릭터 이름 설정
        if main_character_name:
            self.main_character_name = main_character_name
        elif self.detailed_story and self.detailed_story.get("characters"):
            # 첫 번째 캐릭터를 주인공으로 가정
            self.main_character_name = self.detailed_story["characters"][0]["name"]
    
    def check_cloned_voice(self) -> bool:
        """
        클론된 음성 사용 가능 여부 확인
        
        Returns:
            bool: 클론된 음성 사용 가능 여부
        """
        # ElevenLabs API 키 확인
        if not elevenlabs_api_key:
            return False
            
        # 음성 ID 확인
        if not self.child_voice_id:
            # 클론된 음성 파일 확인
            voice_file = self.output_dir / "user_voice.wav"
            if voice_file.exists():
                # 음성 파일이 존재하면 사용 가능으로 간주
                return True
                
        return self.has_cloned_voice 

    def create_storybook_audio(self) -> str:
        """
        내레이션과 대사를 조합하여 완성된 스토리북 오디오 생성
        
        Returns:
            str: 생성된 스토리북 오디오 파일 경로
            
        Note:
            - 생성된 내레이션과 대사 오디오 파일을 장면별로 병합
            - 전체 스토리를 하나의 오디오 파일로 제공
        """
        if not self.narration_audio:
            raise ValueError("음성 파일이 먼저 생성되어야 합니다.")
            
        try:
            # 필요한 패키지 설치 확인
            try:
                from pydub import AudioSegment
            except ImportError:
                print("pydub 패키지가 설치되어 있지 않습니다. 오디오 병합 기능을 사용할 수 없습니다.")
                return None
            
            # 스토리북 오디오 파일 경로
            storybook_audio_path = self.output_dir / "audio" / f"{self.detailed_story['title']}.mp3"
            
            # 장면별 오디오 병합
            combined_audio = AudioSegment.empty()
            
            # 각 장면마다 내레이션과 대사 순서대로 병합
            for i, scene in enumerate(self.detailed_story["scenes"]):
                scene_index = i + 1
                
                # 1초 공백 추가
                silence = AudioSegment.silent(duration=1000)
                combined_audio += silence
                
                # 내레이션 추가
                narration_key = f"scene_{scene_index}"
                if narration_key in self.narration_audio["narration"]:
                    narration_path = self.narration_audio["narration"][narration_key]
                    narration_audio = AudioSegment.from_file(narration_path)
                    combined_audio += narration_audio
                
                # 0.5초 공백 추가
                silence = AudioSegment.silent(duration=500)
                combined_audio += silence
                
                # 대사 추가 (원래 순서대로)
                for j, dialogue in enumerate(scene.get("dialogues", [])):
                    character_name = dialogue["character"]
                    dialogue_key = f"scene_{scene_index}_dialogue_{j+1}"
                    
                    # 해당 캐릭터의 대사 찾기
                    if character_name in self.narration_audio["characters"]:
                        character_data = self.narration_audio["characters"][character_name]
                        if "dialogues" in character_data and dialogue_key in character_data["dialogues"]:
                            dialogue_path = character_data["dialogues"][dialogue_key]
                            dialogue_audio = AudioSegment.from_file(dialogue_path)
                            combined_audio += dialogue_audio
                            
                            # 0.3초 공백 추가
                            silence = AudioSegment.silent(duration=300)
                            combined_audio += silence
            
            # 오디오 파일 저장
            combined_audio.export(str(storybook_audio_path), format="mp3")
            
            print(f"스토리북 오디오가 생성되었습니다: {storybook_audio_path}")
            return str(storybook_audio_path)
            
        except Exception as e:
            print(f"스토리북 오디오 생성 중 오류 발생: {str(e)}")
            return None
    
    def export_storybook(self, output_path: Optional[str] = None) -> Dict:
        """
        완성된 스토리북을 묶어서 내보내기
        
        Args:
            output_path (str, optional): 내보낼 경로. 지정하지 않으면 기본 output 폴더 사용
            
        Returns:
            Dict: 스토리북 정보
                - title: 제목
                - story_data: 스토리 데이터
                - images: 이미지 파일 경로 목록
                - audio: 오디오 파일 경로
                - output_path: 내보내기 경로
        """
        if not self.detailed_story:
            raise ValueError("스토리가 먼저 생성되어야 합니다.")
            
        try:
            # 내보내기 경로 설정
            if output_path:
                export_dir = Path(output_path)
            else:
                export_dir = self.output_dir / "storybooks" / self.detailed_story["title"].replace(" ", "_")
            
            # 디렉토리 생성
            export_dir.mkdir(parents=True, exist_ok=True)
            
            # 스토리 데이터 저장
            story_json_path = export_dir / "story.json"
            with open(story_json_path, "w", encoding="utf-8") as f:
                json.dump(self.detailed_story, f, ensure_ascii=False, indent=2)
            
            # 이미지 파일 복사
            images_dir = export_dir / "images"
            images_dir.mkdir(exist_ok=True)
            
            copied_images = []
            for i, img_path in enumerate(self.generated_images):
                src_path = Path(img_path)
                if src_path.exists():
                    dst_path = images_dir / f"scene_{i+1}.png"
                    # 이미지 파일 복사
                    import shutil
                    shutil.copy2(src_path, dst_path)
                    copied_images.append(str(dst_path))
            
            # 스토리북 오디오 생성 (아직 생성되지 않은 경우)
            audio_path = None
            storybook_audio_path = self.output_dir / "audio" / f"{self.detailed_story['title']}.mp3"
            
            if storybook_audio_path.exists():
                # 이미 생성된 오디오 파일 복사
                dst_audio_path = export_dir / f"{self.detailed_story['title']}.mp3"
                import shutil
                shutil.copy2(storybook_audio_path, dst_audio_path)
                audio_path = str(dst_audio_path)
            else:
                # 새로 오디오 파일 생성
                try:
                    audio_path = self.create_storybook_audio()
                    if audio_path:
                        dst_audio_path = export_dir / f"{self.detailed_story['title']}.mp3"
                        import shutil
                        shutil.copy2(audio_path, dst_audio_path)
                        audio_path = str(dst_audio_path)
                except Exception as audio_err:
                    print(f"스토리북 오디오 생성 실패: {audio_err}")
            
            # 결과 정보 반환
            return {
                "title": self.detailed_story["title"],
                "story_data": str(story_json_path),
                "images": copied_images,
                "audio": audio_path,
                "output_path": str(export_dir)
            }
            
        except Exception as e:
            print(f"스토리북 내보내기 중 오류 발생: {str(e)}")
            return {
                "title": self.detailed_story["title"],
                "error": str(e)
            }
    
    def create_storybook(self, story_outline: Optional[Dict] = None, target_age: Optional[int] = None) -> Dict:
        """
        줄거리를 바탕으로, 스토리, 삽화, 음성을 포함한 전체 스토리북 생성
        
        Args:
            story_outline (Dict, optional): 스토리 아웃라인. 없으면 이미 설정된 아웃라인 사용
            target_age (int, optional): 대상 연령. 없으면 이미 설정된 연령 사용
            
        Returns:
            Dict: 스토리북 생성 결과
                - success: 성공 여부
                - title: 스토리북 제목
                - images: 생성된 이미지 경로 목록
                - audio: 생성된 오디오 경로
                - story_data: 스토리 데이터 경로
                - output_path: 스토리북 출력 경로
                - error: 오류 메시지 (실패 시)
        """
        try:
            # 1. 스토리 아웃라인 설정
            if story_outline:
                self.set_story_outline(story_outline)
            
            if not self.story_outline:
                return {"success": False, "error": "스토리 아웃라인이 설정되지 않았습니다."}
            
            # 2. 대상 연령 설정
            if target_age:
                self.set_target_age(target_age)
            
            print("1. 스토리 아웃라인 처리 완료")
            
            # 3. 상세 스토리 생성
            detailed_story = self.generate_detailed_story()
            if not detailed_story:
                return {"success": False, "error": "상세 스토리 생성에 실패했습니다."}
            
            print(f"2. 상세 스토리 생성 완료: {detailed_story['title']}")
            
            # 4. 삽화 생성
            images = self.generate_illustrations()
            if not images:
                return {"success": False, "error": "삽화 생성에 실패했습니다."}
            
            print(f"3. 삽화 생성 완료: {len(images)}개")
            
            # 5. 음성 생성
            audio = self.generate_voice()
            if not audio:
                return {"success": False, "error": "음성 생성에 실패했습니다."}
            
            print("4. 음성 생성 완료")
            
            # 6. 스토리북 내보내기
            export_result = self.export_storybook()
            if "error" in export_result:
                return {"success": False, "error": export_result["error"]}
            
            print(f"5. 스토리북 내보내기 완료: {export_result['output_path']}")
            
            # 7. 결과 반환
            return {
                "success": True,
                "title": detailed_story["title"],
                "images": images,
                "audio": export_result.get("audio"),
                "story_data": export_result.get("story_data"),
                "output_path": export_result.get("output_path")
            }
            
        except Exception as e:
            print(f"스토리북 생성 중 오류 발생: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            } 