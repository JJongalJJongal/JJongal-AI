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
from .rag_system import RAGSystem

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
        rag_system (RAGSystem): RAG 시스템 인스턴스
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
        
        # RAG 시스템 초기화
        self.rag_system = RAGSystem()
    
    def load_prompts(self):
        """JSON 파일에서 프롬프트를 로드하는 함수"""
        try:
            prompts_path = os.path.join('data', 'prompts', 'chatbot_b_prompts.json')
            with open(prompts_path, 'r', encoding='utf-8') as f:
                self.prompts = json.load(f)
        except Exception as e:
            print(f"프롬프트 로드 중 오류 발생: {str(e)}")
            # 기본 프롬프트 설정
            self.prompts = {
                "system": {
                    "role": [
                        "너는 꼬기라는 이름을 가진 챗봇이야. 너는 부기 (chatbot_a) 와 대화를 하면서 동화를 만들어줄거야.",
                        "너는 부기 (chatbot_a) 가 만들어 준 대략적인 동화 스토리를 통해 상세한 스토리를 만들어줄거야.",
                        "너는 상세한 스토리를 만들고 그 상세한 스토리를 바탕으로 동화 이미지와 내레이션, 동화 인물의 대사를 만들어줄거야.",
                        "너는 동화 이미지를 만들 때 아이들의 연령대, 관심사에 맞게 이미지를 만들어야 해. 예를 들어, 공룡을 좋아하는 5세 아이에게는 귀엽고 친근한 공룡 이미지를 제공해.",
                        "너는 동화 내래이션을 만들 때 동화의 스토리, 이미지에 맞게 내레이션을 만들어야 해. 내레이션은 아이들이 쉽게 이해할 수 있는 언어로 구성해야 해.",
                        "너는 동화 인물의 대사를 만들 때 동화 인물의 성격, 감정에 맞게 대사를 작성해야 해. 대사는 아이들이 공감할 수 있도록 감정이 풍부해야 해."
                    ],
                    "instructions": [
                        "아이들의 연령대와 관심사를 반영하여 이야기를 더욱 흥미롭게 만들어야 해.",
                        "아이들이 참여할 수 있도록 질문을 던지거나 상상력을 자극하는 요소를 추가해.",
                        "명확하고 간결한 지침을 통해 챗봇이 일관된 이야기를 생성할 수 있도록 해.",
                        "아이들이 쉽게 이해할 수 있는 언어를 사용하고, 이야기의 흐름을 자연스럽게 이어가야 해."
                    ]
                }
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
            prompt = self.prompts["image_generation_prompt_template"].format(
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
            prompt = self.prompts["narration_generation_prompt_template"].format(
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
            # GPT-4o를 사용하여 줄거리를 장면으로 분리
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "당신은 동화 줄거리를 장면으로 나누는 전문가입니다."},
                    {"role": "user", "content": f"""
                    다음 줄거리를 3-5개의 장면으로 나누어주세요:
                    {self.story_outline['plot_summary']}
                    
                    각 장면은 일러스트 생성에 적합한 구체적인 설명을 포함해야 합니다.
                    """}
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
            - RAG 시스템을 활용하여 유사한 동화에서 참고 예시 검색
            - Few-shot 학습 방식으로 더 질 높은 스토리 생성
            - 줄거리를 바탕으로 전체 동화 구성
            - 각 장면에 대한 상세 설명 제공
            - 교육적 가치와 주제 강화
            - 주인공 캐릭터 표시 (아이의 음성 사용)
        """
        if not self.story_outline:
            raise ValueError("동화 줄거리가 설정되지 않았습니다.")
            
        try:
            # 연령대 및 주제 확인
            age_group = self.target_age or self.story_outline.get("target_age", 5)
            theme = self.story_outline.get("theme", "모험과 우정")
            
            # RAG 시스템을 통한 Few-shot 프롬프트 생성
            few_shot_prompt = self.rag_system.get_few_shot_prompt(age_group, theme)
            
            # RAG 시스템을 사용하여 주제 풍부화
            enriched_theme = self.rag_system.enrich_story_theme(theme, age_group)
            
            # 시스템 메시지 생성
            system_message = "\n".join(self.prompts["system"]["role"])
            
            # Few-shot 예시와 함께 이야기 생성 요청
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": f"""
                    다음 줄거리를 바탕으로 {age_group}세 아이에게 적합한 동화를 작성해주세요:
                    
                    줄거리: {self.story_outline.get('plot_summary', '')}
                    주제: {enriched_theme}
                    교육적 가치: {self.story_outline.get('educational_value', '협동과 문제 해결 능력')}
                    
                    {few_shot_prompt}
                    
                    다음 형식으로 JSON 응답을 제공해주세요:
                    {{
                        "title": "동화 제목",
                        "target_age": {age_group},
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
                        "theme": "주제 상세 설명",
                        "educational_value": "교육적 가치 설명"
                    }}
                    """}
                ],
                temperature=0.7,
                max_tokens=10000,
                response_format={"type": "json_object"}
            )
            
            # JSON 응답 파싱
            detailed_story = json.loads(response.choices[0].message.content)
            self.detailed_story = detailed_story
            
            return detailed_story
            
        except Exception as e:
            print(f"상세 이야기 생성 중 오류 발생: {str(e)}")
            # 기본 응답 생성
            default_story = {
                "title": self.story_outline.get("theme", "이야기"),
                "target_age": self.target_age or self.story_outline.get("target_age", 5),
                "characters": [{"name": "주인공", "description": "이야기의 주인공", "role": "주인공"}],
                "scenes": [{"title": "장면 1", "description": self.story_outline.get("plot_summary", ""), "narration": self.story_outline.get("plot_summary", ""), "dialogues": []}],
                "educational_value": self.story_outline.get("educational_value", ""),
                "theme": self.story_outline.get("theme", "")
            }
            self.detailed_story = default_story
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
            - RAG 시스템을 통한 유사 이야기의 이미지 참조
        """
        if not self.detailed_story:
            raise ValueError("상세 이야기가 먼저 생성되어야 합니다.")
            
        # 이미지를 저장할 디렉토리 확인
        image_dir = self.output_dir / "images"
        image_dir.mkdir(exist_ok=True)
        
        # 이미지 생성 결과 저장
        generated_images = []
        
        try:
            # 유사한 이야기 검색 (이미지 생성 참고용)
            age_group = self.detailed_story.get("target_age", 5)
            theme = self.detailed_story.get("theme", "")
            similar_stories = self.rag_system.get_similar_stories(theme, age_group, n_results=2)
            
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
                
                # 유사 이야기 정보가 있으면 참고 정보 추가
                if similar_stories:
                    prompt += "\n\n참고할 수 있는 비슷한 동화:"
                    for story in similar_stories:
                        prompt += f"\n- {story['title']}: {story['summary'][:100]}..."
                
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
            - ElevenLabs API를 사용하여 고품질 음성 합성
            - 내레이션과 캐릭터 대사를 구분하여 생성
            - 장면별로 오디오 생성
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
                        voice="Bella",  # 아이들을 위한 친근한 여성 목소리
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
                    
                    # 캐릭터별 음성 설정
                    voice = "Antoni"  # 기본 남성 목소리
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
                        # OpenAI TTS 사용
                        dialogue_response = client.audio.speech.create(
                            model="tts-1",
                            voice="alloy",
                            input=dialogue_text
                        )
                        dialogue_audio = dialogue_response.content
                    
                    # 대사 오디오 저장
                    dialogue_path = audio_dir / f"scene_{i+1}_{character_name}_{j+1}.mp3"
                    with open(dialogue_path, "wb") as f:
                        f.write(dialogue_audio)
                    
                    if "dialogues" not in generated_audio["characters"][character_name]:
                        generated_audio["characters"][character_name]["dialogues"] = {}
                    
                    generated_audio["characters"][character_name]["dialogues"][f"scene_{i+1}_dialogue_{j+1}"] = str(dialogue_path)
            
            # 최종 내레이션 경로 저장
            self.narration_audio = generated_audio
            
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
                "title": self.story_outline["theme"],
                "summary": preview_text,
                "image_count": len(self.generated_images),
                "duration": f"{len(self.generated_images) * 30}초"  # 각 이미지당 30초 가정
            }
            
        except Exception as e:
            print(f"스토리 미리보기 생성 중 오류 발생: {str(e)}")
            return {
                "title": self.story_outline["theme"],
                "summary": self.story_outline["plot_summary"],
                "image_count": len(self.generated_images),
                "duration": "알 수 없음"
            } 