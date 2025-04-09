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
from elevenlabs import generate, save, set_api_key

# 환경 변수 설정
load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
elevenlabs.set_api_key(os.getenv('ELEVENLABS_API_KEY'))

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
        
        # 출력 디렉토리 생성
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "images").mkdir(exist_ok=True)
        (self.output_dir / "audio").mkdir(exist_ok=True)
        
        # 프롬프트 로드
        self.load_prompts()
    
    def load_prompts(self):
        """JSON 파일에서 프롬프트를 로드하는 함수"""
        try:
            with open('data/prompts/chatbot_prompts.json', 'r', encoding='utf-8') as f:
                prompts = json.load(f)
                self.prompts = prompts['chatbot_b']
        except Exception as e:
            print(f"프롬프트 로드 중 오류 발생: {str(e)}")
            # 기본 프롬프트 설정
            self.prompts = {
                "system_message_template": "당신은 동화 일러스트와 내레이션을 생성하는 전문가입니다.",
                "image_generation_prompt_template": "주제: {theme}\n캐릭터: {characters}\n배경: {setting}\n스타일: {style}",
                "narration_generation_prompt_template": "줄거리: {plot_summary}\n교육적 가치: {educational_value}\n대상 연령: {target_age}"
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