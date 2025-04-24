from openai import OpenAI
from typing import List, Dict, Optional, Tuple
from dotenv import load_dotenv
import os
import json
import requests
import base64
from pathlib import Path
import time

# 환경 변수 설정
load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
ELEVENLABS_API_KEY = os.getenv('ELEVENLABS_API_KEY')

class StoryGenerationChatBot:
    """
    줄거리를 바탕으로 상세 동화, 삽화, 음성을 생성하는 AI 챗봇 클래스
    
    Attributes:
        chatbot_name (str): 챗봇의 이름 ("꼬기")
        story_outline (Dict): Chat-bot A에서 수집한 이야기 줄거리
        detailed_story (Dict): 생성된 상세 동화 내용
        illustrations (List[str]): 생성된 삽화 파일 경로 목록
        voice_files (List[str]): 생성된 음성 파일 경로 목록
        voice_id (str): ElevenLabs에서 사용할 음성 ID
        target_age (int): 동화의 대상 연령대
    """
    
    def __init__(self):
        """챗봇 초기화 및 기본 속성 설정"""
        self.chatbot_name = "꼬기"  # 챗봇 이름 설정
        self.story_outline = None   # 이야기 줄거리 (Chat-bot A에서 수집)
        self.detailed_story = {}    # 상세 동화 내용
        self.illustrations = []     # 삽화 파일 경로
        self.voice_files = []       # 음성 파일 경로
        self.voice_id = None        # ElevenLabs 음성 ID
        self.target_age = 5         # 기본 대상 연령
        
        # 결과물을 저장할 디렉토리 생성
        self._create_output_directories()
    
    def _create_output_directories(self):
        """결과물을 저장할 디렉토리 생성"""
        base_dir = Path(__file__).parent.parent.parent  # CCB-AI 디렉토리
        
        # 삽화 저장 디렉토리
        self.image_dir = base_dir / "output" / "images"
        self.image_dir.mkdir(parents=True, exist_ok=True)
        
        # 음성 저장 디렉토리
        self.voice_dir = base_dir / "output" / "voices"
        self.voice_dir.mkdir(parents=True, exist_ok=True)
        
        # 전체 동화 저장 디렉토리
        self.story_dir = base_dir / "output" / "stories"
        self.story_dir.mkdir(parents=True, exist_ok=True)
    
    def set_story_outline(self, outline: Dict):
        """
        이야기 줄거리 설정
        
        Args:
            outline (Dict): Chat-bot A에서 수집한 이야기 줄거리
        """
        self.story_outline = outline
    
    def set_voice_id(self, voice_id: str):
        """
        ElevenLabs 음성 ID 설정
        
        Args:
            voice_id (str): ElevenLabs에서 사용할 음성 ID
        """
        self.voice_id = voice_id
    
    def set_target_age(self, age: int):
        """
        대상 연령대 설정
        
        Args:
            age (int): 동화의 대상 연령대 (4-9세)
        """
        self.target_age = age
    
    def generate_detailed_story(self) -> Dict:
        """
        줄거리를 바탕으로 상세 동화 생성
        
        Returns:
            Dict: 생성된 상세 동화 내용
        """
        if not self.story_outline:
            raise ValueError("이야기 줄거리가 설정되지 않았습니다.")
        
        try:
            summary_text = self.story_outline.get("summary_text", "")
            tags = self.story_outline.get("tags", "")
            
            # 상세 동화 생성 프롬프트
            prompt = f"""
            다음 정보를 바탕으로 {self.target_age}세 아이들을 위한 상세한 동화를 만들어주세요:
            
            줄거리: {summary_text}
            태그: {tags}
            
            다음 형식의 JSON으로 응답해주세요:
            {{
                "title": "동화 제목",
                "scenes": [
                    {{
                        "scene_number": 1,
                        "description": "장면 설명",
                        "text": "장면에 해당하는 동화 텍스트",
                        "narration": "내레이션 텍스트",
                        "dialogues": [
                            {{"character": "캐릭터 이름", "text": "대사 내용"}}
                        ],
                        "image_prompt": "DALL-E 3를 위한 이미지 생성 프롬프트"
                    }}
                ],
                "characters": [
                    {{"name": "캐릭터 이름", "description": "캐릭터 설명"}}
                ],
                "moral": "동화의 교훈",
                "target_age": {self.target_age}
            }}
            """
            
            # GPT-4o를 사용하여 상세 동화 생성
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "당신은 아이들을 위한 동화 작가입니다. 주어진 줄거리와 태그를 바탕으로 창의적이고 교육적인 동화를 만들어주세요."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=2000,
                response_format={"type": "json_object"}
            )
            
            # 응답에서 JSON 파싱
            self.detailed_story = json.loads(response.choices[0].message.content)
            
            return self.detailed_story
            
        except Exception as e:
            print(f"상세 동화 생성 중 오류 발생: {str(e)}")
            return {}
    
    def generate_illustrations(self) -> List[str]:
        """
        상세 동화 내용을 바탕으로 삽화 생성
        
        Returns:
            List[str]: 생성된 삽화 파일 경로 목록
        """
        if not self.detailed_story:
            raise ValueError("상세 동화가 생성되지 않았습니다.")
        
        try:
            illustrations = []
            scenes = self.detailed_story.get("scenes", [])
            
            for idx, scene in enumerate(scenes):
                image_prompt = scene.get("image_prompt", "")
                
                # DALL-E 3를 사용하여 삽화 생성
                style_prompt = f"""
                아이들을 위한 동화책 삽화. {self.target_age}세 아이들이 좋아할 만한 밝고 따뜻한 스타일로 그려주세요.
                다음 장면을 묘사해주세요: {image_prompt}
                """
                
                response = client.images.generate(
                    model="dall-e-3",
                    prompt=style_prompt,
                    size="1024x1024",
                    quality="standard",
                    n=1
                )
                
                # 이미지 URL 추출
                image_url = response.data[0].url
                
                # 이미지 다운로드
                image_response = requests.get(image_url)
                
                # 이미지 저장
                image_path = self.image_dir / f"scene_{idx+1}.png"
                with open(image_path, "wb") as f:
                    f.write(image_response.content)
                
                illustrations.append(str(image_path))
            
            self.illustrations = illustrations
            return illustrations
            
        except Exception as e:
            print(f"삽화 생성 중 오류 발생: {str(e)}")
            return []
    
    def generate_voice(self) -> List[str]:
        """
        상세 동화 내용을 바탕으로 음성 생성
        
        Returns:
            List[str]: 생성된 음성 파일 경로 목록
        """
        if not self.detailed_story:
            raise ValueError("상세 동화가 생성되지 않았습니다.")
        
        try:
            voice_files = []
            scenes = self.detailed_story.get("scenes", [])
            
            for idx, scene in enumerate(scenes):
                narration = scene.get("narration", "")
                dialogues = scene.get("dialogues", [])
                
                # 내레이션과 대사를 하나의 텍스트로 합치기
                full_text = narration + "\n"
                for dialogue in dialogues:
                    character = dialogue.get("character", "")
                    text = dialogue.get("text", "")
                    full_text += f"{character}: {text}\n"
                
                # OpenAI TTS를 사용하여 음성 생성
                response = client.audio.speech.create(
                    model="tts-1",
                    voice="shimmer",
                    input=full_text
                )
                
                # 음성 저장
                voice_path = self.voice_dir / f"scene_{idx+1}.mp3"
                response.stream_to_file(str(voice_path))
                
                voice_files.append(str(voice_path))
            
            self.voice_files = voice_files
            return voice_files
            
        except Exception as e:
            print(f"음성 생성 중 오류 발생: {str(e)}")
            return []
    
    def elevenlabs_voice_generation(self, text: str, voice_id: str) -> str:
        """
        ElevenLabs API를 사용하여 음성 생성
        
        Args:
            text (str): 음성으로 변환할 텍스트
            voice_id (str): ElevenLabs 음성 ID
            
        Returns:
            str: 생성된 음성 파일 경로
        """
        try:
            url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
            
            headers = {
                "Accept": "audio/mpeg",
                "Content-Type": "application/json",
                "xi-api-key": ELEVENLABS_API_KEY
            }
            
            data = {
                "text": text,
                "model_id": "eleven_monolingual_v1",
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.5
                }
            }
            
            response = requests.post(url, json=data, headers=headers)
            
            if response.status_code == 200:
                # 음성 저장
                timestamp = int(time.time())
                voice_path = self.voice_dir / f"elevenlabs_{timestamp}.mp3"
                with open(voice_path, "wb") as f:
                    f.write(response.content)
                
                return str(voice_path)
            else:
                print(f"ElevenLabs API 오류: {response.status_code} - {response.text}")
                return ""
                
        except Exception as e:
            print(f"ElevenLabs 음성 생성 중 오류 발생: {str(e)}")
            return ""
    
    def generate_complete_story(self) -> Dict:
        """
        상세 동화, 삽화, 음성을 하나로 통합
        
        Returns:
            Dict: 통합된 동화 정보
        """
        try:
            # 1. 상세 동화 생성
            if not self.detailed_story:
                self.generate_detailed_story()
            
            # 2. 삽화 생성
            if not self.illustrations:
                self.generate_illustrations()
            
            # 3. 음성 생성
            if not self.voice_files:
                self.generate_voice()
            
            # 4. 결과물 통합
            complete_story = {
                "title": self.detailed_story.get("title", "무제"),
                "scenes": []
            }
            
            scenes = self.detailed_story.get("scenes", [])
            
            for idx, scene in enumerate(scenes):
                if idx < len(self.illustrations) and idx < len(self.voice_files):
                    complete_scene = {
                        "scene_number": scene.get("scene_number", idx + 1),
                        "description": scene.get("description", ""),
                        "text": scene.get("text", ""),
                        "image_path": self.illustrations[idx],
                        "voice_path": self.voice_files[idx]
                    }
                    complete_story["scenes"].append(complete_scene)
            
            # 기타 정보 추가
            complete_story["characters"] = self.detailed_story.get("characters", [])
            complete_story["moral"] = self.detailed_story.get("moral", "")
            complete_story["target_age"] = self.target_age
            
            # 결과 저장
            story_path = self.story_dir / f"{complete_story['title']}.json"
            with open(story_path, "w", encoding="utf-8") as f:
                json.dump(complete_story, f, ensure_ascii=False, indent=2)
            
            return complete_story
            
        except Exception as e:
            print(f"동화 통합 중 오류 발생: {str(e)}")
            return {} 