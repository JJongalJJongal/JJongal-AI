import openai
from typing import Dict, List
from dotenv import load_dotenv
import os
import json

# 환경 변수 설정
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

class ImageGenerator:
    """
    DALL-E 3를 활용한 동화 삽화 생성 클래스
    
    Attributes:
        style_guide (Dict[str, str]): 이미지 스타일 가이드
        prompt_templates (Dict[str, str]): 프롬프트 템플릿
    """
    
    def __init__(self):
        """ImageGenerator 초기화"""
        self.style_guide = {
            'color_scheme': 'warm and vibrant',
            'art_style': 'child-friendly illustration',
            'mood': 'positive and cheerful',
            'composition': 'balanced and dynamic',
            'details': 'clear and engaging'
        }
        
        self.prompt_templates = {
            'character': """
            Create a friendly and lovable character illustration of {character} with:
            - Warm and inviting expression
            - Age-appropriate features
            - Distinctive personality traits
            - Child-friendly proportions
            """,
            'scene': """
            Create a child-friendly scene showing {scene} with:
            - Safe and welcoming environment
            - Clear focal point
            - Engaging background details
            - Appropriate lighting
            """,
            'emotion': """
            Create an illustration expressing {emotion} emotion with:
            - Clear emotional cues
            - Relatable facial expressions
            - Supportive visual elements
            - Age-appropriate intensity
            """
        }
    
    def create_style_prompt(self, scene_description: str, prompt_type: str = 'scene') -> str:
        """
        스타일 가이드에 맞는 프롬프트를 생성하는 함수
        
        Args:
            scene_description (str): 장면 설명
            prompt_type (str): 프롬프트 타입 ('character', 'scene', 'emotion')
            
        Returns:
            str: 생성된 프롬프트
        """
        base_prompt = self.prompt_templates.get(prompt_type, self.prompt_templates['scene'])
        prompt = base_prompt.format(scene=scene_description)
        
        return f"""Create a child-friendly illustration with the following characteristics:
        - Style: {self.style_guide['art_style']}
        - Colors: {self.style_guide['color_scheme']}
        - Mood: {self.style_guide['mood']}
        - Composition: {self.style_guide['composition']}
        - Details: {self.style_guide['details']}
        - Prompt: {prompt}
        - Additional details: {scene_description}
        
        Guidelines:
        1. Ensure all elements are child-appropriate
        2. Maintain consistent style across all images
        3. Use clear, bold lines and shapes
        4. Include engaging details that support the story
        5. Keep the composition balanced and dynamic
        """
    
    def generate_image(self, scene_description: str, prompt_type: str = 'scene') -> str:
        """
        DALL-E 3를 사용하여 이미지를 생성하는 함수
        
        Args:
            scene_description (str): 장면 설명
            prompt_type (str): 프롬프트 타입
            
        Returns:
            str: 생성된 이미지 URL
        """
        prompt = self.create_style_prompt(scene_description, prompt_type)
        
        try:
            response = openai.Image.create(
                model="dall-e-3",
                prompt=prompt,
                size="1024x1024",
                quality="standard",
                n=1,
                response_format="url"
            )
            
            return response.data[0].url
        except Exception as e:
            print(f"Error generating image: {str(e)}")
            return ""
    
    def generate_story_images(self, story: str) -> List[str]:
        """
        동화의 각 장면에 대한 이미지를 생성하는 함수
        
        Args:
            story (str): 동화 내용
            
        Returns:
            List[str]: 생성된 이미지 URL 목록
        """
        try:
            # GPT-4를 사용하여 장면 분리 및 설명 생성
            scene_prompt = f"""
            Break down this story into key scenes and provide detailed visual descriptions for each:
            {story}
            
            Format each scene as:
            Scene X: [Visual Description]
            """
            
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a storyboard artist."},
                    {"role": "user", "content": scene_prompt}
                ]
            )
            
            scenes = response.choices[0].message.content.split('\n\n')
            images = []
            
            for scene in scenes:
                if scene.strip():
                    # 장면 설명에서 시각적 설명 추출
                    visual_desc = scene.split(':')[1].strip() if ':' in scene else scene.strip()
                    image_url = self.generate_image(visual_desc)
                    if image_url:
                        images.append(image_url)
            
            return images
            
        except Exception as e:
            print(f"Error generating story images: {str(e)}")
            return [] 