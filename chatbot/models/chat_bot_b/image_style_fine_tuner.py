"""
동화 이미지 스타일 Fine-tuning을 위한 모듈
"""
import os
import json
import base64
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import time
import random
import requests
from PIL import Image
import io

from shared.utils.logging_utils import get_module_logger
from shared.utils.openai_utils import generate_chat_completion
from shared.utils.file_utils import ensure_directory

logger = get_module_logger(__name__)

class ImageStyleFineTuner:
    """
    동화 이미지 스타일을 학습하고 Fine-tuning된 스타일로 이미지를 생성하는 클래스
    """
    
    def __init__(self, openai_client=None, style_name: str = "fairy_tale_style",
                 training_images_dir: Union[str, Path] = None,
                 style_config_path: Union[str, Path] = None):
        """
        이미지 스타일 Fine-tuner 초기화
        
        Args:
            openai_client: OpenAI API 클라이언트
            style_name: 스타일 이름
            training_images_dir: 학습용 이미지 디렉토리
            style_config_path: 스타일 설정 파일 경로
        """
        self.openai_client = openai_client
        self.style_name = style_name
        self.training_images_dir = Path(training_images_dir) if training_images_dir else Path("data/training_images")
        self.style_config_path = Path(style_config_path) if style_config_path else Path("data/style_configs")
        
        # 스타일 설정 파일
        self.style_config_file = self.style_config_path / f"{style_name}.json"
        
        # 스타일 분석 결과 저장
        self.style_analysis = {}
        
        # 기본 스타일 프롬프트 템플릿
        self.base_style_prompts = {
            "character_style": "cute, child-friendly characters with expressive eyes and warm colors",
            "color_palette": "soft pastels, warm tones, bright but not overwhelming colors",
            "art_style": "illustrated storybook style, digital art, clean lines",
            "atmosphere": "magical, wholesome, safe and comforting for children",
            "composition": "simple composition suitable for young children, clear focal points"
        }
        
        # 학습된 스타일 특성
        self.learned_style_features = {}
        
    def analyze_training_images(self, image_paths: List[Union[str, Path]]) -> Dict[str, Any]:
        """
        학습용 이미지들을 분석하여 스타일 특성을 추출
        
        Args:
            image_paths: 분석할 이미지 파일 경로 리스트
            
        Returns:
            Dict: 분석된 스타일 특성
        """
        if not self.openai_client:
            logger.error("OpenAI 클라이언트가 초기화되지 않았습니다.")
            return {}
        
        style_features = {
            "color_analysis": [],
            "art_style_analysis": [],
            "character_features": [],
            "composition_patterns": [],
            "common_elements": []
        }
        
        for i, image_path in enumerate(image_paths):
            try:
                # 이미지를 base64로 인코딩
                with open(image_path, "rb") as image_file:
                    base64_image = base64.b64encode(image_file.read()).decode('utf-8')
                
                # GPT-4 Vision을 사용하여 이미지 분석
                analysis_prompt = """
                이 동화 이미지를 분석하여 다음 특성들을 추출해주세요:
                
                1. 색상 팔레트 (주요 색상들과 색조)
                2. 그림 스타일 (일러스트레이션 기법, 선 스타일, 질감)
                3. 캐릭터 특징 (얼굴 표현, 체형, 의상 스타일)
                4. 구도와 레이아웃 (배경과 전경의 관계, 시각적 균형)
                5. 전체적인 분위기와 느낌
                
                JSON 형식으로 분석 결과를 제공해주세요:
                {
                    "colors": ["색상1", "색상2", ...],
                    "art_style": "스타일 설명",
                    "character_features": "캐릭터 특징 설명",
                    "composition": "구도 설명",
                    "mood": "분위기 설명",
                    "notable_elements": ["특징적 요소1", "특징적 요소2", ...]
                }
                """
                
                response = self.openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": analysis_prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}"
                                    }
                                }
                            ]
                        }
                    ],
                    max_tokens=1000
                )
                
                # 응답 파싱
                response_text = response.choices[0].message.content
                
                # JSON 추출
                try:
                    json_start = response_text.find('{')
                    json_end = response_text.rfind('}') + 1
                    
                    if json_start >= 0 and json_end > json_start:
                        json_str = response_text[json_start:json_end]
                        analysis_result = json.loads(json_str)
                        
                        # 분석 결과를 카테고리별로 저장
                        style_features["color_analysis"].extend(analysis_result.get("colors", []))
                        style_features["art_style_analysis"].append(analysis_result.get("art_style", ""))
                        style_features["character_features"].append(analysis_result.get("character_features", ""))
                        style_features["composition_patterns"].append(analysis_result.get("composition", ""))
                        style_features["common_elements"].extend(analysis_result.get("notable_elements", []))
                        
                        logger.info(f"이미지 {i+1}/{len(image_paths)} 분석 완료: {image_path}")
                        
                except json.JSONDecodeError as e:
                    logger.warning(f"이미지 {image_path} 분석 결과 JSON 파싱 실패: {e}")
                    
            except Exception as e:
                logger.error(f"이미지 {image_path} 분석 중 오류 발생: {e}")
                continue
        
        # 공통 특성 추출
        self.learned_style_features = self._extract_common_features(style_features)
        
        logger.info("이미지 스타일 분석 완료")
        return self.learned_style_features
    
    def _extract_common_features(self, style_features: Dict[str, List]) -> Dict[str, Any]:
        """
        분석된 특성들에서 공통점을 추출하여 스타일 특성으로 정리
        
        Args:
            style_features: 개별 이미지 분석 결과들
            
        Returns:
            Dict: 공통 스타일 특성
        """
        # 색상 빈도 분석
        color_frequency = {}
        for colors in style_features["color_analysis"]:
            for color in colors:
                color_frequency[color.lower()] = color_frequency.get(color.lower(), 0) + 1
        
        # 가장 빈번한 색상들 추출
        common_colors = sorted(color_frequency.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # 공통 요소 빈도 분석
        element_frequency = {}
        for elements in style_features["common_elements"]:
            for element in elements:
                element_frequency[element.lower()] = element_frequency.get(element.lower(), 0) + 1
        
        common_elements = sorted(element_frequency.items(), key=lambda x: x[1], reverse=True)[:15]
        
        return {
            "dominant_colors": [color for color, _ in common_colors],
            "art_style_patterns": style_features["art_style_analysis"],
            "character_patterns": style_features["character_features"],
            "composition_patterns": style_features["composition_patterns"],
            "common_visual_elements": [element for element, _ in common_elements],
            "analysis_summary": {
                "total_images_analyzed": len(style_features["art_style_analysis"]),
                "color_diversity": len(color_frequency),
                "element_diversity": len(element_frequency)
            }
        }
    
    def create_style_prompt_template(self) -> str:
        """
        학습된 스타일 특성을 바탕으로 이미지 생성용 프롬프트 템플릿 생성
        
        Returns:
            str: 스타일이 적용된 프롬프트 템플릿
        """
        if not self.learned_style_features:
            logger.warning("학습된 스타일 특성이 없습니다. 기본 템플릿을 사용합니다.")
            return self._get_default_style_template()
        
        # 학습된 특성을 바탕으로 프롬프트 생성
        colors = ", ".join(self.learned_style_features.get("dominant_colors", [])[:5])
        
        # 아트 스타일 패턴에서 공통점 추출
        art_patterns = self.learned_style_features.get("art_style_patterns", [])
        art_style_desc = self._summarize_patterns(art_patterns)
        
        # 캐릭터 패턴에서 공통점 추출
        character_patterns = self.learned_style_features.get("character_patterns", [])
        character_style_desc = self._summarize_patterns(character_patterns)
        
        # 공통 요소들
        common_elements = ", ".join(self.learned_style_features.get("common_visual_elements", [])[:8])
        
        style_template = f"""
        Style: {art_style_desc}
        Color palette: {colors}
        Character style: {character_style_desc}
        Common elements: {common_elements}
        Art technique: Illustrated children's book style, soft and warm illustration
        Mood: Friendly, magical, child-appropriate, wholesome
        Quality: High-quality digital illustration, detailed but not overwhelming for children
        """
        
        return style_template.strip()
    
    def _summarize_patterns(self, patterns: List[str]) -> str:
        """
        패턴 리스트에서 공통적인 키워드를 추출하여 요약
        
        Args:
            patterns: 패턴 설명 리스트
            
        Returns:
            str: 요약된 패턴 설명
        """
        if not patterns:
            return "soft, child-friendly illustration style"
        
        # 간단한 키워드 추출 (실제로는 더 정교한 NLP 기법 사용 가능)
        common_words = []
        for pattern in patterns:
            words = pattern.lower().split()
            common_words.extend(words)
        
        # 빈도 기반 키워드 추출
        word_freq = {}
        for word in common_words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # 상위 키워드들로 설명 구성
        top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        summary_words = [word for word, _ in top_words if len(word) > 3]  # 3글자 이상 단어만
        
        return " ".join(summary_words[:5]) if summary_words else "soft, child-friendly illustration style"
    
    def _get_default_style_template(self) -> str:
        """
        기본 동화 스타일 템플릿 반환
        
        Returns:
            str: 기본 스타일 템플릿
        """
        return """
        Style: Soft children's book illustration, watercolor and digital art style
        Color palette: Warm pastels, soft blues, gentle pinks, creamy yellows, forest greens
        Character style: Cute, expressive characters with large friendly eyes, rounded features
        Common elements: Natural settings, magical elements, simple backgrounds
        Art technique: Clean lines, soft shading, child-friendly details
        Mood: Warm, safe, magical, encouraging, age-appropriate
        Quality: Professional children's book illustration quality
        """
    
    def generate_styled_image(self, scene_description: str, chapter_info: Dict = None) -> Optional[Dict]:
        """
        학습된 스타일을 적용하여 이미지 생성
        
        Args:
            scene_description: 장면 설명
            chapter_info: 챕터 정보 (선택사항)
            
        Returns:
            Optional[Dict]: 생성된 이미지 정보
        """
        if not self.openai_client:
            logger.error("OpenAI 클라이언트가 초기화되지 않았습니다.")
            return None
        
        # 스타일 템플릿 가져오기
        style_template = self.create_style_prompt_template()
        
        # 최종 프롬프트 구성
        styled_prompt = f"""
        {scene_description}
        
        {style_template}
        
        Create this scene as a children's storybook illustration with the learned artistic style.
        Ensure the image is appropriate for young children and matches the established visual style.
        """
        
        try:
            # DALL-E를 사용하여 스타일이 적용된 이미지 생성
            response = self.openai_client.images.generate(
                model="dall-e-3",
                prompt=styled_prompt,
                size="1024x1024",
                quality="standard",
                n=1,
                response_format="b64_json"
            )
            
            # 이미지 정보 추출
            image_data = response.data[0]
            image_b64 = image_data.b64_json
            
            result = {
                "image_b64": image_b64,
                "prompt": styled_prompt,
                "style_applied": self.style_name,
                "timestamp": time.time()
            }
            
            logger.info(f"스타일이 적용된 이미지 생성 완료: {self.style_name}")
            return result
            
        except Exception as e:
            logger.error(f"스타일 적용 이미지 생성 중 오류 발생: {e}")
            return None
    
    def save_style_config(self, config_data: Dict = None) -> bool:
        """
        학습된 스타일 설정을 파일로 저장
        
        Args:
            config_data: 저장할 설정 데이터 (기본값: 학습된 특성)
            
        Returns:
            bool: 저장 성공 여부
        """
        try:
            # 디렉토리 생성
            ensure_directory(self.style_config_path)
            
            # 저장할 데이터 준비
            if config_data is None:
                config_data = {
                    "style_name": self.style_name,
                    "learned_features": self.learned_style_features,
                    "style_template": self.create_style_prompt_template(),
                    "created_at": time.time(),
                    "training_images_count": self.learned_style_features.get("analysis_summary", {}).get("total_images_analyzed", 0)
                }
            
            # JSON 파일로 저장
            with open(self.style_config_file, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"스타일 설정 저장 완료: {self.style_config_file}")
            return True
            
        except Exception as e:
            logger.error(f"스타일 설정 저장 중 오류 발생: {e}")
            return False
    
    def load_style_config(self) -> bool:
        """
        저장된 스타일 설정을 로드
        
        Returns:
            bool: 로드 성공 여부
        """
        try:
            if not self.style_config_file.exists():
                logger.warning(f"스타일 설정 파일이 존재하지 않습니다: {self.style_config_file}")
                return False
            
            with open(self.style_config_file, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            self.learned_style_features = config_data.get("learned_features", {})
            
            logger.info(f"스타일 설정 로드 완료: {self.style_name}")
            return True
            
        except Exception as e:
            logger.error(f"스타일 설정 로드 중 오류 발생: {e}")
            return False
    
    def train_from_directory(self, images_directory: Union[str, Path]) -> bool:
        """
        지정된 디렉토리의 모든 이미지로 스타일 학습
        
        Args:
            images_directory: 학습용 이미지가 있는 디렉토리
            
        Returns:
            bool: 학습 성공 여부
        """
        images_dir = Path(images_directory)
        
        if not images_dir.exists():
            logger.error(f"이미지 디렉토리가 존재하지 않습니다: {images_dir}")
            return False
        
        # 이미지 파일들 찾기
        image_extensions = ['.jpg', '.jpeg', '.png', '.webp']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(images_dir.glob(f"*{ext}"))
            image_files.extend(images_dir.glob(f"*{ext.upper()}"))
        
        if not image_files:
            logger.error(f"이미지 디렉토리에 이미지 파일이 없습니다: {images_dir}")
            return False
        
        logger.info(f"{len(image_files)}개의 이미지로 스타일 학습을 시작합니다...")
        
        # 이미지 분석 및 학습
        self.analyze_training_images(image_files)
        
        # 스타일 설정 저장
        return self.save_style_config() 