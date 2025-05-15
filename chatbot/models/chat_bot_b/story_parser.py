"""
이야기 콘텐츠 파싱 및 포맷팅을 담당하는 모듈
"""
from typing import Dict, List, Any, Optional
import json
import re

from shared.utils.logging_utils import get_module_logger

logger = get_module_logger(__name__)

class StoryParser:
    """
    이야기 콘텐츠 파싱 및 포맷팅을 담당하는 클래스
    """
    
    def __init__(self):
        """
        스토리 파서 초기화
        """
        pass
    
    def extract_json_from_response(self, response_text: str) -> Optional[Dict]:
        """
        OpenAI 응답에서 JSON 데이터 추출
        
        Args:
            response_text: OpenAI 응답 텍스트
            
        Returns:
            Optional[Dict]: 추출된 JSON 데이터 또는 None
        """
        try:
            # JSON 패턴 찾기
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                return json.loads(json_str)
            else:
                logger.error("응답에서 JSON 형식을 찾을 수 없습니다.")
                return None
                
        except json.JSONDecodeError as e:
            logger.error(f"JSON 파싱 오류: {e}")
            return None
        except Exception as e:
            logger.error(f"JSON 추출 중 오류 발생: {e}")
            return None
    
    def validate_story_structure(self, story_data: Dict) -> bool:
        """
        이야기 데이터 구조 검증
        
        Args:
            story_data: 이야기 데이터
            
        Returns:
            bool: 검증 결과 (유효하면 True, 아니면 False)
        """
        try:
            # 필수 필드 확인
            required_fields = ["title", "chapters"]
            for field in required_fields:
                if field not in story_data:
                    logger.error(f"필수 필드가 누락되었습니다: {field}")
                    return False
            
            # 챕터 구조 확인
            chapters = story_data.get("chapters", [])
            if not chapters or not isinstance(chapters, list):
                logger.error("유효한 챕터 목록이 없습니다.")
                return False
            
            # 각 챕터 구조 확인
            for chapter in chapters:
                if not isinstance(chapter, dict):
                    logger.error("챕터가 딕셔너리 형태가 아닙니다.")
                    return False
                
                # 챕터 필수 필드 확인
                chapter_fields = ["chapter_number", "title", "narration"]
                for field in chapter_fields:
                    if field not in chapter:
                        logger.error(f"챕터에 필수 필드가 누락되었습니다: {field}")
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"이야기 구조 검증 중 오류 발생: {e}")
            return False
    
    def format_chapter_for_display(self, chapter: Dict) -> str:
        """
        챕터 데이터를 표시용으로 포맷팅
        
        Args:
            chapter: 챕터 데이터
            
        Returns:
            str: 포맷팅된 챕터 텍스트
        """
        try:
            # 챕터 기본 정보
            chapter_number = chapter.get("chapter_number", "")
            title = chapter.get("title", "")
            narration = chapter.get("narration", "")
            dialogues = chapter.get("dialogues", [])
            moral_lesson = chapter.get("moral_lesson", "")
            
            # 포맷팅된 결과 구성
            formatted_text = f"# {chapter_number}. {title}\n\n"
            
            # 내레이션 추가
            formatted_text += f"{narration}\n\n"
            
            # 대화 추가
            if dialogues:
                for dialogue in dialogues:
                    speaker = dialogue.get("speaker", "")
                    text = dialogue.get("text", "")
                    formatted_text += f"**{speaker}**: {text}\n"
                
                formatted_text += "\n"
            
            # 교훈 추가 (있을 경우)
            if moral_lesson:
                formatted_text += f"*교훈: {moral_lesson}*\n"
            
            return formatted_text
            
        except Exception as e:
            logger.error(f"챕터 포맷팅 중 오류 발생: {e}")
            return "챕터 포맷팅 오류"
    
    def format_story_for_display(self, story_data: Dict) -> str:
        """
        전체 이야기를 표시용으로 포맷팅
        
        Args:
            story_data: 이야기 데이터
            
        Returns:
            str: 포맷팅된 이야기 텍스트
        """
        try:
            # 이야기 기본 정보
            title = story_data.get("title", "제목 없음")
            theme = story_data.get("theme", "")
            educational_value = story_data.get("educational_value", "")
            characters = story_data.get("characters", [])
            setting = story_data.get("setting", "")
            chapters = story_data.get("chapters", [])
            
            # 포맷팅된 결과 구성
            formatted_text = f"# {title}\n\n"
            
            # 기본 정보 추가
            if theme:
                formatted_text += f"**주제**: {theme}\n"
            if educational_value:
                formatted_text += f"**교육적 가치**: {educational_value}\n"
            if characters:
                formatted_text += f"**등장인물**: {', '.join(characters)}\n"
            if setting:
                formatted_text += f"**배경**: {setting}\n"
                
            formatted_text += "\n---\n\n"
            
            # 챕터 추가
            for chapter in chapters:
                formatted_text += self.format_chapter_for_display(chapter)
                formatted_text += "\n---\n\n"
            
            return formatted_text
            
        except Exception as e:
            logger.error(f"이야기 포맷팅 중 오류 발생: {e}")
            return "이야기 포맷팅 오류"
    
    def format_story_for_html(self, story_data: Dict, image_paths: Dict = None, audio_paths: Dict = None) -> str:
        """
        이야기를 HTML 형식으로 포맷팅
        
        Args:
            story_data: 이야기 데이터
            image_paths: 챕터별 이미지 경로 매핑
            audio_paths: 챕터별 오디오 경로 매핑
            
        Returns:
            str: HTML 형식의 이야기
        """
        try:
            # 이야기 기본 정보
            title = story_data.get("title", "제목 없음")
            theme = story_data.get("theme", "")
            educational_value = story_data.get("educational_value", "")
            characters = story_data.get("characters", [])
            setting = story_data.get("setting", "")
            chapters = story_data.get("chapters", [])
            
            # 기본 HTML 템플릿
            html = f"""
            <!DOCTYPE html>
            <html lang="ko">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>{title}</title>
                <style>
                    body {{
                        font-family: 'Noto Sans KR', sans-serif;
                        line-height: 1.6;
                        color: #333;
                        max-width: 800px;
                        margin: 0 auto;
                        padding: 20px;
                    }}
                    h1, h2 {{
                        color: #2c3e50;
                    }}
                    .story-info {{
                        background-color: #f8f9fa;
                        padding: 15px;
                        border-radius: 5px;
                        margin-bottom: 20px;
                    }}
                    .chapter {{
                        margin-bottom: 30px;
                        border-bottom: 1px solid #eee;
                        padding-bottom: 20px;
                    }}
                    .chapter-image {{
                        max-width: 100%;
                        border-radius: 5px;
                        margin: 15px 0;
                    }}
                    .dialogue {{
                        margin-left: 20px;
                        margin-bottom: 10px;
                    }}
                    .speaker {{
                        font-weight: bold;
                        color: #3498db;
                    }}
                    .moral-lesson {{
                        font-style: italic;
                        color: #7f8c8d;
                        margin-top: 15px;
                    }}
                    audio {{
                        width: 100%;
                        margin: 10px 0;
                    }}
                    .audio-section {{
                        background-color: #f1f1f1;
                        padding: 10px;
                        border-radius: 5px;
                        margin: 15px 0;
                    }}
                </style>
            </head>
            <body>
                <h1>{title}</h1>
                
                <div class="story-info">
                    <p><strong>주제:</strong> {theme}</p>
                    <p><strong>교육적 가치:</strong> {educational_value}</p>
                    <p><strong>등장인물:</strong> {', '.join(characters)}</p>
                    <p><strong>배경:</strong> {setting}</p>
                </div>
            """
            
            # 챕터별 HTML 추가
            for chapter in chapters:
                chapter_number = chapter.get("chapter_number", "")
                chapter_title = chapter.get("title", "")
                narration = chapter.get("narration", "")
                dialogues = chapter.get("dialogues", [])
                moral_lesson = chapter.get("moral_lesson", "")
                
                html += f"""
                <div class="chapter" id="chapter-{chapter_number}">
                    <h2>{chapter_number}. {chapter_title}</h2>
                """
                
                # 챕터 이미지 추가 (있을 경우)
                if image_paths and str(chapter_number) in image_paths:
                    img_path = image_paths[str(chapter_number)]
                    html += f'<img src="{img_path}" alt="Chapter {chapter_number} illustration" class="chapter-image">'
                
                # 내레이션 추가
                narration_paragraphs = narration.split('\n')
                for paragraph in narration_paragraphs:
                    if paragraph.strip():
                        html += f"<p>{paragraph}</p>\n"
                
                # 내레이션 오디오 추가 (있을 경우)
                if audio_paths and f"chapter_{chapter_number}_narration" in audio_paths:
                    narration_audio = audio_paths[f"chapter_{chapter_number}_narration"]
                    html += f"""
                    <div class="audio-section">
                        <p><strong>내레이션 듣기:</strong></p>
                        <audio controls>
                            <source src="{narration_audio}" type="audio/mpeg">
                            Your browser does not support the audio element.
                        </audio>
                    </div>
                    """
                
                # 대화 추가
                if dialogues:
                    html += "<div class='dialogues'>\n"
                    for idx, dialogue in enumerate(dialogues):
                        speaker = dialogue.get("speaker", "")
                        text = dialogue.get("text", "")
                        html += f"""
                        <div class="dialogue">
                            <span class="speaker">{speaker}:</span> {text}
                        """
                        
                        # 대화 오디오 추가 (있을 경우)
                        audio_key = f"dialogue_{idx+1}_{speaker}"
                        if audio_paths and audio_key in audio_paths:
                            dialogue_audio = audio_paths[audio_key]
                            html += f"""
                            <audio controls>
                                <source src="{dialogue_audio}" type="audio/mpeg">
                                Your browser does not support the audio element.
                            </audio>
                            """
                            
                        html += "</div>\n"
                    
                    html += "</div>\n"
                
                # 교훈 추가 (있을 경우)
                if moral_lesson:
                    html += f'<p class="moral-lesson">교훈: {moral_lesson}</p>\n'
                
                html += "</div>\n"
            
            # HTML 종료
            html += """
            </body>
            </html>
            """
            
            return html
            
        except Exception as e:
            logger.error(f"HTML 포맷팅 중 오류 발생: {e}")
            return f"<html><body><h1>오류 발생</h1><p>{e}</p></body></html>" 