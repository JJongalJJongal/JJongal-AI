"""
이야기 분석 및 구조화를 담당하는 모듈
"""
from typing import Dict, List, Any, Optional
import json

from shared.utils.logging_utils import get_module_logger
from shared.utils.openai_utils import generate_chat_completion

logger = get_module_logger(__name__)

class StoryAnalyzer:
    """
    대화 내용에서 이야기 요소를 분석하고 구조화하는 클래스
    """
    
    def __init__(self, openai_client=None, rag_system=None):
        """
        스토리 분석기 초기화
        
        Args:
            openai_client: OpenAI API 클라이언트
            rag_system: RAG 시스템 인스턴스
        """
        self.openai_client = openai_client
        self.rag_system = rag_system
        self.story_outline = None
    
    def get_conversation_summary(self, conversation_history: List[Dict], 
                                 child_name: str = "", age_group: int = 5) -> str:
        """
        대화 내용을 요약하는 함수
        
        Args:
            conversation_history: 대화 내역 목록
            child_name: 아이 이름
            age_group: 아이 연령대
            
        Returns:
            str: 대화 내용 요약
        """
        if not self.openai_client:
            return "OpenAI 클라이언트가 초기화되지 않아 대화 요약을 생성할 수 없습니다."
            
        try:
            # 대화 내용 요약 프롬프트 생성
            prompt = f"""
            {child_name}({age_group}세)와의 대화를 요약해줘.
            다음 사항에 중점을 두고 요약해줘:
            1. 주요 토픽과 관심사
            2. 이야기 수집 단계별 내용 (캐릭터, 배경, 문제, 해결책)
            3. 아이의 창의적인 아이디어
            4. 교육적 가치나 교훈
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "당신은 대화 분석 전문가입니다."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=300
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"대화 요약 중 오류 발생: {str(e)}")
            return "대화 내용을 요약할 수 없습니다."
    
    def suggest_story_theme(self, conversation_history: List[Dict], 
                          child_name: str = "", age_group: int = 5,
                          interests: List[str] = None, 
                          story_collection_prompt: str = "") -> Dict:
        """
        수집된 대화 내용을 바탕으로 이야기 주제 제안
        
        Args:
            conversation_history: 대화 내역 목록
            child_name: 아이 이름
            age_group: 아이 연령대
            interests: 아이의 관심사 목록
            story_collection_prompt: 이야기 수집을 위한 프롬프트 템플릿
            
        Returns:
            Dict: 이야기 주제 및 줄거리 포맷
        """
        # 대화 내용이 충분한지 확인
        if len(conversation_history) < 5:
                return {
                "theme": "아직 충분한 대화가 수집되지 않았습니다",
                "characters": ["미정"],
                "setting": "미정",
                "plot_summary": "더 많은 대화가 필요합니다",
                "educational_value": "미정",
                "target_age": age_group
            }
        
        # 관심사 문자열 준비
        interests_str = ", ".join(interests) if interests else "다양한 주제"
        prompt = story_collection_prompt
        
        # RAG 시스템을 사용하여 주제 풍부화
        if self.rag_system and interests:
            # 기본 주제 (아이의 관심사 기반)
            base_theme = f"{interests[0]}와 관련된 모험" if interests else "모험과 우정"
            # 주제 풍부화
            enriched_theme = self.rag_system.enrich_story_theme(base_theme, age_group)
            # 프롬프트에 정보 추가
            prompt += f"\n\n참고 정보:\n주제: {enriched_theme}"
            
        system_message = "당신은 아이들과의 대화를 바탕으로 동화 줄거리를 구성하는 전문가입니다."
        
        # 대화 내용 요약
        conversation_summary = self.get_conversation_summary(conversation_history, child_name, age_group)
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt + "\n\n대화 내용 요약:\n" + conversation_summary}
        ]
        
        try:
            # GPT 요청
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=800,
                temperature=0.7
            )
            
            # 응답 파싱 시도
            response_text = response.choices[0].message.content
            
            try:
                # JSON 응답 파싱
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                
                if json_start >= 0 and json_end > json_start:
                    json_str = response_text[json_start:json_end]
                    story_data = json.loads(json_str)
                    
                    self.story_outline = story_data
                    
                    # RAG 시스템에 스토리 추가
                    if self.rag_system:
                        try:
                            self.rag_system.add_story_to_vectordb(story_data)
                        except Exception as e:
                            logger.error(f"RAG 시스템에 스토리 추가 실패: {e}")
                    
                    return story_data
                else:
                    # JSON이 아닌 경우 수동 파싱 시도
                    return self._manual_parse_story(response_text)
            
            except json.JSONDecodeError:
                # JSON 파싱 실패 시 수동 파싱
                return self._manual_parse_story(response_text)
            
        except Exception as e:
            # 오류 발생 시 기본 응답
            logger.error(f"이야기 주제 생성 중 오류 발생: {e}")
            return {
                "theme": f"오류 발생: {str(e)}",
                "characters": ["미정"],
                "setting": "미정",
                "plot_summary": "오류로 인해 생성할 수 없습니다",
                "educational_value": "미정",
                "target_age": age_group
            }
    
    def _manual_parse_story(self, response_text: str) -> Dict:
        """
        GPT 응답 텍스트에서 이야기 요소를 수동으로 파싱
        
        Args:
            response_text (str): GPT 응답 텍스트
            
        Returns:
            Dict: 이야기 주제 및 줄거리 포맷
        """
        try:
            # 기본 이야기 구조 생성
            story_data = {
                "theme": "",
                "characters": [],
                "setting": "",
                "plot_summary": "",
                "educational_value": "",
                "target_age": 5
            }
            
            # 텍스트에서 주제 추출 시도
            theme_match = None
            for pattern in ["주제:", "테마:", "이야기 주제:", "theme:"]:
                index = response_text.lower().find(pattern.lower())
                if index >= 0:
                    line_end = response_text.find("\n", index)
                    if line_end < 0:
                        line_end = len(response_text)
                    theme_match = response_text[index + len(pattern):line_end].strip()
                    if theme_match:
                        break
            
            if theme_match:
                story_data["theme"] = theme_match
            else:
                # 첫 줄을 주제로 가정
                first_line_end = response_text.find("\n")
                if first_line_end > 0:
                    story_data["theme"] = response_text[:first_line_end].strip()
                else:
                    story_data["theme"] = "추출된 주제 없음"
            
            # 텍스트에서 캐릭터 추출 시도
            characters_match = None
            for pattern in ["캐릭터:", "등장인물:", "characters:"]:
                index = response_text.lower().find(pattern.lower())
                if index >= 0:
                    line_end = response_text.find("\n", index)
                    if line_end < 0:
                        line_end = len(response_text)
                    characters_match = response_text[index + len(pattern):line_end].strip()
                    if characters_match:
                        # 쉼표로 분리된 캐릭터 목록 생성
                        story_data["characters"] = [c.strip() for c in characters_match.split(",")]
                        break
            
            # 텍스트에서 배경 추출 시도
            setting_match = None
            for pattern in ["배경:", "장소:", "setting:"]:
                index = response_text.lower().find(pattern.lower())
                if index >= 0:
                    line_end = response_text.find("\n", index)
                    if line_end < 0:
                        line_end = len(response_text)
                    setting_match = response_text[index + len(pattern):line_end].strip()
                    if setting_match:
                        story_data["setting"] = setting_match
                        break
            
            # 텍스트에서 줄거리 추출 시도
            plot_match = None
            for pattern in ["줄거리:", "스토리:", "plot:", "plot summary:"]:
                index = response_text.lower().find(pattern.lower())
                if index >= 0:
                    next_heading = float('inf')
                    for p in ["교육적 가치:", "educational value:", "포인트:"]:
                        next_idx = response_text.lower().find(p.lower(), index)
                        if next_idx > 0 and next_idx < next_heading:
                            next_heading = next_idx
                    
                    if next_heading < float('inf'):
                        plot_match = response_text[index + len(pattern):next_heading].strip()
                    else:
                        plot_match = response_text[index + len(pattern):].strip()
                    
                    if plot_match:
                        story_data["plot_summary"] = plot_match
                        break
            
            # 텍스트에서 교육적 가치 추출 시도
            value_match = None
            for pattern in ["교육적 가치:", "교훈:", "educational value:"]:
                index = response_text.lower().find(pattern.lower())
                if index >= 0:
                    value_match = response_text[index + len(pattern):].strip()
                    story_data["educational_value"] = value_match
                    break
            
            # 줄거리가 여전히 비어있으면 전체 텍스트를 줄거리로 설정
            if not story_data["plot_summary"]:
                story_data["plot_summary"] = response_text
            
            self.story_outline = story_data
            return story_data
            
        except Exception as e:
            logger.error(f"이야기 수동 파싱 중 오류 발생: {str(e)}")
            # 오류 시 기본 응답
            return {
                "theme": "파싱 오류",
                "characters": ["미정"],
                "setting": "미정",
                "plot_summary": response_text[:200] + "...",  # 일부만 반환
                "educational_value": "미정",
                "target_age": 5
            }
    
    def get_story_outline(self) -> Optional[Dict]:
        """
        생성된 이야기 개요 반환
        
        Returns:
            Optional[Dict]: 이야기 개요 또는 None
        """
        return self.story_outline 