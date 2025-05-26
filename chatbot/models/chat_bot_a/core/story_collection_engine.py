"""
부기 (ChatBot A) 이야기 수집 엔진

이야기 요소 수집, 단계 관리, 분석 및 구조화를 담당하는 핵심 엔진
기존 StoryCollector와 StoryAnalyzer의 기능을 통합
"""
from typing import Dict, List, Any, Optional, Set
import random
import json

from shared.utils.logging_utils import get_module_logger
from shared.utils.openai_utils import generate_chat_completion

logger = get_module_logger(__name__)

class StoryCollectionEngine:
    """
    이야기 수집, 분석 및 구조화를 담당하는 통합 엔진
    
    기존 StoryCollector와 StoryAnalyzer의 기능을 하나의 엔진으로 통합하여
    이야기 수집 과정을 더 효율적으로 관리
    """
    
    def __init__(self, openai_client=None, rag_system=None):
        """
        이야기 수집 엔진 초기화
        
        Args:
            openai_client: OpenAI API 클라이언트
            rag_system: RAG 시스템 인스턴스
        """
        # 이야기 수집 단계 (character, setting, problem, resolution)
        self.story_stage = "character"
        
        # 이야기 요소 수집 상태
        self.story_elements = {
            "character": {"count": 0, "topics": set()},
            "setting": {"count": 0, "topics": set()},
            "problem": {"count": 0, "topics": set()},
            "resolution": {"count": 0, "topics": set()}
        }
        
        # 마지막 단계 전환 대화 턴 수
        self.last_stage_transition = 0
        
        # 분석 관련 속성
        self.openai_client = openai_client
        self.rag_system = rag_system
        self.story_outline = None
    
    # === 이야기 수집 관련 메서드 (기존 StoryCollector) ===
    
    def get_current_stage(self) -> str:
        """
        현재 이야기 수집 단계 반환
        
        Returns:
            str: 현재 이야기 수집 단계
        """
        return self.story_stage
    
    def analyze_user_response(self, user_input: str, openai_client=None) -> None:
        """
        사용자 응답을 분석하여 이야기 요소를 추출
        
        Args:
            user_input (str): 사용자 입력 텍스트
            openai_client: OpenAI API 클라이언트 (선택적)
        """
        client = openai_client or self.openai_client
        
        if not client:
            self.story_elements[self.story_stage]["count"] += 1
            return
            
        # GPT를 통해 사용자 응답 분석
        system_message = "사용자의 응답에서 주요 키워드와 토픽을 추출하세요."
        
        try:
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": f"다음 텍스트에서 3-5개의 주요 키워드를 콤마로 구분하여 추출하세요: '{user_input}'"}
            ]
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=100,
                temperature=0.3
            )
            
            key_parts = response.choices[0].message.content.split(',')
            keywords = {k.strip() for k in key_parts if len(k.strip()) > 1}
            
            # 현재 단계에 키워드 추가
            if keywords:
                self.story_elements[self.story_stage]["topics"].update(keywords)
                self.story_elements[self.story_stage]["count"] += 1
        
        except Exception as e:
            logger.error(f"사용자 응답 분석 중 오류 발생: {e}")
            # 오류 시 기본값 설정
            self.story_elements[self.story_stage]["count"] += 1

    def should_transition_to_next_stage(self, conversation_length: int) -> bool:
        """
        현재 단계에서 다음 단계로 전환해야 하는지 결정
        
        Args:
            conversation_length: 현재까지의 전체 대화 메시지 수
            
        Returns:
            bool: 다음 단계로 전환해야 하면 True, 아니면 False
        """
        current_stage = self.story_stage
        current_turn = conversation_length // 2  # 대화 턴 수 (질문-답변 쌍)
        
        # 마지막 전환 이후 최소 2턴 이상 지났는지 확인
        if current_turn - self.last_stage_transition < 2:
            return False
            
        # 단계별 전환 기준
        transition_criteria = {
            "character": lambda: self.story_elements["character"]["count"] >= 3 or current_turn > 4,
            "setting": lambda: self.story_elements["setting"]["count"] >= 2 or current_turn > 8,
            "problem": lambda: self.story_elements["problem"]["count"] >= 2 or current_turn > 12,
            "resolution": lambda: False  # 마지막 단계는 전환하지 않음
        }
        
        # 현재 단계에 대한 전환 기준 확인
        should_transition = transition_criteria.get(current_stage, lambda: False)()
        
        # 확률적 요소 추가 (단계가 진행될수록 전환 확률 증가)
        stages = ["character", "setting", "problem", "resolution"]
        current_index = stages.index(current_stage)
        
        # 이미 충분한 대화가 이루어졌다면 전환 확률 증가
        if current_index < len(stages) - 1:
            # 단계별 기본 전환 확률
            base_transition_prob = 0.1 * (current_index + 1)
            
            # 대화가 길어질수록 전환 확률 증가
            turn_factor = min(0.5, 0.05 * (current_turn // 2))
            
            # 최종 전환 확률
            transition_prob = base_transition_prob + turn_factor
            
            # 전환 기준을 충족하면 확률 추가 증가
            if should_transition:
                transition_prob += 0.3
                
            return random.random() < transition_prob
            
        return False

    def transition_to_next_stage(self, conversation_length: int) -> bool:
        """
        현재 단계에서 다음 단계로 전환
        
        Args:
            conversation_length: 현재까지의 전체 대화 메시지 수
            
        Returns:
            bool: 전환 성공 여부
        """
        stages = ["character", "setting", "problem", "resolution"]
        current_index = stages.index(self.story_stage)
        
        if current_index < len(stages) - 1:
            self.story_stage = stages[current_index + 1]
            self.last_stage_transition = conversation_length // 2
            logger.info(f"이야기 수집 단계 전환: {stages[current_index]} -> {self.story_stage}")
            return True
        
        return False
    
    def get_story_elements(self) -> Dict[str, Dict[str, Any]]:
        """
        수집된 이야기 요소 반환
        
        Returns:
            Dict: 수집된 이야기 요소 (단계별 카운트 및 토픽)
        """
        # 출력용 딕셔너리 생성 (set을 list로 변환)
        result = {}
        for stage, data in self.story_elements.items():
            result[stage] = {
                "count": data["count"],
                "topics": list(data["topics"])
            }
        
        return result
    
    # === 이야기 분석 관련 메서드 (기존 StoryAnalyzer) ===
    
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
        
        if len(conversation_history) < 3:
            return f"{child_name}와의 대화가 아직 충분하지 않습니다."
        
        # 대화 내용을 텍스트로 변환
        conversation_text = ""
        for message in conversation_history[-10:]:  # 최근 10개 메시지만 사용
            role = "아이" if message["role"] == "user" else "부기"
            conversation_text += f"{role}: {message['content']}\n"
        
        system_message = f"""
        당신은 {age_group}세 아이 {child_name}와의 대화를 요약하는 전문가입니다.
        대화에서 나타난 아이의 관심사, 언급된 이야기 요소들, 그리고 전반적인 대화 흐름을 간단히 요약해주세요.
        """
        
        try:
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": f"다음 대화를 요약해주세요:\n\n{conversation_text}"}
            ]
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=300,
                temperature=0.5
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"대화 요약 생성 중 오류 발생: {e}")
            return f"대화 요약 생성 중 오류가 발생했습니다: {str(e)}"
    
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
    
    # === 상태 관리 메서드 ===
    
    def update_from_saved_data(self, saved_data: Dict) -> None:
        """
        저장된 데이터로부터 상태 업데이트
        
        Args:
            saved_data: 저장된 스토리 수집 데이터
        """
        if "story_stage" in saved_data:
            self.story_stage = saved_data["story_stage"]
        
        story_elements = saved_data.get("story_elements", {})
        for key, value in story_elements.items():
            if key in self.story_elements:
                self.story_elements[key]["count"] = value.get("count", 0)
                self.story_elements[key]["topics"] = set(value.get("topics", []))
        
        self.last_stage_transition = saved_data.get("last_stage_transition", 0)
        
        # 스토리 개요 복원
        if "story_outline" in saved_data:
            self.story_outline = saved_data["story_outline"]
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        이야기 수집 통계 정보 반환
        
        Returns:
            Dict[str, Any]: 수집 통계 정보
        """
        total_elements = sum(data["count"] for data in self.story_elements.values())
        total_topics = sum(len(data["topics"]) for data in self.story_elements.values())
        
        return {
            "current_stage": self.story_stage,
            "total_elements_collected": total_elements,
            "total_unique_topics": total_topics,
            "stage_progress": {
                stage: {
                    "count": data["count"],
                    "topics_count": len(data["topics"])
                }
                for stage, data in self.story_elements.items()
            },
            "last_stage_transition": self.last_stage_transition,
            "has_story_outline": self.story_outline is not None
        } 