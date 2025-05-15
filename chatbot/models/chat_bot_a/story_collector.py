"""
이야기 요소 수집을 담당하는 모듈
"""
from typing import Dict, List, Any, Optional, Set
import random

from shared.utils.logging_utils import get_module_logger

logger = get_module_logger(__name__)

class StoryCollector:
    """
    이야기 요소 수집 및 단계 관리를 담당하는 클래스
    """
    
    def __init__(self):
        """
        스토리 수집기 초기화
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
    
    def get_current_stage(self) -> str:
        """
        현재 이야기 수집 단계 반환
        
        Returns:
            str: 현재 이야기 수집 단계
        """
        return self.story_stage
    
    def analyze_user_response(self, user_input: str, openai_client) -> None:
        """
        사용자 응답을 분석하여 이야기 요소를 추출
        
        Args:
            user_input (str): 사용자 입력 텍스트
            openai_client: OpenAI API 클라이언트
        """
        if not openai_client:
            self.story_elements[self.story_stage]["count"] += 1
            return
            
        # GPT를 통해 사용자 응답 분석
        system_message = "사용자의 응답에서 주요 키워드와 토픽을 추출하세요."
        
        try:
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": f"다음 텍스트에서 3-5개의 주요 키워드를 콤마로 구분하여 추출하세요: '{user_input}'"}
            ]
            
            response = openai_client.chat.completions.create(
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