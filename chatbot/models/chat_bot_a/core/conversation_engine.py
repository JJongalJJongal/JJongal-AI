"""
부기 (ChatBot A) 대화 엔진

대화 관리, 토큰 제한 처리, 대화 저장/로드 기능을 담당하는 핵심 엔진
"""
from typing import List, Dict, Any, Optional
import json
from pathlib import Path

from shared.utils.logging_utils import get_module_logger

logger = get_module_logger(__name__)

class ConversationEngine:
    """
    대화 관리 및 토큰 제한 처리를 담당하는 핵심 엔진
    
    기존 ConversationManager의 기능을 엔진 형태로 재구성
    """
    
    def __init__(self, token_limit: int = 10000):
        """
        대화 엔진 초기화
        
        Args:
            token_limit (int): 전체 대화에서 사용 가능한 최대 토큰 수
        """
        self.conversation_history = []
        self.token_limit = token_limit
        self.token_usage = {
            "total_prompt": 0,
            "total_completion": 0,
            "total": 0
        }
        self.token_limit_reached_message = "죄송해요. 오늘 대화가 너무 길어져서 여기서 마무리해야 할 것 같아요. 다음에 또 재미있는 이야기를 만들어봐요!"
        
    def add_message(self, role: str, content: str) -> None:
        """
        대화 내역에 메시지 추가
        
        Args:
            role (str): 메시지 역할 ('user', 'assistant', 'system')
            content (str): 메시지 내용
        """
        if not isinstance(role, str) or role not in ['user', 'assistant', 'system']:
            logger.warning(f"잘못된 역할: {role}. 'user', 'assistant', 'system' 중 하나여야 합니다.")
            return
            
        if not isinstance(content, str) or not content.strip():
            logger.warning("메시지 내용이 비어있거나 문자열이 아닙니다.")
            return
            
        message = {
            "role": role,
            "content": content.strip()
        }
        
        self.conversation_history.append(message)
        logger.debug(f"메시지 추가됨: {role} - {content[:50]}...")
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """
        전체 대화 내역 반환
        
        Returns:
            List[Dict[str, str]]: 대화 내역 리스트
        """
        return self.conversation_history.copy()
    
    def get_recent_messages(self, count: int = 5) -> List[Dict[str, str]]:
        """
        최근 메시지들 반환
        
        Args:
            count (int): 반환할 메시지 수
            
        Returns:
            List[Dict[str, str]]: 최근 메시지 리스트
        """
        if count <= 0:
            return []
        return self.conversation_history[-count:]
    
    def update_token_usage(self, prompt_tokens: int, completion_tokens: int) -> None:
        """
        토큰 사용량 업데이트
        
        Args:
            prompt_tokens (int): 프롬프트 토큰 수
            completion_tokens (int): 완성 토큰 수
        """
        self.token_usage["total_prompt"] += prompt_tokens
        self.token_usage["total_completion"] += completion_tokens
        self.token_usage["total"] = self.token_usage["total_prompt"] + self.token_usage["total_completion"]
        
        logger.debug(f"토큰 사용량 업데이트: 프롬프트={prompt_tokens}, 완성={completion_tokens}, 총합={self.token_usage['total']}")
    
    def get_token_usage(self) -> Dict[str, int]:
        """
        현재 토큰 사용량 반환
        
        Returns:
            Dict[str, int]: 토큰 사용량 정보
        """
        return self.token_usage.copy()
    
    def is_token_limit_reached(self) -> bool:
        """
        토큰 제한에 도달했는지 확인
        
        Returns:
            bool: 토큰 제한 도달 여부
        """
        return self.token_usage["total"] >= self.token_limit
    
    def get_remaining_tokens(self) -> int:
        """
        남은 토큰 수 반환
        
        Returns:
            int: 남은 토큰 수
        """
        return max(0, self.token_limit - self.token_usage["total"])
    
    def clear_conversation(self) -> None:
        """
        대화 내역 초기화
        """
        self.conversation_history.clear()
        logger.info("대화 내역이 초기화되었습니다.")
    
    def save_conversation(self, file_path: str, additional_data: Optional[Dict] = None) -> bool:
        """
        대화 내역을 JSON 파일로 저장
        
        Args:
            file_path (str): 저장할 파일 경로
            additional_data (Optional[Dict]): 추가로 저장할 데이터
            
        Returns:
            bool: 저장 성공 여부
        """
        try:
            # 저장할 데이터 구성
            save_data = {
                "conversation_history": self.conversation_history,
                "token_usage": self.token_usage,
                "token_limit": self.token_limit
            }
            
            # 추가 데이터가 있으면 병합
            if additional_data:
                save_data.update(additional_data)
            
            # 디렉토리 생성
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            
            # JSON 파일로 저장
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"대화 내역이 저장되었습니다: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"대화 내역 저장 실패: {e}")
            return False
    
    def load_conversation(self, file_path: str) -> Optional[Dict]:
        """
        대화 내역 JSON 파일 로드
        
        Args:
            file_path (str): 로드할 파일 경로
            
        Returns:
            Optional[Dict]: 로드된 데이터 또는 None
        """
        try:
            if not Path(file_path).exists():
                logger.warning(f"파일이 존재하지 않습니다: {file_path}")
                return None
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 대화 내역 복원
            self.conversation_history = data.get("conversation_history", [])
            self.token_usage = data.get("token_usage", {"total_prompt": 0, "total_completion": 0, "total": 0})
            self.token_limit = data.get("token_limit", self.token_limit)
            
            logger.info(f"대화 내역이 로드되었습니다: {file_path}")
            return data
            
        except Exception as e:
            logger.error(f"대화 내역 로드 실패: {e}")
            return None
    
    def get_conversation_stats(self) -> Dict[str, Any]:
        """
        대화 통계 정보 반환
        
        Returns:
            Dict[str, Any]: 대화 통계 정보
        """
        user_messages = [msg for msg in self.conversation_history if msg["role"] == "user"]
        assistant_messages = [msg for msg in self.conversation_history if msg["role"] == "assistant"]
        
        return {
            "total_messages": len(self.conversation_history),
            "user_messages": len(user_messages),
            "assistant_messages": len(assistant_messages),
            "token_usage": self.token_usage,
            "token_limit": self.token_limit,
            "remaining_tokens": self.get_remaining_tokens(),
            "token_usage_percentage": (self.token_usage["total"] / self.token_limit) * 100
        } 