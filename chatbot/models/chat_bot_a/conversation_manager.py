"""
대화 관리를 담당하는 모듈
"""
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import sys

sys.path.append("/Users/b._.chan/Documents/University/캡스톤디자인/AI/CCB_AI")
from shared.utils.logging_utils import get_module_logger
from shared.utils.file_utils import save_json, load_json

logger = get_module_logger(__name__)

class ConversationManager:
    """
    대화 내역 관리 및 상태 유지를 담당하는 클래스
    """
    
    def __init__(self, token_limit: int = 10000):
        """
        대화 관리자 초기화
        
        Args:
            token_limit: 전체 대화에서 사용 가능한 최대 토큰 수
        """
        self.conversation_history = []  # 대화 내역 저장 (role, content)
        
        # 토큰 사용량 추적
        self.token_usage = {
            "total_prompt": 0,
            "total_completion": 0,
            "total": 0
        }
        
        # 토큰 제한
        self.token_limit = token_limit
        
        # 토큰 제한 도달 시 메시지
        self.token_limit_reached_message = "토큰 제한에 걸렸으니 그만 써라 좀..."
    
    def add_message(self, role: str, content: str) -> None:
        """
        대화 내역에 메시지 추가
        
        Args:
            role: 메시지 역할 (user, assistant, system)
            content: 메시지 내용
        """
        self.conversation_history.append({
            "role": role,
            "content": content
        })
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """
        대화 내역 반환
        
        Returns:
            List[Dict[str, str]]: 대화 내역 (role, content)
        """
        return self.conversation_history
    
    def get_recent_messages(self, count: int = 5) -> List[Dict[str, str]]:
        """
        최근 대화 내역 반환
        
        Args:
            count: 반환할 메시지 수
            
        Returns:
            List[Dict[str, str]]: 최근 대화 내역
        """
        return self.conversation_history[-count:] if len(self.conversation_history) >= count else self.conversation_history
    
    def update_token_usage(self, prompt_tokens: int, completion_tokens: int) -> bool:
        """
        토큰 사용량 업데이트
        
        Args:
            prompt_tokens: 프롬프트 토큰 수
            completion_tokens: 응답 토큰 수
            
        Returns:
            bool: 토큰 제한 내에 있으면 True, 아니면 False
        """
        self.token_usage["total_prompt"] += prompt_tokens
        self.token_usage["total_completion"] += completion_tokens
        self.token_usage["total"] = self.token_usage["total_prompt"] + self.token_usage["total_completion"]
        
        # 토큰 제한 체크
        if self.token_usage["total"] >= self.token_limit * 0.85:
            logger.warning(f"토큰 사용량 85% 도달: {self.token_usage['total']}/{self.token_limit}")
        
        # 토큰 제한 도달 여부 반환
        return self.token_usage["total"] < self.token_limit
    
    def is_token_limit_reached(self) -> bool:
        """
        토큰 제한 도달 여부 확인
        
        Returns:
            bool: 토큰 제한에 도달했으면 True, 아니면 False
        """
        return self.token_usage["total"] >= self.token_limit
    
    def get_token_usage(self) -> Dict[str, int]:
        """
        현재 토큰 사용량 반환
        
        Returns:
            Dict[str, int]: 토큰 사용량 정보
        """
        return self.token_usage
    
    def save_conversation(self, file_path: Union[str, Path], additional_data: Optional[Dict] = None) -> bool:
        """
        대화 내역 및 상태 저장
        
        Args:
            file_path: 저장할 파일 경로
            additional_data: 함께 저장할 추가 데이터
            
        Returns:
            bool: 성공 여부
        """
        try:
            # 저장할 데이터 구성
            data = {
                "conversation_history": self.conversation_history,
                "token_usage": self.token_usage
            }
            
            # 추가 데이터가 있으면 병합
            if additional_data:
                data.update(additional_data)
            
            # 파일 저장
            result = save_json(data, file_path)
            
            if result:
                logger.info(f"대화 내역 저장 완료: {file_path}")
            
            return result
            
        except Exception as e:
            logger.error(f"대화 내역 저장 실패: {e}")
            return False
    
    def load_conversation(self, file_path: Union[str, Path]) -> Optional[Dict]:
        """
        대화 내역 및 상태 로드
        
        Args:
            file_path: 로드할 파일 경로
            
        Returns:
            Optional[Dict]: 로드된 데이터 (실패 시 None)
        """
        try:
            # 파일 로드
            data = load_json(file_path)
            
            if not data:
                logger.error(f"대화 내역 로드 실패: 파일이 없거나 빈 파일 - {file_path}")
                return None
            
            # 대화 내역 복원
            self.conversation_history = data.get("conversation_history", [])
            
            # 토큰 사용량 복원
            self.token_usage = data.get("token_usage", {"total_prompt": 0, "total_completion": 0, "total": 0})
            
            logger.info(f"대화 내역 로드 완료: {file_path}")
            return data
            
        except Exception as e:
            logger.error(f"대화 내역 로드 실패: {e}")
            return None
    
    def clear_conversation(self) -> None:
        """대화 내역 초기화"""
        self.conversation_history = []
        self.token_usage = {
            "total_prompt": 0,
            "total_completion": 0,
            "total": 0
        }
        logger.info("대화 내역 초기화 완료") 