"""
WebSocket 세션 관리자

클라이언트 세션 상태, 데이터, 활동 타임스탬프 등을 관리합니다.
"""
import time
from typing import Dict, Any, Optional, List

from shared.utils.logging_utils import get_module_logger

logger = get_module_logger(__name__)

SESSION_CLEANUP_INTERVAL = 10 * 60  # 10 minutes, for example
DEFAULT_SESSION_TIMEOUT = 30 * 60   # 30 minutes

class SessionManager:
    def __init__(self, default_timeout: int = DEFAULT_SESSION_TIMEOUT):
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.default_timeout = default_timeout
        logger.info(f"SessionManager 초기화 완료. 기본 타임아웃: {default_timeout}초")

    def create_session(self, client_id: str, initial_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """새로운 세션을 생성하거나 기존 세션을 반환하고 활동 시간을 갱신합니다."""
        current_time = time.time()
        if client_id in self.sessions:
            logger.warning(f"이미 존재하는 세션 ({client_id}), 활동 시간 갱신.")
            self.sessions[client_id]["last_activity"] = current_time
            return self.sessions[client_id]
        
        session_data = {
            "created_at": current_time,
            "last_activity": current_time,
            "data": initial_data or {}
        }
        self.sessions[client_id] = session_data
        logger.info(f"세션 생성 완료: {client_id}")
        return session_data

    def get_session_data(self, client_id: str) -> Optional[Dict[str, Any]]:
        """세션 데이터를 반환하고 활동 시간을 갱신합니다."""
        session = self.sessions.get(client_id)
        if session:
            session["last_activity"] = time.time()
            logger.debug(f"세션 데이터 조회 및 활동 시간 갱신: {client_id}")
            return session["data"]
        logger.debug(f"조회할 세션 없음: {client_id}")
        return None

    def update_session_data(self, client_id: str, data_to_update: Dict[str, Any]) -> bool:
        """세션 데이터를 업데이트하고 활동 시간을 갱신합니다."""
        session = self.sessions.get(client_id)
        if session:
            session["data"].update(data_to_update)
            session["last_activity"] = time.time()
            logger.info(f"세션 데이터 업데이트: {client_id}")
            return True
        logger.warning(f"업데이트할 세션 없음: {client_id}")
        return False

    def remove_session(self, client_id: str) -> bool:
        """세션을 제거합니다."""
        if client_id in self.sessions:
            del self.sessions[client_id]
            logger.info(f"세션 제거: {client_id}")
            return True
        logger.warning(f"제거할 세션 없음: {client_id}")
        return False

    def list_active_sessions(self) -> List[str]:
        """현재 활성화된 모든 세션 ID 목록을 반환합니다."""
        return list(self.sessions.keys())

    def get_session_activity_info(self, client_id: str) -> Optional[Dict[str, float]]:
        """세션의 생성 및 마지막 활동 시간을 반환합니다."""
        session = self.sessions.get(client_id)
        if session:
            return {
                "created_at": session["created_at"],
                "last_activity": session["last_activity"]
            }
        return None

    def cleanup_inactive_sessions(self, timeout: Optional[int] = None) -> int:
        """지정된 시간 동안 비활성 상태인 세션을 정리합니다."""
        effective_timeout = timeout if timeout is not None else self.default_timeout
        current_time = time.time()
        cleaned_count = 0
        
        inactive_session_ids = [
            client_id for client_id, session_info in self.sessions.items()
            if current_time - session_info.get("last_activity", 0) > effective_timeout
        ]
        
        for client_id in inactive_session_ids:
            if self.remove_session(client_id):
                cleaned_count += 1
                logger.info(f"비활성 세션 정리됨: {client_id} (타임아웃: {effective_timeout}초)")
        
        if cleaned_count > 0:
            logger.info(f"비활성 세션 정리 완료. 총 {cleaned_count}개 세션 제거됨.")
        else:
            logger.debug("정리할 비활성 세션 없음.")
            
        return cleaned_count

# 비동기 백그라운드 태스크로 세션 정리 (예시, app.py 등에 통합 필요)
# async def periodic_session_cleanup(session_manager: SessionManager, interval_seconds: int = SESSION_CLEANUP_INTERVAL):
#     import asyncio
#     logger.info(f"주기적 세션 정리 태스크 시작 (주기: {interval_seconds}초).")
#     while True:
#         await asyncio.sleep(interval_seconds)
#         logger.debug("주기적 세션 정리 실행...")
#         session_manager.cleanup_inactive_sessions() 