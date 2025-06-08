"""
WebSocket 세션 관리자

클라이언트 세션 상태, 데이터, 활동 타임스탬프 등을 관리합니다.
"""
import time
from typing import Dict, Any, Optional, List
from pathlib import Path
import json
from datetime import datetime

from shared.utils.logging_utils import get_module_logger

logger = get_module_logger(__name__)

SESSION_CLEANUP_INTERVAL = 10 * 60  # 10 minutes, for example
DEFAULT_SESSION_TIMEOUT = 30 * 60   # 30 minutes

# 글로벌 세션 스토어 (WebSocket과 REST API 간 데이터 공유)
class GlobalSessionStore:
    """
    WebSocket과 REST API 간 conversation_data 공유를 위한 글로벌 스토어
    메모리 + 파일 백업으로 데이터 안전성 보장
    """
    
    def __init__(self):
        self.conversation_data_store = {}  # {child_name: conversation_data}
        self.session_metadata = {}  # {child_name: metadata}
        
        # 파일 백업 경로 설정
        self.backup_dir = Path("output") / "session_backup"
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # 시작 시 백업 파일에서 복원
        self._restore_from_backup()
        
        logger.info(f"GlobalSessionStore 초기화 완료 (백업 경로: {self.backup_dir})")
    
    def _get_backup_file_path(self, child_name: str) -> Path:
        """백업 파일 경로 생성"""
        safe_name = "".join(c for c in child_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        return self.backup_dir / f"{safe_name}_session.json"
    
    def _save_to_backup(self, child_name: str, conversation_data: Dict[str, Any]) -> None:
        """메모리 데이터를 파일에 백업"""
        try:
            backup_file = self._get_backup_file_path(child_name)
            backup_data = {
                "child_name": child_name,
                "conversation_data": conversation_data,
                "metadata": self.session_metadata.get(child_name, {}),
                "backup_timestamp": datetime.now().isoformat()
            }
            
            with open(backup_file, 'w', encoding='utf-8') as f:
                json.dump(backup_data, f, ensure_ascii=False, indent=2)
            
            logger.debug(f"[BACKUP] 세션 데이터 백업 완료: {child_name}")
            
        except Exception as e:
            logger.error(f"[BACKUP] 세션 백업 실패 {child_name}: {e}")
    
    def _load_from_backup(self, child_name: str) -> Optional[Dict[str, Any]]:
        """백업 파일에서 데이터 복원"""
        try:
            backup_file = self._get_backup_file_path(child_name)
            if backup_file.exists():
                with open(backup_file, 'r', encoding='utf-8') as f:
                    backup_data = json.load(f)
                
                logger.info(f"[BACKUP] 백업에서 세션 복원: {child_name}")
                return backup_data.get("conversation_data")
                
        except Exception as e:
            logger.error(f"[BACKUP] 백업 복원 실패 {child_name}: {e}")
        
        return None
    
    def _restore_from_backup(self) -> None:
        """시작 시 모든 백업 파일에서 데이터 복원"""
        try:
            backup_files = list(self.backup_dir.glob("*_session.json"))
            restored_count = 0
            
            for backup_file in backup_files:
                try:
                    with open(backup_file, 'r', encoding='utf-8') as f:
                        backup_data = json.load(f)
                    
                    child_name = backup_data.get("child_name")
                    conversation_data = backup_data.get("conversation_data")
                    metadata = backup_data.get("metadata", {})
                    
                    if child_name and conversation_data:
                        self.conversation_data_store[child_name] = conversation_data
                        self.session_metadata[child_name] = metadata
                        restored_count += 1
                        
                except Exception as e:
                    logger.warning(f"[BACKUP] 백업 파일 복원 실패 {backup_file}: {e}")
            
            if restored_count > 0:
                logger.info(f"[BACKUP] 시작 시 백업에서 {restored_count}개 세션 복원")
                
        except Exception as e:
            logger.error(f"[BACKUP] 백업 복원 과정 오류: {e}")

    def store_conversation_data(self, child_name: str, conversation_data: Dict[str, Any], client_id: str = None) -> None:
        """
        대화 데이터 저장 (메모리 + 파일 백업)
        
        Args:
            child_name: 아이 이름 (키)
            conversation_data: 대화 데이터
            client_id: 클라이언트 ID (선택사항)
        """
        # 메모리에 저장
        self.conversation_data_store[child_name] = conversation_data
        self.session_metadata[child_name] = {
            "client_id": client_id,
            "stored_at": time.time(),
            "last_accessed": time.time()
        }
        
        # 파일 백업
        self._save_to_backup(child_name, conversation_data)
        
        logger.info(f"[GLOBAL_STORE] 대화 데이터 저장 (메모리+백업): {child_name} ({len(conversation_data.get('messages', []))}개 메시지)")
    
    def get_conversation_data(self, child_name: str) -> Optional[Dict[str, Any]]:
        """
        대화 데이터 조회 (메모리 우선, 백업 파일 fallback)
        
        Args:
            child_name: 아이 이름
            
        Returns:
            Optional[Dict[str, Any]]: 대화 데이터 (없으면 None)
        """
        # 1. 메모리에서 조회 시도
        if child_name in self.conversation_data_store:
            # 접근 시간 업데이트
            if child_name in self.session_metadata:
                self.session_metadata[child_name]["last_accessed"] = time.time()
            
            conversation_data = self.conversation_data_store[child_name]
            logger.info(f"[GLOBAL_STORE] 메모리에서 대화 데이터 조회: {child_name} ({len(conversation_data.get('messages', []))}개 메시지)")
            return conversation_data
        
        # 2. 백업 파일에서 조회 시도
        backup_data = self._load_from_backup(child_name)
        if backup_data:
            # 백업에서 복원한 데이터를 메모리에도 저장
            self.store_conversation_data(child_name, backup_data)
            logger.info(f"[GLOBAL_STORE] 백업에서 대화 데이터 복원: {child_name} ({len(backup_data.get('messages', []))}개 메시지)")
            return backup_data
        
        logger.warning(f"[GLOBAL_STORE] 대화 데이터 없음 (메모리+백업): {child_name}")
        return None

    def remove_conversation_data(self, child_name: str) -> bool:
        """
        대화 데이터 삭제 (메모리 + 백업 파일)
        
        Args:
            child_name: 아이 이름
            
        Returns:
            bool: 삭제 성공 여부
        """
        removed = False
        
        # 메모리에서 삭제
        if child_name in self.conversation_data_store:
            del self.conversation_data_store[child_name]
            if child_name in self.session_metadata:
                del self.session_metadata[child_name]
            removed = True
        
        # 백업 파일 삭제
        try:
            backup_file = self._get_backup_file_path(child_name)
            if backup_file.exists():
                backup_file.unlink()
                removed = True
        except Exception as e:
            logger.error(f"[BACKUP] 백업 파일 삭제 실패 {child_name}: {e}")
        
        if removed:
            logger.info(f"[GLOBAL_STORE] 대화 데이터 삭제 (메모리+백업): {child_name}")
        
        return removed
    
    def cleanup_expired_sessions(self, expiry_time: int = 3600) -> int:
        """
        만료된 세션 정리
        
        Args:
            expiry_time: 만료 시간 (초, 기본값: 1시간)
            
        Returns:
            int: 정리된 세션 수
        """
        current_time = time.time()
        expired_sessions = []
        
        for child_name, metadata in self.session_metadata.items():
            if current_time - metadata["stored_at"] > expiry_time:
                expired_sessions.append(child_name)
        
        for child_name in expired_sessions:
            self.remove_conversation_data(child_name)
        
        if expired_sessions:
            logger.info(f"[GLOBAL_STORE] 만료된 세션 {len(expired_sessions)}개 정리")
        
        return len(expired_sessions)
    
    def get_all_active_sessions(self) -> List[str]:
        """
        활성 세션 목록 반환
        
        Returns:
            List[str]: 활성 세션 아이 이름 목록
        """
        return list(self.conversation_data_store.keys())
    
    def get_session_stats(self) -> Dict[str, Any]:
        """
        세션 통계 정보 반환
        
        Returns:
            Dict[str, Any]: 세션 통계
        """
        current_time = time.time()
        total_sessions = len(self.conversation_data_store)
        total_messages = sum(len(data.get('messages', [])) for data in self.conversation_data_store.values())
        
        recent_sessions = 0
        for metadata in self.session_metadata.values():
            if current_time - metadata["last_accessed"] < 300:  # 5분 이내
                recent_sessions += 1
        
        return {
            "total_sessions": total_sessions,
            "recent_sessions": recent_sessions,
            "total_messages": total_messages,
            "store_keys": list(self.conversation_data_store.keys())
        }

# 글로벌 인스턴스
global_session_store = GlobalSessionStore()

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

    def update_session_data(self, client_id: str, data: Dict[str, Any]) -> bool:
        """세션 데이터를 업데이트합니다."""
        if client_id in self.sessions:
            self.sessions[client_id]["data"].update(data)
            self.sessions[client_id]["last_activity"] = time.time()
            logger.debug(f"세션 데이터 업데이트: {client_id}")
            return True
        logger.warning(f"업데이트할 세션이 없음: {client_id}")
        return False

    def remove_session(self, client_id: str) -> bool:
        """세션을 제거합니다."""
        if client_id in self.sessions:
            del self.sessions[client_id]
            logger.info(f"세션 제거 완료: {client_id}")
            return True
        logger.warning(f"제거할 세션이 없음: {client_id}")
        return False

    def cleanup_expired_sessions(self) -> int:
        """만료된 세션들을 정리합니다."""
        current_time = time.time()
        expired_sessions = []
        
        for client_id, session in self.sessions.items():
            if current_time - session["last_activity"] > self.default_timeout:
                expired_sessions.append(client_id)
        
        for client_id in expired_sessions:
            self.remove_session(client_id)
        
        if expired_sessions:
            logger.info(f"만료된 세션 {len(expired_sessions)}개 정리")
        
        return len(expired_sessions)

    def get_active_sessions(self) -> List[str]:
        """활성 세션 목록을 반환합니다."""
        current_time = time.time()
        active_sessions = []
        
        for client_id, session in self.sessions.items():
            if current_time - session["last_activity"] <= self.default_timeout:
                active_sessions.append(client_id)
        
        return active_sessions

    def get_session_count(self) -> int:
        """현재 세션 수를 반환합니다."""
        return len(self.sessions)

    def get_session_stats(self) -> Dict[str, Any]:
        """세션 통계를 반환합니다."""
        current_time = time.time()
        active_count = len(self.get_active_sessions())
        
        oldest_session = None
        newest_session = None
        
        if self.sessions:
            creation_times = [session["created_at"] for session in self.sessions.values()]
            oldest_session = current_time - min(creation_times)
            newest_session = current_time - max(creation_times)
        
        return {
            "total_sessions": len(self.sessions),
            "active_sessions": active_count,
            "oldest_session_age": oldest_session,
            "newest_session_age": newest_session,
            "default_timeout": self.default_timeout
        }

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

