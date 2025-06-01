"""
WebSocket 연결 관리 엔진

기존 connection.py의 기능을 통합하고 개선한 연결 관리 엔진
"""
import time
import asyncio
from typing import Dict, List, Any, Optional
from fastapi import WebSocket

from shared.utils.logging_utils import get_module_logger
from chatbot.utils.conversation_utils import save_conversation
from shared.utils.file_utils import cleanup_temp_files

logger = get_module_logger(__name__)

class ConnectionEngine:
    """
    WebSocket 연결 관리를 담당하는 통합 엔진
    
    주요 기능:
    - 클라이언트 연결 관리
    - 세션 상태 추적
    - 비활성 연결 정리
    - ChatBot 인스턴스 관리
    """
    
    def __init__(self, connection_timeout: int = 30 * 60):
        """
        연결 엔진 초기화
        
        Args:
            connection_timeout: 연결 타임아웃 시간 (초, 기본값: 30분)
        """
        self.active_connections: Dict[str, Dict[str, Any]] = {}
        self.chatbot_b_instances: Dict[str, Dict[str, Any]] = {}
        self.connection_timeout = connection_timeout
        self.shutdown_event = asyncio.Event()
        
        logger.info("연결 엔진 초기화 완료")
    
    # ==========================================
    # 기본 연결 관리
    # ==========================================
    
    def get_client_count(self) -> int:
        """활성 연결 수 반환"""
        return len(self.active_connections)
    
    def get_client_info(self, client_id: str) -> Optional[Dict[str, Any]]:
        """클라이언트 정보 반환"""
        return self.active_connections.get(client_id)
    
    def add_client(self, client_id: str, connection_info: Dict[str, Any]) -> None:
        """
        클라이언트 연결 추가
        
        Args:
            client_id: 클라이언트 식별자
            connection_info: 연결 정보 딕셔너리
        """
        self.active_connections[client_id] = connection_info
        logger.info(f"클라이언트 추가: {client_id} (총 {len(self.active_connections)}개 연결)")
    
    def remove_client(self, client_id: str) -> None:
        """클라이언트 연결 제거"""
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            logger.info(f"클라이언트 제거: {client_id} (총 {len(self.active_connections)}개 연결)")
    
    async def close_all_connections(self) -> None:
        """모든 연결 종료"""
        for client_id, connection_info in self.active_connections.items():
            try:
                if "websocket" in connection_info:
                    await connection_info["websocket"].close()
            except Exception as e:
                logger.error(f"연결 종료 중 오류: {e}")
                
        self.active_connections.clear()
        self.chatbot_b_instances.clear()
        logger.info("모든 연결 종료 완료")
    
    async def handle_disconnect(self, client_id: str) -> None:
        """
        클라이언트 연결 종료 처리
        
        Args:
            client_id: 클라이언트 식별자
        """
        if client_id not in self.active_connections:
            return
            
        connection_info = self.active_connections[client_id]
        
        try:
            # 대화 내용 저장
            if "chatbot" in connection_info and "child_name" in connection_info:
                chatbot = connection_info["chatbot"]
                child_name = connection_info.get("child_name", "unknown")
                if not hasattr(chatbot, "get_conversation_history"):
                    logger.error(f"[DISCONNECT] chatbot에 get_conversation_history 없음! 타입: {type(chatbot)} client_id: {client_id}")
                else:
                    await save_conversation(chatbot, child_name, client_id)
            
            # 임시 파일 정리
            if "temp_files" in connection_info:
                cleanup_temp_files(connection_info["temp_files"])
            
            # ChatBot B 인스턴스 정리
            if client_id in self.chatbot_b_instances:
                del self.chatbot_b_instances[client_id]
                logger.info(f"ChatBot B 인스턴스 정리: {client_id}")
            
            # 연결 정보 삭제
            self.remove_client(client_id)
            logger.info(f"클라이언트 연결 종료 처리 완료: {client_id}")
            
        except Exception as e:
            logger.error(f"연결 종료 처리 중 오류: {e}")
            # 오류가 발생해도 연결은 제거
            self.remove_client(client_id)
    
    # ==========================================
    # ChatBot B 인스턴스 관리
    # ==========================================
    
    def add_chatbot_b_instance(self, client_id: str, instance_data: Dict[str, Any]) -> None:
        """ChatBot B 인스턴스 추가"""
        self.chatbot_b_instances[client_id] = instance_data
        logger.info(f"ChatBot B 인스턴스 추가: {client_id}")
    
    def get_chatbot_b_instance(self, client_id: str) -> Optional[Dict[str, Any]]:
        """ChatBot B 인스턴스 조회"""
        return self.chatbot_b_instances.get(client_id)
    
    def update_chatbot_b_activity(self, client_id: str) -> None:
        """ChatBot B 활동 시간 업데이트"""
        if client_id in self.chatbot_b_instances:
            self.chatbot_b_instances[client_id]["last_activity"] = time.time()
    
    # ==========================================
    # 비활성 연결 정리
    # ==========================================
    
    async def cleanup_inactive_clients(self) -> None:
        """비활성 클라이언트 정리 함수"""
        logger.info("비활성 클라이언트 정리 태스크 시작")
        
        while not self.shutdown_event.is_set():
            try:
                # 5분마다 실행하되, 1초마다 종료 이벤트 확인
                for _ in range(300):
                    await asyncio.sleep(1)
                    if self.shutdown_event.is_set():
                        break
                        
                # 종료 이벤트가 설정되었다면 루프 종료
                if self.shutdown_event.is_set():
                    break
                    
                await self._cleanup_inactive_connections()
                await self._cleanup_inactive_chatbot_b_instances()
            
            except Exception as e:
                logger.error(f"비활성 클라이언트 정리 중 오류 발생: {str(e)}")
        
        logger.info("비활성 클라이언트 정리 태스크 완전 종료")
    
    async def _cleanup_inactive_connections(self) -> None:
        """비활성 일반 연결 정리"""
        current_time = time.time()
        
        # 30분 이상 비활성인 클라이언트 식별
        inactive_clients = []
        for client_id, connection_info in self.active_connections.items():
            if current_time - connection_info.get("start_time", 0) > self.connection_timeout:
                inactive_clients.append(client_id)
        
        # 비활성 클라이언트 정리
        for client_id in inactive_clients:
            logger.info(f"타임아웃으로 인한 연결 종료: {client_id}")
            await self.handle_disconnect(client_id)
    
    async def _cleanup_inactive_chatbot_b_instances(self) -> None:
        """비활성 ChatBot B 인스턴스 정리"""
        current_time = time.time()
        
        inactive_chatbot_bs = []
        for client_id, client_data in self.chatbot_b_instances.items():
            if current_time - client_data.get("last_activity", 0) > self.connection_timeout:
                inactive_chatbot_bs.append(client_id)
        
        for client_id in inactive_chatbot_bs:
            del self.chatbot_b_instances[client_id]
            logger.info(f"비활성 ChatBot B 인스턴스 정리: {client_id}")
    
    # ==========================================
    # 종료 관리
    # ==========================================
    
    def set_shutdown_event(self) -> None:
        """종료 이벤트 설정"""
        self.shutdown_event.set()
        logger.info("종료 이벤트 설정됨")
    
    # ==========================================
    # 상태 조회
    # ==========================================
    
    def get_active_connections_info(self) -> List[Dict[str, Any]]:
        """활성 연결 정보 요약"""
        result = []
        for client_id, connection_info in self.active_connections.items():
            result.append({
                "client_id": client_id,
                "child_name": connection_info.get("child_name", "unknown"),
                "age": connection_info.get("age", 0),
                "connected_since": time.strftime(
                    "%Y-%m-%d %H:%M:%S", 
                    time.localtime(connection_info.get("start_time", 0))
                ),
                "temp_files_count": len(connection_info.get("temp_files", []))
            })
        return result
    
    def get_system_stats(self) -> Dict[str, Any]:
        """시스템 통계 반환"""
        return {
            "active_connections": len(self.active_connections),
            "chatbot_b_instances": len(self.chatbot_b_instances),
            "connection_timeout": self.connection_timeout,
            "shutdown_requested": self.shutdown_event.is_set()
        } 