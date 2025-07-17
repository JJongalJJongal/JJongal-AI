from fastapi import WebSocket
from typing import Dict, List, Any


class ConnectionManager: 
    def __init__(self):
        # client_id 별로 Websocket 연결 관리
        self.active_connections: Dict[str, WebSocket] = {}
        # 사용자별 연결 관리
        self.user_connections: Dict[str, List[str]] = {}
        # 연결 메타데이터
        self.connection_metadata: Dict[str, Dict[str, Any]] = {}
        
    async def connect(self, client_id: str, websocket: WebSocket) -> None:
        """WebSocket 연결 수락 및 등록"""
        await websocket.accept() # 연결 수락
        self.active_connections[client_id] = websocket # 사용자 연결 등록
        
    def disconnect(self, client_id: str) -> None:
        """WebSocket 연결 해제"""
        if client_id in self.active_connections: # 사용자 연결 존재 시
            del self.active_connections[client_id] # 연결 해제
            
        # 특정 사용자 연결 해제
        if client_id in 
        