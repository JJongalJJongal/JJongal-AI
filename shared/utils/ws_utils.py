"""
웹소켓 관련 유틸리티 모듈
"""
import logging
import os
import time
import json
from typing import Dict, Any, Optional, Set, Callable, Awaitable, Tuple
from fastapi import WebSocket, WebSocketDisconnect
import asyncio
import jwt
from datetime import datetime, timedelta
from ..configs.app_config import get_env_vars

logger = logging.getLogger(__name__)

def generate_jwt_token(payload: Dict[str, Any], expiry_hours: int = 24) -> str:
    """
    JWT 토큰 생성 함수

    Args:
        payload (Dict[str, Any]): 토큰에 포함될 데이터
        expiry_hours (int): 토큰 만료 시간

    Returns:
        str: 생성된 JWT 토큰
    """

    env_vars = get_env_vars()
    secret_key = env_vars.get("jwt_secret_key", "쫑알쫑알_기본_시크릿키")
    
    # 만료 시간 추가
    expiry = datetime.utcnow() + timedelta(hours=expiry_hours)
    token_data = {
        "exp": expiry,
        "iat": datetime.utcnow(),
        **payload
    }

    token = jwt.encode(token_data, secret_key, algorithm="HS256")
    return token

def decode_jwt_token(token: str) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
    """
    JWT 토큰 디코딩 함수
    
    Args:
        token (str): 디코딩할 JWT 토큰
        
    Returns:
        Tuple[bool, Optional[Dict], Optional[str]]:
            - 성공 여부
            - payload (실패 시 None)
            - 오류 메시지 (성공 시 None)
    """
    
    env_vars = get_env_vars()
    secret_key = env_vars.get("jwt_secret_key", "쫑알쫑알_기본_시크릿키")
    
    try:
        payload = jwt.decode(token, secret_key, algorithms=["HS256"])
        return True, payload, None
    except jwt.ExpiredSignatureError:
        return False, None, "토큰이 만료되었습니다"
    except jwt.InvalidTokenError:
        return False, None, "유효하지 않은 토큰입니다"
    except Exception as e:
        return False, None, f"토큰 디코딩 중 오류 발생: {e}"

def validate_token(token: str) -> bool:
    """
    인증 토큰 검증 함수
    
    Args:
        token (str): 검증할 토큰
    
    Returns:
        bool: 유효한 토큰이면 True, 아니면 False
    """
    env_vars = get_env_vars()
    
    # 개발 모드
    if env_vars.get("enviroment") == "development":
        valid_token = env_vars.get("ws_auth_token", "valid_token")
        if token == valid_token:
            logger.debug("개발 모드 : 기본 토큰 인증 성공")
            return True
    
    # JWT 토큰 검증
    success, payload, error = decode_jwt_token(token)
    
    if success:
        # 추가 검증 로직 구현중

        logger.debug(f"JWT 토큰 검증 성공 : sub={payload.get('sub', 'unknown')}")
        return True
    else:
        logger.warning(f"JWT 토큰 검증 실패 : {error}")
        return False
    
# 테스트 토큰 생성 함수
def create_test_token(user_id: str = "test_user", role: str = "user") -> str:
    """
    테스트용 JWT 토큰 생성

    Args:
        user_id (str): 사용자 ID.
        role (str): 사용자 역할.

    Returns:
        str: 생성된 JWT 토큰
    """

    payload = {
        "sub": user_id, # 토큰 주체
        "role": role, # 사용자 역할
        "test": True # 테스트 토큰 여부
    }
    
    return generate_jwt_token(payload)
    
class ConnectionManager:
    """
    웹소켓 연결 관리자 클래스
    """
    
    def __init__(self, connection_timeout: int = 1800):
        """
        초기화
        
        Args:
            connection_timeout (int): 연결 타임아웃 (초, 기본 30분)
        """
        self.active_connections: Dict[str, Dict[str, Any]] = {}
        self.connection_timeout = connection_timeout
        self.shutdown_event = asyncio.Event()
        
    async def connect(self, websocket: WebSocket, client_id: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        웹소켓 연결 수락 및 관리
        
        Args:
            websocket (WebSocket): 웹소켓 객체
            client_id (str): 클라이언트 ID
            metadata (Optional[Dict[str, Any]]): 메타데이터
        """
        await websocket.accept()
        self.active_connections[client_id] = {
            "websocket": websocket,
            "connected_at": time.time(),
            "last_active": time.time(),
            "metadata": metadata or {}
        }
        logger.info(f"클라이언트 연결됨: {client_id}")
        
    def disconnect(self, client_id: str) -> None:
        """
        웹소켓 연결 해제 및 관리에서 제거
        
        Args:
            client_id (str): 클라이언트 ID
        """
        if client_id in self.active_connections:
            # 연결 시간 계산
            connected_at = self.active_connections[client_id].get("connected_at", 0)
            duration = time.time() - connected_at
            
            logger.info(f"클라이언트 연결 해제: {client_id} (연결 시간: {duration:.1f}초)")
            self.active_connections.pop(client_id)
            
    def update_activity(self, client_id: str) -> None:
        """
        클라이언트 활동 시간 업데이트
        
        Args:
            client_id (str): 클라이언트 ID
        """
        if client_id in self.active_connections:
            self.active_connections[client_id]["last_active"] = time.time()
            
    async def send_json(self, client_id: str, data: Dict[str, Any]) -> bool:
        """
        JSON 데이터 전송
        
        Args:
            client_id (str): 클라이언트 ID
            data (Dict[str, Any]): 전송할 데이터
            
        Returns:
            bool: 전송 성공 여부
        """
        if client_id not in self.active_connections:
            logger.warning(f"존재하지 않는 클라이언트에 데이터 전송 시도: {client_id}")
            return False
            
        try:
            websocket = self.active_connections[client_id]["websocket"]
            await websocket.send_json(data)
            self.update_activity(client_id)
            return True
        except Exception as e:
            logger.error(f"데이터 전송 실패 ({client_id}): {e}")
            return False
            
    async def send_text(self, client_id: str, text: str) -> bool:
        """
        텍스트 데이터 전송
        
        Args:
            client_id (str): 클라이언트 ID
            text (str): 전송할 텍스트
            
        Returns:
            bool: 전송 성공 여부
        """
        if client_id not in self.active_connections:
            logger.warning(f"존재하지 않는 클라이언트에 텍스트 전송 시도: {client_id}")
            return False
            
        try:
            websocket = self.active_connections[client_id]["websocket"]
            await websocket.send_text(text)
            self.update_activity(client_id)
            return True
        except Exception as e:
            logger.error(f"텍스트 전송 실패 ({client_id}): {e}")
            return False
            
    async def broadcast(self, message: Dict[str, Any], exclude: Optional[Set[str]] = None) -> None:
        """
        모든 클라이언트에 메시지 브로드캐스트
        
        Args:
            message (Dict[str, Any]): 전송할 메시지
            exclude (Optional[Set[str]]): 제외할 클라이언트 ID 목록
        """
        exclude_set = exclude or set()
        for client_id in list(self.active_connections.keys()):
            if client_id not in exclude_set:
                await self.send_json(client_id, message)
                
    async def cleanup_inactive_clients(self) -> None:
        """
        비활성 클라이언트 정리
        """
        current_time = time.time()
        to_remove = []
        
        for client_id, conn_info in self.active_connections.items():
            last_active = conn_info.get("last_active", 0)
            if current_time - last_active > self.connection_timeout:
                to_remove.append(client_id)
                
        for client_id in to_remove:
            logger.info(f"비활성 클라이언트 제거: {client_id} (시간 초과)")
            self.disconnect(client_id)
            
    async def start_cleanup_task(self) -> None:
        """
        정리 태스크 시작
        """
        try:
            while not self.shutdown_event.is_set():
                await self.cleanup_inactive_clients()
                await asyncio.sleep(60)  # 1분마다 확인
        except asyncio.CancelledError:
            logger.info("정리 태스크 취소됨")
        except Exception as e:
            logger.error(f"정리 태스크 오류: {e}")
            
    def stop_cleanup_task(self) -> None:
        """
        정리 태스크 중지
        """
        self.shutdown_event.set()
        logger.info("정리 태스크 중지 신호 전송")
        
    def get_client_count(self) -> int:
        """
        현재 연결된 클라이언트 수 반환
        
        Returns:
            int: 클라이언트 수
        """
        return len(self.active_connections)
        
    def get_client_metadata(self, client_id: str) -> Optional[Dict[str, Any]]:
        """
        클라이언트 메타데이터 반환
        
        Args:
            client_id (str): 클라이언트 ID
            
        Returns:
            Optional[Dict[str, Any]]: 메타데이터 (존재하지 않으면 None)
        """
        conn_info = self.active_connections.get(client_id)
        return conn_info.get("metadata") if conn_info else None 