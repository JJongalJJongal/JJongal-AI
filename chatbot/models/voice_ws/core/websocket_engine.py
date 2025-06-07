"""
WebSocket Core Engine

Handles underlying WebSocket logic, message parsing, and custom exceptions.
"""

import json
import asyncio
from typing import Dict, Any, Optional, Union
from fastapi import WebSocket, WebSocketDisconnect as FastAPIWebSocketDisconnect

from shared.utils.logging_utils import get_module_logger

logger = get_module_logger(__name__)

class WebSocketDisconnect(Exception):
    """WebSocket 연결 끊김 예외"""
    def __init__(self, code: int = 1000, reason: str = None):
        self.code = code
        self.reason = reason
        super().__init__(f"WebSocket disconnected with code {code}: {reason}")

class WebSocketEngine:
    """
    WebSocket 통신, 메시지 프레임 관리, 오류 처리 관리
    
    주요 기능:
    - JSON 메시지 송수신
    - 연결 상태 검증
    - 메시지 유효성 검증
    - 오류 처리 및 로깅
    """
    
    def __init__(self):
        logger.info("WebSocketEngine 초기화")
        self.message_validators = {}
        self.error_handlers = {}
    
    async def send_json(self, websocket: WebSocket, data: Dict[str, Any]) -> bool:
        """
        WebSocket에 JSON 데이터 전송
        
        Args:
            websocket: WebSocket 연결 객체
            data: 전송할 데이터 딕셔너리
            
        Returns:
            전송 성공 여부
        """
        try:
            # WebSocket 연결 상태 확인
            if websocket.client_state.value != 1:  # CONNECTED = 1
                logger.warning(f"WebSocket 연결 상태가 유효하지 않음: {websocket.client_state}")
                return False
            
            # 데이터 유효성 검증
            if not isinstance(data, dict) or not data:
                logger.error(f"잘못된 데이터 타입: {type(data)}, dict여야 함")
                return False
            
            # 필수 필드 확인
            if "type" not in data:
                data["type"] = "message"
                logger.warning("메시지 타입이 누락되어 기본값으로 설정됨")
            
            # JSON 직렬화 테스트
            try:
                json_str = json.dumps(data, ensure_ascii=False)
            except (TypeError, ValueError) as e:
                logger.error(f"JSON 직렬화 실패: {e}")
                return False
            
            # WebSocket 전송
            await websocket.send_json(data)
            logger.debug(f"JSON 메시지 전송 성공: {data.get('type', 'unknown')}")
            return True
            
        except FastAPIWebSocketDisconnect:
            logger.warning("WebSocket 연결이 끊어져 메시지 전송 실패")
            raise WebSocketDisconnect(1000, "Connection closed")
        except RuntimeError as e:
            if "websocket.send" in str(e) and "websocket.close" in str(e):
                logger.warning("이미 닫힌 WebSocket에 메시지 전송 시도")
                return False
            logger.error(f"WebSocket 런타임 오류: {e}")
            return False
        except Exception as e:
            logger.error(f"JSON 메시지 전송 중 오류: {e}")
            return False

    async def receive_json(self, websocket: WebSocket, timeout: float = 30.0) -> Optional[Dict[str, Any]]:
        """
        WebSocket에서 JSON 데이터 수신
        
        Args:
            websocket: WebSocket 연결 객체
            timeout: 수신 타임아웃 (초)
            
        Returns:
            수신된 데이터 딕셔너리 또는 None
        """
        try:
            # 타임아웃과 함께 메시지 수신
            data = await asyncio.wait_for(websocket.receive_json(), timeout=timeout)
            
            # 데이터 유효성 검증
            if not isinstance(data, dict):
                logger.warning(f"잘못된 메시지 형식: {type(data)}")
                return None
            
            # 메시지 타입별 유효성 검증
            if "type" in data and data["type"] in self.message_validators:
                if not self.message_validators[data["type"]](data):
                    logger.warning(f"메시지 유효성 검증 실패: {data.get('type')}")
                    return None
            
            logger.debug(f"JSON 메시지 수신 성공: {data.get('type', 'unknown')}")
            return data
            
        except asyncio.TimeoutError:
            logger.debug(f"메시지 수신 타임아웃 ({timeout}초)")
            return None
        except FastAPIWebSocketDisconnect:
            logger.info("WebSocket 연결 끊김 감지")
            raise WebSocketDisconnect(1000, "Connection closed")
        except json.JSONDecodeError as e:
            logger.error(f"JSON 디코딩 오류: {e}")
            return None
        except Exception as e:
            logger.error(f"메시지 수신 중 오류: {e}")
            return None

    async def send_error(self, websocket: WebSocket, error_message: str, error_code: str = "unknown_error") -> bool:
        """
        오류 메시지 전송
        
        Args:
            websocket: WebSocket 연결 객체
            error_message: 오류 메시지
            error_code: 오류 코드
            
        Returns:
            전송 성공 여부
        """
        error_data = {
            "type": "error",
            "error_message": error_message,
            "error_code": error_code,
            "status": "error"
        }
        return await self.send_json(websocket, error_data)

    async def send_status(self, websocket: WebSocket, status: str, message: str = "") -> bool:
        """
        상태 메시지 전송
        
        Args:
            websocket: WebSocket 연결 객체
            status: 상태 코드
            message: 상태 메시지
            
        Returns:
            전송 성공 여부
        """
        status_data = {
            "type": "status",
            "status": status,
            "message": message
        }
        return await self.send_json(websocket, status_data)

    async def ping(self, websocket: WebSocket) -> bool:
        """
        연결 상태 확인용 ping 전송
        
        Args:
            websocket: WebSocket 연결 객체
            
        Returns:
            전송 성공 여부
        """
        ping_data = {
            "type": "ping",
            "message": "connection_check"
        }
        return await self.send_json(websocket, ping_data)

    def add_message_validator(self, message_type: str, validator_func) -> None:
        """
        메시지 타입별 유효성 검증 함수 추가
        
        Args:
            message_type: 메시지 타입
            validator_func: 검증 함수 (data -> bool)
        """
        self.message_validators[message_type] = validator_func
        logger.info(f"메시지 검증기 추가: {message_type}")

    def add_error_handler(self, error_type: str, handler_func) -> None:
        """
        오류 타입별 처리 함수 추가
        
        Args:
            error_type: 오류 타입
            handler_func: 처리 함수
        """
        self.error_handlers[error_type] = handler_func
        logger.info(f"오류 핸들러 추가: {error_type}")

    async def is_connected(self, websocket: WebSocket) -> bool:
        """
        WebSocket 연결 상태 확인
        
        Args:
            websocket: WebSocket 연결 객체
            
        Returns:
            연결 상태 (True: 연결됨, False: 끊어짐)
        """
        try:
            # 1. 기본 연결 상태 확인
            if websocket.client_state.value != 1:  # CONNECTED = 1
                return False
            
            # 2. ping 메시지로 실제 연결 확인
            ping_data = {
                "type": "ping",
                "message": "connection_check"
            }
            await websocket.send_json(ping_data)
            return True
        except:
            return False

    async def handle_disconnect(self, websocket: WebSocket, code: int = 1000, reason: str = "Normal closure") -> None:
        """
        WebSocket 연결 종료 처리
        
        Args:
            websocket: WebSocket 연결 객체
            code: 종료 코드
            reason: 종료 이유
        """
        try:
            await websocket.close(code=code, reason=reason)
            logger.info(f"WebSocket 연결 종료: code={code}, reason={reason}")
        except Exception as e:
            logger.error(f"WebSocket 연결 종료 중 오류: {e}")

    def validate_audio_message(self, data: Dict[str, Any]) -> bool:
        """오디오 메시지 유효성 검증"""
        required_fields = ["type"]
        return all(field in data for field in required_fields)

    def validate_text_message(self, data: Dict[str, Any]) -> bool:
        """텍스트 메시지 유효성 검증"""
        required_fields = ["type", "text"]
        return all(field in data for field in required_fields)

    def validate_story_message(self, data: Dict[str, Any]) -> bool:
        """스토리 메시지 유효성 검증"""
        required_fields = ["type"]
        return all(field in data for field in required_fields)

# 기본 메시지 검증기들을 등록하는 전역 인스턴스
default_websocket_engine = WebSocketEngine()
default_websocket_engine.add_message_validator("audio", default_websocket_engine.validate_audio_message)
default_websocket_engine.add_message_validator("text", default_websocket_engine.validate_text_message)
default_websocket_engine.add_message_validator("story", default_websocket_engine.validate_story_message)

# WebSocket 유틸리티 함수들
async def safe_websocket_send(websocket: WebSocket, data: Dict[str, Any]) -> bool:
    """안전한 WebSocket 메시지 전송 (전역 함수)"""
    return await default_websocket_engine.send_json(websocket, data)

async def safe_websocket_receive(websocket: WebSocket, timeout: float = 30.0) -> Optional[Dict[str, Any]]:
    """안전한 WebSocket 메시지 수신 (전역 함수)"""
    return await default_websocket_engine.receive_json(websocket, timeout) 