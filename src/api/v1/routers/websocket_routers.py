# WebSocket 통합
"""
JWT 인증 음성 WebSocket APIRouter
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query, Depends
from typing import Optional, Dict, Any
import asyncio
import json
from datetime import datetime

from ..dependencies import (
    verify_jwt_token,
    # verify_child_permission,
    get_connection_manager,
    get_audio_processor
)

from shared.utils import ConnectionManager
from ..processors.audio_processor import AudioProcessor
from src.shared.utils.logging import get_module_logger

logger = get_module_logger(__name__) # Module Logger 상속 받아 생성

# APIRouter 생성
router = APIRouter(
    prefix="/wss", # WebSocket 접두사 설정
    tags=["websocket", "voice", "authenticated"] # API 문서용 Tag 설정
)

@router.websocket("/voice/{child_name}")
async def voice_websocket_endpoint(
    websocket: WebSocket,
    child_name: str,
    # JWT 인증 의존성
    user_info: Dict[str, Any] = Depends(verify_jwt_token),
    # child_permission: Dict[str, Any] = Depends(verify_child_permission),
    # 서비스 의존성
    connection_manager: ConnectionManager = Depends(get_connection_manager),
):
    """
    JWT 인증 음성 WebSocket endpoint

    Query Parameters:
        token: JWT 인증 토큰 (필수)
    
    Path Parameters:
        child_name: 아이 이름
    
    Example:
        wss://AWS_IP/voice_ws/ws/voice/병찬?=token=jwt_token&age=7&interests=여행,자유
    """

    await websocket.accept() # WebSocket 연결 수락
    
    client_id = f"{user_info['user_id']}_{child_name}_story" # Client 고유 ID 
    
    try:
        # 연결 등록
        await connection_manager.register_client(client_id, websocket, {
            "user_info": user_info, # 아이 정보
            # "child_info": child_permission,
            "connection_type": "voice", # 연결 유형
            "connection_at": datetime.now().isoformat() # 연결 시간
        })
        
        logger.info(f"인증된 음성 WebSocket 연결 : child={child_name}")

        # 부모님 권한 기능 추가하면 사용
        # logger.info(f"인증된 음성 WebSocket 연결 : parent={user_info['user_id']}, child={child_name}")

    
        # 연결 성공
        await websocket.send_json({
            "type": "connection_success",
            "message": f"{child_name} 연결 성공",
            "child_name": child_name,
            # "parent_id": user_info['user_id'],
            # "permissions": child_permission['permissions'],
            "timestamp": datetime.now().isoformat()
        })

        # Message 처리 loop
        while True:
            try:
                # JSON
                data = await websocket.receive_json()
                await handle_voice_message(
                    data, client_id, websocket,
                    connection_manager, audio_processor
                )
            
            except:
                # binary audio data
                audio_data = await websocket.receive_bytes()
                await handle_audio_stream(
                    audio_data, client_id, websocket,
                    connection_manager, audio_processor
                )
    
    except WebSocketDisconnect:
        logger.info(f"음성 WebSocket 정상 연결 해제 : {client_id}")
        
    except Exception as e:
        logger.error(f"음성 WebSocket 오류 : {client_id} - {e}")
        await websocket.close()
        
    finally:
        # 연결 정리
        await connection_manager.unregister_client(client_id)
        
@router.websocket("/story/{child_name}")
async def story_websocket_endpoint(
    websocket: WebSocket,
    child_name: str,
    age: Optional[int] = Query(None, description="아이 나이"),
    interests: Optional[str] = Query(None, description="관심사 (쉼표 구분)"),
    # JWT 인증 의존성
    user_info: Dict[str, Any] = Depends(verify_jwt_token),
    # child_permission: Dict[str, Any] = Depends(verify_child_permission),
    # 서비스 의존성
    connection_manager: ConnectionManager = Depends(get_connection_manager)
    
):
    
    """
    JWT 인증된 동화 생성 WebSocket endpoint
    
    Query Parameters:
        token: JWT 인증 토큰 (필수)
        age: 아이 나이 (선택)
        interests: 관심사 목록 (선택)
        
    Example:
        wss://AWS_IP/voice_ws/ws/story/병찬?=token=jwt_token&age=7&interests=여행,자유
    """
    
    await websocket.accept() # WebSocket 연결 수락
    
    client_id = f"{user_info['user_id']}_{child_name}_story"
    
    try:
        # connection register
        await connection_manager.register_client(client_id, websocket, {
            "user_info": user_info,
            "connection_type": "story",
            # "child_info": child_permission,
            "child_age": age,
            "interests": interests.split(',') if interests else [],
            "connection_at": datetime.now().isoformat()
        })
        
        # parent permission 은 추후 추가 예정
        logger.info(f"인증된 동화 WebSocket 연결: child={child_name}, age={age}")
        
        await websocket.send_json({
            "type": "connection_success"
        })

@router.websocket("/voice-training/{child_name}")
async def voice_training_endpoint(
    websocket: WebSocket,
    child_name: str, 
    user_info: Dict[str, Any] = Depends(verify_jwt_token), # JWT 인증 의존성
    connection_manager: ConnectionManager = Depends(get_connection_manager) # 서비스 의존성
    
):
    """
    아이 음성 수집 및 Voice Clone 생성 WebSocket endpoint

    - 부모 인증 필수는 추후 추가 예정
    - 아이 음성 샘플 수집
    - 실시간 품질 체크
    """
    
    await websocket.accept() # WebSocket 연결 수락
    
    try:
        # Parent Authentication check 는 추후 추가 예정
        # parent_id = user_info.get('user_id')
        
        # 음성 수집 session 시작
        await websocket.send_json({
            "type": "voice_training_start",
            "child_name": child_name,
            # "parent_id": parent_id,
            "child_name": child_name,
            "required_samples": 5, # 필요 샘플 수
            "sample_duration": 30, # 각 샘플 30초
            "instructions": [
                "아이의"
            ]
        })
        