"""
WebSocket 연결 관리 엔진

기존 connection.py의 기능을 통합하고 개선한 연결 관리 엔진
"""
import time
import asyncio
import gc
import psutil
import os
from typing import Dict, List, Any, Optional
from fastapi import WebSocket
from fastapi.websockets import WebSocketState

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
    - 음성 정보 공유 및 동기화
    - 성능 모니터링 및 최적화
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
        
        # 음성 정보 공유를 위한 새로운 데이터 구조 추가
        self.voice_mappings: Dict[str, Dict[str, Any]] = {}  # {client_id: voice_info}
        self.audio_processors: Dict[str, Any] = {}  # {client_id: audio_processor_ref}
        
        # 성능 모니터링
        self.performance_stats = {
            "start_time": time.time(),
            "total_connections": 0,
            "cleanup_count": 0,
            "memory_cleanups": 0,
            "health_checks": 0
        }
        
        # 리소스 정리 최적화를 위한 배치 처리
        self.cleanup_batch_size = 10
        self.last_memory_check = time.time()
        self.memory_check_interval = 60  # 1분마다 메모리 체크
        
        logger.info("🚀 연결 엔진 초기화 완료 (성능 최적화 포함)")
    
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
        self.performance_stats["total_connections"] += 1
        logger.info(f"✅ 클라이언트 추가: {client_id} (총 {len(self.active_connections)}개 연결)")
    
    def remove_client(self, client_id: str) -> None:
        """클라이언트 연결 제거 (최적화된 정리)"""
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            
            # 관련된 모든 리소스를 배치로 정리
            resources_to_clean = []
            
            if client_id in self.voice_mappings:
                del self.voice_mappings[client_id]
                resources_to_clean.append("voice_mapping")
                
            if client_id in self.audio_processors:
                del self.audio_processors[client_id]
                resources_to_clean.append("audio_processor")
            
            if client_id in self.chatbot_b_instances:
                del self.chatbot_b_instances[client_id]
                resources_to_clean.append("chatbot_b_instance")
            
            logger.info(f"🗑️ 클라이언트 제거: {client_id} (정리된 리소스: {', '.join(resources_to_clean)})")
            
            # 메모리 압축 트리거 (주기적으로)
            if len(self.active_connections) % 5 == 0:
                asyncio.create_task(self._trigger_memory_cleanup())
    
    async def close_all_connections(self) -> None:
        """모든 연결 종료 (최적화된 배치 처리)"""
        logger.info(f"🔄 {len(self.active_connections)}개 연결 일괄 종료 시작...")
        
        # 배치 단위로 연결 종료
        connection_items = list(self.active_connections.items())
        
        for i in range(0, len(connection_items), self.cleanup_batch_size):
            batch = connection_items[i:i + self.cleanup_batch_size]
            
            # 배치 내 연결들을 비동기로 동시 처리
            close_tasks = []
            for client_id, connection_info in batch:
                if "websocket" in connection_info:
                    close_tasks.append(self._close_websocket_safely(connection_info["websocket"], client_id))
            
            if close_tasks:
                await asyncio.gather(*close_tasks, return_exceptions=True)
                logger.info(f"📦 배치 {i//self.cleanup_batch_size + 1} 연결 종료 완료 ({len(close_tasks)}개)")
        
        # 모든 데이터 구조 일괄 정리
        self.active_connections.clear()
        self.chatbot_b_instances.clear()
        self.voice_mappings.clear()
        self.audio_processors.clear()
        
        # 메모리 정리
        await self._trigger_memory_cleanup()
        logger.info("✅ 모든 연결 및 리소스 정리 완료")
    
    async def _close_websocket_safely(self, websocket: WebSocket, client_id: str) -> None:
        """WebSocket 안전 종료"""
        try:
            await websocket.close()
        except Exception as e:
            logger.warning(f"⚠️ 연결 종료 중 오류 (클라이언트 {client_id}): {e}")
    
    async def handle_disconnect(self, client_id: str) -> None:
        """
        클라이언트 연결 종료 처리 (성능 최적화)
        
        Args:
            client_id: 클라이언트 식별자
        """
        if client_id not in self.active_connections:
            return
            
        connection_info = self.active_connections[client_id]
        
        try:
            logger.info(f"🔌 연결 종료 처리 시작: {client_id}")
            
            # 대화 내용 저장 (비동기 처리로 최적화)
            save_task = None
            if "chatbot" in connection_info and "child_name" in connection_info:
                chatbot = connection_info["chatbot"]
                child_name = connection_info.get("child_name", "unknown")
                
                if chatbot and hasattr(chatbot, "get_conversation_history"):
                    save_task = asyncio.create_task(
                        save_conversation(chatbot, child_name, client_id)
                    )
                    logger.info(f"💾 대화 저장 작업 시작: {client_id}")
            
            # 임시 파일 정리 (백그라운드에서)
            if "temp_files" in connection_info:
                cleanup_task = asyncio.create_task(
                    self._cleanup_temp_files_async(connection_info["temp_files"], client_id)
                )
            
            # 연결 정보 삭제 (음성 정보도 함께 정리됨)
            self.remove_client(client_id)
            self.performance_stats["cleanup_count"] += 1
            
            # 저장 작업 완료 대기 (타임아웃 적용)
            if save_task:
                try:
                    await asyncio.wait_for(save_task, timeout=5.0)
                    logger.info(f"✅ 대화 저장 완료: {client_id}")
                except asyncio.TimeoutError:
                    logger.warning(f"⏰ 대화 저장 타임아웃: {client_id}")
                except Exception as e:
                    logger.error(f"❌ 대화 저장 실패: {client_id}, 오류: {e}")
            
            logger.info(f"✅ 클라이언트 연결 종료 처리 완료: {client_id}")
            
        except Exception as e:
            logger.error(f"❌ 연결 종료 처리 중 오류: {e}")
            # 오류가 발생해도 연결은 제거
            self.remove_client(client_id)
    
    async def _cleanup_temp_files_async(self, temp_files: List[str], client_id: str) -> None:
        """임시 파일 비동기 정리"""
        try:
            cleanup_temp_files(temp_files)
            logger.info(f"🗂️ 임시 파일 정리 완료: {client_id} ({len(temp_files)}개 파일)")
        except Exception as e:
            logger.error(f"❌ 임시 파일 정리 실패: {client_id}, 오류: {e}")
    
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
    # 음성 정보 공유 기능
    # ==========================================
    
    def set_client_voice_mapping(self, client_id: str, voice_id: str, voice_settings: dict = None, user_name: str = None) -> None:
        """
        클라이언트의 음성 매핑 정보 설정
        
        Args:
            client_id (str): 클라이언트 식별자
            voice_id (str): 음성 ID (클론 음성 포함)
            voice_settings (dict): 음성 설정 (옵션)
            user_name (str): 사용자 이름 (옵션)
        """
        self.voice_mappings[client_id] = {
            "voice_id": voice_id,
            "voice_settings": voice_settings or {},
            "user_name": user_name,
            "created_at": time.time(),
            "last_used": time.time()
        }
        
        logger.info(f"클라이언트 {client_id}의 음성 매핑 설정: {voice_id} (사용자: {user_name})")
        
        # 등록된 AudioProcessor에 자동으로 음성 매핑 적용
        self._sync_voice_mapping_to_audio_processor(client_id, voice_id, voice_settings)
    
    def get_client_voice_mapping(self, client_id: str) -> Optional[Dict[str, Any]]:
        """
        클라이언트의 음성 매핑 정보 조회
        
        Args:
            client_id (str): 클라이언트 식별자
            
        Returns:
            Optional[Dict]: 음성 매핑 정보 또는 None
        """
        voice_info = self.voice_mappings.get(client_id)
        if voice_info:
            # 마지막 사용 시간 업데이트
            voice_info["last_used"] = time.time()
        return voice_info
    
    def remove_client_voice_mapping(self, client_id: str) -> None:
        """
        클라이언트의 음성 매핑 정보 제거
        
        Args:
            client_id (str): 클라이언트 식별자
        """
        if client_id in self.voice_mappings:
            del self.voice_mappings[client_id]
            logger.info(f"클라이언트 {client_id}의 음성 매핑 제거")
            
            # AudioProcessor에서도 제거
            if client_id in self.audio_processors:
                audio_processor = self.audio_processors[client_id]
                if hasattr(audio_processor, 'remove_user_voice_mapping'):
                    audio_processor.remove_user_voice_mapping(client_id)
    
    def register_audio_processor(self, client_id: str, audio_processor) -> None:
        """
        클라이언트의 AudioProcessor 등록
        
        Args:
            client_id (str): 클라이언트 식별자
            audio_processor: AudioProcessor 인스턴스
        """
        self.audio_processors[client_id] = audio_processor
        logger.info(f"클라이언트 {client_id}의 AudioProcessor 등록")
        
        # 기존 음성 매핑이 있다면 자동 적용
        if client_id in self.voice_mappings:
            voice_info = self.voice_mappings[client_id]
            self._sync_voice_mapping_to_audio_processor(
                client_id, 
                voice_info["voice_id"], 
                voice_info["voice_settings"]
            )
    
    def _sync_voice_mapping_to_audio_processor(self, client_id: str, voice_id: str, voice_settings: dict = None) -> None:
        """
        음성 매핑 정보를 AudioProcessor에 동기화
        
        Args:
            client_id (str): 클라이언트 식별자
            voice_id (str): 음성 ID
            voice_settings (dict): 음성 설정
        """
        if client_id in self.audio_processors:
            audio_processor = self.audio_processors[client_id]
            if hasattr(audio_processor, 'set_user_voice_mapping'):
                audio_processor.set_user_voice_mapping(client_id, voice_id, voice_settings)
                logger.info(f"AudioProcessor에 음성 매핑 동기화 완료: {client_id} -> {voice_id}")
            else:
                logger.warning(f"AudioProcessor에 set_user_voice_mapping 메서드가 없습니다: {client_id}")
    
    def get_all_voice_mappings(self) -> Dict[str, Dict[str, Any]]:
        """
        모든 클라이언트의 음성 매핑 정보 반환
        
        Returns:
            Dict: 모든 음성 매핑 정보
        """
        return self.voice_mappings.copy()
    
    def update_voice_mapping_usage(self, client_id: str) -> None:
        """
        음성 매핑 사용 시간 업데이트
        
        Args:
            client_id (str): 클라이언트 식별자
        """
        if client_id in self.voice_mappings:
            self.voice_mappings[client_id]["last_used"] = time.time()
    
    # ==========================================
    # 비활성 연결 정리 (성능 최적화)
    # ==========================================
    
    async def cleanup_inactive_clients(self) -> None:
        """비활성 클라이언트 정리 태스크 (최적화된 버전)"""
        logger.info("🔄 비활성 클라이언트 정리 태스크 시작 (최적화)")
        
        # 정리 주기 조정 (메모리 사용량에 따라 동적 조정)
        base_interval = 120  # 기본 2분
        
        while not self.shutdown_event.is_set():
            try:
                # 메모리 사용량에 따른 정리 주기 조정
                memory_usage = psutil.virtual_memory().percent
                if memory_usage > 70:
                    cleanup_interval = base_interval // 2  # 메모리 높으면 더 자주 정리
                    logger.info(f"🚨 메모리 사용량 높음 ({memory_usage}%), 정리 주기 단축: {cleanup_interval}초")
                else:
                    cleanup_interval = base_interval
                
                # 동적 대기 (1초씩 체크하되 종료 이벤트 우선 확인)
                for _ in range(cleanup_interval):
                    await asyncio.sleep(1)
                    if self.shutdown_event.is_set():
                        break
                        
                # 종료 이벤트가 설정되었다면 루프 종료
                if self.shutdown_event.is_set():
                    break
                
                # 모든 정리 작업을 병렬로 실행
                cleanup_tasks = [
                    self._cleanup_inactive_connections(),
                    self._cleanup_inactive_chatbot_b_instances(),
                    self._cleanup_inactive_voice_mappings(),
                    self._check_websocket_health()
                ]
                
                # 배치 정리 실행
                await asyncio.gather(*cleanup_tasks, return_exceptions=True)
                
                # 주기적 메모리 정리
                await self._trigger_memory_cleanup()
                
                self.performance_stats["health_checks"] += 1
                logger.info(f"🔍 정리 사이클 완료 (활성 연결: {len(self.active_connections)}개)")
            
            except Exception as e:
                logger.error(f"❌ 비활성 클라이언트 정리 중 오류 발생: {str(e)}")
        
        logger.info("✅ 비활성 클라이언트 정리 태스크 완전 종료")
    
    async def _cleanup_inactive_connections(self) -> None:
        """비활성 일반 연결 정리 (배치 처리 최적화)"""
        current_time = time.time()
        
        # 30분 이상 비활성인 클라이언트 식별
        inactive_clients = [
            client_id for client_id, connection_info in self.active_connections.items()
            if current_time - connection_info.get("start_time", 0) > self.connection_timeout
        ]
        
        if not inactive_clients:
            return
        
        logger.info(f"🕐 {len(inactive_clients)}개 비활성 연결 발견, 정리 시작...")
        
        # 배치 단위로 비활성 클라이언트 정리
        for i in range(0, len(inactive_clients), self.cleanup_batch_size):
            batch = inactive_clients[i:i + self.cleanup_batch_size]
            
            # 배치 정리 작업들을 병렬로 실행
            disconnect_tasks = [
                self.handle_disconnect(client_id) for client_id in batch
            ]
            
            await asyncio.gather(*disconnect_tasks, return_exceptions=True)
            logger.info(f"📦 비활성 연결 배치 {i//self.cleanup_batch_size + 1} 정리 완료 ({len(batch)}개)")
    
    async def _cleanup_inactive_chatbot_b_instances(self) -> None:
        """비활성 ChatBot B 인스턴스 정리 (최적화)"""
        current_time = time.time()
        
        inactive_chatbot_bs = [
            client_id for client_id, client_data in self.chatbot_b_instances.items()
            if current_time - client_data.get("last_activity", 0) > self.connection_timeout
        ]
        
        for client_id in inactive_chatbot_bs:
            # ChatBot B 인스턴스 정리 시 리소스 해제
            if client_id in self.chatbot_b_instances:
                chatbot_data = self.chatbot_b_instances[client_id]
                if "instance" in chatbot_data and hasattr(chatbot_data["instance"], "cleanup"):
                    try:
                        chatbot_data["instance"].cleanup()
                    except Exception as e:
                        logger.warning(f"⚠️ ChatBot B 리소스 정리 실패: {client_id}, 오류: {e}")
                
                del self.chatbot_b_instances[client_id]
                logger.info(f"🤖 비활성 ChatBot B 인스턴스 정리: {client_id}")
    
    async def _cleanup_inactive_voice_mappings(self) -> None:
        """비활성 음성 매핑 정리"""
        current_time = time.time()
        
        # 1시간 이상 사용되지 않은 음성 매핑 정리
        voice_timeout = 3600  # 1시간
        
        inactive_voice_mappings = [
            client_id for client_id, voice_info in self.voice_mappings.items()
            if current_time - voice_info.get("last_used", 0) > voice_timeout
        ]
        
        for client_id in inactive_voice_mappings:
            if client_id not in self.active_connections:  # 연결이 없는 경우에만 정리
                del self.voice_mappings[client_id]
                logger.info(f"🎤 비활성 음성 매핑 정리: {client_id}")
    
    async def _check_websocket_health(self) -> None:
        """WebSocket 연결 상태 건강성 체크 (최적화)"""
        if not self.active_connections:
            return
        
        # 배치 단위로 헬스 체크
        connection_items = list(self.active_connections.items())
        disconnected_clients = []
        
        for i in range(0, len(connection_items), self.cleanup_batch_size):
            batch = connection_items[i:i + self.cleanup_batch_size]
            
            # 배치 내 연결들을 병렬로 헬스 체크
            health_check_tasks = []
            for client_id, connection_info in batch:
                health_check_tasks.append(
                    self._check_single_websocket_health(client_id, connection_info)
                )
            
            # 병렬 헬스 체크 실행
            results = await asyncio.gather(*health_check_tasks, return_exceptions=True)
            
            # 결과 수집
            for j, result in enumerate(results):
                if isinstance(result, Exception):
                    client_id = batch[j][0]
                    logger.warning(f"⚠️ WebSocket 헬스 체크 실패: {client_id}")
                    disconnected_clients.append(client_id)
                elif result is False:  # 연결이 끊어진 경우
                    client_id = batch[j][0]
                    disconnected_clients.append(client_id)
        
        # 끊어진 연결들 정리
        if disconnected_clients:
            logger.info(f"🔌 헬스 체크로 {len(disconnected_clients)}개 연결 정리 예정")
            disconnect_tasks = [
                self.handle_disconnect(client_id) for client_id in disconnected_clients
            ]
            await asyncio.gather(*disconnect_tasks, return_exceptions=True)
    
    async def _check_single_websocket_health(self, client_id: str, connection_info: Dict[str, Any]) -> bool:
        """단일 WebSocket 연결 헬스 체크"""
        websocket = connection_info.get("websocket")
        if not websocket:
            return True  # WebSocket이 없으면 정상으로 간주
            
        try:
            # WebSocket 상태 체크
            if websocket.client_state != WebSocketState.CONNECTED:
                logger.warning(f"🔗 끊어진 WebSocket 연결 감지: {client_id}")
                return False
            
            # 가벼운 ping 전송으로 실제 연결 테스트 (타임아웃 적용)
            ping_data = {"type": "ping", "message": "health_check", "timestamp": time.time()}
            await asyncio.wait_for(websocket.send_json(ping_data), timeout=3.0)
            return True
                
        except Exception as e:
            logger.warning(f"⚠️ WebSocket 헬스 체크 실패: {client_id}, 오류: {e}")
            return False
    
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

    async def _trigger_memory_cleanup(self) -> None:
        """메모리 정리 트리거"""
        current_time = time.time()
        if current_time - self.last_memory_check > self.memory_check_interval:
            self.last_memory_check = current_time
            await self._cleanup_memory()
    
    async def _cleanup_memory(self) -> None:
        """메모리 정리 함수"""
        gc.collect()
        self.performance_stats["memory_cleanups"] += 1
        logger.info("🧹 메모리 정리 완료")

        # 메모리 사용량 확인
        memory_usage = psutil.virtual_memory().percent
        logger.info(f"💾 메모리 사용량: {memory_usage}%")

        # 메모리 사용량이 일정 수준을 초과하면 추가 정리 필요
        if memory_usage > 80:
            logger.warning("🚨 메모리 사용량이 높음! 추가 정리 필요")
            await self._additional_cleanup()
    
    async def _additional_cleanup(self) -> None:
        """추가적인 메모리 정리 함수"""
        # 추가적인 메모리 정리 로직을 구현해야 합니다.
        logger.warning("추가적인 메모리 정리 로직을 구현해야 합니다.")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """성능 통계 반환"""
        return self.performance_stats.copy() 