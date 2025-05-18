"""
WebSocket 연결 관리 모듈

이 모듈은 WebSocket 연결 관리 및 클라이언트 세션 관리 기능을 제공합니다.
"""
import os
import time
import logging
import asyncio
from fastapi import WebSocket, status
from .utils import save_conversation, cleanup_temp_files

# 활성 연결 저장소
active_connections = {}
CONNECTION_TIMEOUT = 30 * 60  # 30분 타임아웃

# Chatbot B 인스턴스 저장소 (꼬기)
chatbot_b_instances = {}

# 종료 이벤트
shutdown_event = asyncio.Event()

class ConnectionManager:
    """WebSocket 연결 관리 클래스"""
    
    @staticmethod
    def get_client_count():
        """활성 연결 수 반환"""
        return len(active_connections)
    
    @staticmethod
    def get_client_info(client_id):
        """클라이언트 정보 반환"""
        return active_connections.get(client_id)
    
    @staticmethod
    def add_client(client_id, connection_info):
        """클라이언트 연결 추가"""
        active_connections[client_id] = connection_info
        logging.info(f"클라이언트 추가: {client_id} (총 {len(active_connections)}개 연결)")
    
    @staticmethod
    def remove_client(client_id):
        """클라이언트 연결 제거"""
        if client_id in active_connections:
            del active_connections[client_id]
            logging.info(f"클라이언트 제거: {client_id} (총 {len(active_connections)}개 연결)")
    
    @staticmethod
    async def close_all_connections():
        """모든 연결 종료"""
        for client_id, connection_info in active_connections.items():
            try:
                if "websocket" in connection_info:
                    await connection_info["websocket"].close()
            except Exception as e:
                logging.error(f"연결 종료 중 오류: {e}")
                
        active_connections.clear()
        logging.info("모든 연결 종료 완료")
        
    @staticmethod
    async def handle_disconnect(client_id):
        """
        클라이언트 연결 종료 처리
        
        Args:
            client_id (str): 클라이언트 식별자
        """
        if client_id in active_connections:
            connection_info = active_connections[client_id]
            
            # 대화 내용 저장
            if "chatbot" in connection_info and "child_name" in connection_info:
                chatbot = connection_info["chatbot"]
                child_name = connection_info.get("child_name", "unknown")
                await save_conversation(chatbot, child_name, client_id)
            
            # 임시 파일 정리
            if "temp_files" in connection_info:
                cleanup_temp_files(connection_info["temp_files"])
            
            # 연결 정보 삭제
            ConnectionManager.remove_client(client_id)
            logging.info(f"클라이언트 연결 종료 처리 완료: {client_id}")

    @staticmethod
    async def cleanup_inactive_clients():
        """비활성 클라이언트 정리 함수"""
        while not shutdown_event.is_set():
            try:
                # 5분마다 실행하되, 1초마다 종료 이벤트 확인
                for _ in range(300):
                    await asyncio.sleep(1)
                    if shutdown_event.is_set():
                        break
                        
                # 종료 이벤트가 설정되었다면 루프 종료
                if shutdown_event.is_set():
                    logging.info("비활성 클라이언트 정리 태스크 종료")
                    break
                    
                current_time = time.time()
                
                # 30분 이상 비활성인 클라이언트 식별
                inactive_clients = []
                for client_id, connection_info in active_connections.items():
                    if current_time - connection_info.get("start_time", 0) > CONNECTION_TIMEOUT:
                        inactive_clients.append(client_id)
                
                # 비활성 클라이언트 정리
                for client_id in inactive_clients:
                    logging.info(f"타임아웃으로 인한 연결 종료: {client_id}")
                    await ConnectionManager.handle_disconnect(client_id)
                
                # 꼬기(ChatbotB) 인스턴스 정리
                inactive_chatbot_bs = []
                for client_id, client_data in chatbot_b_instances.items():
                    if current_time - client_data.get("last_activity", 0) > CONNECTION_TIMEOUT:
                        inactive_chatbot_bs.append(client_id)
                
                for client_id in inactive_chatbot_bs:
                    del chatbot_b_instances[client_id]
                    logging.info(f"비활성 꼬기 인스턴스 정리: {client_id}")
            
            except Exception as e:
                logging.error(f"비활성 클라이언트 정리 중 오류 발생: {str(e)}")
        
        logging.info("비활성 클라이언트 정리 태스크 완전 종료")

    @staticmethod
    def set_shutdown_event():
        """종료 이벤트 설정"""
        shutdown_event.set()
        logging.info("종료 이벤트 설정됨")
    
    @staticmethod
    def add_chatbot_b_instance(client_id, instance_data):
        """꼬기(ChatbotB) 인스턴스 추가"""
        chatbot_b_instances[client_id] = instance_data
        logging.info(f"꼬기 인스턴스 추가: {client_id}")
    
    @staticmethod
    def get_chatbot_b_instance(client_id):
        """꼬기(ChatbotB) 인스턴스 조회"""
        return chatbot_b_instances.get(client_id)
    
    @staticmethod
    def update_chatbot_b_activity(client_id):
        """꼬기(ChatbotB) 활동 시간 업데이트"""
        if client_id in chatbot_b_instances:
            chatbot_b_instances[client_id]["last_activity"] = time.time()
            
    @staticmethod
    def get_active_connections_info():
        """활성 연결 정보 요약"""
        result = []
        for client_id, connection_info in active_connections.items():
            result.append({
                "client_id": client_id,
                "child_name": connection_info.get("child_name", "unknown"),
                "age": connection_info.get("age", 0),
                "connected_since": time.strftime("%Y-%m-%d %H:%M:%S", 
                                               time.localtime(connection_info.get("start_time", 0)))
            })
        return result 