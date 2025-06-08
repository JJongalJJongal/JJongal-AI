"""
부기 (ChatBot A) 영구 메모리 시스템

LangChain의 최신 메모리 패턴과 영구 저장소를 통합한 메모리 관리 시스템
"""
import sqlite3
import json
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Sequence
from pathlib import Path
import threading

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import trim_messages
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

from shared.utils.logging_utils import get_module_logger

logger = get_module_logger(__name__)

class PersistentChatMessageHistory(BaseChatMessageHistory):
    """
    SQLite 기반 영구 채팅 메시지 히스토리
    LangChain의 BaseChatMessageHistory를 확장하여 데이터베이스 저장 기능 제공
    """
    
    def __init__(self, session_id: str, db_path: str = "chatbot/data/conversations.db"):
        """
        영구 메모리 초기화
        
        Args:
            session_id: 세션 식별자 (아이 이름 + 타임스탬프)
            db_path: SQLite 데이터베이스 파일 경로
        """
        self.session_id = session_id
        self.db_path = Path(db_path)
        self._lock = threading.Lock()
        
        # 데이터베이스 디렉토리 생성
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 데이터베이스 초기화
        self._init_database()
        
        logger.info(f"영구 메모리 초기화 완료: session_id={session_id}")
    
    def _init_database(self):
        """데이터베이스 테이블 초기화"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    message_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    INDEX(session_id),
                    INDEX(timestamp)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    session_id TEXT PRIMARY KEY,
                    child_name TEXT,
                    age_group INTEGER,
                    interests TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    last_activity DATETIME DEFAULT CURRENT_TIMESTAMP,
                    message_count INTEGER DEFAULT 0,
                    total_tokens INTEGER DEFAULT 0
                )
            """)
            
            conn.commit()
    
    @property
    def messages(self) -> List[BaseMessage]:
        """현재 세션의 모든 메시지 반환"""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT message_type, content, metadata FROM chat_messages "
                    "WHERE session_id = ? ORDER BY timestamp ASC",
                    (self.session_id,)
                )
                
                messages = []
                for msg_type, content, metadata_str in cursor.fetchall():
                    metadata = json.loads(metadata_str) if metadata_str else {}
                    
                    if msg_type == "human":
                        messages.append(HumanMessage(content=content, additional_kwargs=metadata))
                    elif msg_type == "ai":
                        messages.append(AIMessage(content=content, additional_kwargs=metadata))
                    elif msg_type == "system":
                        messages.append(SystemMessage(content=content, additional_kwargs=metadata))
                
                return messages
    
    def add_message(self, message: BaseMessage) -> None:
        """메시지 추가"""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                # 메시지 타입 결정
                if isinstance(message, HumanMessage):
                    msg_type = "human"
                elif isinstance(message, AIMessage):
                    msg_type = "ai"
                elif isinstance(message, SystemMessage):
                    msg_type = "system"
                else:
                    msg_type = "unknown"
                
                # 메타데이터 직렬화
                metadata = json.dumps(message.additional_kwargs) if message.additional_kwargs else None
                
                # 메시지 저장
                conn.execute(
                    "INSERT INTO chat_messages (session_id, message_type, content, metadata) "
                    "VALUES (?, ?, ?, ?)",
                    (self.session_id, msg_type, message.content, metadata)
                )
                
                # 세션 정보 업데이트
                conn.execute(
                    "INSERT OR REPLACE INTO chat_sessions "
                    "(session_id, last_activity, message_count) "
                    "VALUES (?, CURRENT_TIMESTAMP, "
                    "  (SELECT COUNT(*) FROM chat_messages WHERE session_id = ?))",
                    (self.session_id, self.session_id)
                )
                
                conn.commit()
        
        logger.debug(f"메시지 저장 완료: {msg_type} - {message.content[:50]}...")
    
    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        """여러 메시지 일괄 추가"""
        for message in messages:
            self.add_message(message)
    
    def clear(self) -> None:
        """현재 세션의 모든 메시지 삭제"""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM chat_messages WHERE session_id = ?", (self.session_id,))
                conn.execute("DELETE FROM chat_sessions WHERE session_id = ?", (self.session_id,))
                conn.commit()
        
        logger.info(f"세션 메시지 삭제 완료: {self.session_id}")
    
    def get_recent_messages(self, limit: int = 10) -> List[BaseMessage]:
        """최근 메시지 조회"""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT message_type, content, metadata FROM chat_messages "
                    "WHERE session_id = ? ORDER BY timestamp DESC LIMIT ?",
                    (self.session_id, limit)
                )
                
                messages = []
                for msg_type, content, metadata_str in reversed(cursor.fetchall()):
                    metadata = json.loads(metadata_str) if metadata_str else {}
                    
                    if msg_type == "human":
                        messages.append(HumanMessage(content=content, additional_kwargs=metadata))
                    elif msg_type == "ai":
                        messages.append(AIMessage(content=content, additional_kwargs=metadata))
                    elif msg_type == "system":
                        messages.append(SystemMessage(content=content, additional_kwargs=metadata))
                
                return messages
    
    def update_session_info(self, child_name: str, age_group: int, interests: List[str]):
        """세션 정보 업데이트"""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO chat_sessions "
                    "(session_id, child_name, age_group, interests, last_activity) "
                    "VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)",
                    (self.session_id, child_name, age_group, json.dumps(interests))
                )
                conn.commit()
    
    def get_session_stats(self) -> Dict[str, Any]:
        """세션 통계 정보 반환"""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT child_name, age_group, interests, created_at, "
                    "last_activity, message_count, total_tokens "
                    "FROM chat_sessions WHERE session_id = ?",
                    (self.session_id,)
                )
                
                row = cursor.fetchone()
                if row:
                    return {
                        "session_id": self.session_id,
                        "child_name": row[0],
                        "age_group": row[1],
                        "interests": json.loads(row[2]) if row[2] else [],
                        "created_at": row[3],
                        "last_activity": row[4],
                        "message_count": row[5],
                        "total_tokens": row[6]
                    }
                return {}

class ConversationMemoryManager:
    """
    LangChain 기반 대화 메모리 관리자
    RunnableWithMessageHistory와 토큰 관리 기능 통합
    """
    
    def __init__(self, openai_client=None, model_name: str = "gpt-4o-mini", 
                 max_tokens: int = 4000, db_path: str = "chatbot/data/conversations.db"):
        """
        메모리 관리자 초기화
        
        Args:
            openai_client: OpenAI 클라이언트
            model_name: 사용할 LLM 모델
            max_tokens: 최대 토큰 수
            db_path: 데이터베이스 경로
        """
        self.db_path = db_path
        self.max_tokens = max_tokens
        self.model_name = model_name
        
        # LLM 설정
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0.9,
            api_key=openai_client.api_key if openai_client else None
        )
        
        # 프롬프트 템플릿
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 아이들과 재미있는 동화를 만드는 친근한 AI 친구 '부기'입니다.

아이의 상상력을 자극하고 창의적인 이야기 요소를 수집하여 함께 멋진 동화를 만들어보세요.
항상 긍정적이고 격려하는 말투로 대화하며, 아이의 연령에 맞는 쉬운 언어를 사용하세요.

대화 기억사항:
- 아이의 이름과 관심사를 기억하세요
- 이전 대화 내용을 참고하여 연속성있게 대화하세요
- 아이가 제안한 이야기 요소들을 기억하고 활용하세요"""),
            
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}")
        ])
        
        # 체인 구성
        self.chain = self.prompt | self.llm | StrOutputParser()
        
        # 세션별 메모리 저장소
        self._session_store = {}
        
        logger.info(f"대화 메모리 관리자 초기화 완료: model={model_name}")
    
    def get_session_history(self, session_id: str) -> PersistentChatMessageHistory:
        """세션 히스토리 가져오기 또는 생성"""
        if session_id not in self._session_store:
            self._session_store[session_id] = PersistentChatMessageHistory(
                session_id=session_id, 
                db_path=self.db_path
            )
        return self._session_store[session_id]
    
    def create_conversation_chain(self, session_id: str):
        """세션별 대화 체인 생성"""
        # RunnableWithMessageHistory로 메모리 통합
        return RunnableWithMessageHistory(
            self.chain,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="history"
        )
    
    async def generate_response(self, session_id: str, user_input: str, 
                              child_name: str = "친구", age_group: int = 5, 
                              interests: List[str] = None) -> str:
        """
        LangChain 기반 응답 생성 (토큰 관리 포함)
        
        Args:
            session_id: 세션 ID
            user_input: 사용자 입력
            child_name: 아이 이름
            age_group: 연령대
            interests: 관심사 목록
            
        Returns:
            생성된 응답
        """
        try:
            # 세션 정보 업데이트
            history = self.get_session_history(session_id)
            history.update_session_info(child_name, age_group, interests or [])
            
            # 대화 체인 생성
            conversation = self.create_conversation_chain(session_id)
            
            # 토큰 제한을 위한 메시지 트리밍
            current_messages = history.messages
            if len(current_messages) > 0:
                # 토큰 수가 많으면 메시지 트리밍
                trimmed_messages = trim_messages(
                    current_messages,
                    max_tokens=self.max_tokens,
                    strategy="last",
                    token_counter=self.llm,
                    include_system=True,
                    start_on="human"
                )
                
                # 트리밍된 메시지가 원본과 다르면 로그
                if len(trimmed_messages) < len(current_messages):
                    logger.info(f"메시지 트리밍: {len(current_messages)} -> {len(trimmed_messages)}")
            
            # LangChain으로 응답 생성
            response = await conversation.ainvoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}}
            )
            
            logger.info(f"LangChain 응답 생성 완료: {session_id}")
            return response
            
        except Exception as e:
            logger.error(f"LangChain 응답 생성 실패: {e}")
            # 폴백 응답
            return f"안녕 {child_name}아! 재미있는 이야기를 함께 만들어보자!"
    
    def get_conversation_summary(self, session_id: str) -> Dict[str, Any]:
        """대화 요약 정보 반환"""
        history = self.get_session_history(session_id)
        stats = history.get_session_stats()
        
        messages = history.get_recent_messages(5)
        recent_topics = []
        
        for msg in messages:
            if isinstance(msg, HumanMessage):
                # 간단한 키워드 추출
                words = msg.content.split()
                topics = [w for w in words if len(w) > 2 and w.isalpha()]
                recent_topics.extend(topics[:3])
        
        return {
            "session_stats": stats,
            "recent_topics": list(set(recent_topics)),
            "message_count": len(history.messages),
            "conversation_health": "active" if stats.get("message_count", 0) > 0 else "inactive"
        }
    
    def cleanup_old_sessions(self, days: int = 30):
        """오래된 세션 정리"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        with sqlite3.connect(self.db_path) as conn:
            # 오래된 세션 조회
            cursor = conn.execute(
                "SELECT session_id FROM chat_sessions "
                "WHERE last_activity < ?",
                (cutoff_date.isoformat(),)
            )
            
            old_sessions = [row[0] for row in cursor.fetchall()]
            
            # 오래된 메시지 및 세션 삭제
            for session_id in old_sessions:
                conn.execute("DELETE FROM chat_messages WHERE session_id = ?", (session_id,))
                conn.execute("DELETE FROM chat_sessions WHERE session_id = ?", (session_id,))
                
                # 메모리에서도 제거
                if session_id in self._session_store:
                    del self._session_store[session_id]
            
            conn.commit()
            
        logger.info(f"오래된 세션 정리 완료: {len(old_sessions)}개 세션 삭제")
        return len(old_sessions)

# 유틸리티 함수
def create_session_id(child_name: str) -> str:
    """세션 ID 생성"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{child_name}_{timestamp}_{uuid.uuid4().hex[:8]}"

def restore_session_from_file(file_path: str, memory_manager: ConversationMemoryManager) -> str:
    """파일에서 세션 복원"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 새 세션 ID 생성
        child_name = data.get("child_info", {}).get("name", "unknown")
        session_id = create_session_id(child_name)
        
        # 메시지 복원
        history = memory_manager.get_session_history(session_id)
        
        for msg_data in data.get("messages", []):
            role = msg_data.get("role")
            content = msg_data.get("content", "")
            
            if role == "user":
                history.add_message(HumanMessage(content=content))
            elif role == "assistant":
                history.add_message(AIMessage(content=content))
            elif role == "system":
                history.add_message(SystemMessage(content=content))
        
        # 세션 정보 업데이트
        child_info = data.get("child_info", {})
        history.update_session_info(
            child_name=child_info.get("name", "unknown"),
            age_group=child_info.get("age", 5),
            interests=child_info.get("interests", [])
        )
        
        logger.info(f"세션 복원 완료: {session_id} ({len(data.get('messages', []))}개 메시지)")
        return session_id
        
    except Exception as e:
        logger.error(f"세션 복원 실패: {e}")
        return None