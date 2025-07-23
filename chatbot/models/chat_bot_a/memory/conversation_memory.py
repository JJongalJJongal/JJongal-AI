"""
Enhanced Conversation Memory Management

LangChain-based persistent memory with intelligent context management,
automatic summarization, and child-appropriate conversation tracking.
"""

from typing import Dict, Any, List, Optional
import json
import sqlite3
from datetime import datetime, timedelta
import uuid

from langchain.memory import ConversationSummaryBufferMemory
from langchain.memory.chat_message_histories import SQLChatMessageHistory
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

from shared.utils.logging_utils import get_module_logger

logger = get_module_logger(__name__)


class ConversationMemoryManager:
    """
    Advanced conversation memory with LangChain integration.
    
    Features:
    - Persistent SQLite storage
    - Automatic conversation summarization
    - Child profile tracking
    - Context-aware memory retrieval
    - Token-aware memory management
    """
    
    def __init__(
        self, 
        db_path: str = "chatbot/data/conversations.db",
        model_name: str = "gpt-4o-mini",
        max_token_limit: int = 8000,
        summary_threshold: int = 20
    ):
        self.db_path = db_path
        self.model_name = model_name
        self.max_token_limit = max_token_limit
        self.summary_threshold = summary_threshold
        
        self.llm = ChatOpenAI(model=model_name, temperature=0.3)
        self.active_sessions = {}
        
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize SQLite database with enhanced schema"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Enhanced message history table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS message_store (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    message_id TEXT UNIQUE,
                    message_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    token_count INTEGER DEFAULT 0
                )
            """)
            
            # Create indexes separately
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_session_id ON message_store(session_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON message_store(timestamp)")
            
            # Session metadata table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS session_metadata (
                    session_id TEXT PRIMARY KEY,
                    child_name TEXT,
                    child_age INTEGER,
                    child_interests TEXT,
                    session_start DATETIME,
                    last_activity DATETIME,
                    total_messages INTEGER DEFAULT 0,
                    story_elements_collected INTEGER DEFAULT 0,
                    conversation_summary TEXT,
                    is_active BOOLEAN DEFAULT 1
                )
            """)
            
            # Story elements table for tracking collection progress
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS story_elements (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    element_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    confidence_score REAL DEFAULT 0.8,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    source_message_id TEXT,
                    FOREIGN KEY (session_id) REFERENCES session_metadata (session_id)
                )
            """)
            
            conn.commit()
            conn.close()
            
            logger.info("Conversation memory database initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize conversation memory database: {e}")
            raise
    
    def create_session(
        self, 
        child_name: str, 
        child_age: int, 
        child_interests: List[str] = None
    ) -> str:
        """Create a new conversation session"""
        session_id = f"conv_{uuid.uuid4().hex[:12]}_{datetime.now().strftime('%Y%m%d_%H%M')}"
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO session_metadata 
                (session_id, child_name, child_age, child_interests, session_start, last_activity)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                session_id,
                child_name,
                child_age,
                json.dumps(child_interests or []),
                datetime.now(),
                datetime.now()
            ))
            
            conn.commit()
            conn.close()
            
            # Initialize LangChain memory for this session
            chat_history = SQLChatMessageHistory(
                session_id=session_id,
                connection_string=f"sqlite:///{self.db_path}",
                table_name="message_store"
            )
            
            memory = ConversationSummaryBufferMemory(
                llm=self.llm,
                chat_memory=chat_history,
                max_token_limit=self.max_token_limit,
                return_messages=True
            )
            
            self.active_sessions[session_id] = {
                "memory": memory,
                "chat_history": chat_history,
                "child_name": child_name,
                "child_age": child_age,
                "child_interests": child_interests or [],
                "message_count": 0,
                "last_summary_at": 0
            }
            
            logger.info(f"Created conversation session: {session_id} for {child_name} ({child_age}세)")
            return session_id
            
        except Exception as e:
            logger.error(f"Failed to create conversation session: {e}")
            raise
    
    def add_message(self, session_id: str, message: BaseMessage, metadata: Dict[str, Any] = None):
        """Add message to conversation memory with enhanced tracking"""
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        try:
            session = self.active_sessions[session_id]
            
            # Add to LangChain memory
            session["chat_history"].add_message(message)
            
            # Update session metadata
            session["message_count"] += 1
            
            # Update database metadata
            self._update_session_activity(session_id)
            
            # Check if summarization is needed
            if (session["message_count"] - session["last_summary_at"]) >= self.summary_threshold:
                self._create_conversation_summary(session_id)
            
            logger.debug(f"Added message to session {session_id}: {type(message).__name__}")
            
        except Exception as e:
            logger.error(f"Failed to add message to session {session_id}: {e}")
            raise
    
    def get_conversation_context(
        self, 
        session_id: str, 
        include_summary: bool = True,
        max_messages: int = 10
    ) -> Dict[str, Any]:
        """Get conversation context with intelligent memory retrieval"""
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        try:
            session = self.active_sessions[session_id]
            memory = session["memory"]
            
            # Get buffer from memory (recent messages + summary if needed)
            memory_buffer = memory.chat_memory.messages
            
            # Get recent messages
            recent_messages = memory_buffer[-max_messages:] if memory_buffer else []
            
            context = {
                "session_id": session_id,
                "child_profile": {
                    "name": session["child_name"],
                    "age": session["child_age"],
                    "interests": session["child_interests"]
                },
                "recent_messages": [
                    {
                        "type": type(msg).__name__,
                        "content": msg.content,
                        "timestamp": getattr(msg, "timestamp", None)
                    }
                    for msg in recent_messages
                ],
                "message_count": session["message_count"],
                "memory_summary": memory.buffer if include_summary else None
            }
            
            return context
            
        except Exception as e:
            logger.error(f"Failed to get conversation context for session {session_id}: {e}")
            raise
    
    def get_story_elements(self, session_id: str) -> List[Dict[str, Any]]:
        """Retrieve collected story elements for the session"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT element_type, content, confidence_score, timestamp
                FROM story_elements
                WHERE session_id = ?
                ORDER BY timestamp ASC
            """, (session_id,))
            
            elements = []
            for row in cursor.fetchall():
                elements.append({
                    "element_type": row[0],
                    "content": row[1],
                    "confidence_score": row[2],
                    "timestamp": row[3]
                })
            
            conn.close()
            return elements
            
        except Exception as e:
            logger.error(f"Failed to get story elements for session {session_id}: {e}")
            return []
    
    def add_story_element(
        self, 
        session_id: str, 
        element_type: str, 
        content: str, 
        confidence_score: float = 0.8,
        source_message_id: str = None
    ):
        """Add a story element to the session"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO story_elements 
                (session_id, element_type, content, confidence_score, source_message_id)
                VALUES (?, ?, ?, ?, ?)
            """, (session_id, element_type, content, confidence_score, source_message_id))
            
            # Update story elements count in session metadata
            cursor.execute("""
                UPDATE session_metadata 
                SET story_elements_collected = (
                    SELECT COUNT(*) FROM story_elements WHERE session_id = ?
                )
                WHERE session_id = ?
            """, (session_id, session_id))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Added story element ({element_type}): {content[:50]}... to session {session_id}")
            
        except Exception as e:
            logger.error(f"Failed to add story element to session {session_id}: {e}")
            raise
    
    def _update_session_activity(self, session_id: str):
        """Update session last activity timestamp"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE session_metadata 
                SET last_activity = ?, total_messages = total_messages + 1
                WHERE session_id = ?
            """, (datetime.now(), session_id))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to update session activity: {e}")
    
    def _create_conversation_summary(self, session_id: str):
        """Create conversation summary using LangChain memory"""
        try:
            session = self.active_sessions[session_id]
            memory = session["memory"]
            
            # Force memory to create summary
            summary = memory.predict_new_summary(
                memory.chat_memory.messages,
                memory.buffer
            )
            
            # Store summary in database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE session_metadata 
                SET conversation_summary = ?
                WHERE session_id = ?
            """, (summary, session_id))
            
            conn.commit()
            conn.close()
            
            # Update last summary point
            session["last_summary_at"] = session["message_count"]
            
            logger.info(f"Created conversation summary for session {session_id}")
            
        except Exception as e:
            logger.error(f"Failed to create conversation summary for session {session_id}: {e}")
    
    def close_session(self, session_id: str) -> Dict[str, Any]:
        """Close conversation session and return summary"""
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        try:
            # Create final summary
            self._create_conversation_summary(session_id)
            
            # Mark session as inactive
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE session_metadata 
                SET is_active = 0
                WHERE session_id = ?
            """, (session_id,))
            
            conn.commit()
            conn.close()
            
            # Get session summary
            session_data = self.active_sessions[session_id]
            summary = {
                "session_id": session_id,
                "child_name": session_data["child_name"],
                "message_count": session_data["message_count"],
                "story_elements": self.get_story_elements(session_id),
                "conversation_summary": session_data["memory"].buffer
            }
            
            # Clean up active session
            del self.active_sessions[session_id]
            
            logger.info(f"Closed conversation session: {session_id}")
            return summary
            
        except Exception as e:
            logger.error(f"Failed to close session {session_id}: {e}")
            raise
    
    def restore_session(self, session_id: str) -> bool:
        """Restore a previous conversation session"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get session metadata
            cursor.execute("""
                SELECT child_name, child_age, child_interests, total_messages
                FROM session_metadata
                WHERE session_id = ?
            """, (session_id,))
            
            row = cursor.fetchone()
            if not row:
                logger.warning(f"Session {session_id} not found in database")
                return False
            
            child_name, child_age, child_interests, total_messages = row
            child_interests = json.loads(child_interests) if child_interests else []
            
            # Initialize LangChain memory
            chat_history = SQLChatMessageHistory(
                session_id=session_id,
                connection_string=f"sqlite:///{self.db_path}",
                table_name="message_store"
            )
            
            memory = ConversationSummaryBufferMemory(
                llm=self.llm,
                chat_memory=chat_history,
                max_token_limit=self.max_token_limit,
                return_messages=True
            )
            
            self.active_sessions[session_id] = {
                "memory": memory,
                "chat_history": chat_history,
                "child_name": child_name,
                "child_age": child_age,
                "child_interests": child_interests,
                "message_count": total_messages,
                "last_summary_at": 0
            }
            
            conn.close()
            
            logger.info(f"Restored conversation session: {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to restore session {session_id}: {e}")
            return False
    
    def cleanup_old_sessions(self, days_old: int = 30) -> int:
        """Clean up old inactive sessions"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_old)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get sessions to delete
            cursor.execute("""
                SELECT session_id FROM session_metadata
                WHERE last_activity < ? AND is_active = 0
            """, (cutoff_date,))
            
            sessions_to_delete = [row[0] for row in cursor.fetchall()]
            
            # Delete associated data
            for session_id in sessions_to_delete:
                cursor.execute("DELETE FROM story_elements WHERE session_id = ?", (session_id,))
                cursor.execute("DELETE FROM message_store WHERE session_id = ?", (session_id,))
                cursor.execute("DELETE FROM session_metadata WHERE session_id = ?", (session_id,))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Cleaned up {len(sessions_to_delete)} old conversation sessions")
            return len(sessions_to_delete)
            
        except Exception as e:
            logger.error(f"Failed to cleanup old sessions: {e}")
            return 0