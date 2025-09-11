"""
JJongal-AI Conversation Data Models

Pydantic models for conversation sessions and message management.
"""

import uuid
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum

from src.shared.utils.age_group_utils import AgeGroupManager


class MessageType(str, Enum):
    """Message type enumeration"""
    HUMAN = "HumanMessage"
    AI = "AIMessage"
    SYSTEM = "SystemMessage"
    FUNCTION = "FunctionMessage"


class SessionStatus(str, Enum):
    """Conversation session status"""
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    ABANDONED = "abandoned"
    ERROR = "error"


class ConversationMessage(BaseModel):
    """Individual conversation message model"""
    
    message_id: str = Field(..., description="Unique message identifier")
    session_id: str = Field(..., description="Associated session ID")
    message_type: MessageType = Field(..., description="Type of message")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Optional[Dict[str, Any]] = Field(default=None)


class StoryElementProgress(BaseModel):
    """Progress tracking for story element collection"""
    
    element_type: str = Field(..., description="Story element type")
    content: str = Field(..., description="Element content")
    confidence: float = Field(..., ge=0.0, le=1.0)
    collected_at: datetime = Field(default_factory=datetime.now)
    source_message_id: Optional[str] = Field(default=None)


class ConversationSession(BaseModel):
    """Conversation session model"""
    
    session_id: str = Field(..., description="Unique session identifier")
    child_name: str = Field(..., description="Child name")
    child_age: int = Field(..., ge=3, le=12, description="Child age")
    child_interests: List[str] = Field(default_factory=list)
    age_group: str = Field(..., description="Child's age group")
    
    status: SessionStatus = Field(default=SessionStatus.ACTIVE)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    messages: List[ConversationMessage] = Field(default_factory=list)
    story_elements: List[StoryElementProgress] = Field(default_factory=list)
    
    max_messages: int = Field(default=50)
    session_timeout_minutes: int = Field(default=30)
    
    conversation_summary: Optional[str] = Field(default=None)
    engagement_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    completion_percentage: float = Field(default=0.0, ge=0.0, le=1.0)
    
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def add_message(
        self, 
        message_type: MessageType, 
        content: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Add a new message to the session"""
        from uuid import uuid4
        
        message_id = f"msg_{uuid4().hex[:12]}"
        message = ConversationMessage(
            message_id=message_id,
            session_id=self.session_id,
            message_type=message_type,
            content=content,
            metadata=metadata or {}
        )
        
        self.messages.append(message)
        self.updated_at = datetime.now()
        
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]
        
        return message_id
    
    def add_story_element(
        self, 
        element_type: str, 
        content: str, 
        confidence: float = 1.0,
        source_message_id: Optional[str] = None
    ) -> None:
        """Add a collected story element"""
        element = StoryElementProgress(
            element_type=element_type,
            content=content,
            confidence=confidence,
            source_message_id=source_message_id
        )
        
        self.story_elements.append(element)
        self.updated_at = datetime.now()


def create_conversation_session(
    child_name: str,
    child_age: int,
    child_interests: Optional[List[str]] = None
) -> ConversationSession:
    """
    Creates and initializes a new conversation session.
    """
    session_id = f"session_{uuid.uuid4().hex[:12]}"
    
    age_group_manager = AgeGroupManager()
    age_group = age_group_manager.classify_age_group(child_age)
    
    session = ConversationSession(
        session_id=session_id,
        child_name=child_name,
        child_age=child_age,
        child_interests=child_interests or [],
        age_group=age_group
    )
    
    return session

def create_message(
    session: ConversationSession,
    message_type: MessageType,
    content: str,
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """
    Creates and adds a new message to a session.
    (Backward compatibility for the legacy adapter)
    """
    return session.add_message(
        message_type=message_type,
        content=content,
        metadata=metadata
    )


# Type aliases for convenience
SessionData = ConversationSession
MessageData = ConversationMessage
ElementData = StoryElementProgress