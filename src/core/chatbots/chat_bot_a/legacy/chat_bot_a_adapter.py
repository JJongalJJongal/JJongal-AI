"""
Legacy ChatBot A Adapter for JJongal-AI

Provides backward compatibility and migration support for
legacy ChatBot A implementations and API interfaces.
"""

from typing import Dict, Any, Optional
from datetime import datetime

from src.shared.utils.logging import get_module_logger
from src.data.models.conversation import (
    ConversationSession, create_conversation_session
)

logger = get_module_logger(__name__)

class LegacyChatBotAAdapter:
    """
    Legacy adapter for ChatBot A compatibility
    
    Provides compatibility layer for older API interfaces
    while maintaining integration with new system components.
    """
    
    def __init__(self):
        """Initialize legacy adapter"""
        self.active_sessions: Dict[str, ConversationSession] = {}
        self.legacy_session_mapping: Dict[str, str] = {}  # old_id -> new_id
        
        logger.info("Legacy ChatBot A Adapter initialized")
    
    async def create_legacy_session(
        self,
        child_name: str,
        child_age: int,
        legacy_session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create session using legacy API format
        
        Args:
            child_name: Child's name
            child_age: Child's age
            legacy_session_id: Optional legacy session identifier
            
        Returns:
            Dict: Legacy format session data
        """
        try:
            # Create new session
            session = create_conversation_session(
                child_name=child_name,
                child_age=child_age
            )
            
            # Store session
            self.active_sessions[session.session_id] = session
            
            # Map legacy ID if provided
            if legacy_session_id:
                self.legacy_session_mapping[legacy_session_id] = session.session_id
            
            # Return legacy format
            return {
                "session_id": session.session_id,
                "status": "created",
                "child_info": {
                    "name": child_name,
                    "age": child_age,
                    "age_group": session.age_group
                },
                "conversation_state": "initialized",
                "timestamp": session.start_time.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to create legacy session: {e}")
            raise

# Global legacy adapter instance
_legacy_adapter = None

def get_legacy_adapter() -> LegacyChatBotAAdapter:
    """Get global legacy adapter instance"""
    global _legacy_adapter
    if _legacy_adapter is None:
        _legacy_adapter = LegacyChatBotAAdapter()
    return _legacy_adapter