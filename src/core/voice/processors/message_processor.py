"""
Message Processor for WebSocket communications
"""

from typing import Dict, Any, Optional
from src.shared.utils.logging import get_module_logger

logger = get_module_logger(__name__)

class MessageProcessor:
    """Processes WebSocket messages"""
    
    VALID_MESSAGE_TYPES = [
        "start_conversation",
        "end_conversation", 
        "user_message"
    ]
    
    def __init__(self):
        self.active_sessions = {}
        logger.info("MessageProcessor initialized")
    
    def validate_message(self, message: Dict[str, Any]) -> bool:
        """Validate message format"""
        if not isinstance(message, dict):
            return False
            
        message_type = message.get("type")
        if not message_type or message_type not in self.VALID_MESSAGE_TYPES:
            return False
        
        return True