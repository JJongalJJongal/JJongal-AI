"""
Memory Tool for ChatBot A

Manages conversation memory and story element collection.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
from src.shared.utils.logging import get_module_logger

logger = get_module_logger(__name__)

class MemoryTool:
    """Basic memory management for ChatBot A conversations"""
    
    def __init__(self):
        self.sessions: Dict[str, Dict[str, Any]] = {}
        logger.info("Memory Tool initialized")
    
    async def initialize_session_memory(self, session_id: str, child_name: str, child_age: int):
        """Initialize memory for new session"""
        self.sessions[session_id] = {
            "child_name": child_name,
            "child_age": child_age,
            "story_elements": {},
            "conversation_history": [],
            "created_at": datetime.now()
        }
        logger.info(f"Initialized memory for session {session_id}")
    
    async def store_message_memory(self, session_id: str, message: str, message_type: str):
        """Store message in session memory"""
        if session_id in self.sessions:
            self.sessions[session_id]["conversation_history"].append({
                "message": message,
                "type": message_type,
                "timestamp": datetime.now()
            })
    
    async def get_story_elements(self, session_id: str) -> Dict[str, Any]:
        """Get collected story elements"""
        return self.sessions.get(session_id, {}).get("story_elements", {})
    
    async def suggest_next_question(self, session_id: str) -> Optional[str]:
        """Suggest next question based on context"""
        if session_id not in self.sessions:
            return None
        
        elements = self.sessions[session_id]["story_elements"]
        
        if "character" not in elements:
            return "이야기의 주인공은 누구인가요?"
        elif "setting" not in elements:
            return "이야기가 일어나는 장소는 어디인가요?"
        elif "conflict" not in elements:
            return "주인공에게 어떤 문제가 생겼나요?"
        else:
            return "정말 재미있는 이야기네요! 더 들려주세요."

_memory_tool = None

def get_memory_tool() -> MemoryTool:
    global _memory_tool
    if _memory_tool is None:
        _memory_tool = MemoryTool()
    return _memory_tool