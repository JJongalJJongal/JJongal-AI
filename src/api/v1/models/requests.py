from pydantic import BaseModel
from typing import Optional, Dict, Any, Literal, List
from datetime import datetime

# Standard Response Models
class StandardResponse(BaseModel):
    success: bool # Success or Failure
    message: str # Message
    data: Optional[Dict[str, Any]] = None # Data

# Error Response Model
class ErrorResponse(BaseModel):
    success: bool = False # Success or Failure
    error: Dict[str, str] # {"code": "ERROR_CODE", "message": "..."}

# Authentication Models
class TokenRequest(BaseModel):
    user_id: str # User ID

class TokenResponse(BaseModel):
    token_type: Literal["Bearer"] = "Bearer"
    access_token: str
    expires_in: int = 3600

class StoryRequest(BaseModel):
    """Story generation request"""
    child_name: str # Child Name
    age: int
    interests: List[str] = []
    conversation_summary: str
    story_elements: Dict[str, Any]
    child_voice_id: Optional[str] = None

class Chapter(BaseModel):
    """Story chapter"""
    chapter_number: int
    title: str
    content: str
    image_url: Optional[str] = None
    audio_url: Optional[str] = None

class Story(BaseModel):
    """Complete story response"""
    story_id: str
    title: str
    status: Literal["generating", "completed", "failed"]
    chapters: List[Chapter]
    created_at: datetime
    generation_time: Optional[float] = None
    
# WebSocket Message Models
class AudioConfigMessage(BaseModel):
    event: Literal["start_conversation"]
    payload: Dict[str, Any]
    
class TranscriptionMessage(BaseModel):
    event: Literal["interim_transcription", "final_transcription", "voice_with_text"]
    payload: Dict[str, Any]
    
class ChatBotResponse(BaseModel):
    event: Literal["jjong_response"]
    payload: Dict[str, Any]

class VoiceClineStatus(BaseModel):
    event: Literal["voice_clone_update"]
    payload: Dict[str, Any]

# Legacy Models
class Users(BaseModel):
    id: str
    object: Literal["user"] = "user"
    name: str
    age: int
    created_at: datetime

class File(BaseModel):
    object: Literal["file"] = "file"
    type: Literal["internal"] = "internal"
    url: str
    expiry_time: datetime
    

# Legacy Ari models
class AriStoryRequest(BaseModel):
    """Legacy: Complete fairy tale generation request"""
    conversation_summary: str
    story_elements: Dict[str, Any]
    child_name: str
    age: int
    conversation_analysis: Optional[Dict[str, Any]] = None
    extracted_keywords: Optional[List[str]] = None
    child_voice_id: Optional[str] = None
    main_character_name: Optional[str] = None
    

# New Story Creation Request
class StoryCreateRequest(BaseModel):
    """Legacy: Story creation request"""
    user_preferences: Dict[str, Any]
    conversation_id: str