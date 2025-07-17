from pydantic import BaseModel
from typing import Optional, Dict, Any, Literal, List
from datetime import datetime

# 표준 응답 모델
class StandardResponse(BaseModel):
    success: bool # 성공 여부
    message: str # 메시지
    data: Optional[Dict[str, Any]] = None # 데이터

# 에러 응답 모델
class ErrorResponse(BaseModel):
    success: bool = False # 성공 여부
    error: Dict[str, str] # {"code": "ERROR_CODE", "message": "..."}


# User Object
class User(BaseModel):
    id: str # User UUID
    object: Literal["user"] = "user" # 객체 타입
    name: str # 아이 이름
    age: int # 아이 나이
    created_at: datetime # 생성 시간
    
# File Object
class File(BaseModel):
    object: Literal["file"] = "file" # 객체 타입
    type: Literal["internal"] = "internal" # 파일 타입
    url: str # File URL
    expiry_time: datetime # 만료 시간

# Chapter Object
class Chapter(BaseModel):
    id: str # Chapter UUID
    object: Literal["chapter"] = "chapter" # 객체 타입
    chapter_number: int # Chapter Number
    title: str # Chapter Title
    content: str # Chapter Content
    image: Optional[File] = None # Chapter Image
    audio: Optional[File] = None # Chapter Audio

# Story Object
class Story(BaseModel):
    id: str # Story UUID
    object: Literal["story"] = "story" # 객체 타입
    owner_id: str # Owner User ID
    title: str # Story Title
    status: Literal["pending", "in_progress", "completed", "failed"] # Story Status
    chapters: List[Chapter] = [] # Chapter List
    created_at: datetime # 생성 시간
    updated_at: datetime # 수정 시간

# Authentification Object
class TokenRequest(BaseModel):
    user_id: str # User ID
    password: str # 향후 확장예정
    
class TokenResponse(BaseModel):
    token_type: Literal["Bearer"] = "Bearer" # Token Type
    access_token: str # JWT Token
    expires_in: int = 3600 # Token 만료 시간 (초)
    refresh_token: str
    
# Story Creation Object
class StoryCreateRequest(BaseModel):
    user_preferences: Dict[str, Any] # User Preferences
    conversation_id: str # UUID
    
# WebSocket Message Object
class AudioConfigMessage(BaseModel):
    event: Literal["start_conversation"]
    payload: Dict[str, Any]

# Transcription Message Object
class TranscriptionMessage(BaseModel):
    event: Literal["interim_transcription", "final_transcription", "voice_with_text"]
    payload: Dict[str, Any]
    
class ChatBotResponse(BaseModel):
    event: Literal["jjong_response"]
    payload: Dict[str, Any]
    
class VoiceClineStatus(BaseModel):
    event: Literal["voice_clone_update"]
    payload: Dict[str, Any]
