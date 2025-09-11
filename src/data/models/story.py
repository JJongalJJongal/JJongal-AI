"""
Story Data Models for JJongal-AI

Pydantic models for story entities, metadata, and related data structures
used throughout the JJongal-AI storytelling system.
"""

from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field, field_validator
from datetime import datetime
from enum import Enum
import uuid


class StoryStatus(str, Enum):
    """Story generation and processing status"""
    DRAFT = "draft"
    GENERATING = "generating"
    COMPLETED = "completed"
    FAILED = "failed"
    ARCHIVED = "archived"


class AgeGroup(str, Enum):
    """Age group classification for stories"""
    YOUNG = "4-6"  # Young children (4-6 years)
    OLD = "7-9"    # Older children (7-9 years)


class StoryGenre(str, Enum):
    """Story genre types"""
    ADVENTURE = "adventure"
    FANTASY = "fantasy"
    FRIENDSHIP = "friendship"
    FAMILY = "family"
    ANIMAL = "animal"
    EDUCATIONAL = "educational"
    MYSTERY = "mystery"


class StoryElement(BaseModel):
    """Individual story element (character, setting, etc.)"""
    
    element_type: str = Field(..., description="Type of story element")
    name: str = Field(..., description="Element name")
    description: str = Field(..., description="Element description")
    importance: float = Field(default=1.0, ge=0.0, le=1.0, description="Importance score")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class StoryMetadata(BaseModel):
    """Story metadata and analytics"""
    
    word_count: int = Field(default=0, description="Total word count")
    reading_time_minutes: float = Field(default=0.0, description="Estimated reading time")
    age_appropriateness_score: float = Field(default=1.0, ge=0.0, le=1.0)
    educational_value: Optional[str] = Field(default=None, description="Educational aspects")
    moral_lesson: Optional[str] = Field(default=None, description="Moral or lesson")
    
    # Generation metrics
    generation_time_seconds: float = Field(default=0.0, description="Time taken to generate")
    ai_model_used: Optional[str] = Field(default=None, description="AI model identifier")
    quality_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Generated quality score")


class Story(BaseModel):
    """Complete story model with all associated data"""
    
    # Core identifiers
    story_id: str = Field(default_factory=lambda: f"story_{uuid.uuid4().hex[:12]}")
    session_id: Optional[str] = Field(default=None, description="Associated conversation session")
    
    # Story content
    title: str = Field(..., description="Story title")
    content: str = Field(..., description="Complete story text")
    summary: str = Field(..., description="Story summary")
    
    # Classification
    genre: StoryGenre = Field(..., description="Story genre")
    age_group: AgeGroup = Field(..., description="Target age group")
    status: StoryStatus = Field(default=StoryStatus.DRAFT, description="Current status")
    
    # Story elements
    characters: List[StoryElement] = Field(default_factory=list, description="Story characters")
    settings: List[StoryElement] = Field(default_factory=list, description="Story settings")
    themes: List[str] = Field(default_factory=list, description="Story themes")
    
    # Child information
    child_name: str = Field(..., description="Child's name")
    child_age: int = Field(..., ge=3, le=12, description="Child's age")
    child_interests: List[str] = Field(default_factory=list, description="Child's interests")
    
    # Multimedia content
    image_urls: List[str] = Field(default_factory=list, description="Generated image URLs")
    audio_urls: List[str] = Field(default_factory=list, description="Generated audio URLs")
    voice_clone_id: Optional[str] = Field(default=None, description="Child's voice clone ID")
    
    # Metadata
    metadata: StoryMetadata = Field(default_factory=StoryMetadata)
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = Field(default=None)
    
    # Additional data
    tags: List[str] = Field(default_factory=list, description="Story tags")
    language: str = Field(default="ko", description="Story language")
    
    @field_validator('child_age')
    @classmethod
    def validate_age_group_consistency(cls, v, info):
        """Ensure child age matches the age group"""
        if hasattr(info, 'data') and 'age_group' in info.data:
            age_group = info.data['age_group']
            if age_group == AgeGroup.YOUNG and not (4 <= v <= 6):
                raise ValueError("Child age must be 4-6 for YOUNG age group")
            elif age_group == AgeGroup.OLD and not (7 <= v <= 9):
                raise ValueError("Child age must be 7-9 for OLD age group")
        return v
    
    def mark_completed(self):
        """Mark story as completed"""
        self.status = StoryStatus.COMPLETED
        self.completed_at = datetime.now()
        self.updated_at = datetime.now()
    
    def mark_failed(self, error_message: str = None):
        """Mark story as failed"""
        self.status = StoryStatus.FAILED
        self.updated_at = datetime.now()
        if error_message:
            self.metadata.metadata["error_message"] = error_message
    
    def add_character(self, name: str, description: str, importance: float = 1.0):
        """Add a character to the story"""
        character = StoryElement(
            element_type="character",
            name=name,
            description=description,
            importance=importance
        )
        self.characters.append(character)
        self.updated_at = datetime.now()
    
    def add_setting(self, name: str, description: str, importance: float = 1.0):
        """Add a setting to the story"""
        setting = StoryElement(
            element_type="setting",
            name=name,
            description=description,
            importance=importance
        )
        self.settings.append(setting)
        self.updated_at = datetime.now()
    
    def get_primary_character(self) -> Optional[StoryElement]:
        """Get the primary character (highest importance)"""
        if not self.characters:
            return None
        return max(self.characters, key=lambda c: c.importance)
    
    def get_primary_setting(self) -> Optional[StoryElement]:
        """Get the primary setting (highest importance)"""
        if not self.settings:
            return None
        return max(self.settings, key=lambda s: s.importance)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return self.dict(by_alias=True)


# Request/Response models for API
class CreateStoryRequest(BaseModel):
    """Request model for creating a new story"""
    
    session_id: str = Field(..., description="Conversation session ID")
    child_name: str = Field(..., description="Child's name")
    child_age: int = Field(..., ge=3, le=12)
    child_interests: List[str] = Field(default_factory=list)
    
    title: Optional[str] = Field(default=None)
    genre: StoryGenre = Field(default=StoryGenre.ADVENTURE)
    
    # Story elements from conversation
    characters: List[Dict[str, Any]] = Field(default_factory=list)
    settings: List[Dict[str, Any]] = Field(default_factory=list)
    themes: List[str] = Field(default_factory=list)
    
    # Generation preferences
    include_images: bool = Field(default=True)
    include_audio: bool = Field(default=True)
    voice_clone_id: Optional[str] = Field(default=None)


class UpdateStoryRequest(BaseModel):
    """Request model for updating an existing story"""
    
    title: Optional[str] = Field(default=None)
    content: Optional[str] = Field(default=None)
    summary: Optional[str] = Field(default=None)
    status: Optional[StoryStatus] = Field(default=None)
    tags: Optional[List[str]] = Field(default=None)
    metadata: Optional[Dict[str, Any]] = Field(default=None)


class StoryResponse(BaseModel):
    """Response model for story data"""
    
    story_id: str
    title: str
    status: StoryStatus
    summary: str
    content: Optional[str] = None
    
    genre: StoryGenre
    age_group: AgeGroup
    child_name: str
    
    image_urls: List[str] = Field(default_factory=list)
    audio_urls: List[str] = Field(default_factory=list)
    
    created_at: datetime
    updated_at: datetime
    completed_at: Optional[datetime] = None
    
    metadata: Dict[str, Any] = Field(default_factory=dict)