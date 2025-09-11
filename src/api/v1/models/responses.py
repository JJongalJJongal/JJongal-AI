"""
API Response Models for JJongal-AI

Standardized response models for all API endpoints with consistent
error handling, status codes, and data structures.
"""

from typing import Any, Dict, List, Optional, Union, Generic, TypeVar
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum

T = TypeVar('T')

class ResponseStatus(str, Enum):
    """Response status types"""
    SUCCESS = "success"
    ERROR = "error"
    PARTIAL = "partial"
    PENDING = "pending"

class ErrorCode(str, Enum):
    """Standardized error codes"""
    # Authentication errors
    AUTH_TOKEN_INVALID = "AUTH_TOKEN_INVALID"
    AUTH_TOKEN_EXPIRED = "AUTH_TOKEN_EXPIRED"
    AUTH_PERMISSION_DENIED = "AUTH_PERMISSION_DENIED"
    
    # Validation errors
    VALIDATION_ERROR = "VALIDATION_ERROR"
    INVALID_INPUT = "INVALID_INPUT"
    MISSING_REQUIRED_FIELD = "MISSING_REQUIRED_FIELD"
    
    # Business logic errors
    WORKFLOW_NOT_FOUND = "WORKFLOW_NOT_FOUND"
    WORKFLOW_ALREADY_COMPLETED = "WORKFLOW_ALREADY_COMPLETED"
    CONVERSATION_SESSION_NOT_FOUND = "CONVERSATION_SESSION_NOT_FOUND"
    STORY_GENERATION_FAILED = "STORY_GENERATION_FAILED"
    VOICE_CLONING_FAILED = "VOICE_CLONING_FAILED"
    
    # Resource errors
    RESOURCE_NOT_FOUND = "RESOURCE_NOT_FOUND"
    RESOURCE_ALREADY_EXISTS = "RESOURCE_ALREADY_EXISTS"
    RESOURCE_LIMIT_EXCEEDED = "RESOURCE_LIMIT_EXCEEDED"
    
    # External service errors
    EXTERNAL_API_ERROR = "EXTERNAL_API_ERROR"
    AI_SERVICE_UNAVAILABLE = "AI_SERVICE_UNAVAILABLE"
    VOICE_SERVICE_UNAVAILABLE = "VOICE_SERVICE_UNAVAILABLE"
    
    # System errors
    INTERNAL_SERVER_ERROR = "INTERNAL_SERVER_ERROR"
    SERVICE_TEMPORARILY_UNAVAILABLE = "SERVICE_TEMPORARILY_UNAVAILABLE"
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"

class ErrorDetail(BaseModel):
    """Detailed error information"""
    code: ErrorCode = Field(..., description="Error code")
    message: str = Field(..., description="Human-readable error message")
    field: Optional[str] = Field(None, description="Field causing the error (for validation errors)")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")

class BaseResponse(BaseModel, Generic[T]):
    """Base response model for all API endpoints"""
    status: ResponseStatus = Field(..., description="Response status")
    message: str = Field(..., description="Response message")
    data: Optional[T] = Field(None, description="Response data")
    errors: Optional[List[ErrorDetail]] = Field(None, description="Error details")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")
    request_id: Optional[str] = Field(None, description="Request identifier for tracking")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

# Health Check Responses
class HealthCheckData(BaseModel):
    """Health check response data"""
    service_name: str = Field(..., description="Service name")
    version: str = Field(..., description="Service version")
    status: str = Field(..., description="Service status")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")
    dependencies: Dict[str, str] = Field(..., description="Dependency status")
    performance_metrics: Dict[str, Any] = Field(..., description="Performance metrics")

class HealthCheckResponse(BaseResponse[HealthCheckData]):
    """Health check endpoint response"""
    pass

# Authentication Responses
class AuthTokenData(BaseModel):
    """Authentication token data"""
    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field("bearer", description="Token type")
    expires_in: int = Field(..., description="Token expiration time in seconds")
    refresh_token: Optional[str] = Field(None, description="Refresh token")
    user_id: str = Field(..., description="User identifier")
    permissions: List[str] = Field(..., description="User permissions")

class AuthTokenResponse(BaseResponse[AuthTokenData]):
    """Authentication token response"""
    pass

class UserProfileData(BaseModel):
    """User profile data"""
    user_id: str = Field(..., description="User identifier")
    child_name: str = Field(..., description="Child's name")
    child_age: int = Field(..., description="Child's age")
    child_interests: List[str] = Field(..., description="Child's interests")
    voice_clone_available: bool = Field(..., description="Voice cloning availability")
    created_at: datetime = Field(..., description="Account creation date")
    last_active: datetime = Field(..., description="Last activity timestamp")

class UserProfileResponse(BaseResponse[UserProfileData]):
    """User profile response"""
    pass

# Story Generation Responses
class StoryElementsData(BaseModel):
    """Story elements data"""
    character: Optional[str] = Field(None, description="Main character")
    character_traits: List[str] = Field(default_factory=list, description="Character traits")
    setting: Optional[str] = Field(None, description="Story setting")
    theme: Optional[str] = Field(None, description="Story theme")
    conflict: Optional[str] = Field(None, description="Story conflict")
    resolution: Optional[str] = Field(None, description="Conflict resolution")
    additional_elements: Dict[str, Any] = Field(default_factory=dict, description="Additional story elements")

class ConversationSessionData(BaseModel):
    """Conversation session data"""
    session_id: str = Field(..., description="Session identifier")
    child_name: str = Field(..., description="Child's name")
    child_age: int = Field(..., description="Child's age")
    status: str = Field(..., description="Session status")
    current_stage: str = Field(..., description="Current conversation stage")
    completion_percentage: float = Field(..., description="Completion percentage")
    story_elements: StoryElementsData = Field(..., description="Collected story elements")
    duration_minutes: float = Field(..., description="Session duration in minutes")
    voice_samples_collected: int = Field(..., description="Voice samples collected")
    is_active: bool = Field(..., description="Session active status")

class ConversationSessionResponse(BaseResponse[ConversationSessionData]):
    """Conversation session response"""
    pass

class StoryChapterData(BaseModel):
    """Story chapter data"""
    chapter_number: int = Field(..., description="Chapter number")
    title: str = Field(..., description="Chapter title")
    content: str = Field(..., description="Chapter content")
    image_url: Optional[str] = Field(None, description="Chapter image URL")
    audio_url: Optional[str] = Field(None, description="Chapter audio URL")

class GeneratedStoryData(BaseModel):
    """Generated story data"""
    story_id: str = Field(..., description="Story identifier")
    title: str = Field(..., description="Story title")
    chapters: List[StoryChapterData] = Field(..., description="Story chapters")
    total_chapters: int = Field(..., description="Total number of chapters")
    genre: str = Field(..., description="Story genre")
    age_group: str = Field(..., description="Target age group")
    moral_lesson: str = Field(..., description="Story moral lesson")
    created_at: datetime = Field(..., description="Creation timestamp")
    estimated_reading_time: int = Field(..., description="Estimated reading time in minutes")

class StoryGenerationResponse(BaseResponse[GeneratedStoryData]):
    """Story generation response"""
    pass

# Workflow Responses
class WorkflowProgressData(BaseModel):
    """Workflow progress data"""
    workflow_id: str = Field(..., description="Workflow identifier")
    child_name: str = Field(..., description="Child's name")
    current_stage: str = Field(..., description="Current workflow stage")
    progress_percentage: float = Field(..., description="Progress percentage")
    estimated_completion_time: Optional[datetime] = Field(None, description="Estimated completion time")
    last_update: datetime = Field(..., description="Last update timestamp")
    is_active: bool = Field(..., description="Workflow active status")

class WorkflowStatusResponse(BaseResponse[WorkflowProgressData]):
    """Workflow status response"""
    pass

class WorkflowResultData(BaseModel):
    """Workflow result data"""
    workflow_id: str = Field(..., description="Workflow identifier")
    success: bool = Field(..., description="Success status")
    duration_seconds: float = Field(..., description="Execution duration")
    story: Optional[GeneratedStoryData] = Field(None, description="Generated story")
    error_message: Optional[str] = Field(None, description="Error message if failed")

class WorkflowResultResponse(BaseResponse[WorkflowResultData]):
    """Workflow result response"""
    pass

# Voice Processing Responses
class VoiceCloneStatusData(BaseModel):
    """Voice clone status data"""
    session_id: str = Field(..., description="Voice cloning session ID")
    child_name: str = Field(..., description="Child's name")
    status: str = Field(..., description="Cloning status")
    samples_collected: int = Field(..., description="Samples collected")
    samples_needed: int = Field(..., description="Samples still needed")
    quality_score: float = Field(..., description="Average quality score")
    voice_id: Optional[str] = Field(None, description="Generated voice ID")
    ready: bool = Field(..., description="Voice clone ready status")

class VoiceCloneStatusResponse(BaseResponse[VoiceCloneStatusData]):
    """Voice clone status response"""
    pass

class AudioProcessingData(BaseModel):
    """Audio processing result data"""
    audio_id: str = Field(..., description="Audio identifier")
    transcription: Optional[str] = Field(None, description="Speech transcription")
    duration: float = Field(..., description="Audio duration in seconds")
    quality_score: float = Field(..., description="Audio quality score")
    voice_clone_suitable: bool = Field(..., description="Suitable for voice cloning")
    processed_audio_url: Optional[str] = Field(None, description="Processed audio URL")

class AudioProcessingResponse(BaseResponse[AudioProcessingData]):
    """Audio processing response"""
    pass

# List Responses
class PaginationMeta(BaseModel):
    """Pagination metadata"""
    page: int = Field(..., description="Current page number")
    per_page: int = Field(..., description="Items per page")
    total_items: int = Field(..., description="Total number of items")
    total_pages: int = Field(..., description="Total number of pages")
    has_next: bool = Field(..., description="Has next page")
    has_prev: bool = Field(..., description="Has previous page")

class StoryListData(BaseModel):
    """Story list data"""
    stories: List[GeneratedStoryData] = Field(..., description="List of stories")
    pagination: PaginationMeta = Field(..., description="Pagination information")

class StoryListResponse(BaseResponse[StoryListData]):
    """Story list response"""
    pass

class WorkflowListData(BaseModel):
    """Workflow list data"""
    workflows: List[WorkflowProgressData] = Field(..., description="List of workflows")
    active_count: int = Field(..., description="Number of active workflows")
    completed_count: int = Field(..., description="Number of completed workflows")
    pagination: PaginationMeta = Field(..., description="Pagination information")

class WorkflowListResponse(BaseResponse[WorkflowListData]):
    """Workflow list response"""
    pass

# WebSocket Responses
class WebSocketEventData(BaseModel):
    """WebSocket event data"""
    event_type: str = Field(..., description="Event type")
    session_id: str = Field(..., description="Session identifier")
    data: Dict[str, Any] = Field(..., description="Event data")
    timestamp: datetime = Field(default_factory=datetime.now, description="Event timestamp")

class WebSocketResponse(BaseModel):
    """WebSocket message response"""
    type: str = Field(..., description="Message type")
    data: WebSocketEventData = Field(..., description="Event data")

# System Information Responses
class SystemMetricsData(BaseModel):
    """System metrics data"""
    cpu_usage_percent: float = Field(..., description="CPU usage percentage")
    memory_usage_mb: float = Field(..., description="Memory usage in MB")
    active_connections: int = Field(..., description="Active connections count")
    active_workflows: int = Field(..., description="Active workflows count")
    total_stories_generated: int = Field(..., description="Total stories generated")
    uptime_seconds: float = Field(..., description="System uptime in seconds")

class SystemMetricsResponse(BaseResponse[SystemMetricsData]):
    """System metrics response"""
    pass

# Utility Response Creators
def create_success_response(
    data: T, 
    message: str = "Operation completed successfully",
    request_id: Optional[str] = None
) -> BaseResponse[T]:
    """Create a success response"""
    return BaseResponse(
        status=ResponseStatus.SUCCESS,
        message=message,
        data=data,
        request_id=request_id
    )

def create_error_response(
    error_code: ErrorCode,
    message: str,
    field: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
    request_id: Optional[str] = None
) -> BaseResponse[None]:
    """Create an error response"""
    return BaseResponse(
        status=ResponseStatus.ERROR,
        message="Request failed",
        data=None,
        errors=[ErrorDetail(
            code=error_code,
            message=message,
            field=field,
            details=details
        )],
        request_id=request_id
    )

def create_validation_error_response(
    field: str,
    message: str,
    request_id: Optional[str] = None
) -> BaseResponse[None]:
    """Create a validation error response"""
    return create_error_response(
        error_code=ErrorCode.VALIDATION_ERROR,
        message=message,
        field=field,
        request_id=request_id
    )

def create_not_found_response(
    resource_type: str,
    resource_id: str,
    request_id: Optional[str] = None
) -> BaseResponse[None]:
    """Create a not found error response"""
    return create_error_response(
        error_code=ErrorCode.RESOURCE_NOT_FOUND,
        message=f"{resource_type} with ID '{resource_id}' not found",
        details={"resource_type": resource_type, "resource_id": resource_id},
        request_id=request_id
    )

def create_unauthorized_response(
    message: str = "Authentication required",
    request_id: Optional[str] = None
) -> BaseResponse[None]:
    """Create an unauthorized error response"""
    return create_error_response(
        error_code=ErrorCode.AUTH_TOKEN_INVALID,
        message=message,
        request_id=request_id
    )

def create_forbidden_response(
    message: str = "Permission denied",
    request_id: Optional[str] = None
) -> BaseResponse[None]:
    """Create a forbidden error response"""
    return create_error_response(
        error_code=ErrorCode.AUTH_PERMISSION_DENIED,
        message=message,
        request_id=request_id
    )

def create_rate_limit_response(
    retry_after_seconds: int,
    request_id: Optional[str] = None
) -> BaseResponse[None]:
    """Create a rate limit error response"""
    return create_error_response(
        error_code=ErrorCode.RATE_LIMIT_EXCEEDED,
        message="Rate limit exceeded. Please try again later.",
        details={"retry_after_seconds": retry_after_seconds},
        request_id=request_id
    )