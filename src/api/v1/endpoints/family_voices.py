"""
Family Voice Management API Endpoints
Handles family member voice cloning (mom, dad) and management
"""

import base64
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.shared.utils.logging import get_module_logger
from ..dependencies import verify_jwt_token, get_voice_cloning_processor

logger = get_module_logger(__name__)

router = APIRouter()

# ⭐ Request/Response Models
class AudioSample(BaseModel):
    sample_number: int = Field(..., description="Sample number (1-5)")
    audio_data: str = Field(..., description="Base64 encoded audio data")
    duration: float = Field(..., description="Audio duration in seconds")
    format: str = Field(default="mp3", description="Audio format (mp3, wav, m4a)")

class FamilyVoiceRequest(BaseModel):
    family_member: str = Field(..., description="Family member type", pattern="^(mom|dad)$")
    voice_name: str = Field(..., description="Custom voice name", min_length=1, max_length=50)
    audio_samples: List[AudioSample] = Field(..., description="Audio samples (5 required)", min_items=5, max_items=5)

class FamilyVoice(BaseModel):
    voice_id: str
    family_member: str
    voice_name: str
    status: str  # "uploading" | "processing" | "completed" | "failed"
    samples_count: int
    created_at: str
    processing_time: Optional[float] = None
    sample_quality: Optional[Dict[str, Any]] = None

class FamilyVoicesListResponse(BaseModel):
    family_voices: List[Dict[str, Any]]
    total_count: int
    max_allowed: int = Field(default=4, description="Maximum allowed voices (2 per family member)")

# ⭐ API Endpoints
@router.post("/voices/family", response_model=Dict[str, Any])
async def upload_family_voice(
    request: FamilyVoiceRequest,
    background_tasks: BackgroundTasks,
    user_info: Dict[str, Any] = Depends(verify_jwt_token),
    voice_processor = Depends(get_voice_cloning_processor)
) -> Dict[str, Any]:
    """
    Upload and clone family member voice (API v2.0)
    
    Creates a cloned voice for a family member (mom or dad) using 5 audio samples.
    The voice cloning process is performed in the background.
    
    Args:
        request: FamilyVoiceRequest with audio samples
        background_tasks: FastAPI background tasks
        user_info: Authenticated user information
        voice_processor: VoiceCloningProcessor instance
        
    Returns:
        Family voice creation result with processing status
    """
    try:
        user_id = user_info.get("user_id")
        family_member = request.family_member
        voice_name = request.voice_name
        
        logger.info(f"Family voice upload request - user: {user_id}, member: {family_member}, name: {voice_name}")
        
        # Check if user already has too many voices for this family member
        existing_voices = voice_processor.list_family_voices(user_id)
        member_voices = [v for v in existing_voices if v["family_member"] == family_member]
        
        if len(member_voices) >= voice_processor.max_voices_per_family_member:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Maximum {voice_processor.max_voices_per_family_member} voices allowed per family member"
            )
        
        # Validate and process audio samples
        sample_results = []
        for sample in request.audio_samples:
            try:
                # Decode base64 audio data
                audio_data = base64.b64decode(sample.audio_data)
                
                # Collect voice sample
                result = await voice_processor.collect_family_voice_sample(
                    user_id=user_id,
                    family_member=family_member,
                    voice_name=voice_name,
                    audio_data=audio_data,
                    file_format=sample.format
                )
                
                sample_results.append(result)
                
                if not result["success"]:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Sample {sample.sample_number} failed: {result['error']}"
                    )
                    
            except Exception as e:
                logger.error(f"Error processing sample {sample.sample_number}: {e}")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid audio data in sample {sample.sample_number}"
                )
        
        # Start background voice cloning
        background_tasks.add_task(
            _process_family_voice_cloning,
            voice_processor,
            user_id,
            family_member,
            voice_name
        )
        
        # Return immediate response
        voice_id = f"voice_{family_member}_{user_id}_{len(member_voices) + 1}"
        
        return {
            "voice_id": voice_id,
            "family_member": family_member,
            "voice_name": voice_name,
            "status": "processing",
            "samples_count": len(request.audio_samples),
            "estimated_completion": "2-3 minutes",
            "message": "Voice cloning started in background"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Family voice upload failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during voice upload"
        )

@router.get("/voices/family", response_model=FamilyVoicesListResponse)
async def list_family_voices(
    user_info: Dict[str, Any] = Depends(verify_jwt_token),
    voice_processor = Depends(get_voice_cloning_processor)
) -> FamilyVoicesListResponse:
    """
    List all family voices for the authenticated user
    
    Returns a list of all uploaded and cloned family member voices
    with their current status and metadata.
    
    Args:
        user_info: Authenticated user information
        voice_processor: VoiceCloningProcessor instance
        
    Returns:
        List of family voices with metadata
    """
    try:
        user_id = user_info.get("user_id")
        
        logger.info(f"Family voices list request - user: {user_id}")
        
        # Get all family voices for user
        family_voices = voice_processor.list_family_voices(user_id)
        
        # Format response according to API spec
        formatted_voices = []
        for voice in family_voices:
            formatted_voices.append({
                "voice_id": voice["voice_id"],
                "family_member": voice["family_member"],
                "voice_name": voice["voice_name"],
                "status": voice["status"],
                "created_at": voice["created_at"],
                "samples_count": voice.get("sample_count", 5)
            })
        
        return FamilyVoicesListResponse(
            family_voices=formatted_voices,
            total_count=len(formatted_voices),
            max_allowed=len(voice_processor.supported_family_members) * voice_processor.max_voices_per_family_member
        )
        
    except Exception as e:
        logger.error(f"Family voices list failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error retrieving family voices"
        )

@router.delete("/voices/family/{voice_id}")
async def delete_family_voice(
    voice_id: str,
    user_info: Dict[str, Any] = Depends(verify_jwt_token),
    voice_processor = Depends(get_voice_cloning_processor)
) -> Dict[str, Any]:
    """
    Delete a family member voice
    
    Removes the specified family voice and cleans up associated files.
    This action cannot be undone.
    
    Args:
        voice_id: ID of the voice to delete
        user_info: Authenticated user information
        voice_processor: VoiceCloningProcessor instance
        
    Returns:
        Deletion confirmation
    """
    try:
        user_id = user_info.get("user_id")
        
        logger.info(f"Family voice deletion request - user: {user_id}, voice_id: {voice_id}")
        
        # Find the voice and family member
        family_voices = voice_processor.list_family_voices(user_id)
        target_voice = None
        
        for voice in family_voices:
            if voice["voice_id"] == voice_id:
                target_voice = voice
                break
        
        if not target_voice:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Family voice not found"
            )
        
        # Delete the voice
        result = await voice_processor.delete_family_voice(
            user_id=user_id,
            family_member=target_voice["family_member"]
        )
        
        if not result["success"]:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result["error"]
            )
        
        return {
            "success": True,
            "message": "가족 음성이 성공적으로 삭제되었습니다.",
            "deleted_voice_id": voice_id,
            "family_member": target_voice["family_member"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Family voice deletion failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during voice deletion"
        )

@router.get("/voices/family/{voice_id}/status")
async def get_family_voice_status(
    voice_id: str,
    user_info: Dict[str, Any] = Depends(verify_jwt_token),
    voice_processor = Depends(get_voice_cloning_processor)
) -> Dict[str, Any]:
    """
    Get detailed status of a specific family voice
    
    Args:
        voice_id: ID of the voice to check
        user_info: Authenticated user information
        voice_processor: VoiceCloningProcessor instance
        
    Returns:
        Detailed voice status information
    """
    try:
        user_id = user_info.get("user_id")
        
        # Find the voice
        family_voices = voice_processor.list_family_voices(user_id)
        target_voice = None
        
        for voice in family_voices:
            if voice["voice_id"] == voice_id:
                target_voice = voice
                break
        
        if not target_voice:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Family voice not found"
            )
        
        # Get detailed status
        detailed_status = voice_processor.get_family_voice_status(
            user_id=user_id,
            family_member=target_voice["family_member"]
        )
        
        return detailed_status
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Family voice status check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error checking voice status"
        )

# ⭐ Background task for voice cloning
async def _process_family_voice_cloning(
    voice_processor,
    user_id: str,
    family_member: str,
    voice_name: str
):
    """
    Background task to process family voice cloning
    
    Args:
        voice_processor: VoiceCloningProcessor instance
        user_id: User identifier
        family_member: Family member type
        voice_name: Voice name
    """
    try:
        logger.info(f"Starting background family voice cloning - user: {user_id}, member: {family_member}")
        
        # Perform voice cloning
        result = await voice_processor.create_family_voice_clone(
            user_id=user_id,
            family_member=family_member
        )
        
        if result["success"]:
            logger.info(f"Family voice cloning completed - user: {user_id}, member: {family_member}, voice_id: {result['voice_id']}")
        else:
            logger.error(f"Family voice cloning failed - user: {user_id}, member: {family_member}, error: {result['error']}")
        
    except Exception as e:
        logger.error(f"Background family voice cloning error - user: {user_id}, member: {family_member}, error: {e}")