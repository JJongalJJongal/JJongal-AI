"""
ChatBot Voice Options API Endpoints
Manages voice configuration for 쫑이 (ChatBot A)
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from src.core.chatbots.chat_bot_a.chat_bot_a import ChatBotA
from src.core.chatbots.chat_bot_a.voice_options import JjongiVoiceType
from src.api.v1.dependencies import get_authenticated_user
from src.shared.utils.logging import get_module_logger

logger = get_module_logger(__name__)

# Create router
router = APIRouter(
    prefix="/chatbot-voice", 
    tags=["ChatBot Voice Options"]
)

# Global ChatBot A instance for voice management
chatbot_a = ChatBotA(enable_monitoring=False)

# === Request/Response Models ===

class VoiceOptionResponse(BaseModel):
    """Voice option information"""
    voice_type: str = Field(..., description="Voice type identifier")
    display_name: str = Field(..., description="User-friendly display name")
    description: str = Field(..., description="Voice description")
    is_current: bool = Field(..., description="Whether this is the current active voice")

class VoiceOptionsResponse(BaseModel):
    """List of available voice options"""
    voice_options: List[VoiceOptionResponse]
    current_voice: str = Field(..., description="Currently active voice type")
    total_options: int = Field(..., description="Total number of voice options")

class VoiceChangeRequest(BaseModel):
    """Request to change voice option"""
    voice_type: str = Field(..., description="Voice type to switch to", 
                           pattern="^(cheerful|gentle|playful)$")

class VoiceChangeResponse(BaseModel):
    """Response after voice change"""
    success: bool = Field(..., description="Whether the change was successful")
    new_voice: str = Field(..., description="New active voice type")
    message: str = Field(..., description="Success/error message")

class VoiceConfigResponse(BaseModel):
    """Current voice configuration"""
    voice_id: str = Field(..., description="ElevenLabs Voice ID")
    voice_type: str = Field(..., description="Voice type")
    display_name: str = Field(..., description="Display name")
    voice_settings: Dict[str, Any] = Field(..., description="ElevenLabs voice settings")
    conversation_optimizations: Optional[Dict[str, Any]] = Field(None, description="Conversation-specific settings")

class VoiceValidationResponse(BaseModel):
    """Voice configuration validation results"""
    validation_results: Dict[str, bool] = Field(..., description="Validation status per voice type")
    needs_setup: bool = Field(..., description="Whether any voices need setup")
    current_voice: str = Field(..., description="Current active voice")
    total_voices: int = Field(..., description="Total number of configured voices")

class AdminVoiceUpdateRequest(BaseModel):
    """Admin request to update voice IDs"""
    voice_id_mapping: Dict[str, str] = Field(..., description="Mapping of voice types to ElevenLabs Voice IDs")

# === API Endpoints ===

@router.get("/options", response_model=VoiceOptionsResponse)
async def get_voice_options() -> VoiceOptionsResponse:
    """
    Get available voice options for 쫑이
    
    Returns list of available voice personalities with their descriptions.
    """
    try:
        voice_options = chatbot_a.get_voice_options()
        current_voice = chatbot_a.voice_manager.current_voice.value
        
        return VoiceOptionsResponse(
            voice_options=[VoiceOptionResponse(**option) for option in voice_options],
            current_voice=current_voice,
            total_options=len(voice_options)
        )
    except Exception as e:
        logger.error(f"Error getting voice options: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve voice options")

@router.post("/change", response_model=VoiceChangeResponse)
async def change_voice_option(request: VoiceChangeRequest) -> VoiceChangeResponse:
    """
    Change 쫑이's current voice option
    
    Allows switching between different voice personalities:
    - cheerful: 밝고 활발한 목소리
    - gentle: 부드럽고 따뜻한 목소리  
    - playful: 장난기 있고 재미있는 목소리
    """
    try:
        success = chatbot_a.set_voice_option(request.voice_type)
        
        if success:
            return VoiceChangeResponse(
                success=True,
                new_voice=request.voice_type,
                message=f"쫑이의 목소리가 '{request.voice_type}'로 변경되었습니다."
            )
        else:
            return VoiceChangeResponse(
                success=False,
                new_voice=chatbot_a.voice_manager.current_voice.value,
                message=f"유효하지 않은 음성 옵션입니다: {request.voice_type}"
            )
    except Exception as e:
        logger.error(f"Error changing voice option: {e}")
        raise HTTPException(status_code=500, detail="Failed to change voice option")

@router.get("/current", response_model=VoiceConfigResponse)
async def get_current_voice_config(session_id: Optional[str] = None) -> VoiceConfigResponse:
    """
    Get current voice configuration for TTS generation
    
    Returns detailed voice configuration including ElevenLabs settings.
    Optionally optimized for specific conversation context.
    """
    try:
        if session_id:
            # Get conversation-optimized config
            config = chatbot_a.get_voice_config_for_conversation(session_id)
        else:
            # Get basic current config
            config = chatbot_a.get_current_voice_config()
        
        return VoiceConfigResponse(**config)
    except Exception as e:
        logger.error(f"Error getting voice config: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve voice configuration")

@router.get("/validation", response_model=VoiceValidationResponse)
async def validate_voice_configuration() -> VoiceValidationResponse:
    """
    Validate current voice configuration setup
    
    Checks if all voice options have valid ElevenLabs Voice IDs configured.
    Useful for admin monitoring and setup verification.
    """
    try:
        validation_data = chatbot_a.validate_voice_configuration()
        return VoiceValidationResponse(**validation_data)
    except Exception as e:
        logger.error(f"Error validating voice configuration: {e}")
        raise HTTPException(status_code=500, detail="Failed to validate voice configuration")

@router.post("/admin/update-voice-ids")
async def update_voice_ids(
    request: AdminVoiceUpdateRequest,
    user: Dict[str, Any] = Depends(get_authenticated_user)
) -> Dict[str, str]:
    """
    Update ElevenLabs Voice IDs (Admin only)
    
    Allows administrators to configure the actual ElevenLabs Voice IDs
    for each voice personality option.
    
    **Admin access required**
    """
    try:
        # Validate voice types in the mapping
        valid_voice_types = {vt.value for vt in JjongiVoiceType}
        invalid_types = set(request.voice_id_mapping.keys()) - valid_voice_types
        
        if invalid_types:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid voice types: {invalid_types}. Valid types: {valid_voice_types}"
            )
        
        # Update voice IDs
        chatbot_a.update_voice_ids(request.voice_id_mapping)
        
        # Validate the update
        validation = chatbot_a.validate_voice_configuration()
        
        logger.info(f"Voice IDs updated by admin user: {user.get('user_id', 'unknown')}")
        
        return {
            "message": "Voice IDs updated successfully",
            "updated_voices": list(request.voice_id_mapping.keys()),
            "validation_passed": not validation["needs_setup"]
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating voice IDs: {e}")
        raise HTTPException(status_code=500, detail="Failed to update voice IDs")

@router.get("/demo/sample", response_model=Dict[str, Any])
async def get_voice_sample_for_demo() -> Dict[str, Any]:
    """
    Get sample voice configuration for demo/testing purposes
    
    Returns example voice settings that can be used for testing
    without requiring actual ElevenLabs Voice IDs.
    """
    try:
        sample_config = {
            "demo_voice_settings": {
                "cheerful": {
                    "stability": 0.75,
                    "similarity_boost": 0.85,
                    "style": 0.2,
                    "use_speaker_boost": True
                },
                "gentle": {
                    "stability": 0.85,
                    "similarity_boost": 0.80,
                    "style": 0.1,
                    "use_speaker_boost": False
                },
                "playful": {
                    "stability": 0.70,
                    "similarity_boost": 0.90,
                    "style": 0.4,
                    "use_speaker_boost": True
                }
            },
            "demo_message": "이 설정들은 데모용입니다. 실제 음성 생성을 위해서는 유효한 ElevenLabs Voice ID가 필요합니다.",
            "voice_types": [vt.value for vt in JjongiVoiceType]
        }
        
        return sample_config
    except Exception as e:
        logger.error(f"Error getting demo sample: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve demo sample")

# Health check for this module
@router.get("/health")
async def voice_module_health() -> Dict[str, Any]:
    """Voice module health check"""
    try:
        validation = chatbot_a.validate_voice_configuration()
        
        return {
            "status": "healthy",
            "voice_manager_initialized": chatbot_a.voice_manager is not None,
            "total_voice_options": validation["total_voices"],
            "voices_configured": sum(validation["validation_results"].values()),
            "current_voice": validation["current_voice"],
            "needs_setup": validation["needs_setup"]
        }
    except Exception as e:
        logger.error(f"Voice module health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }