"""
Stories API Endpoints - API v2.0 Compliant
Handles direct story generation from StoryRequest format
"""

from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from fastapi.responses import JSONResponse

from src.shared.utils.logging import get_module_logger
from src.core.chatbots.collaboration.jjong_ari_collaborator import ModernJjongAriCollaborator

logger = get_module_logger(__name__)

router = APIRouter()

# Initialize collaborator
collaborator = ModernJjongAriCollaborator()

@router.post("/stories", response_model=Dict[str, Any])
async def create_story(
    story_request: Dict[str, Any],
    background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """
    Generate story directly from API v2.0 StoryRequest format
    
    Args:
        story_request: StoryRequest format from API v2.0
        
    Returns:
        Story format compliant with API v2.0
    """
    try:
        logger.info(f"Story creation request received: {story_request.get('child_name', 'unknown')}")
        
        # Validate required fields
        required_fields = ["child_name", "age"]
        for field in required_fields:
            if field not in story_request:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Missing required field: {field}"
                )
        
        # Validate age range
        age = story_request.get("age")
        if not isinstance(age, int) or age < 4 or age > 9:
            raise HTTPException(
                status_code=400,
                detail="Age must be an integer between 4 and 9"
            )
        
        # Generate story using ModernAri directly
        story_result = await collaborator.create_story_from_request(story_request)
        
        logger.info(f"Story generated successfully: {story_result['story_id']}")
        return story_result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Story generation failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during story generation")

@router.get("/stories/{story_id}", response_model=Dict[str, Any])
async def get_story(story_id: str) -> Dict[str, Any]:
    """
    Get story by ID
    
    Args:
        story_id: Story identifier
        
    Returns:
        Story format compliant with API v2.0
    """
    try:
        # For now, return a simple response since we don't have persistent storage
        # In a real implementation, this would query a database
        logger.info(f"Story retrieval request: {story_id}")
        
        # This would be replaced with actual database lookup
        return {
            "story_id": story_id,
            "title": f"Story {story_id}",
            "status": "completed",
            "chapters": [],
            "created_at": "2024-01-01T00:00:00Z",
            "generation_time": 0.0
        }
        
    except Exception as e:
        logger.error(f"Story retrieval failed for {story_id}: {e}")
        raise HTTPException(status_code=404, detail="Story not found")

@router.get("/stories/{story_id}/status", response_model=Dict[str, Any])
async def get_story_status(story_id: str) -> Dict[str, Any]:
    """
    Get story generation status
    
    Args:
        story_id: Story identifier
        
    Returns:
        Story status information
    """
    try:
        # Get status from collaborator
        status_info = await collaborator.get_collaboration_status(story_id)
        
        return {
            "story_id": story_id,
            "status": status_info.get("status", "unknown"),
            "last_updated": status_info.get("last_updated")
        }
        
    except Exception as e:
        logger.error(f"Status check failed for {story_id}: {e}")
        raise HTTPException(status_code=500, detail="Status check failed")