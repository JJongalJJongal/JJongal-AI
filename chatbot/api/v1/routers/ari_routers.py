"""
Ari (Chatbot B) Internal Management API Router

REST API endpoints for complete multimedia fairy tale generation
- Text-only generation
- Complete fairy tale generation (text + images + voice)
- Status monitoring and regeneration
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from fastapi.responses import JSONResponse
from typing import Dict, Any, Optional, List
import asyncio
import uuid
from datetime import datetime

# Core imports
from chatbot.models.chat_bot_b import ChatBotB
from chatbot.workflow.orchestrator import WorkflowOrchestrator
from chatbot.workflow.story_schema import ChildProfile, StoryDataSchema

# API models
from chatbot.api.v1.models import (
    AriStoryRequest, AriTextOnlyRequest, AriRegenerateRequest,
    AriStoryResponse, AriStatusResponse, AriErrorResponse
)

# Dependencies
from chatbot.api.v1.dependencies import get_chatbot_b, get_workflow_orchestrator

# Logging
from shared.utils.logging_utils import get_module_logger

logger = get_module_logger(__name__)

# Create router
router = APIRouter(
    prefix="/ari",
    tags=["Ari (Chatbot B)"],
    responses={
        404: {"description": "Not found"},
        422: {"description": "Validation Error"},
        500: {"description": "Internal Server Error"}
    }
)

# Track active generation tasks
active_generations: Dict[str, Dict[str, Any]] = {}

@router.post(
    "/generate/complete",
    response_model=AriStoryResponse,
    summary="Generate Complete Fairy Tale",
    description="Generate complete multimedia fairy tale including text, images, and voice"
)

async def generate_complete_story(
    request: AriStoryRequest,
    background_tasks: BackgroundTasks,
    chatbot_b: ChatBotB = Depends(get_chatbot_b),
    orchestrator: WorkflowOrchestrator = Depends(get_workflow_orchestrator)
) -> AriStoryResponse:
    """
    Generate complete multimedia fairy tale
    
    Args:
        request: Fairy tale generation request data
        background_tasks: Background task manager
        chatbot_b: ChatBot B instance
        orchestrator: Workflow orchestrator
        
    Returns:
        Generated fairy tale data and metadata
    """
    
    generation_id = str(uuid.uuid4())
    
    try:
        logger.info(f"Starting complete fairy tale generation: {generation_id}")
        
        # Initialize progress tracking
        active_generations[generation_id] = {
            "status": "starting",
            "progress": 0.0,
            "stage": "initialization",
            "start_time": datetime.now(),
        }
    
        # Convert request to story outline format
        story_outline = {
            "conversation_summary": request.conversation_summary,
            "child_profile": {
                "name": request.child_name,
                "age": request.child_age
            },
            "conversation_analysis": request.conversation_analysis or {},
            "extracted_keywords": request.extracted_keywords or [],
            "story_generation_method": "complete"
        }