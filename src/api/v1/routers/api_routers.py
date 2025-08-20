"""
JJongal Fairy Tale chatbot REST API router
API v1.0 implementation
"""

from fastapi import APIRouter, HTTPException, Depends, Query, Path
from fastapi.responses import JSONResponse
from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel
import asyncio
import os

# Dependencies import
from ..dependencies import verify_jwt_token, get_vector_db, get_connection_manager
from src.shared.utils.logging import get_module_logger
from src.shared.configs.app import AppConfig

logger = get_module_logger(__name__)

# APIRouter creation - API v1.0
router = APIRouter(
    prefix="/api/v1",
    tags=["api-v1", "rest-endpoints"],
    responses={
        404: {"description": "Not found"},
        500: {"description": "Internal server error"}
    }
)

# ====== Data Model Definitions =======
class HealthResponse(BaseModel):
    """System health status response model"""
    status: str
    timestamp: str
    services: Dict[str, str]
    version: str = "1.0.0"

class StoryMetadata(BaseModel):
    """Stroy metadata model"""
    story_id: str
    child_name: str
    title: str
    created_at: str
    age_group: str
    status: str # "generating", "completed", "failed"

class VoiceCloneStatus(BaseModel):
    """Voice cloning status model"""
    child_name: str 
    samples_collected: int
    required_samples: int = 5
    voice_ready: bool
    last_updated: str

# ====== Import Endpoints ======
from ..endpoints import stories, auth, health

# ====== Health Check Endpoint ======
@router.get("/health", response_model=HealthResponse)
async def health_check():
    """System health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        services={
            "api": "healthy",
            "chatbot_a": "healthy", 
            "chatbot_b": "healthy",
            "vector_db": "healthy"
        }
    )

# ====== Include Sub-routers ======
# Include stories endpoints
router.include_router(stories.router, tags=["stories"])

# Include auth endpoints if they exist
try:
    router.include_router(auth.router, tags=["auth"])
except AttributeError:
    logger.info("Auth endpoints not available")

# Include health endpoints if they exist
try:
    router.include_router(health.router, tags=["health"])
except AttributeError:
    logger.info("Additional health endpoints not available")




