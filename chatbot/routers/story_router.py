"""
Router for Story Management API Endpoints

Handles creating, retrieving, updating, and deleting stories,
and managing the story generation workflow via ChatBotB.
"""
import time # Ensure time is imported at the top
from fastapi import APIRouter, HTTPException, status, Body, BackgroundTasks
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
import asyncio # For async operations with ChatBotB

from shared.utils.logging_utils import get_module_logger
# Assuming ChatBotB and RAGSystem are accessible; adjust imports as needed
from ..models.chat_bot_b import StoryGenerationChatBot
# from ..models.rag_system import RAGSystem # If RAG interactions are part of story management
# Import the new DB functions
from chatbot.db import ( 
    upsert_task,
    get_task as get_task_from_db,
    update_task_status as update_task_status_in_db,
    # delete_task # if needed later
)

logger = get_module_logger("story_router")
router = APIRouter()

# --- Pydantic Models for Story API ---
class StoryOutlineRequest(BaseModel):
    child_name: str = Field(..., description="아이 이름")
    age: int = Field(..., ge=3, le=12, description="아이 나이 (3-12세)")
    interests: Optional[List[str]] = Field(None, description="아이 관심사 목록")
    story_theme_summary: str = Field(..., description="ChatBot A 또는 사용자가 제공한 이야기 주제/줄거리 요약")
    # Add other fields from ChatBotA's story_outline if needed, e.g., tags, characters
    initial_tags: Optional[List[str]] = Field(None, description="초기 태그")

class StoryGenerationStatus(BaseModel):
    story_id: str
    status: str # e.g., "pending", "generating_text", "generating_images", "generating_voice", "completed", "failed"
    message: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    input_outline: Optional[Dict[str, Any]] = None # To store the original input outline

class GeneratedStoryResponse(BaseModel):
    story_id: str
    title: Optional[str]
    content: Dict[str, Any] # Full story content (scenes, characters etc.)
    illustrations: Optional[List[Dict[str, str]]] # list of {scene_id: ..., image_url: ...}
    audio_files: Optional[List[Dict[str, str]]]  # list of {narration/character: ..., audio_url: ...}

# --- Background Task Function ---
async def process_story_generation_task_background(story_id: str):
    """
    Background task to process story generation.
    This function contains the logic previously in generate_full_story_endpoint.
    """
    logger.info(f"백그라운드 작업 시작: 전체 이야기 생성 (ID: {story_id})")
    
    task_data = get_task_from_db(story_id)
    if not task_data:
        logger.error(f"백그라운드 작업: Story task ID {story_id}를 DB에서 찾을 수 없습니다. 작업 중단.")
        return

    # Convert task_data (dict from DB) to StoryGenerationStatus Pydantic model for consistency if needed
    # Or directly use fields from task_data dict. For status updates, we call DB functions directly.
    current_status_str = task_data.get("status")
    current_details = task_data.get("details", {}) # Ensure details is a dict
    input_outline_data = task_data.get("input_outline")

    if current_status_str not in ["queued_for_generation", "failed_text_generation", "failed_image_generation", "failed_voice_generation", "failed_runtime_error"]:
        logger.warning(f"백그라운드 작업: Task {story_id}가 전체 생성을 위한 실행 가능한 상태가 아닙니다. 현재 상태: {current_status_str}. 작업 중단.")
        return

    if not input_outline_data:
        logger.error(f"백그라운드 작업: 입력 개요 정보 누락 (ID: {story_id}). 작업 중단.")
        update_task_status_in_db(story_id, "failed_missing_outline", "Error: Input outline data is missing for background task.")
        return

    chatbot_b = None
    try:
        chatbot_b = StoryGenerationChatBot()
        chatbot_b.set_target_age(input_outline_data.get("age"))
        story_outline_for_b = {
            "summary_text": input_outline_data.get("story_theme_summary"),
            "tags": input_outline_data.get("initial_tags", []),
        }
        chatbot_b.set_story_outline(story_outline_for_b)
        logger.info(f"백그라운드 작업: ChatBot B 인스턴스 생성 및 초기화 완료 (ID: {story_id})")

    except Exception as e:
        logger.error(f"백그라운드 작업: ChatBot B 인스턴스 생성 또는 초기화 실패 (ID: {story_id}): {e}", exc_info=True)
        update_task_status_in_db(story_id, "failed_init_chatbot_b", f"Error initializing ChatBot B in background: {str(e)}")
        return

    try:
        update_task_status_in_db(story_id, "generating_text", "백그라운드: 상세 이야기 텍스트 생성 중...")
        logger.info(f"백그라운드 작업 [{story_id}]: 상세 텍스트 생성 시작")
        
        detailed_story_content = await asyncio.to_thread(chatbot_b.generate_detailed_story)
        if not detailed_story_content:
            update_task_status_in_db(story_id, "failed_text_generation", "백그라운드: 상세 이야기 텍스트 생성 실패")
            logger.error(f"백그라운드 작업 [{story_id}]: 상세 텍스트 생성 실패")
            return
        logger.info(f"백그라운드 작업 [{story_id}]: 상세 텍스트 생성 완료")
        # current_details is already a dict from task_data or initialized to {}
        current_details["text_content_preview"] = str(detailed_story_content)[:200]
        current_details["full_text_content"] = detailed_story_content
        update_task_status_in_db(story_id, "generating_text", "텍스트 생성 완료, 삽화 생성 준비 중...", details_update=current_details)

        update_task_status_in_db(story_id, "generating_images", "백그라운드: 삽화 생성 중...", details_update=current_details)
        logger.info(f"백그라운드 작업 [{story_id}]: 삽화 생성 시작")
        illustrations_result = await asyncio.to_thread(chatbot_b.generate_illustrations)
        if not illustrations_result:
            logger.warning(f"백그라운드 작업 [{story_id}]: 삽화 생성 실패 또는 결과 없음. 계속 진행.")
            current_details["illustration_status"] = "failed_or_empty"
        else:
            logger.info(f"백그라운드 작업 [{story_id}]: 삽화 생성 완료. 결과: {illustrations_result}")
            current_details["illustrations"] = illustrations_result
        update_task_status_in_db(story_id, "generating_images", "삽화 생성 완료, 음성 합성 준비 중...", details_update=current_details)
        
        update_task_status_in_db(story_id, "generating_voice", "백그라운드: 음성 합성 중...", details_update=current_details)
        logger.info(f"백그라운드 작업 [{story_id}]: 음성 합성 시작")
        voice_data_result = await asyncio.to_thread(chatbot_b.generate_voice)
        if not voice_data_result:
            logger.warning(f"백그라운드 작업 [{story_id}]: 음성 합성 실패 또는 결과 없음. 계속 진행.")
            current_details["voice_status"] = "failed_or_empty"
        else:
            logger.info(f"백그라운드 작업 [{story_id}]: 음성 합성 완료. 결과: {voice_data_result}")
            current_details["voice_data"] = voice_data_result
        
        update_task_status_in_db(story_id, "completed", "백그라운드: 이야기 생성 완료 (텍스트, 삽화, 음성)", details_update=current_details)
        logger.info(f"백그라운드 작업: 이야기 생성 작업 완료 (ID: {story_id})" )

    except Exception as e:
        logger.error(f"백그라운드 작업: 전체 이야기 생성 중 오류 발생 (ID: {story_id}): {e}", exc_info=True)
        update_task_status_in_db(story_id, "failed_runtime_error", f"Error during background generation process: {str(e)}", details_update=current_details)
    finally:
        if chatbot_b:
            logger.debug(f"백그라운드 작업: ChatBot B 인스턴스 정리 (ID: {story_id})")
            del chatbot_b

# --- API Endpoints ---

@router.post("/initiate", 
             response_model=StoryGenerationStatus, 
             status_code=status.HTTP_202_ACCEPTED,
             summary="이야기 생성 작업 등록 및 백그라운드 처리 시작")
async def initiate_story_generation(
    outline_request: StoryOutlineRequest = Body(...),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    story_id = f"story_{outline_request.child_name.replace(' ','_')}_{int(time.time())}"
    logger.info(f"이야기 생성 작업 등록 요청 수신 (ID: {story_id}): {outline_request.child_name}, {outline_request.age}세")

    existing_task_data = get_task_from_db(story_id)
    if existing_task_data:
        existing_status = StoryGenerationStatus(**existing_task_data) # Convert dict to Pydantic model
        if existing_status.status not in ["failed", "failed_text_generation", "failed_image_generation", "failed_voice_generation", "failed_missing_outline", "failed_init_chatbot_b", "failed_runtime_error"]:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, 
                                detail=f"Story generation for ID {story_id} is already active or completed: {existing_status.status}")
        logger.info(f"기존 실패한 작업 (ID: {story_id})에 대한 생성 재시도.")

    try:
        input_outline_dict = outline_request.model_dump()
        # Initial upsert with input_outline, details will be populated by background task
        upsert_task(
            story_id=story_id,
            status="queued_for_generation",
            message="이야기 생성 작업이 대기열에 추가되어 백그라운드 처리를 시작합니다.",
            input_outline=input_outline_dict,
            details={}
        )
        
        background_tasks.add_task(process_story_generation_task_background, story_id=story_id)
        
        logger.info(f"이야기 생성 작업 백그라운드 처리 시작됨 (ID: {story_id})")
        # Fetch the just-created task to return its initial state as StoryGenerationStatus
        # This ensures the returned object is consistent with what get_status would return
        # (though details will be minimal initially)
        created_task_data = get_task_from_db(story_id)
        if not created_task_data:
             # This should ideally not happen if upsert_task succeeded
            logger.error(f"Failed to retrieve task {story_id} immediately after creation for response.")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Task created but could not be retrieved.")
        return StoryGenerationStatus(**created_task_data)

    except Exception as e:
        logger.error(f"이야기 생성 작업 등록 또는 백그라운드 시작 실패 (ID: {story_id}): {e}", exc_info=True)
        # No partial record to clean up from DB here as upsert_task is one operation
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
                            detail=f"Failed to register or start story generation task: {str(e)}")

@router.get("/{story_id}/status", 
            response_model=StoryGenerationStatus,
            summary="이야기 생성 상태 조회")
async def get_story_status(story_id: str):
    logger.debug(f"이야기 상태 조회 요청: {story_id}")
    task_data = get_task_from_db(story_id)
    if not task_data:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Story ID {story_id} not found.")
    return StoryGenerationStatus(**task_data) # Convert dict from DB to Pydantic model

@router.get("/{story_id}/content", 
            response_model=GeneratedStoryResponse, 
            summary="생성된 이야기 내용 조회")
async def get_generated_story_content(story_id: str):
    logger.debug(f"생성된 이야기 내용 조회 요청: {story_id}")
    task_data = get_task_from_db(story_id)

    if not task_data:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Story ID {story_id} not found.")

    # Convert to Pydantic model for easier field access and validation
    status_info = StoryGenerationStatus(**task_data) 

    if status_info.status != "completed":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, 
                            detail=f"Story generation for ID {story_id} is not yet completed. Current status: {status_info.status}")
    
    if not status_info.details:
        logger.error(f"완료된 이야기 {story_id}에 대한 상세 정보가 없습니다.")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Completed story data is missing details.")

    try:
        # Details from DB (already a dict) contains full_text_content, illustrations, voice_data
        full_text_content = status_info.details.get("full_text_content", {})
        title = full_text_content.get("title", "제목 미정")
        content = full_text_content
        
        illustrations_payload = status_info.details.get("illustrations", []) 
        audio_files_payload = status_info.details.get("voice_data", [])

        return GeneratedStoryResponse(
            story_id=story_id,
            title=title,
            content=content,
            illustrations=illustrations_payload,
            audio_files=audio_files_payload
        )
    except Exception as e:
        logger.error(f"생성된 이야기 내용 조회 중 오류 (ID: {story_id}): {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
                            detail=f"Error retrieving story content: {str(e)}")