"""
Router for Story Management API Endpoints

Handles creating, retrieving, updating, and deleting stories,
and managing the story generation workflow via ChatBotB.
"""
import time # 시간 관련 모듈
from fastapi import APIRouter, HTTPException, status, Body, BackgroundTasks
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
import asyncio # 비동기 작업을 위한 모듈

from shared.utils.logging_utils import get_module_logger
# ChatBotB 관련 함수
from chatbot.models.chat_bot_b.chat_bot_b import ChatBotB

# DB 관련 함수
from chatbot.db import ( 
    upsert_task,
    get_task as get_task_from_db,
    update_task_status as update_task_status_in_db
)

logger = get_module_logger("story_router") # 로깅 설정
router = APIRouter() # APIRouter 설정

# --- Pydantic Models for Story API ---
class StoryOutlineRequest(BaseModel):
    child_name: str = Field(..., description="아이 이름")
    age: int = Field(..., ge=3, le=12, description="아이 나이 (3-12세)")
    interests: Optional[List[str]] = Field(None, description="아이 관심사 목록")
    story_summary: str = Field(..., description="ChatBot A 또는 사용자가 제공한 이야기 줄거리 요약")
    initial_tags: Optional[List[str]] = Field(None, description="초기 태그")

class StoryGenerationStatus(BaseModel):
    story_id: str
    status: str # 상태 (예: "pending", "generating_text", "generating_images", "generating_voice", "completed", "failed")
    message: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    input_outline: Optional[Dict[str, Any]] = None # 원본 입력 개요 저장

class GeneratedStoryResponse(BaseModel):
    story_id: str
    title: Optional[str]
    content: Dict[str, Any] # 전체 이야기 내용 (장면, 캐릭터 등)
    illustrations: Optional[List[Dict[str, str]]] # 장면 ID와 이미지 URL 리스트
    audio_files: Optional[List[Dict[str, str]]]  # 음성 파일 리스트 (예: {narration/character: ..., audio_url: ...})

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

    # task_data를 StoryGenerationStatus Pydantic 모델로 변환하여 일관성을 유지합니다.
    # 또는 task_data 딕셔너리의 필드를 직접 사용합니다. 상태 업데이트의 경우 DB 함수를 직접 호출합니다.
    current_status_str = task_data.get("status")
    current_details = task_data.get("details", {}) # 상세 정보 확인
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
        # ChatBotB 인스턴스 생성 (환경변수에서 설정 읽기)
        import os
        chatbot_b = ChatBotB(
            vector_db_path=os.getenv("CHROMA_DB_PATH", "/app/chatbot/data/vector_db"),
            collection_name="fairy_tales",
            use_enhanced_generators=True,
            enable_performance_tracking=True
        )
        
        # 대상 연령 설정
        chatbot_b.set_target_age(input_outline_data.get("age"))
        
        # 스토리 개요 설정 (올바른 키 구조 사용)
        story_outline_for_b = {
            "plot_summary": input_outline_data.get("story_summary"),  # plot_summary 키 사용
            "tags": input_outline_data.get("initial_tags", []),
            "child_name": input_outline_data.get("child_name", "친구"),
            "age_group": input_outline_data.get("age"),
            "interests": input_outline_data.get("interests", [])
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
        
        detailed_story_content = await chatbot_b.generate_detailed_story()
        if not detailed_story_content:
            update_task_status_in_db(story_id, "failed_text_generation", "백그라운드: 상세 이야기 텍스트 생성 실패")
            logger.error(f"백그라운드 작업 [{story_id}]: 상세 텍스트 생성 실패")
            return
        logger.info(f"백그라운드 작업 [{story_id}]: 상세 텍스트 생성 완료")
        # current_details는 task_data에서 이미 딕셔너리이거나 {}로 초기화되어 있습니다.
        current_details["text_content_preview"] = str(detailed_story_content)[:200]
        current_details["full_text_content"] = detailed_story_content
        update_task_status_in_db(story_id, "generating_text", "텍스트 생성 완료, 삽화 생성 준비 중...", details_update=current_details)

        update_task_status_in_db(story_id, "generating_images", "백그라운드: 삽화 생성 중...", details_update=current_details)
        logger.info(f"백그라운드 작업 [{story_id}]: 삽화 생성 시작")
        # 이미지 생성 (ChatBotB의 image_generator 사용)
        try:
            if hasattr(chatbot_b, 'image_generator') and chatbot_b.image_generator:
                image_input = {
                    "story_data": detailed_story_content,
                    "story_id": story_id
                }
                illustrations_result = await chatbot_b.image_generator.generate(image_input)
                logger.info(f"백그라운드 작업 [{story_id}]: 삽화 생성 완료. 결과: {illustrations_result}")
                current_details["illustrations"] = illustrations_result
            else:
                logger.warning(f"백그라운드 작업 [{story_id}]: 이미지 생성기를 찾을 수 없음. 건너뛰기.")
                current_details["illustration_status"] = "generator_not_found"
        except Exception as e:
            logger.warning(f"백그라운드 작업 [{story_id}]: 삽화 생성 실패: {e}. 계속 진행.")
            current_details["illustration_status"] = "failed_or_empty"
        update_task_status_in_db(story_id, "generating_images", "삽화 생성 완료, 음성 합성 준비 중...", details_update=current_details)
        
        update_task_status_in_db(story_id, "generating_voice", "백그라운드: 음성 합성 중...", details_update=current_details)
        logger.info(f"백그라운드 작업 [{story_id}]: 음성 합성 시작")
        # 음성 생성 (ChatBotB의 voice_generator 사용)
        try:
            if hasattr(chatbot_b, 'voice_generator') and chatbot_b.voice_generator:
                voice_input = {
                    "story_data": detailed_story_content,
                    "story_id": story_id
                }
                voice_data_result = await chatbot_b.voice_generator.generate(voice_input)
                logger.info(f"백그라운드 작업 [{story_id}]: 음성 합성 완료. 결과: {voice_data_result}")
                current_details["voice_data"] = voice_data_result
            else:
                logger.warning(f"백그라운드 작업 [{story_id}]: 음성 생성기를 찾을 수 없음. 건너뛰기.")
                current_details["voice_status"] = "generator_not_found"
        except Exception as e:
            logger.warning(f"백그라운드 작업 [{story_id}]: 음성 합성 실패: {e}. 계속 진행.")
            current_details["voice_status"] = "failed_or_empty"
        
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
        existing_status = StoryGenerationStatus(**existing_task_data) # 딕셔너리를 Pydantic 모델로 변환
        if existing_status.status not in ["failed", "failed_text_generation", "failed_image_generation", "failed_voice_generation", "failed_missing_outline", "failed_init_chatbot_b", "failed_runtime_error"]:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, 
                                detail=f"Story generation for ID {story_id} is already active or completed: {existing_status.status}")
        logger.info(f"기존 실패한 작업 (ID: {story_id})에 대한 생성 재시도.")

    try:
        input_outline_dict = outline_request.model_dump()
        # 초기 upsert (input_outline, 상세 정보는 백그라운드 작업에서 채워짐)
        upsert_task(
            story_id=story_id,
            status="queued_for_generation",
            message="이야기 생성 작업이 대기열에 추가되어 백그라운드 처리를 시작합니다.",
            input_outline=input_outline_dict,
            details={}
        )
        
        background_tasks.add_task(process_story_generation_task_background, story_id=story_id)
        
        logger.info(f"이야기 생성 작업 백그라운드 처리 시작됨 (ID: {story_id})")
        # 방금 생성된 작업을 가져와서 StoryGenerationStatus 형태로 반환
        # 이는 get_status가 반환하는 객체와 일관성을 유지합니다.
        # (상세 정보는 초기에 최소한으로 반환됩니다.)
        created_task_data = get_task_from_db(story_id)
        if not created_task_data:
            # 이는 upsert_task가 성공했다면 발생해서는 안 됩니다.
            logger.error(f"Failed to retrieve task {story_id} immediately after creation for response.")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Task created but could not be retrieved.")
        return StoryGenerationStatus(**created_task_data)

    except Exception as e:
        logger.error(f"이야기 생성 작업 등록 또는 백그라운드 시작 실패 (ID: {story_id}): {e}", exc_info=True)
    
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
    return StoryGenerationStatus(**task_data) # 딕셔너리를 Pydantic 모델로 변환

@router.get("/{story_id}/content", 
            response_model=GeneratedStoryResponse, 
            summary="생성된 이야기 내용 조회")
async def get_generated_story_content(story_id: str):
    logger.debug(f"생성된 이야기 내용 조회 요청: {story_id}")
    task_data = get_task_from_db(story_id)

    if not task_data:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Story ID {story_id} not found.")

    # Pydantic 모델로 변환하여 필드 접근과 검증을 용이하게 합니다.
    status_info = StoryGenerationStatus(**task_data) 

    if status_info.status != "completed":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, 
                            detail=f"Story generation for ID {story_id} is not yet completed. Current status: {status_info.status}")
    
    if not status_info.details:
        logger.error(f"완료된 이야기 {story_id}에 대한 상세 정보가 없습니다.")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Completed story data is missing details.")

    try:
        # DB에서 가져온 상세 정보 (이미 딕셔너리)에는 full_text_content, illustrations, voice_data가 포함됩니다.
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