"""
CCB_AI 통합 FastAPI 애플리케이션

WebSocket 음성 인터페이스와 스토리 생성 API를 통합한 메인 서버입니다.
"""
import asyncio
import os
import sys
from contextlib import asynccontextmanager
from typing import Optional
from datetime import datetime
from fastapi import FastAPI, WebSocket, HTTPException, Response, Query, status, Depends, Request
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.websockets import WebSocketDisconnect, WebSocketState
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

# 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)
load_dotenv(os.path.join(project_root, '.env'))

# 로깅
from shared.utils.logging_utils import get_module_logger

# Voice WebSocket 컴포넌트
from chatbot.models.voice_ws.core.connection_engine import ConnectionEngine
from chatbot.models.voice_ws.processors.auth_processor import AuthProcessor
from chatbot.models.voice_ws.processors.audio_processor import AudioProcessor 
from chatbot.models.voice_ws.handlers.audio_handler import handle_audio_websocket
from chatbot.models.voice_ws.handlers.story_handler import handle_story_generation_websocket
from chatbot.data.vector_db.core import VectorDB

# Integration API 컴포넌트
from chatbot.workflow.orchestrator import WorkflowOrchestrator
from chatbot.workflow.story_schema import ChildProfile, AgeGroup

# Integration API 모델들
from chatbot.workflow.integration_api import (
    StoryCreationRequest, StoryResponse, StandardResponse, HealthResponse,
    verify_auth
)

logger = get_module_logger(__name__)
logger.info("=== CHATBOT.APP.PY Module Loaded ===")

# 전역 컴포넌트
connection_engine = ConnectionEngine()
auth_processor = AuthProcessor()
audio_processor = AudioProcessor()
orchestrator = None

@asynccontextmanager
async def lifespan_manager(app: FastAPI):
    """서비스 생명주기 관리"""
    global orchestrator
    
    # 시작 시 초기화
    logger.info("꼬꼬북 AI 시스템 시작 중...")
    logger.info(f"작업 디렉토리: {os.getcwd()}")
    logger.info(f"Python 버전: {sys.version.split()[0]}")
    
    # 필수 디렉토리 생성
    try:
        ensure_required_directories()
        logger.info("필수 디렉토리 확인/생성 완료")
    except Exception as e:
        logger.error(f"디렉토리 생성 중 오류: {e}")
    
    # 파일 권한 설정
    try:
        from shared.utils.file_permissions import ensure_readable_output
        if ensure_readable_output():
            logger.info("출력 폴더 권한 설정 완료")
        else:
            logger.warning("출력 폴더 권한 설정 실패")
    except Exception as e:
        logger.error(f"파일 권한 설정 중 오류: {e}")
    
    # VectorDB 사전 로드
    logger.info("설치된 패키지 확인...")
    try:
        app.state.vector_db = VectorDB()
        logger.info("VectorDB 사전 로드 완료")
    except Exception as e:
        logger.error(f"VectorDB 사전 로드 실패: {e}")
        app.state.vector_db = None
    
    # 워크플로우 시스템 초기화
    try:
        logger.info("워크플로우 시스템 초기화 중...")
        
        # 출력 디렉토리 설정
        output_dir = os.getenv("MULTIMEDIA_OUTPUT_DIR", "/app/output")
        
        # WorkflowOrchestrator 초기화 (내부에서 StateManager, PipelineManager, MultimediaCoordinator 생성)
        orchestrator = WorkflowOrchestrator(
            output_dir=output_dir,
            enable_multimedia=os.getenv("ENABLE_MULTIMEDIA", "true").lower() == "true"
        )
        
        logger.info("워크플로우 시스템 초기화 완료")
        
    except Exception as e:
        logger.error(f"워크플로우 시스템 초기화 실패: {e}")
        orchestrator = None
    
    logger.info("FastAPI 서버 시작 중... (포트: 8000)")
    
    # WebSocket 정리 태스크 시작
    asyncio.create_task(connection_engine.cleanup_inactive_clients())
    
    yield
    
    # 종료 시 정리
    logger.info("꼬꼬북 AI 시스템 종료 중...")
    
    if orchestrator:
        # 활성 스토리 정리
        active_stories = orchestrator.get_active_stories()
        if active_stories:
            logger.info(f"활성 스토리 정리 중: {len(active_stories)}개")
            for story_id in active_stories:
                try:
                    await orchestrator.cancel_story(story_id)
                except:
                    pass
    
    # WebSocket 연결 정리
    try:
        await connection_engine.disconnect_all()
        logger.info("WebSocket 연결 정리 완료")
    except:
        pass
    
    logger.info("꼬꼬북 AI 시스템 종료 완료")

def ensure_required_directories():
    """도커 환경에서 필요한 디렉토리들을 확인하고 생성"""
    base_output_dir = os.getenv("MULTIMEDIA_OUTPUT_DIR", "/app/output")
    
    # 통일된 벡터DB 경로 설정
    vector_db_base = os.getenv("CHROMA_DB_PATH", "/app/chatbot/data/vector_db")
    
    required_directories = [
        base_output_dir,                                                  # /app/output
        os.path.join(base_output_dir, "workflow_states"),                # workflow_states 
        os.path.join(base_output_dir, "metadata"),                       # metadata
        os.path.join(base_output_dir, "stories"),                        # stories
        os.path.join(base_output_dir, "temp"),                           # temp
        os.path.join(base_output_dir, "temp", "images"),                 # temp/images
        os.path.join(base_output_dir, "temp", "audio"),                  # temp/audio
        os.path.join(base_output_dir, "temp", "voice_samples"),          # temp/voice_samples
        os.path.join(base_output_dir, "conversations"),                  # conversations
        "/app/logs",                                                     # logs (절대 경로)
        vector_db_base,                                                  # 벡터DB 기본 경로
        os.path.join(vector_db_base, "main"),                            # vector_db/main
        os.path.join(vector_db_base, "detailed"),                        # vector_db/detailed  
        os.path.join(vector_db_base, "summary"),                         # vector_db/summary
    ]
    
    created_count = 0
    for directory in required_directories:
        try:
            if not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
                logger.info(f"디렉토리 생성: {directory}")
                created_count += 1
            else:
                logger.debug(f"디렉토리 확인: {directory}")
        except PermissionError as e:
            logger.error(f"디렉토리 생성 권한 오류: {directory} - {e}")
        except OSError as e:
            logger.error(f"디렉토리 생성 실패: {directory} - {e}")
        except Exception as e:
            logger.error(f"예상치 못한 디렉토리 생성 오류: {directory} - {e}")
    
    if created_count > 0:
        logger.info(f"총 {created_count}개의 새로운 디렉토리가 생성되었습니다")
    else:
        logger.info("모든 필수 디렉토리가 이미 존재합니다")
    
    # VectorDB 경로 로깅
    logger.info(f"벡터DB 기본 경로: {vector_db_base}")
    logger.info(f"  - Main DB: {os.path.join(vector_db_base, 'main')}")
    logger.info(f"  - Detailed DB: {os.path.join(vector_db_base, 'detailed')}")
    logger.info(f"  - Summary DB: {os.path.join(vector_db_base, 'summary')}")

# FastAPI 애플리케이션 생성
app = FastAPI(
    title="꼬꼬북 AI 통합 서버",
    description="아동 음성 인터페이스 및 동화 생성 통합 API",
    version="1.0.0",
    lifespan=lifespan_manager
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", 
        "http://localhost:8080", 
        "http://52.78.92.115",      # AWS 고정 IP HTTP
        "https://52.78.92.115",     # AWS 고정 IP HTTPS
        "ws://52.78.92.115",        # AWS 고정 IP WebSocket
        "wss://52.78.92.115",       # AWS 고정 IP Secure WebSocket
        "*"                         # 개발 단계에서 모든 origin 허용
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# 정적 파일 서빙 설정 (output 폴더)
try:
    output_dir = "/app/output"
    if os.path.exists(output_dir):
        app.mount("/output", StaticFiles(directory=output_dir), name="output")
        logger.info(f"정적 파일 서빙 활성화: /output -> {output_dir}")
    else:
        logger.warning(f"출력 디렉토리가 존재하지 않음: {output_dir}")
except Exception as e:
    logger.error(f"정적 파일 서빙 설정 실패: {e}")

# 전역 오류 처리
@app.middleware("http")
async def catch_exceptions_middleware(request, call_next):
    """전역 오류 처리 미들웨어"""
    try:
        return await call_next(request)
    except Exception as e:
        logger.error(f"처리되지 않은 서버 오류: {e}")
        return Response(
            content='{"detail": "서버 내부 오류가 발생했습니다"}',
            status_code=500,
            media_type="application/json"
        )

# ===========================================
# WebSocket 엔드포인트
# ===========================================

@app.websocket("/ws/audio")
async def audio_endpoint(
    websocket: WebSocket,
    child_name: str = Query(None),
    age: int = Query(None),
    interests: Optional[str] = Query(None),
):
    """WebSocket 오디오 처리 엔드포인트"""
    logger.info(f"=== 웹소켓 연결 시도 ===")
    logger.info(f"클라이언트 IP: {websocket.client.host if websocket.client else 'Unknown'}")
    logger.info(f"요청 파라미터: child_name={child_name}, age={age}, interests={interests}")
    logger.info(f"Headers: {dict(websocket.headers) if hasattr(websocket, 'headers') else 'None'}")
    
    try:
        # WebSocket 연결 수락
        await websocket.accept()
        logger.info("WebSocket 연결 수락 완료")
        
        # 파라미터 검증
        if not child_name:
            logger.warning("필수 파라미터 누락: child_name")
            await websocket.close(code=1003, reason="Missing child_name parameter")
            return
            
        if not age or not (4 <= age <= 9):
            logger.warning(f"잘못된 age 파라미터: {age}")
            await websocket.close(code=1003, reason="Invalid age parameter (4-9)")
            return
        
        # 인증 확인 (WebSocket 수락 후)
        if not await auth_processor.validate_connection(websocket):
            logger.warning(f"인증 실패로 연결 거부")
            await websocket.close(code=1008, reason="Authentication failed")
            return
            
        logger.info(f"인증 성공, audio_handler로 전달")
        
        # 오디오 핸들러로 전달
        await handle_audio_websocket(
            websocket,
            child_name,
            age,
            interests,
            connection_engine=connection_engine,
            audio_processor=audio_processor
        )
        
    except Exception as e:
        logger.error(f"웹소켓 엔드포인트 오류: {e}")
        try:
            await websocket.close(code=1011, reason="Internal server error")
        except:
            pass

@app.websocket("/ws/story_generation")
async def story_generation_endpoint(
    websocket: WebSocket,
    child_name: str = Query(None),
    age: int = Query(None),
    interests: Optional[str] = Query(None),
    token: Optional[str] = Query(None)
):
    """WebSocket 스토리 생성 엔드포인트"""
    try:
        # WebSocket 연결 수락
        await websocket.accept()
        logger.info("WebSocket 스토리 생성 연결 수락 완료")
        
        # 인증 확인 (WebSocket 수락 후)
        if not await auth_processor.validate_connection(websocket):
            await websocket.close(code=1008, reason="Authentication failed")
            return
            
        await handle_story_generation_websocket(
            websocket,
            child_name,
            age,
            interests,
            token
        )
    except Exception as e:
        logger.error(f"스토리 생성 웹소켓 엔드포인트 오류: {e}")
        try:
            await websocket.close(code=1011, reason="Internal server error")
        except:
            pass

# ===========================================
# HTTP API 엔드포인트
# ===========================================

@app.get("/health")
async def health_check():
    """서버 상태 확인 엔드포인트"""
    if audio_processor.whisper_model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Whisper 모델이 초기화되지 않았습니다"
        )
    return {
        "status": "online", 
        "whisper_model": "loaded",
        "orchestrator": orchestrator is not None
    }

# 활성 연결 정보 조회 엔드포인트
@app.get("/connections")
async def get_connections_info(): 
    """활성 연결 정보 조회 엔드포인트 (ConnectionEngine 사용)"""
    return {
        "connections": connection_engine.get_active_connections_info(),
        "count": connection_engine.get_client_count()
    }

@app.get("/api/test-token")
async def api_test_token():
    """JWT 토큰 테스트 발급 엔드포인트 (AuthProcessor 사용)"""
    token_info = auth_processor.get_test_token()
    if not token_info["token"]:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="토큰 생성 실패"
        )
    return token_info

# 헬퍼 함수들
async def _create_story_with_orchestrator(
    child_profile: ChildProfile,
    conversation_data: Optional[dict],
    story_preferences: Optional[dict]
) -> str:
    """Orchestrator를 통해 이야기 생성하고 실제 story_id 반환"""
    try:
        if not orchestrator:
            raise RuntimeError("오케스트레이터가 초기화되지 않았습니다")
        
        logger.info("이야기 생성 시작")
        logger.info(f"아이 프로필: {child_profile.name}, 나이: {child_profile.age}")
        logger.info(f"관심사: {child_profile.interests}")
        
        # Orchestrator가 story_id를 생성하고 실제 이야기 생성을 백그라운드에서 시작
        logger.info("오케스트레이터 create_story 호출 중...")
        story_schema = await orchestrator.create_story(
            child_profile=child_profile,
            conversation_data=conversation_data,
            story_preferences=story_preferences
        )
        
        logger.info(f"오케스트레이터 create_story 반환됨: {story_schema is not None}")
        if story_schema:
            logger.info(f"스토리 스키마 단계: {story_schema.current_stage}")
            logger.info(f"생성된 스토리 내용 길이: {len(story_schema.generated_story.content) if story_schema.generated_story else 'None'}")
        
        # Orchestrator가 생성한 실제 story_id 반환
        actual_story_id = story_schema.metadata.story_id
        logger.info(f"이야기 생성 완료: {actual_story_id}")
        
        return actual_story_id
        
    except Exception as e:
        logger.error(f"이야기 생성 실패: {e}", exc_info=True)
        raise

def _determine_age_group(age: int) -> AgeGroup:
    """나이에 따른 연령대 결정"""
    if age <= 7:
        return AgeGroup.YOUNG_CHILDREN
    else:
        return AgeGroup.ELEMENTARY

# ===========================================
# 대화 내역 API 엔드포인트
# ===========================================

logger.info("=== 대화 내역 API 엔드포인트 등록 시작 ===")

@app.get("/api/v1/conversations")
async def list_conversations(auth: dict = Depends(verify_auth)):
    """대화 내역 목록 조회"""
    try:
        import glob
        from datetime import datetime
        
        conversations_dir = "/app/output/conversations"
        conversation_files = []
        
        # 대화 파일들 검색 (JSON 파일만)
        pattern = os.path.join(conversations_dir, "**", "*.json")
        files = glob.glob(pattern, recursive=True)
        
        for file_path in files:
            try:
                # 파일 정보 추출
                rel_path = os.path.relpath(file_path, conversations_dir)
                stat = os.stat(file_path)
                
                # 파일명에서 정보 추출 시도
                filename = os.path.basename(file_path)
                parts = filename.replace('.json', '').split('_')
                
                conversation_info = {
                    "file_path": rel_path,
                    "filename": filename,
                    "size": stat.st_size,
                    "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat()
                }
                
                # 파일명에서 추가 정보 추출
                if len(parts) >= 3:
                    conversation_info["child_name"] = parts[0]
                    conversation_info["timestamp"] = f"{parts[1]}_{parts[2]}"
                    if len(parts) >= 4:
                        conversation_info["client_id"] = parts[3]
                
                conversation_files.append(conversation_info)
                
            except Exception as e:
                logger.warning(f"파일 정보 추출 실패: {file_path} - {e}")
                continue
        
        # 수정일 기준 내림차순 정렬
        conversation_files.sort(key=lambda x: x["modified_at"], reverse=True)
        
        return StandardResponse(
            success=True,
            message="대화 내역 목록 조회 성공",
            data={
                "conversations": conversation_files,
                "count": len(conversation_files)
            }
        )
        
    except Exception as e:
        logger.error(f"대화 내역 목록 조회 실패: {e}", exc_info=True)
        return StandardResponse(
            success=False,
            message=f"대화 내역 조회 중 오류가 발생했습니다: {str(e)}",
            error_code="CONVERSATION_LIST_FAILED"
        )

@app.get("/api/v1/conversations/{file_path:path}")
async def get_conversation_file(file_path: str, auth: dict = Depends(verify_auth)):
    """특정 대화 내역 파일 조회"""
    try:
        import json
        
        # 보안을 위한 경로 검증
        if ".." in file_path or file_path.startswith("/"):
            raise HTTPException(
                status_code=400, 
                detail="잘못된 파일 경로입니다"
            )
        
        conversations_dir = "/app/output/conversations"
        full_path = os.path.join(conversations_dir, file_path)
        
        # 파일 존재 확인
        if not os.path.exists(full_path):
            raise HTTPException(
                status_code=404, 
                detail="대화 파일을 찾을 수 없습니다"
            )
        
        # JSON 파일만 허용
        if not full_path.endswith('.json'):
            raise HTTPException(
                status_code=400, 
                detail="JSON 파일만 조회할 수 있습니다"
            )
        
        # 파일 읽기
        with open(full_path, 'r', encoding='utf-8') as f:
            conversation_data = json.load(f)
        
        return StandardResponse(
            success=True,
            message="대화 내역 조회 성공",
            data=conversation_data
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"대화 파일 조회 실패: {e}", exc_info=True)
        return StandardResponse(
            success=False,
            message=f"대화 파일 조회 중 오류가 발생했습니다: {str(e)}",
            error_code="CONVERSATION_FILE_FAILED"
        )

# ===========================================
# 임시 파일 API 엔드포인트
# ===========================================

logger.info("=== 임시 파일 API 엔드포인트 등록 시작 ===")

@app.get("/api/v1/temp")
async def list_temp_files(auth: dict = Depends(verify_auth)):
    """임시 파일 목록 조회"""
    try:
        import glob
        from datetime import datetime
        
        temp_dir = "/app/output/temp"
        temp_files = []
        
        # 모든 파일 검색 (재귀적으로)
        pattern = os.path.join(temp_dir, "**", "*")
        all_paths = glob.glob(pattern, recursive=True)
        
        for file_path in all_paths:
            try:
                # 디렉토리는 제외
                if os.path.isdir(file_path):
                    continue
                
                # 숨김 파일 제외 (.DS_Store 등)
                if os.path.basename(file_path).startswith('.'):
                    continue
                
                # 파일 정보 추출
                rel_path = os.path.relpath(file_path, temp_dir)
                stat = os.stat(file_path)
                
                # 파일 타입 및 카테고리 결정
                file_ext = os.path.splitext(file_path)[1].lower()
                if file_ext in ['.mp3', '.wav', '.m4a', '.ogg']:
                    file_type = 'audio'
                elif file_ext in ['.png', '.jpg', '.jpeg', '.gif', '.webp']:
                    file_type = 'image'
                else:
                    file_type = 'other'
                
                # 파일명에서 스토리 ID 추출 시도
                filename = os.path.basename(file_path)
                story_id_match = None
                
                # UUID 형태의 ID 찾기 (8자리-4자리-4자리-4자리-12자리 또는 8자리)
                import re
                uuid_pattern = r'[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}|[0-9a-f]{8}'
                match = re.search(uuid_pattern, filename)
                if match:
                    story_id_match = match.group()
                
                temp_file_info = {
                    "file_path": rel_path,
                    "filename": filename,
                    "size": stat.st_size,
                    "type": file_type,
                    "extension": file_ext,
                    "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat()
                }
                
                if story_id_match:
                    temp_file_info["story_id"] = story_id_match
                
                temp_files.append(temp_file_info)
                
            except Exception as e:
                logger.warning(f"파일 정보 추출 실패: {file_path} - {e}")
                continue
        
        # 수정일 기준 내림차순 정렬
        temp_files.sort(key=lambda x: x["modified_at"], reverse=True)
        
        # 파일 타입별 통계
        stats = {
            'audio': len([f for f in temp_files if f['type'] == 'audio']),
            'image': len([f for f in temp_files if f['type'] == 'image']),
            'other': len([f for f in temp_files if f['type'] == 'other'])
        }
        
        return StandardResponse(
            success=True,
            message="임시 파일 목록 조회 성공",
            data={
                "files": temp_files,
                "count": len(temp_files),
                "stats": stats
            }
        )
        
    except Exception as e:
        logger.error(f"임시 파일 목록 조회 실패: {e}", exc_info=True)
        return StandardResponse(
            success=False,
            message=f"임시 파일 조회 중 오류가 발생했습니다: {str(e)}",
            error_code="TEMP_FILE_LIST_FAILED"
        )

@app.get("/api/v1/temp/{file_path:path}")
async def get_temp_file(file_path: str, auth: dict = Depends(verify_auth)):
    """특정 임시 파일 다운로드"""
    try:
        from fastapi.responses import FileResponse
        
        # 보안을 위한 경로 검증
        if ".." in file_path or file_path.startswith("/"):
            raise HTTPException(
                status_code=400, 
                detail="잘못된 파일 경로입니다"
            )
        
        temp_dir = "/app/output/temp"
        full_path = os.path.join(temp_dir, file_path)
        
        # 파일 존재 확인
        if not os.path.exists(full_path):
            raise HTTPException(
                status_code=404, 
                detail="임시 파일을 찾을 수 없습니다"
            )
        
        # 디렉토리 접근 방지
        if os.path.isdir(full_path):
            raise HTTPException(
                status_code=400, 
                detail="디렉토리는 다운로드할 수 없습니다"
            )
        
        # 파일 확장자 검증 (허용된 파일 타입만)
        allowed_extensions = ['.mp3', '.wav', '.m4a', '.ogg', '.png', '.jpg', '.jpeg', '.gif', '.webp', '.json', '.txt']
        file_ext = os.path.splitext(full_path)[1].lower()
        
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail="허용되지 않은 파일 타입입니다"
            )
        
        # 적절한 Content-Type 설정
        media_type = "application/octet-stream"
        if file_ext in ['.mp3', '.wav', '.m4a', '.ogg']:
            media_type = f"audio/{file_ext[1:]}"
        elif file_ext in ['.png', '.jpg', '.jpeg', '.gif', '.webp']:
            media_type = f"image/{file_ext[1:]}" if file_ext != '.jpg' else "image/jpeg"
        elif file_ext == '.json':
            media_type = "application/json"
        elif file_ext == '.txt':
            media_type = "text/plain"
        
        return FileResponse(
            path=full_path,
            media_type=media_type,
            filename=os.path.basename(full_path)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"임시 파일 다운로드 실패: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"임시 파일 다운로드 중 오류가 발생했습니다: {str(e)}"
        )

@app.get("/api/v1/temp/by-story/{story_id}")
async def get_temp_files_by_story(story_id: str, auth: dict = Depends(verify_auth)):
    """특정 스토리 ID의 임시 파일들 조회"""
    try:
        import glob
        from datetime import datetime
        
        temp_dir = "/app/output/temp"
        story_files = []
        
        # 해당 스토리 ID가 포함된 파일들 검색
        pattern = os.path.join(temp_dir, "**", f"*{story_id}*")
        files = glob.glob(pattern, recursive=True)
        
        for file_path in files:
            try:
                # 디렉토리는 제외
                if os.path.isdir(file_path):
                    continue
                
                # 파일 정보 추출
                rel_path = os.path.relpath(file_path, temp_dir)
                stat = os.stat(file_path)
                
                # 파일 타입 결정
                file_ext = os.path.splitext(file_path)[1].lower()
                if file_ext in ['.mp3', '.wav', '.m4a', '.ogg']:
                    file_type = 'audio'
                elif file_ext in ['.png', '.jpg', '.jpeg', '.gif', '.webp']:
                    file_type = 'image'
                else:
                    file_type = 'other'
                
                story_file_info = {
                    "file_path": rel_path,
                    "filename": os.path.basename(file_path),
                    "size": stat.st_size,
                    "type": file_type,
                    "extension": file_ext,
                    "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "download_url": f"/api/v1/temp/{rel_path}"
                }
                
                story_files.append(story_file_info)
                
            except Exception as e:
                logger.warning(f"파일 정보 추출 실패: {file_path} - {e}")
                continue
        
        # 파일명 기준 정렬
        story_files.sort(key=lambda x: x["filename"])
        
        # 파일 타입별 분류
        files_by_type = {
            'audio': [f for f in story_files if f['type'] == 'audio'],
            'image': [f for f in story_files if f['type'] == 'image'],
            'other': [f for f in story_files if f['type'] == 'other']
        }
        
        return StandardResponse(
            success=True,
            message=f"스토리 {story_id}의 임시 파일 조회 성공",
            data={
                "story_id": story_id,
                "files": story_files,
                "files_by_type": files_by_type,
                "count": len(story_files)
            }
        )
        
    except Exception as e:
        logger.error(f"스토리별 임시 파일 조회 실패: {e}", exc_info=True)
        return StandardResponse(
            success=False,
            message=f"스토리별 임시 파일 조회 중 오류가 발생했습니다: {str(e)}",
            error_code="STORY_TEMP_FILES_FAILED"
        )

# ===========================================
# 시스템 모니터링 및 자동 백업 API
# ===========================================

logger.info("=== 시스템 모니터링 API 엔드포인트 등록 시작 ===")

@app.get("/api/v1/system/disk-usage")
async def get_disk_usage(auth: dict = Depends(verify_auth)):
    """디스크 사용량 확인"""
    try:
        from shared.utils.s3_manager import S3Manager
        import os
        
        # 환경변수에서 S3 설정 가져오기
        bucket_name = os.getenv("AWS_S3_BUCKET_NAME")
        if not bucket_name:
            return StandardResponse(
                success=False,
                message="S3 버킷 이름이 설정되지 않았습니다",
                error_code="S3_BUCKET_NOT_CONFIGURED"
            )
        
        # S3Manager 초기화
        s3_manager = S3Manager()
        if not s3_manager.is_healthy():
            return StandardResponse(
                success=False,
                message="S3 연결에 실패했습니다",
                error_code="S3_CONNECTION_FAILED"
            )
        
        # temp 폴더 업로드
        temp_dir = "/app/output/temp"
        result = s3_manager.upload_temp_files_to_s3(temp_dir, bucket_name)
        
        if result["success"]:
            return StandardResponse(
                success=True,
                message="temp 폴더 업로드 완료",
                data=result
            )
        else:
            return StandardResponse(
                success=False,
                message=f"업로드 실패: {result.get('error', 'Unknown error')}",
                error_code="S3_UPLOAD_FAILED"
            )
            
    except Exception as e:
        logger.error(f"S3 업로드 실패: {e}", exc_info=True)
        return StandardResponse(
            success=False,
            message=f"S3 업로드 중 오류가 발생했습니다: {str(e)}",
            error_code="S3_UPLOAD_ERROR"
        )

@app.post("/api/v1/s3/upload-story/{story_id}")
async def upload_story_to_s3(story_id: str, auth: dict = Depends(verify_auth)):
    """특정 스토리의 파일들을 S3에 업로드"""
    try:
        from shared.utils.s3_manager import S3Manager
        import os
        
        # 환경변수에서 S3 설정 가져오기
        bucket_name = os.getenv("AWS_S3_BUCKET_NAME")
        if not bucket_name:
            return StandardResponse(
                success=False,
                message="S3 버킷 이름이 설정되지 않았습니다",
                error_code="S3_BUCKET_NOT_CONFIGURED"
            )
        
        # S3Manager 초기화
        s3_manager = S3Manager()
        if not s3_manager.is_healthy():
            return StandardResponse(
                success=False,
                message="S3 연결에 실패했습니다",
                error_code="S3_CONNECTION_FAILED"
            )
        
        # 특정 스토리 파일들 업로드
        temp_dir = "/app/output/temp"
        result = s3_manager.upload_story_files_to_s3(temp_dir, bucket_name, story_id)
        
        if result["success"]:
            return StandardResponse(
                success=True,
                message=f"스토리 {story_id} 파일 업로드 완료",
                data=result
            )
        else:
            return StandardResponse(
                success=False,
                message=f"업로드 실패: {result.get('error', 'Unknown error')}",
                error_code="S3_UPLOAD_FAILED"
            )
            
    except Exception as e:
        logger.error(f"스토리 S3 업로드 실패: {e}", exc_info=True)
        return StandardResponse(
            success=False,
            message=f"스토리 S3 업로드 중 오류가 발생했습니다: {str(e)}",
            error_code="S3_STORY_UPLOAD_ERROR"
        )

@app.post("/api/v1/system/auto-backup")
async def auto_backup_to_s3(
    force: bool = Query(False, description="강제 백업 실행 여부"),
    auth: dict = Depends(verify_auth)
):
    """디스크 사용량 기반 자동 S3 백업"""
    try:
        import shutil
        import os
        from shared.utils.s3_manager import S3Manager
        
        # 디스크 사용량 확인
        total, used, free = shutil.disk_usage("/")
        usage_percent = (used / total) * 100
        
        # 백업 실행 조건 확인
        should_backup = force or usage_percent > 85
        
        if not should_backup:
            return StandardResponse(
                success=True,
                message=f"백업 불필요 (디스크 사용량: {usage_percent:.1f}%)",
                data={
                    "backup_executed": False,
                    "disk_usage_percent": round(usage_percent, 2),
                    "threshold_percent": 85
                }
            )
        
        # S3 설정 확인
        bucket_name = os.getenv("AWS_S3_BUCKET_NAME")
        if not bucket_name:
            return StandardResponse(
                success=False,
                message="S3 버킷이 설정되지 않아 백업할 수 없습니다",
                error_code="S3_BUCKET_NOT_CONFIGURED"
            )
        
        # S3Manager 초기화
        s3_manager = S3Manager()
        if not s3_manager.is_healthy():
            return StandardResponse(
                success=False,
                message="S3 연결 실패로 백업할 수 없습니다",
                error_code="S3_CONNECTION_FAILED"
            )
        
        # 백업 실행 (업로드 후 로컬 파일 삭제)
        temp_dir = "/app/output/temp"
        result = s3_manager.sync_temp_to_s3(temp_dir, bucket_name, delete_after_upload=True)
        
        if result["success"]:
            # 백업 후 디스크 사용량 재확인
            total_after, used_after, free_after = shutil.disk_usage("/")
            usage_percent_after = (used_after / total_after) * 100
            
            freed_space = used - used_after
            
            def format_bytes(bytes_size):
                for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
                    if bytes_size < 1024.0:
                        return f"{bytes_size:.2f} {unit}"
                    bytes_size /= 1024.0
                return f"{bytes_size:.2f} PB"
            
            backup_result = {
                "backup_executed": True,
                "uploaded_files": len(result.get("uploaded_files", [])),
                "deleted_files": len(result.get("deleted_files", [])),
                "freed_space": format_bytes(freed_space),
                "disk_usage_before": round(usage_percent, 2),
                "disk_usage_after": round(usage_percent_after, 2),
                "space_saved_percent": round(usage_percent - usage_percent_after, 2)
            }
            
            logger.info(f"자동 백업 완료: {backup_result['uploaded_files']}개 파일 업로드, "
                       f"{backup_result['freed_space']} 공간 확보")
            
            return StandardResponse(
                success=True,
                message=f"자동 백업 완료 - {backup_result['freed_space']} 공간 확보됨",
                data=backup_result
            )
        else:
            return StandardResponse(
                success=False,
                message=f"백업 실패: {result.get('error', 'Unknown error')}",
                error_code="AUTO_BACKUP_FAILED"
            )
            
    except Exception as e:
        logger.error(f"자동 백업 실패: {e}", exc_info=True)
        return StandardResponse(
            success=False,
            message=f"자동 백업 중 오류가 발생했습니다: {str(e)}",
            error_code="AUTO_BACKUP_ERROR"
        )

@app.post("/api/v1/s3/restore-file")
async def restore_file_from_s3(
    s3_key: str = Query(..., description="S3에서 복원할 파일의 키"),
    local_path: Optional[str] = Query(None, description="로컬 저장 경로 (기본: temp 폴더)"),
    auth: dict = Depends(verify_auth)
):
    """S3에서 파일을 로컬로 복원"""
    try:
        from shared.utils.s3_manager import S3Manager
        import os
        
        # S3 설정 확인
        bucket_name = os.getenv("AWS_S3_BUCKET_NAME")
        if not bucket_name:
            return StandardResponse(
                success=False,
                message="S3 버킷이 설정되지 않았습니다",
                error_code="S3_BUCKET_NOT_CONFIGURED"
            )
        
        # S3Manager 초기화
        s3_manager = S3Manager()
        if not s3_manager.is_healthy():
            return StandardResponse(
                success=False,
                message="S3 연결에 실패했습니다",
                error_code="S3_CONNECTION_FAILED"
            )
        
        # 로컬 저장 경로 결정
        if not local_path:
            filename = os.path.basename(s3_key)
            local_path = f"/app/output/temp/{filename}"
        
        # 디렉토리 생성
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        # S3에서 파일 다운로드
        try:
            s3_manager.s3_client.download_file(bucket_name, s3_key, local_path)
            
            file_size = os.path.getsize(local_path)
            
            def format_bytes(bytes_size):
                for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
                    if bytes_size < 1024.0:
                        return f"{bytes_size:.2f} {unit}"
                    bytes_size /= 1024.0
                return f"{bytes_size:.2f} PB"
            
            return StandardResponse(
                success=True,
                message=f"파일 복원 완료: {os.path.basename(local_path)}",
                data={
                    "s3_key": s3_key,
                    "local_path": local_path,
                    "file_size": format_bytes(file_size),
                    "file_size_bytes": file_size
                }
            )
            
        except Exception as e:
            logger.error(f"S3 파일 다운로드 실패: {e}")
            return StandardResponse(
                success=False,
                message=f"S3에서 파일 다운로드 실패: {str(e)}",
                error_code="S3_DOWNLOAD_FAILED"
            )
            
    except Exception as e:
        logger.error(f"파일 복원 실패: {e}", exc_info=True)
        return StandardResponse(
            success=False,
            message=f"파일 복원 중 오류가 발생했습니다: {str(e)}",
            error_code="FILE_RESTORE_ERROR"
        )

# ===========================================
# 스토리 생성 API 엔드포인트
# ===========================================

logger.info("=== 스토리 생성 API 엔드포인트 등록 시작 ===")

@app.post("/api/v1/stories", response_model=StoryResponse)
async def create_story(
    request: Request,
    story_request: StoryCreationRequest, 
    auth: dict = Depends(verify_auth)
):
    """새 이야기 생성"""
    logger.info("CREATE_STORY Function Called")
    logger.info("=== 스토리 생성 API 호출됨 ===")
    logger.info(f"요청 데이터: 아이 이름={story_request.child_profile.name}, 나이={story_request.child_profile.age}")
    
    try:
        logger.info("오케스트레이터 상태 확인 중...")
        if not orchestrator:
            logger.error("오케스트레이터가 None입니다")
            return StoryResponse(
                success=False,
                message="오케스트레이터가 초기화되지 않았습니다",
                error_code="ORCHESTRATOR_NOT_INITIALIZED"
            )
        
        logger.info("오케스트레이터 정상 확인됨. 아이 프로필 변환 중...")
        
        # 아이 프로필 변환
        age_group = _determine_age_group(story_request.child_profile.age)
        child_profile = ChildProfile(
            name=story_request.child_profile.name,
            age=story_request.child_profile.age,
            age_group=age_group,
            interests=story_request.child_profile.interests,
            language_level=story_request.child_profile.language_level,
            special_needs=story_request.child_profile.special_needs
        )
        
        logger.info(f"아이 프로필 변환 완료: {child_profile.name}, 연령대: {age_group}")
        
        # 오케스트레이터에서 story_id 먼저 생성
        logger.info("_create_story_with_orchestrator 호출 중...")
        story_id = await _create_story_with_orchestrator(
            child_profile,
            story_request.conversation_data,
            story_request.story_preferences
        )
        
        logger.info(f"스토리 생성 완료! Story ID: {story_id}")
        
        return StoryResponse(
            success=True,
            story_id=story_id,
            message="이야기 생성이 시작되었습니다",
            data={
                "child_name": child_profile.name,
                "estimated_completion_time": "3-5분"
            }
        )
        
    except Exception as e:
        logger.error(f"이야기 생성 요청 실패: {e}", exc_info=True)
        return StoryResponse(
            success=False,
            message=f"이야기 생성 중 오류가 발생했습니다: {str(e)}",
            error_code="STORY_CREATION_FAILED"
        )

@app.get("/api/v1/stories/{story_id}/status", response_model=StandardResponse)
async def get_story_status(story_id: str, auth: dict = Depends(verify_auth)):
    """이야기 상태 조회"""
    try:
        if not orchestrator:
            return StandardResponse(
                success=False,
                message="오케스트레이터가 초기화되지 않았습니다",
                error_code="ORCHESTRATOR_NOT_INITIALIZED"
            )
        
        # 이야기 상태 조회
        status = await orchestrator.get_story_status(story_id)
        if not status:
            return StandardResponse(
                success=False,
                message="이야기를 찾을 수 없습니다",
                error_code="STORY_NOT_FOUND"
            )
        
        return StandardResponse(
            success=True,
            message="이야기 상태 조회 성공",
            data=status
        )
        
    except Exception as e:
        logger.error(f"이야기 상태 조회 실패: {e}", exc_info=True)
        return StandardResponse(
            success=False,
            message=f"상태 조회 중 오류가 발생했습니다: {str(e)}",
            error_code="STATUS_RETRIEVAL_FAILED"
        )

@app.get("/api/v1/stories/{story_id}", response_model=StandardResponse)
async def get_story(story_id: str, auth: dict = Depends(verify_auth)):
    """이야기 조회"""
    try:
        if not orchestrator:
            return StandardResponse(
                success=False,
                message="오케스트레이터가 초기화되지 않았습니다",
                error_code="ORCHESTRATOR_NOT_INITIALIZED"
            )
        
        # 이야기 상태 로드
        story_schema = await orchestrator.state_manager.load_story_state(story_id)
        if not story_schema:
            return StandardResponse(
                success=False,
                message="이야기를 찾을 수 없습니다",
                error_code="STORY_NOT_FOUND"
            )
        
        return StandardResponse(
            success=True,
            message="이야기 조회 성공",
            data=story_schema.to_dict()
        )
        
    except Exception as e:
        logger.error(f"이야기 조회 실패: {e}", exc_info=True)
        return StandardResponse(
            success=False,
            message=f"이야기 조회 중 오류가 발생했습니다: {str(e)}",
            error_code="STORY_RETRIEVAL_FAILED"
        )

logger.info("=== 인증 API 엔드포인트 등록 시작 ===")

@app.post("/api/v1/auth/token")
async def get_auth_token():
    """JWT 토큰 발급"""
    try:
        token_data = auth_processor.get_test_token()
        return {
            "success": True,
            "message": "토큰 발급 성공",
            "data": token_data
        }
    except Exception as e:
        logger.error(f"토큰 발급 실패: {e}", exc_info=True)
        return {
            "success": False,
            "message": f"토큰 발급 중 오류가 발생했습니다: {str(e)}",
            "error_code": "TOKEN_GENERATION_FAILED"
        }

@app.get("/api/v1/health", response_model=HealthResponse)
async def api_health_check():
    """API 헬스체크"""
    from datetime import datetime
    try:
        active_stories = len(orchestrator.get_active_stories()) if orchestrator else 0
        
        if orchestrator:
            all_stories = await orchestrator.state_manager.list_all_stories()
            total_stories = len(all_stories)
        else:
            total_stories = 0
        
        return HealthResponse(
            status="healthy",
            timestamp=datetime.now().isoformat(),
            version="1.0.0",
            active_stories=active_stories,
            total_stories=total_stories
        )
        
    except Exception as e:
        logger.error(f"헬스체크 실패: {e}", exc_info=True)
        return HealthResponse(
            status="unhealthy",
            timestamp=datetime.now().isoformat(),
            version="1.0.0",
            active_stories=0,
            total_stories=0
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "chatbot.app:app",
        host="0.0.0.0",
        port=8001,
        reload=False,
        log_level="info"
    ) 