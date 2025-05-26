"""
음성 WebSocket 서버 모듈 V2 (모듈화된 구조)

이 모듈은 다음의 주요 구성 요소로 재구성되었습니다:
- app.py: FastAPI 앱 설정 및 라우팅 (업데이트됨)
- voice_ws_server.py: 서버 실행 진입점 (유지)
- core/: 핵심 엔진 (연결, WebSocket 로직, 세션 관리)
- processors/: 데이터 처리 (오디오, 메시지, 인증)
- handlers/: 요청 핸들러 (엔드포인트별 로직)
- utils/: WebSocket 관련 유틸리티
"""

# FastAPI 앱 (라우팅 및 설정)
from .app import app

# 핵심 엔진
from .core.connection_engine import ConnectionEngine
from .core.websocket_engine import WebSocketEngine 
from .core.session_manager import SessionManager  

# 프로세서
from .processors.audio_processor import AudioProcessor
from .processors.auth_processor import AuthProcessor
from .processors.message_processor import MessageProcessor

# 핸들러 (주로 app.py 내부에서 사용)
from .handlers.audio_handler import handle_audio_websocket
from .handlers.story_handler import handle_story_generation_websocket

# 유틸리티
from .utils import setup_logging, retry_operation, cleanup_temp_files, save_conversation

# 서버 실행 함수
from .voice_ws_server import run_server

__all__ = [
    'app',
    'run_server',
    'ConnectionEngine',
    'WebSocketEngine',
    'SessionManager',
    'AudioProcessor',
    'AuthProcessor',
    'MessageProcessor',
    'setup_logging',
    'retry_operation',
    'cleanup_temp_files',
    'save_conversation'
]

logger = setup_logging() # 최상위 로거 설정
logger.info("음성 WebSocket 모듈 로드 완료") 