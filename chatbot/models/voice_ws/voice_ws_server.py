"""
음성 WebSocket 서버 메인 모듈

이 모듈은 꼬꼬북 프로젝트의 음성 WebSocket 서버를 실행하는 진입점입니다.
모듈화된 코드를 사용하며, JWT 인증을 헤더에서 처리합니다.
"""
import os
from shared.utils.logging_utils import get_module_logger
import uvicorn
from dotenv import load_dotenv

# 환경 변수 Load
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
dotenv_path = os.path.join(project_root, '.env')
load_dotenv(dotenv_path=dotenv_path)

logger = get_module_logger(__name__)

# 모듈화된 앱 임포트
from .app import app
from shared.utils.logging_utils import setup_logger

def run_server(host="0.0.0.0", port=8001, log_level="info", reload=False):
    """
    WebSocket 서버를 실행합니다.
    
    Args:
        host (str): 바인딩할 호스트 (기본값: "0.0.0.0")
        port (int): 사용할 포트 (기본값: 8001)
        log_level (str): 로깅 레벨 (기본값: "info")
        reload (bool): 코드 변경 시 자동 리로드 여부 (기본값: False)
    """
    # 로깅 설정
    setup_logger(name="voice_ws")
    
    # 서버 시작 정보 로깅
    logger.info(f"음성 WebSocket 서버 시작: {host}:{port}, Log Level: {log_level}")
    
    # 서버 실행
    uvicorn.run(
        "chatbot.models.voice_ws.app:app",
        host=host,
        port=port,
        log_level=log_level,
        reload=reload
    )

if __name__ == "__main__":
    # 커맨드라인에서 직접 실행 시
    import argparse
    
    parser = argparse.ArgumentParser(description="꼬꼬북 음성 WebSocket 서버")
    parser.add_argument("--host", default="0.0.0.0", help="바인딩할 호스트 (기본값: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8001, help="사용할 포트 (기본값: 8001)")
    parser.add_argument("--log-level", default="info", choices=["debug", "info", "warning", "error", "critical"], 
                        help="로깅 레벨 (기본값: info)")
    parser.add_argument("--reload", action="store_true", help="코드 변경 시 자동 리로드")
    
    args = parser.parse_args()
    
    run_server(
        host=args.host, 
        port=args.port, 
        log_level=args.log_level, 
        reload=args.reload
    )
        
            
