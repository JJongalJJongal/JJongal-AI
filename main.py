# 진입점
import uvicorn
from app import app


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True, # 개발 모드 (실제 프로덕션 환경에서는 False)
        log_level="info", # 로깅 레벨 (debug, info, warning, error, critical)
    )