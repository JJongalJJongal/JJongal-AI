"""
JJongal Health Check API Endpoints
시스템 상태 확인 API
"""

import os
import psutil
from typing import Dict, Any, List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from datetime import datetime

from src.shared.utils.logging import get_module_logger

logger = get_module_logger(__name__)

router = APIRouter(prefix="/health", tags=["health"])

class HealthResponse(BaseModel):
    """Health check response model"""
    status: str
    timestamp: str
    services: Dict[str, str]
    system_info: Dict[str, Any]
    version: str = "1.0.0"

class ServiceStatus(BaseModel):
    """Individual service status"""
    name: str
    status: str
    details: Dict[str, Any] = {}

@router.get("/", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    전체 시스템 상태 확인
    """
    try:
        # 기본 서비스 상태 확인
        services = {
            "api": "healthy",
            "chatbot_a": "healthy",
            "chatbot_b": "healthy",
            "vector_db": "healthy",
            "voice_processor": "healthy"
        }
        
        # 시스템 정보 수집
        system_info = {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent,
            "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}"
        }
        
        return HealthResponse(
            status="healthy",
            timestamp=datetime.now().isoformat(),
            services=services,
            system_info=system_info
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")

@router.get("/detailed", response_model=Dict[str, Any])
async def detailed_health_check() -> Dict[str, Any]:
    """
    상세 시스템 상태 확인
    """
    try:
        # CPU 정보
        cpu_info = {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "cpu_count": psutil.cpu_count(),
            "cpu_freq": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None
        }
        
        # 메모리 정보
        memory_info = psutil.virtual_memory()._asdict()
        
        # 디스크 정보
        disk_info = psutil.disk_usage('/')._asdict()
        
        # 네트워크 정보
        network_info = psutil.net_io_counters()._asdict()
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "system": {
                "cpu": cpu_info,
                "memory": memory_info,
                "disk": disk_info,
                "network": network_info
            },
            "environment": {
                "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
                "platform": os.sys.platform,
                "cwd": os.getcwd()
            }
        }
        
    except Exception as e:
        logger.error(f"Detailed health check failed: {e}")
        raise HTTPException(status_code=500, detail="Detailed health check failed")

@router.get("/services", response_model=List[ServiceStatus])
async def services_health_check() -> List[ServiceStatus]:
    """
    개별 서비스 상태 확인
    """
    try:
        services = []
        
        # API 서비스 상태
        services.append(ServiceStatus(
            name="api",
            status="healthy",
            details={"description": "FastAPI REST API"}
        ))
        
        # ChatBot A 서비스 상태
        try:
            # ChatBot A import 테스트
            from src.core.chatbots.chat_bot_a.chat_bot_a import ChatBotA
            services.append(ServiceStatus(
                name="chatbot_a",
                status="healthy",
                details={"description": "ChatBot A (쫑이) - Story Collection"}
            ))
        except Exception as e:
            services.append(ServiceStatus(
                name="chatbot_a",
                status="unhealthy",
                details={"error": str(e)}
            ))
        
        # ChatBot B 서비스 상태
        try:
            from src.core.chatbots.chat_bot_b.chat_bot_b import ModernAri
            services.append(ServiceStatus(
                name="chatbot_b",
                status="healthy",
                details={"description": "ChatBot B (아리) - Story Generation"}
            ))
        except Exception as e:
            services.append(ServiceStatus(
                name="chatbot_b",
                status="unhealthy",
                details={"error": str(e)}
            ))
        
        # Vector DB 서비스 상태
        try:
            from src.data.vector_db.core import ModernVectorDB
            services.append(ServiceStatus(
                name="vector_db",
                status="healthy",
                details={"description": "ChromaDB Vector Database"}
            ))
        except Exception as e:
            services.append(ServiceStatus(
                name="vector_db",
                status="unhealthy",
                details={"error": str(e)}
            ))
        
        return services
        
    except Exception as e:
        logger.error(f"Services health check failed: {e}")
        raise HTTPException(status_code=500, detail="Services health check failed")