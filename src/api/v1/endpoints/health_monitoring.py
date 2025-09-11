"""
AI Model Health Monitoring API Endpoints

Provides real-time monitoring of AI models running in background,
including memory usage, response times, and health status.
"""

from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any, Optional
from datetime import datetime

from ..dependencies import get_jwt_manager
from ..middleware.auth import verify_token
from src.shared.monitoring.model_health_monitor import get_health_monitor
from src.shared.utils.logging import get_module_logger

logger = get_module_logger(__name__)

router = APIRouter(prefix="/monitoring", tags=["Health", "Monitoring"])


@router.get("/models/status")
async def get_models_status(
    token_data: Dict[str, Any] = Depends(verify_token)
) -> Dict[str, Any]:
    """
    Get current status of all AI models
    
    Returns comprehensive health information for all registered models
    including memory usage, response times, and error counts.
    """
    try:
        monitor = get_health_monitor()
        all_status = monitor.get_all_models_status()
        
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "models_count": len(all_status),
            "models": {
                name: {
                    "model_name": metrics.model_name,
                    "is_loaded": metrics.is_loaded,
                    "memory_usage_mb": metrics.memory_usage_mb,
                    "last_used": metrics.last_used.isoformat() if metrics.last_used else None,
                    "total_requests": metrics.total_requests,
                    "error_count": metrics.error_count,
                    "avg_response_time": metrics.avg_response_time,
                    "status": metrics.status
                }
                for name, metrics in all_status.items()
            }
        }
    except Exception as e:
        logger.error(f"Failed to get models status: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve models status")


@router.get("/models/{model_name}/status")
async def get_model_status(
    model_name: str,
    token_data: Dict[str, Any] = Depends(verify_token)
) -> Dict[str, Any]:
    """
    Get detailed status for a specific model
    
    Args:
        model_name: Name of the model to check (e.g., 'chatbot_a', 'voice_processor')
    """
    try:
        monitor = get_health_monitor()
        metrics = monitor.get_model_status(model_name)
        
        if not metrics:
            raise HTTPException(
                status_code=404, 
                detail=f"Model '{model_name}' not found or not monitored"
            )
        
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "model": {
                "model_name": metrics.model_name,
                "is_loaded": metrics.is_loaded,
                "memory_usage_mb": metrics.memory_usage_mb,
                "last_used": metrics.last_used.isoformat() if metrics.last_used else None,
                "total_requests": metrics.total_requests,
                "error_count": metrics.error_count,
                "avg_response_time": metrics.avg_response_time,
                "status": metrics.status,
                "health_check": "passed" if metrics.status in ["healthy", "idle"] else "failed"
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get status for model {model_name}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve model status")


@router.get("/health/summary")
async def get_health_summary(
    token_data: Dict[str, Any] = Depends(verify_token)
) -> Dict[str, Any]:
    """
    Get comprehensive health summary of the entire AI system
    
    Includes overall system health, memory usage, and recommendations
    for optimization.
    """
    try:
        monitor = get_health_monitor()
        summary = monitor.get_health_summary()
        
        return {
            "status": "success",
            "summary": summary
        }
    except Exception as e:
        logger.error(f"Failed to get health summary: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve health summary")


@router.post("/models/{model_name}/usage")
async def record_model_usage(
    model_name: str,
    usage_data: Dict[str, Any],
    token_data: Dict[str, Any] = Depends(verify_token)
) -> Dict[str, Any]:
    """
    Record model usage for monitoring
    
    Args:
        model_name: Name of the model
        usage_data: Usage information including response_time and error status
    """
    try:
        monitor = get_health_monitor()
        
        response_time = usage_data.get("response_time")
        error = usage_data.get("error", False)
        
        monitor.record_model_usage(
            model_name=model_name,
            response_time=response_time,
            error=error
        )
        
        return {
            "status": "success",
            "message": f"Usage recorded for model {model_name}",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to record usage for model {model_name}: {e}")
        raise HTTPException(status_code=500, detail="Failed to record model usage")


@router.get("/background/persistence")
async def check_background_persistence(
    token_data: Dict[str, Any] = Depends(verify_token)
) -> Dict[str, Any]:
    """
    Check if AI models are properly persisting in background
    
    This endpoint verifies that AI models remain loaded and active
    even when no active WebSocket connections exist.
    """
    try:
        from src.api.v1.dependencies import ServiceFactory
        
        persistence_status = {
            "timestamp": datetime.now().isoformat(),
            "background_persistence": True,
            "services": {}
        }
        
        # Check each registered service
        services_to_check = ["chatbot_a", "voice_processor", "collaborator", "connection_manager"]
        
        for service_name in services_to_check:
            try:
                # Get instance - this will return existing instance if it's already loaded
                instance = await ServiceFactory.get_instance(service_name)
                
                service_info = {
                    "loaded": instance is not None,
                    "instance_id": id(instance) if instance else None,
                    "memory_address": hex(id(instance)) if instance else None
                }
                
                # Try to get more detailed info if available
                if hasattr(instance, '__class__'):
                    service_info["class_name"] = instance.__class__.__name__
                
                if hasattr(instance, 'active_sessions'):
                    service_info["active_sessions"] = len(getattr(instance, 'active_sessions', {}))
                
                persistence_status["services"][service_name] = service_info
                
            except Exception as e:
                persistence_status["services"][service_name] = {
                    "loaded": False,
                    "error": str(e)
                }
                persistence_status["background_persistence"] = False
        
        # Check if ServiceFactory._instances contains our services
        try:
            factory_instances = len(ServiceFactory._instances)
            persistence_status["factory_cache_size"] = factory_instances
            persistence_status["cached_services"] = list(ServiceFactory._instances.keys())
        except:
            pass
        
        return {
            "status": "success",
            "persistence_check": persistence_status
        }
    except Exception as e:
        logger.error(f"Failed to check background persistence: {e}")
        raise HTTPException(status_code=500, detail="Failed to check background persistence")


@router.get("/system/resources")
async def get_system_resources(
    token_data: Dict[str, Any] = Depends(verify_token)
) -> Dict[str, Any]:
    """
    Get current system resource usage
    
    Returns information about memory usage, CPU usage, and other
    system resources related to AI model operations.
    """
    try:
        import psutil
        import gc
        
        # Get process information
        process = psutil.Process()
        memory_info = process.memory_info()
        
        system_info = {
            "timestamp": datetime.now().isoformat(),
            "process": {
                "memory_rss_mb": round(memory_info.rss / (1024 * 1024), 2),
                "memory_vms_mb": round(memory_info.vms / (1024 * 1024), 2),
                "memory_percent": round(process.memory_percent(), 2),
                "cpu_percent": round(process.cpu_percent(), 2),
                "num_threads": process.num_threads()
            },
            "python": {
                "garbage_objects": len(gc.get_objects()),
                "garbage_counts": gc.get_count()
            }
        }
        
        # Add system-wide information if available
        try:
            system_info["system"] = {
                "memory_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
                "memory_available_gb": round(psutil.virtual_memory().available / (1024**3), 2),
                "memory_percent": psutil.virtual_memory().percent,
                "cpu_percent": psutil.cpu_percent(interval=1)
            }
        except:
            pass
        
        return {
            "status": "success",
            "resources": system_info
        }
    except Exception as e:
        logger.error(f"Failed to get system resources: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve system resources")