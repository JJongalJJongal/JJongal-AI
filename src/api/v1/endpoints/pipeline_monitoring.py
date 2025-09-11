"""
Pipeline Monitoring API Endpoints
Provides real-time monitoring of LangChain pipeline performance
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from src.shared.monitoring.pipeline_monitor import pipeline_monitor
from src.shared.utils.logging import get_module_logger

logger = get_module_logger(__name__)

# Create router
router = APIRouter(
    prefix="/pipeline-monitoring", 
    tags=["Pipeline Monitoring"]
)

# === Response Models ===

class PipelineStageStatus(BaseModel):
    """Pipeline stage status"""
    success: Optional[bool] = Field(None, description="Whether stage completed successfully")
    duration: Optional[float] = Field(None, description="Stage duration in seconds")
    error: Optional[str] = Field(None, description="Error message if failed")

class PipelineStatusResponse(BaseModel):
    """Pipeline status response"""
    pipeline_id: str = Field(..., description="Pipeline identifier")
    status: str = Field(..., description="Pipeline status (running, completed)")
    start_time: float = Field(..., description="Pipeline start timestamp")
    duration: Optional[float] = Field(None, description="Total pipeline duration")
    child_name: str = Field(..., description="Child's name")
    child_age: Optional[int] = Field(None, description="Child's age")
    story_theme: Optional[str] = Field(None, description="Story theme")
    stages: Dict[str, PipelineStageStatus] = Field(..., description="Status of each stage")
    success_rate: float = Field(..., description="Success rate of completed stages")
    total_tokens: int = Field(..., description="Total tokens used")
    total_images: int = Field(..., description="Total images generated")
    total_audio_duration: float = Field(..., description="Total audio duration in seconds")
    total_retries: int = Field(..., description="Total retry attempts")

class StagePerformance(BaseModel):
    """Stage performance statistics"""
    avg_duration: float = Field(..., description="Average duration in seconds")
    min_duration: float = Field(..., description="Minimum duration in seconds")
    max_duration: float = Field(..., description="Maximum duration in seconds")
    count: int = Field(..., description="Number of recorded executions")

class PerformanceSummaryResponse(BaseModel):
    """Overall performance summary"""
    total_completed_pipelines: int = Field(..., description="Total completed pipelines")
    active_pipelines: int = Field(..., description="Currently active pipelines")
    overall_stats: Dict[str, float] = Field(..., description="Overall statistics")
    stage_performance: Dict[str, StagePerformance] = Field(..., description="Performance by stage")
    monitoring_level: str = Field(..., description="Current monitoring level")

class ActivePipelineInfo(BaseModel):
    """Active pipeline information"""
    pipeline_id: str = Field(..., description="Pipeline identifier")
    child_name: str = Field(..., description="Child's name")
    child_age: Optional[int] = Field(None, description="Child's age")
    story_theme: Optional[str] = Field(None, description="Story theme")
    start_time: float = Field(..., description="Start timestamp")
    running_duration: float = Field(..., description="Current running duration")
    completed_stages: int = Field(..., description="Number of completed stages")
    total_stages: int = Field(..., description="Total expected stages")

# === API Endpoints ===

@router.get("/status/{pipeline_id}", response_model=PipelineStatusResponse)
async def get_pipeline_status(pipeline_id: str) -> PipelineStatusResponse:
    """
    Get detailed status of a specific pipeline
    
    Returns comprehensive information about pipeline execution including
    stage-by-stage progress, performance metrics, and error details.
    """
    try:
        status_data = pipeline_monitor.get_pipeline_status(pipeline_id)
        
        if not status_data:
            raise HTTPException(status_code=404, detail=f"Pipeline {pipeline_id} not found")
        
        # Convert stages to response model format
        stages = {}
        for stage_name, stage_data in status_data["stages"].items():
            stages[stage_name] = PipelineStageStatus(**stage_data)
        
        return PipelineStatusResponse(
            **{**status_data, "stages": stages}
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting pipeline status: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve pipeline status")

@router.get("/performance", response_model=PerformanceSummaryResponse)
async def get_performance_summary() -> PerformanceSummaryResponse:
    """
    Get comprehensive performance summary
    
    Returns aggregated performance metrics across all completed pipelines
    including average durations, success rates, and resource usage.
    """
    try:
        summary_data = pipeline_monitor.get_performance_summary()
        
        if "message" in summary_data:
            # No completed pipelines yet
            return PerformanceSummaryResponse(
                total_completed_pipelines=0,
                active_pipelines=len(pipeline_monitor.active_pipelines),
                overall_stats={},
                stage_performance={},
                monitoring_level=pipeline_monitor.monitoring_level.value
            )
        
        # Convert stage performance to response model format
        stage_performance = {}
        for stage_name, perf_data in summary_data.get("stage_performance", {}).items():
            stage_performance[stage_name] = StagePerformance(**perf_data)
        
        return PerformanceSummaryResponse(
            **{**summary_data, "stage_performance": stage_performance}
        )
    except Exception as e:
        logger.error(f"Error getting performance summary: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve performance summary")

@router.get("/active", response_model=List[ActivePipelineInfo])
async def get_active_pipelines() -> List[ActivePipelineInfo]:
    """
    Get list of currently active (running) pipelines
    
    Returns information about all pipelines that are currently executing,
    including progress information and runtime statistics.
    """
    try:
        active_pipelines = []
        
        import time
        current_time = time.time()
        
        for pipeline_id, pipeline_metrics in pipeline_monitor.active_pipelines.items():
            running_duration = current_time - pipeline_metrics.start_time
            completed_stages = len([s for s in pipeline_metrics.stages.values() if s.success is not None])
            
            active_pipelines.append(ActivePipelineInfo(
                pipeline_id=pipeline_metrics.pipeline_id,
                child_name=pipeline_metrics.child_name,
                child_age=pipeline_metrics.child_age,
                story_theme=pipeline_metrics.story_theme,
                start_time=pipeline_metrics.start_time,
                running_duration=running_duration,
                completed_stages=completed_stages,
                total_stages=4  # story, image, voice, complete
            ))
        
        return active_pipelines
    except Exception as e:
        logger.error(f"Error getting active pipelines: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve active pipelines")

@router.get("/health")
async def pipeline_monitoring_health() -> Dict[str, Any]:
    """
    Pipeline monitoring system health check
    
    Returns system status and basic monitoring statistics.
    """
    try:
        return {
            "status": "healthy",
            "monitoring_level": pipeline_monitor.monitoring_level.value,
            "active_pipelines_count": len(pipeline_monitor.active_pipelines),
            "completed_pipelines_count": len(pipeline_monitor.completed_pipelines),
            "max_history": pipeline_monitor.max_completed_history,
            "performance_thresholds": {
                stage.value: threshold 
                for stage, threshold in pipeline_monitor.performance_thresholds.items()
            }
        }
    except Exception as e:
        logger.error(f"Pipeline monitoring health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }

@router.get("/metrics/recent")
async def get_recent_performance_metrics(
    limit: int = Query(10, ge=1, le=50, description="Number of recent pipelines to analyze")
) -> Dict[str, Any]:
    """
    Get performance metrics for recent pipelines
    
    Returns detailed performance analysis for the most recently completed pipelines.
    """
    try:
        recent_pipelines = pipeline_monitor.completed_pipelines[-limit:]
        
        if not recent_pipelines:
            return {"message": "No recent pipelines found"}
        
        # Calculate recent metrics
        recent_durations = [p.duration for p in recent_pipelines if p.duration]
        recent_success_rates = [p.success_rate for p in recent_pipelines]
        recent_token_usage = [p.total_tokens for p in recent_pipelines]
        recent_image_counts = [p.total_images for p in recent_pipelines]
        
        return {
            "analyzed_pipelines": len(recent_pipelines),
            "time_range": {
                "oldest": recent_pipelines[0].start_time if recent_pipelines else None,
                "newest": recent_pipelines[-1].start_time if recent_pipelines else None
            },
            "performance_metrics": {
                "avg_duration": sum(recent_durations) / len(recent_durations) if recent_durations else 0,
                "avg_success_rate": sum(recent_success_rates) / len(recent_success_rates) if recent_success_rates else 0,
                "avg_tokens": sum(recent_token_usage) / len(recent_token_usage) if recent_token_usage else 0,
                "avg_images": sum(recent_image_counts) / len(recent_image_counts) if recent_image_counts else 0
            },
            "trends": {
                "fastest_pipeline": min(recent_durations) if recent_durations else None,
                "slowest_pipeline": max(recent_durations) if recent_durations else None,
                "total_tokens_used": sum(recent_token_usage),
                "total_images_generated": sum(recent_image_counts)
            }
        }
    except Exception as e:
        logger.error(f"Error getting recent metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve recent performance metrics")

@router.post("/alerts/test")
async def test_performance_alert() -> Dict[str, str]:
    """
    Test performance alerting system
    
    Triggers a test alert to verify the monitoring system is working correctly.
    """
    try:
        # Create a test pipeline with slow performance to trigger alert
        test_pipeline_id = f"test_alert_{int(time.time())}"
        
        # This is just for testing - normally alerts are triggered by actual slow performance
        logger.warning(f"TEST ALERT: Simulated slow performance for pipeline {test_pipeline_id}")
        
        return {
            "message": f"Test alert triggered for pipeline {test_pipeline_id}",
            "check_logs": "Check application logs for alert message"
        }
    except Exception as e:
        logger.error(f"Error triggering test alert: {e}")
        raise HTTPException(status_code=500, detail="Failed to trigger test alert")

@router.get("/config")
async def get_monitoring_config() -> Dict[str, Any]:
    """
    Get current monitoring configuration
    
    Returns the current monitoring settings and thresholds.
    """
    return {
        "monitoring_level": pipeline_monitor.monitoring_level.value,
        "max_completed_history": pipeline_monitor.max_completed_history,
        "performance_thresholds": {
            stage.value: threshold 
            for stage, threshold in pipeline_monitor.performance_thresholds.items()
        },
        "stage_history_limits": {
            stage.value: len(history)
            for stage, history in pipeline_monitor.stage_performance_history.items()
        }
    }