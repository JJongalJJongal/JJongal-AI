"""
LangChain Pipeline Monitoring for JJongal-AI
Monitors image generation, voice synthesis, and story creation pipeline
"""

import time
import asyncio
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path

from src.shared.utils.logging import get_module_logger
from src.shared.configs.settings import get_settings

logger = get_module_logger(__name__)

class PipelineStage(Enum):
    """Pipeline stages for monitoring"""
    STORY_GENERATION = "story_generation"
    IMAGE_GENERATION = "image_generation"
    VOICE_SYNTHESIS = "voice_synthesis"
    PIPELINE_COMPLETE = "pipeline_complete"

class MonitoringLevel(Enum):
    """Monitoring detail levels"""
    BASIC = "basic"          # Just timing and success/failure
    DETAILED = "detailed"    # Include metrics and parameters
    DEBUG = "debug"          # Full trace with intermediate steps

@dataclass
class StageMetrics:
    """Metrics for a single pipeline stage"""
    stage: PipelineStage
    start_time: float
    end_time: Optional[float] = None
    success: Optional[bool] = None
    error_message: Optional[str] = None
    
    # Stage-specific metrics
    token_count: Optional[int] = None
    image_count: Optional[int] = None
    audio_duration: Optional[float] = None
    file_size_mb: Optional[float] = None
    
    # API/Model specific metrics
    model_name: Optional[str] = None
    api_latency: Optional[float] = None
    retry_count: int = 0
    
    # Quality metrics
    content_length: Optional[int] = None
    quality_score: Optional[float] = None
    
    def __post_init__(self):
        if self.start_time is None:
            self.start_time = time.time()
    
    @property
    def duration(self) -> Optional[float]:
        """Get stage duration in seconds"""
        if self.end_time is None:
            return None
        return self.end_time - self.start_time
    
    def finish(self, success: bool = True, error_message: Optional[str] = None):
        """Mark stage as finished"""
        self.end_time = time.time()
        self.success = success
        self.error_message = error_message

@dataclass
class PipelineMetrics:
    """Complete pipeline execution metrics"""
    pipeline_id: str
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    
    # Pipeline metadata
    child_name: str = ""
    child_age: Optional[int] = None
    story_theme: Optional[str] = None
    
    # Stage metrics
    stages: Dict[PipelineStage, StageMetrics] = field(default_factory=dict)
    
    # Overall metrics
    total_tokens: int = 0
    total_images: int = 0
    total_audio_duration: float = 0.0
    total_retries: int = 0
    
    # Performance metrics
    cache_hits: int = 0
    cache_misses: int = 0
    
    @property
    def duration(self) -> Optional[float]:
        """Get total pipeline duration"""
        if self.end_time is None:
            return None
        return self.end_time - self.start_time
    
    @property
    def success_rate(self) -> float:
        """Get success rate of completed stages"""
        completed = [s for s in self.stages.values() if s.success is not None]
        if not completed:
            return 0.0
        successful = [s for s in completed if s.success]
        return len(successful) / len(completed)
    
    def add_stage(self, stage: PipelineStage) -> StageMetrics:
        """Add a new stage for monitoring"""
        stage_metrics = StageMetrics(stage=stage, start_time=time.time())
        self.stages[stage] = stage_metrics
        return stage_metrics
    
    def finish(self, success: bool = True):
        """Mark pipeline as finished"""
        self.end_time = time.time()
        
        # Calculate totals
        self.total_tokens = sum(s.token_count or 0 for s in self.stages.values())
        self.total_images = sum(s.image_count or 0 for s in self.stages.values())
        self.total_audio_duration = sum(s.audio_duration or 0 for s in self.stages.values())
        self.total_retries = sum(s.retry_count for s in self.stages.values())

class PipelineMonitor:
    """
    LangChain Pipeline Monitor
    
    Monitors the complete story generation pipeline:
    - Story text generation (ChatBot B)
    - Image generation (DALL-E 3)
    - Voice synthesis (ElevenLabs)
    - Pipeline orchestration
    """
    
    def __init__(self, monitoring_level: MonitoringLevel = MonitoringLevel.DETAILED):
        self.settings = get_settings()
        self.monitoring_level = monitoring_level
        self.active_pipelines: Dict[str, PipelineMetrics] = {}
        self.completed_pipelines: List[PipelineMetrics] = []
        self.max_completed_history = 100  # Keep last 100 completed pipelines
        
        # Performance tracking
        self.stage_performance_history: Dict[PipelineStage, List[float]] = {
            stage: [] for stage in PipelineStage
        }
        
        # Alerting thresholds
        self.performance_thresholds = {
            PipelineStage.STORY_GENERATION: 30.0,    # 30 seconds
            PipelineStage.IMAGE_GENERATION: 60.0,    # 60 seconds
            PipelineStage.VOICE_SYNTHESIS: 45.0,     # 45 seconds
            PipelineStage.PIPELINE_COMPLETE: 180.0,  # 3 minutes total
        }
        
        logger.info(f"PipelineMonitor initialized with {monitoring_level.value} monitoring")
    
    def start_pipeline(self, pipeline_id: str, child_name: str = "", child_age: Optional[int] = None, 
                      story_theme: Optional[str] = None) -> PipelineMetrics:
        """Start monitoring a new pipeline execution"""
        
        metrics = PipelineMetrics(
            pipeline_id=pipeline_id,
            child_name=child_name,
            child_age=child_age,
            story_theme=story_theme
        )
        
        self.active_pipelines[pipeline_id] = metrics
        
        if self.monitoring_level != MonitoringLevel.BASIC:
            logger.info(f"Started monitoring pipeline {pipeline_id} for {child_name} (age {child_age})")
        
        return metrics
    
    def start_stage(self, pipeline_id: str, stage: PipelineStage) -> Optional[StageMetrics]:
        """Start monitoring a pipeline stage"""
        
        if pipeline_id not in self.active_pipelines:
            logger.warning(f"Pipeline {pipeline_id} not found for stage {stage.value}")
            return None
        
        pipeline = self.active_pipelines[pipeline_id]
        stage_metrics = pipeline.add_stage(stage)
        
        if self.monitoring_level == MonitoringLevel.DEBUG:
            logger.info(f"Started stage {stage.value} for pipeline {pipeline_id}")
        
        return stage_metrics
    
    def finish_stage(self, pipeline_id: str, stage: PipelineStage, 
                    success: bool = True, error_message: Optional[str] = None,
                    **stage_specific_metrics) -> None:
        """Finish monitoring a pipeline stage"""
        
        if pipeline_id not in self.active_pipelines:
            logger.warning(f"Pipeline {pipeline_id} not found for finishing stage {stage.value}")
            return
        
        pipeline = self.active_pipelines[pipeline_id]
        
        if stage not in pipeline.stages:
            logger.warning(f"Stage {stage.value} not started for pipeline {pipeline_id}")
            return
        
        stage_metrics = pipeline.stages[stage]
        stage_metrics.finish(success, error_message)
        
        # Add stage-specific metrics
        for key, value in stage_specific_metrics.items():
            if hasattr(stage_metrics, key):
                setattr(stage_metrics, key, value)
        
        # Record performance
        if stage_metrics.duration:
            self.stage_performance_history[stage].append(stage_metrics.duration)
            
            # Keep only recent history
            if len(self.stage_performance_history[stage]) > 50:
                self.stage_performance_history[stage] = self.stage_performance_history[stage][-50:]
            
            # Check for performance alerts
            self._check_performance_alert(stage, stage_metrics.duration)
        
        if self.monitoring_level != MonitoringLevel.BASIC:
            status = "✅" if success else "❌"
            duration_str = f" ({stage_metrics.duration:.2f}s)" if stage_metrics.duration else ""
            logger.info(f"{status} Stage {stage.value} for pipeline {pipeline_id}{duration_str}")
            
            if not success and error_message:
                logger.error(f"Stage {stage.value} failed: {error_message}")
    
    def finish_pipeline(self, pipeline_id: str, success: bool = True) -> Optional[PipelineMetrics]:
        """Finish monitoring a pipeline execution"""
        
        if pipeline_id not in self.active_pipelines:
            logger.warning(f"Pipeline {pipeline_id} not found for finishing")
            return None
        
        pipeline = self.active_pipelines[pipeline_id]
        pipeline.finish(success)
        
        # Move to completed pipelines
        self.completed_pipelines.append(pipeline)
        del self.active_pipelines[pipeline_id]
        
        # Keep history manageable
        if len(self.completed_pipelines) > self.max_completed_history:
            self.completed_pipelines = self.completed_pipelines[-self.max_completed_history:]
        
        # Log completion
        status = "✅" if success else "❌"
        duration_str = f" ({pipeline.duration:.2f}s)" if pipeline.duration else ""
        success_rate = f" (Success: {pipeline.success_rate:.0%})"
        
        logger.info(f"{status} Pipeline {pipeline_id} completed{duration_str}{success_rate}")
        
        if self.monitoring_level == MonitoringLevel.DETAILED:
            self._log_detailed_pipeline_summary(pipeline)
        
        return pipeline
    
    def get_pipeline_status(self, pipeline_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of a pipeline"""
        
        pipeline = self.active_pipelines.get(pipeline_id)
        if not pipeline:
            # Check completed pipelines
            completed = [p for p in self.completed_pipelines if p.pipeline_id == pipeline_id]
            if completed:
                pipeline = completed[-1]  # Get most recent
            else:
                return None
        
        return {
            "pipeline_id": pipeline.pipeline_id,
            "status": "completed" if pipeline.end_time else "running",
            "start_time": pipeline.start_time,
            "duration": pipeline.duration,
            "child_name": pipeline.child_name,
            "child_age": pipeline.child_age,
            "story_theme": pipeline.story_theme,
            "stages": {
                stage.value: {
                    "success": stage_metrics.success,
                    "duration": stage_metrics.duration,
                    "error": stage_metrics.error_message
                }
                for stage, stage_metrics in pipeline.stages.items()
            },
            "success_rate": pipeline.success_rate,
            "total_tokens": pipeline.total_tokens,
            "total_images": pipeline.total_images,
            "total_audio_duration": pipeline.total_audio_duration,
            "total_retries": pipeline.total_retries
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get overall performance summary"""
        
        total_completed = len(self.completed_pipelines)
        if total_completed == 0:
            return {"message": "No completed pipelines yet"}
        
        # Calculate averages
        avg_durations = {}
        for stage, durations in self.stage_performance_history.items():
            if durations:
                avg_durations[stage.value] = {
                    "avg_duration": sum(durations) / len(durations),
                    "min_duration": min(durations),
                    "max_duration": max(durations),
                    "count": len(durations)
                }
        
        # Overall pipeline stats
        completed_durations = [p.duration for p in self.completed_pipelines if p.duration]
        success_rates = [p.success_rate for p in self.completed_pipelines]
        
        return {
            "total_completed_pipelines": total_completed,
            "active_pipelines": len(self.active_pipelines),
            "overall_stats": {
                "avg_pipeline_duration": sum(completed_durations) / len(completed_durations) if completed_durations else 0,
                "avg_success_rate": sum(success_rates) / len(success_rates) if success_rates else 0,
                "total_tokens": sum(p.total_tokens for p in self.completed_pipelines),
                "total_images": sum(p.total_images for p in self.completed_pipelines),
                "total_audio_duration": sum(p.total_audio_duration for p in self.completed_pipelines)
            },
            "stage_performance": avg_durations,
            "monitoring_level": self.monitoring_level.value
        }
    
    def _check_performance_alert(self, stage: PipelineStage, duration: float) -> None:
        """Check if stage duration exceeds threshold and alert"""
        
        threshold = self.performance_thresholds.get(stage)
        if threshold and duration > threshold:
            logger.warning(f"Performance alert: {stage.value} took {duration:.2f}s (threshold: {threshold}s)")
    
    def _log_detailed_pipeline_summary(self, pipeline: PipelineMetrics) -> None:
        """Log detailed pipeline summary"""
        
        logger.info(f"Pipeline {pipeline.pipeline_id} Summary:")
        logger.info(f"  Child: {pipeline.child_name} (age {pipeline.child_age})")
        logger.info(f"  Theme: {pipeline.story_theme}")
        logger.info(f"  Duration: {pipeline.duration:.2f}s")
        logger.info(f"  Success Rate: {pipeline.success_rate:.0%}")
        logger.info(f"  Tokens: {pipeline.total_tokens}, Images: {pipeline.total_images}, Audio: {pipeline.total_audio_duration:.1f}s")
        
        for stage, stage_metrics in pipeline.stages.items():
            status = "✅" if stage_metrics.success else "❌"
            duration = f"{stage_metrics.duration:.2f}s" if stage_metrics.duration else "N/A"
            logger.info(f"    {status} {stage.value}: {duration}")

# Global pipeline monitor instance
pipeline_monitor = PipelineMonitor()

# Context managers for easy monitoring
class monitor_pipeline:
    """Context manager for monitoring complete pipeline execution"""
    
    def __init__(self, pipeline_id: str, child_name: str = "", child_age: Optional[int] = None, 
                 story_theme: Optional[str] = None):
        self.pipeline_id = pipeline_id
        self.child_name = child_name
        self.child_age = child_age
        self.story_theme = story_theme
        self.success = True
    
    def __enter__(self) -> PipelineMetrics:
        return pipeline_monitor.start_pipeline(
            self.pipeline_id, self.child_name, self.child_age, self.story_theme
        )
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.success = False
        pipeline_monitor.finish_pipeline(self.pipeline_id, self.success)

class monitor_stage:
    """Context manager for monitoring individual pipeline stages"""
    
    def __init__(self, pipeline_id: str, stage: PipelineStage):
        self.pipeline_id = pipeline_id
        self.stage = stage
        self.stage_metrics = None
        self.success = True
        self.stage_specific_metrics = {}
    
    def __enter__(self) -> 'monitor_stage':
        self.stage_metrics = pipeline_monitor.start_stage(self.pipeline_id, self.stage)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.success = False
            error_message = str(exc_val) if exc_val else None
        else:
            error_message = None
        
        pipeline_monitor.finish_stage(
            self.pipeline_id, 
            self.stage, 
            self.success, 
            error_message,
            **self.stage_specific_metrics
        )
    
    def add_metrics(self, **metrics):
        """Add stage-specific metrics"""
        self.stage_specific_metrics.update(metrics)
        return self