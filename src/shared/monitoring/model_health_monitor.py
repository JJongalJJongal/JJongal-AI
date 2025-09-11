"""
AI Model Health Monitor for JJongal-AI

Monitors AI model performance, memory usage, and background persistence
to ensure optimal resource management and reliability.
Integrates with pipeline monitoring for comprehensive AI system tracking.
"""

import asyncio
import time
import psutil
import gc
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass

from src.shared.utils.logging import get_module_logger
from .pipeline_monitor import pipeline_monitor, PipelineStage

logger = get_module_logger(__name__)


@dataclass
class ModelHealthMetrics:
    """Data class for model health metrics"""
    model_name: str
    is_loaded: bool
    memory_usage_mb: float
    last_used: Optional[datetime]
    total_requests: int
    error_count: int
    avg_response_time: float
    status: str  # 'healthy', 'degraded', 'critical', 'idle'


class ModelHealthMonitor:
    """
    Comprehensive health monitor for AI models running in background
    
    Features:
    - Real-time memory usage tracking
    - Response time monitoring 
    - Error rate analysis
    - Background model persistence verification
    - Resource cleanup recommendations
    """
    
    def __init__(self, check_interval: int = 300):  # 5 minutes
        self.check_interval = check_interval
        self.metrics_history: Dict[str, List[ModelHealthMetrics]] = {}
        self.monitored_models: Dict[str, Dict[str, Any]] = {}
        self.last_cleanup = datetime.now()
        self._monitoring_task = None
        self._is_monitoring = False
        
        logger.info("Model Health Monitor initialized")
    
    def register_model(
        self,
        model_name: str,
        instance_ref: Any,
        check_method: Optional[str] = None
    ):
        """
        Register a model for health monitoring
        
        Args:
            model_name: Name identifier for the model
            instance_ref: Reference to the model instance
            check_method: Method name to check model health
        """
        self.monitored_models[model_name] = {
            'instance': instance_ref,
            'check_method': check_method or 'health_check',
            'registered_at': datetime.now(),
            'total_requests': 0,
            'error_count': 0,
            'response_times': [],
            'last_used': None
        }
        
        logger.info(f"Registered model for monitoring: {model_name}")
    
    async def start_monitoring(self):
        """Start background health monitoring"""
        if self._is_monitoring:
            logger.warning("Health monitoring already running")
            return
        
        self._is_monitoring = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Model health monitoring started")
    
    async def stop_monitoring(self):
        """Stop background health monitoring"""
        self._is_monitoring = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Model health monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self._is_monitoring:
            try:
                await self._perform_health_checks()
                await self._check_memory_usage()
                await self._cleanup_old_metrics()
                
                await asyncio.sleep(self.check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying
    
    async def _perform_health_checks(self):
        """Perform health checks on all registered models"""
        current_time = datetime.now()
        
        for model_name, model_info in self.monitored_models.items():
            try:
                # Check if model instance still exists
                instance = model_info['instance']
                if instance is None:
                    status = 'unavailable'
                    is_loaded = False
                else:
                    is_loaded = True
                    
                    # Try to call health check method if available
                    check_method = model_info.get('check_method')
                    if check_method and hasattr(instance, check_method):
                        start_time = time.time()
                        
                        try:
                            health_result = getattr(instance, check_method)()
                            if asyncio.iscoroutine(health_result):
                                health_result = await health_result
                            
                            response_time = (time.time() - start_time) * 1000  # ms
                            
                            if health_result:
                                status = 'healthy'
                            else:
                                status = 'degraded'
                                model_info['error_count'] += 1
                                
                        except Exception as e:
                            logger.warning(f"Health check failed for {model_name}: {e}")
                            status = 'critical'
                            response_time = -1
                            model_info['error_count'] += 1
                    else:
                        status = 'healthy'  # Assume healthy if no check method
                        response_time = 0
                
                # Calculate memory usage
                memory_usage = await self._estimate_model_memory(instance) if instance else 0
                
                # Calculate average response time
                response_times = model_info.get('response_times', [])
                if response_time > 0:
                    response_times.append(response_time)
                    # Keep only last 100 response times
                    model_info['response_times'] = response_times[-100:]
                
                avg_response_time = sum(response_times) / len(response_times) if response_times else 0
                
                # Create health metrics
                metrics = ModelHealthMetrics(
                    model_name=model_name,
                    is_loaded=is_loaded,
                    memory_usage_mb=memory_usage,
                    last_used=model_info.get('last_used'),
                    total_requests=model_info.get('total_requests', 0),
                    error_count=model_info.get('error_count', 0),
                    avg_response_time=round(avg_response_time, 2),
                    status=status
                )
                
                # Store metrics
                if model_name not in self.metrics_history:
                    self.metrics_history[model_name] = []
                
                self.metrics_history[model_name].append(metrics)
                
                # Log status changes
                if len(self.metrics_history[model_name]) > 1:
                    previous_status = self.metrics_history[model_name][-2].status
                    if previous_status != status:
                        logger.info(f"Model {model_name} status changed: {previous_status} -> {status}")
                
            except Exception as e:
                logger.error(f"Error checking health of {model_name}: {e}")
    
    async def _estimate_model_memory(self, instance: Any) -> float:
        """Estimate memory usage of a model instance"""
        try:
            import sys
            
            # Get memory usage of the instance and its attributes
            memory_size = 0
            
            # Check if instance has memory-intensive attributes
            for attr_name in dir(instance):
                if not attr_name.startswith('_'):
                    try:
                        attr_value = getattr(instance, attr_name)
                        
                        # Estimate memory for common AI model attributes
                        if hasattr(attr_value, '__sizeof__'):
                            memory_size += sys.getsizeof(attr_value)
                        
                        # Special handling for known memory-intensive objects
                        if 'llm' in attr_name.lower() or 'model' in attr_name.lower():
                            memory_size += 100 * 1024 * 1024  # Estimate 100MB for LLM
                        elif 'vector' in attr_name.lower() or 'embed' in attr_name.lower():
                            memory_size += 50 * 1024 * 1024   # Estimate 50MB for vectors
                        
                    except Exception:
                        continue
            
            return memory_size / (1024 * 1024)  # Convert to MB
            
        except Exception as e:
            logger.debug(f"Could not estimate memory for model: {e}")
            return 0.0
    
    async def _check_memory_usage(self):
        """Check overall system memory usage"""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_percent = process.memory_percent()
            
            # Log memory usage if it's high
            if memory_percent > 80:
                logger.warning(f"High memory usage detected: {memory_percent:.1f}% "
                             f"({memory_info.rss / (1024*1024):.1f} MB)")
            
            # Trigger garbage collection if memory usage is very high
            if memory_percent > 90:
                logger.info("Triggering garbage collection due to high memory usage")
                gc.collect()
                
        except Exception as e:
            logger.debug(f"Could not check system memory: {e}")
    
    async def _cleanup_old_metrics(self):
        """Clean up old metrics to prevent memory buildup"""
        cutoff_time = datetime.now() - timedelta(hours=24)  # Keep 24 hours of metrics
        
        for model_name, metrics_list in self.metrics_history.items():
            original_count = len(metrics_list)
            
            # Keep only recent metrics
            self.metrics_history[model_name] = [
                m for m in metrics_list 
                if m.last_used is None or m.last_used > cutoff_time
            ]
            
            cleaned_count = original_count - len(self.metrics_history[model_name])
            if cleaned_count > 0:
                logger.debug(f"Cleaned {cleaned_count} old metrics for {model_name}")
    
    def record_model_usage(self, model_name: str, response_time: float = None, error: bool = False):
        """
        Record model usage for monitoring
        
        Args:
            model_name: Name of the model
            response_time: Response time in milliseconds
            error: Whether an error occurred
        """
        if model_name in self.monitored_models:
            model_info = self.monitored_models[model_name]
            model_info['last_used'] = datetime.now()
            model_info['total_requests'] += 1
            
            if error:
                model_info['error_count'] += 1
            
            if response_time is not None:
                response_times = model_info.get('response_times', [])
                response_times.append(response_time)
                model_info['response_times'] = response_times[-100:]  # Keep last 100
    
    def get_model_status(self, model_name: str) -> Optional[ModelHealthMetrics]:
        """Get current status of a specific model"""
        if model_name in self.metrics_history and self.metrics_history[model_name]:
            return self.metrics_history[model_name][-1]
        return None
    
    def get_all_models_status(self) -> Dict[str, ModelHealthMetrics]:
        """Get current status of all monitored models"""
        status = {}
        for model_name in self.monitored_models:
            latest_metrics = self.get_model_status(model_name)
            if latest_metrics:
                status[model_name] = latest_metrics
        return status
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get comprehensive health summary"""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_models': len(self.monitored_models),
            'models_status': {},
            'system_memory': {},
            'recommendations': []
        }
        
        # Model statuses
        all_status = self.get_all_models_status()
        healthy_count = sum(1 for m in all_status.values() if m.status == 'healthy')
        
        summary['models_status'] = {
            'healthy': healthy_count,
            'degraded': sum(1 for m in all_status.values() if m.status == 'degraded'),
            'critical': sum(1 for m in all_status.values() if m.status == 'critical'),
            'idle': sum(1 for m in all_status.values() if m.status == 'idle')
        }
        
        # System memory
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            summary['system_memory'] = {
                'rss_mb': round(memory_info.rss / (1024*1024), 1),
                'percent': round(process.memory_percent(), 1)
            }
        except Exception:
            pass
        
        # Recommendations
        for model_name, metrics in all_status.items():
            if metrics.status == 'critical':
                summary['recommendations'].append(f"Model {model_name} needs attention - critical status")
            elif metrics.memory_usage_mb > 500:  # 500MB threshold
                summary['recommendations'].append(f"Model {model_name} using high memory: {metrics.memory_usage_mb:.1f}MB")
            elif metrics.error_count > 10:
                summary['recommendations'].append(f"Model {model_name} has high error count: {metrics.error_count}")
        
        return summary


# Global monitor instance
_health_monitor = None


def get_health_monitor() -> ModelHealthMonitor:
    """Get global health monitor instance"""
    global _health_monitor
    if _health_monitor is None:
        _health_monitor = ModelHealthMonitor()
    return _health_monitor


async def initialize_model_monitoring():
    """Initialize model monitoring for all registered services"""
    monitor = get_health_monitor()
    
    try:
        # Import and register models
        from src.api.v1.dependencies import ServiceFactory
        
        # Register ChatBotA if available
        try:
            chatbot_a = await ServiceFactory.get_instance("chatbot_a")
            monitor.register_model("chatbot_a", chatbot_a, "health_check")
            logger.info("Registered ChatBotA for monitoring")
        except Exception as e:
            logger.warning(f"Could not register ChatBotA for monitoring: {e}")
        
        # Register voice processor if available
        try:
            voice_processor = await ServiceFactory.get_instance("voice_processor")
            monitor.register_model("voice_processor", voice_processor, "health_check")
            logger.info("Registered Voice Processor for monitoring")
        except Exception as e:
            logger.warning(f"Could not register Voice Processor for monitoring: {e}")
        
        # Start monitoring
        await monitor.start_monitoring()
        logger.info("Model health monitoring initialized and started")
        
    except Exception as e:
        logger.error(f"Failed to initialize model monitoring: {e}")


async def cleanup_model_monitoring():
    """Cleanup model monitoring on shutdown"""
    global _health_monitor
    if _health_monitor:
        await _health_monitor.stop_monitoring()
        logger.info("Model health monitoring cleaned up")