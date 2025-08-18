"""
Enhanced Performance Optimization System

Advanced performance monitoring and optimization with:
- Intelligent caching strategies
- Token usage optimization
- Chain performance analysis
- Memory management
- Adaptive scaling
- Real-time monitoring
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
import time
import asyncio
import threading
from collections import defaultdict, deque
from datetime import datetime, timedelta
import statistics
from src.shared.utils.logging import get_module_logger

logger = get_module_logger(__name__)

@dataclass
class PerformanceMetrics:
    """Performance metrics data structure"""
    operation_name: str
    duration: float
    token_usage: int
    memory_usage: float
    success: bool
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    value: Any
    created_at: datetime
    access_count: int
    last_accessed: datetime
    size_bytes: int
    ttl_seconds: int

class IntelligentCache:
    """
    Intelligent caching system with LRU, TTL, and adaptive policies
    """
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: Dict[str, CacheEntry] = {}
        self._access_order = deque()
        self._lock = threading.RLock()
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "total_size_bytes": 0
        }
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache with intelligent tracking"""
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                self._stats["misses"] += 1
                return None
            
            # Check TTL
            if self._is_expired(entry):
                self._remove_entry(key)
                self._stats["misses"] += 1
                return None
            
            # Update access metadata
            entry.access_count += 1
            entry.last_accessed = datetime.now()
            
            # Move to end of access order (LRU)
            try:
                self._access_order.remove(key)
            except ValueError:
                pass
            self._access_order.append(key)
            
            self._stats["hits"] += 1
            return entry.value
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache with intelligent eviction"""
        with self._lock:
            ttl = ttl or self.default_ttl
            
            # Calculate approximate size
            size_bytes = self._estimate_size(value)
            
            # Remove existing entry if present
            if key in self._cache:
                self._remove_entry(key)
            
            # Evict if necessary
            while len(self._cache) >= self.max_size:
                self._evict_lru()
            
            # Create new entry
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=datetime.now(),
                access_count=1,
                last_accessed=datetime.now(),
                size_bytes=size_bytes,
                ttl_seconds=ttl
            )
            
            self._cache[key] = entry
            self._access_order.append(key)
            self._stats["total_size_bytes"] += size_bytes
    
    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if cache entry is expired"""
        return (datetime.now() - entry.created_at).seconds > entry.ttl_seconds
    
    def _remove_entry(self, key: str) -> None:
        """Remove entry from cache"""
        if key in self._cache:
            entry = self._cache[key]
            self._stats["total_size_bytes"] -= entry.size_bytes
            del self._cache[key]
            
            try:
                self._access_order.remove(key)
            except ValueError:
                pass
    
    def _evict_lru(self) -> None:
        """Evict least recently used entry"""
        if self._access_order:
            lru_key = self._access_order.popleft()
            self._remove_entry(lru_key)
            self._stats["evictions"] += 1
    
    def _estimate_size(self, obj: Any) -> int:
        """Estimate object size in bytes"""
        try:
            import sys
            return sys.getsizeof(obj)
        except:
            return 1024  # Default estimate
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            total_requests = self._stats["hits"] + self._stats["misses"]
            hit_rate = self._stats["hits"] / total_requests if total_requests > 0 else 0
            
            return {
                **self._stats,
                "hit_rate": hit_rate,
                "cache_size": len(self._cache),
                "max_size": self.max_size
            }
    
    def clear(self) -> None:
        """Clear all cache entries"""
        with self._lock:
            self._cache.clear()
            self._access_order.clear()
            self._stats["total_size_bytes"] = 0

class PerformanceMonitor:
    """
    Real-time performance monitoring and analysis
    """
    
    def __init__(self, max_history: int = 10000):
        self.max_history = max_history
        self._metrics_history = deque(maxlen=max_history)
        self._operation_stats = defaultdict(list)
        self._alerts = []
        self._thresholds = {
            "max_duration": 30.0,  # seconds
            "max_token_usage": 10000,
            "max_memory_mb": 500,
            "min_success_rate": 0.9
        }
    
    def record_metric(self, metric: PerformanceMetrics) -> None:
        """Record a performance metric"""
        self._metrics_history.append(metric)
        self._operation_stats[metric.operation_name].append(metric)
        
        # Check for performance alerts
        self._check_alerts(metric)
    
    def _check_alerts(self, metric: PerformanceMetrics) -> None:
        """Check for performance alerts"""
        alerts = []
        
        if metric.duration > self._thresholds["max_duration"]:
            alerts.append(f"High duration: {metric.duration:.2f}s for {metric.operation_name}")
        
        if metric.token_usage > self._thresholds["max_token_usage"]:
            alerts.append(f"High token usage: {metric.token_usage} for {metric.operation_name}")
        
        if metric.memory_usage > self._thresholds["max_memory_mb"]:
            alerts.append(f"High memory usage: {metric.memory_usage:.2f}MB for {metric.operation_name}")
        
        for alert in alerts:
            self._alerts.append({
                "message": alert,
                "timestamp": datetime.now(),
                "metric": metric
            })
            logger.warning(f"Performance Alert: {alert}")
    
    def get_operation_stats(self, operation_name: str) -> Dict[str, Any]:
        """Get statistics for a specific operation"""
        metrics = self._operation_stats[operation_name]
        if not metrics:
            return {}
        
        durations = [m.duration for m in metrics]
        token_usages = [m.token_usage for m in metrics]
        memory_usages = [m.memory_usage for m in metrics]
        success_rate = sum(1 for m in metrics if m.success) / len(metrics)
        
        return {
            "operation_name": operation_name,
            "total_calls": len(metrics),
            "success_rate": success_rate,
            "duration_stats": {
                "mean": statistics.mean(durations),
                "median": statistics.median(durations),
                "min": min(durations),
                "max": max(durations),
                "std": statistics.stdev(durations) if len(durations) > 1 else 0
            },
            "token_stats": {
                "mean": statistics.mean(token_usages),
                "total": sum(token_usages),
                "max": max(token_usages)
            },
            "memory_stats": {
                "mean": statistics.mean(memory_usages),
                "max": max(memory_usages)
            }
        }
    
    def get_overall_stats(self) -> Dict[str, Any]:
        """Get overall performance statistics"""
        if not self._metrics_history:
            return {}
        
        all_durations = [m.duration for m in self._metrics_history]
        all_token_usage = [m.token_usage for m in self._metrics_history]
        all_memory_usage = [m.memory_usage for m in self._metrics_history]
        
        success_rate = sum(1 for m in self._metrics_history if m.success) / len(self._metrics_history)
        
        # Recent performance trend (last 100 metrics)
        recent_metrics = list(self._metrics_history)[-100:]
        recent_success_rate = sum(1 for m in recent_metrics if m.success) / len(recent_metrics)
        recent_avg_duration = statistics.mean([m.duration for m in recent_metrics])
        
        return {
            "total_operations": len(self._metrics_history),
            "overall_success_rate": success_rate,
            "recent_success_rate": recent_success_rate,
            "recent_avg_duration": recent_avg_duration,
            "duration_stats": {
                "mean": statistics.mean(all_durations),
                "median": statistics.median(all_durations),
                "p95": statistics.quantiles(all_durations, n=20)[18] if len(all_durations) > 20 else max(all_durations),
                "max": max(all_durations)
            },
            "token_usage": {
                "total": sum(all_token_usage),
                "mean": statistics.mean(all_token_usage),
                "max": max(all_token_usage)
            },
            "memory_usage": {
                "mean": statistics.mean(all_memory_usage),
                "max": max(all_memory_usage)
            },
            "active_alerts": len([a for a in self._alerts if 
                                (datetime.now() - a["timestamp"]).seconds < 3600]),
            "operation_breakdown": {op: len(metrics) for op, metrics in self._operation_stats.items()}
        }
    
    def get_recent_alerts(self, hours: int = 1) -> List[Dict[str, Any]]:
        """Get recent performance alerts"""
        cutoff = datetime.now() - timedelta(hours=hours)
        return [alert for alert in self._alerts if alert["timestamp"] > cutoff]

class EnhancedPerformanceOptimizer:
    """
    Enhanced Performance Optimization System
    
    Features:
    - Intelligent caching with adaptive policies
    - Real-time performance monitoring
    - Token usage optimization
    - Memory management
    - Automatic scaling recommendations
    """
    
    def __init__(self):
        self.cache = IntelligentCache(max_size=1000, default_ttl=3600)
        self.monitor = PerformanceMonitor(max_history=10000)
        self.optimization_recommendations = []
        
        # Token optimization settings
        self.token_optimization_enabled = True
        self.max_context_length = 8000
        self.compression_threshold = 0.8
        
        # Caching policies
        self.cache_policies = {
            "prompt_templates": {"ttl": 7200, "enabled": True},
            "rag_results": {"ttl": 1800, "enabled": True},
            "generation_plans": {"ttl": 3600, "enabled": True},
            "evaluation_results": {"ttl": 1800, "enabled": True},
            "cultural_contexts": {"ttl": 14400, "enabled": True}
        }
    
    async def optimize_performance(self, operation_name: str, func, *args, **kwargs):
        """
        Wrapper for performance optimization with monitoring
        """
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        try:
            # Check cache first
            cache_key = self._generate_cache_key(operation_name, args, kwargs)
            if self._should_cache(operation_name):
                cached_result = self.cache.get(cache_key)
                if cached_result is not None:
                    logger.debug(f"Cache hit for {operation_name}")
                    return cached_result
            
            # Execute function
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            # Cache result if appropriate
            if self._should_cache(operation_name) and result is not None:
                ttl = self.cache_policies.get(operation_name, {}).get("ttl", self.cache.default_ttl)
                self.cache.set(cache_key, result, ttl)
            
            # Record performance metrics
            duration = time.time() - start_time
            memory_usage = self._get_memory_usage() - start_memory
            token_usage = self._estimate_token_usage(args, kwargs, result)
            
            metric = PerformanceMetrics(
                operation_name=operation_name,
                duration=duration,
                token_usage=token_usage,
                memory_usage=memory_usage,
                success=True,
                timestamp=datetime.now(),
                metadata={
                    "cache_hit": False,
                    "args_count": len(args),
                    "kwargs_count": len(kwargs)
                }
            )
            
            self.monitor.record_metric(metric)
            
            return result
            
        except Exception as e:
            # Record failure metric
            duration = time.time() - start_time
            memory_usage = self._get_memory_usage() - start_memory
            
            metric = PerformanceMetrics(
                operation_name=operation_name,
                duration=duration,
                token_usage=0,
                memory_usage=memory_usage,
                success=False,
                timestamp=datetime.now(),
                metadata={"error": str(e)}
            )
            
            self.monitor.record_metric(metric)
            raise
    
    def _generate_cache_key(self, operation_name: str, args: Tuple, kwargs: Dict) -> str:
        """Generate cache key for operation"""
        import hashlib
        import json
        
        # Create a deterministic string representation
        key_data = {
            "operation": operation_name,
            "args": str(args),
            "kwargs": {k: str(v) for k, v in kwargs.items()}
        }
        
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _should_cache(self, operation_name: str) -> bool:
        """Determine if operation should be cached"""
        policy = self.cache_policies.get(operation_name, {})
        return policy.get("enabled", False)
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # Convert to MB
        except:
            return 0.0
    
    def _estimate_token_usage(self, args: Tuple, kwargs: Dict, result: Any) -> int:
        """Estimate token usage for operation"""
        # Simple estimation based on text length
        total_text = ""
        
        for arg in args:
            if isinstance(arg, str):
                total_text += arg
            elif isinstance(arg, dict):
                total_text += str(arg)
        
        for value in kwargs.values():
            if isinstance(value, str):
                total_text += value
            elif isinstance(value, dict):
                total_text += str(value)
        
        if isinstance(result, str):
            total_text += result
        elif isinstance(result, dict):
            total_text += str(result)
        
        # Rough estimation: 4 characters per token
        return len(total_text) // 4
    
    def optimize_prompt_length(self, prompt: str, max_length: int = None) -> str:
        """Optimize prompt length while preserving meaning"""
        if not self.token_optimization_enabled:
            return prompt
        
        max_length = max_length or self.max_context_length
        estimated_tokens = len(prompt) // 4
        
        if estimated_tokens <= max_length:
            return prompt
        
        # Simple truncation with ellipsis
        target_chars = max_length * 4 - 100  # Leave buffer
        if len(prompt) > target_chars:
            truncated = prompt[:target_chars] + "..."
            logger.info(f"Truncated prompt from {len(prompt)} to {len(truncated)} characters")
            return truncated
        
        return prompt
    
    def generate_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """Generate performance optimization recommendations"""
        recommendations = []
        
        # Cache performance analysis
        cache_stats = self.cache.get_stats()
        if cache_stats["hit_rate"] < 0.5:
            recommendations.append({
                "type": "cache_optimization",
                "priority": "medium",
                "description": f"Low cache hit rate ({cache_stats['hit_rate']:.2%}). Consider adjusting cache policies.",
                "action": "Review cache policies and TTL settings"
            })
        
        # Performance analysis
        overall_stats = self.monitor.get_overall_stats()
        if overall_stats and overall_stats.get("overall_success_rate", 1.0) < 0.9:
            recommendations.append({
                "type": "reliability",
                "priority": "high",
                "description": f"Low success rate ({overall_stats['overall_success_rate']:.2%})",
                "action": "Investigate error patterns and add retry mechanisms"
            })
        
        if overall_stats and overall_stats.get("recent_avg_duration", 0) > 10.0:
            recommendations.append({
                "type": "performance",
                "priority": "medium",
                "description": f"High average response time ({overall_stats['recent_avg_duration']:.2f}s)",
                "action": "Consider caching, parallel processing, or model optimization"
            })
        
        # Memory usage analysis
        if overall_stats and overall_stats.get("memory_usage", {}).get("max", 0) > 1000:
            recommendations.append({
                "type": "memory",
                "priority": "medium",
                "description": "High memory usage detected",
                "action": "Review memory usage patterns and implement cleanup strategies"
            })
        
        return recommendations
    
    def get_comprehensive_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        return {
            "cache_stats": self.cache.get_stats(),
            "performance_stats": self.monitor.get_overall_stats(),
            "recent_alerts": self.monitor.get_recent_alerts(hours=24),
            "optimization_recommendations": self.generate_optimization_recommendations(),
            "configuration": {
                "token_optimization_enabled": self.token_optimization_enabled,
                "max_context_length": self.max_context_length,
                "cache_policies": self.cache_policies
            }
        }
    
    def reset_metrics(self) -> None:
        """Reset all performance metrics"""
        self.cache.clear()
        self.monitor = PerformanceMonitor(max_history=10000)
        self.optimization_recommendations = []
        logger.info("Performance metrics reset")

# Global performance optimizer instance
performance_optimizer = EnhancedPerformanceOptimizer()