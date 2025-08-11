"""
Observability, logging, and metrics for MiniAI
"""

import structlog
import logging
import time
import uuid
from typing import Dict, Any, Optional
from contextlib import contextmanager
from abc import ABC, abstractmethod


# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)


def get_logger(name: str) -> structlog.BoundLogger:
    """Get a structured logger"""
    return structlog.get_logger(name)


class MetricsCollector(ABC):
    """Abstract metrics collector"""
    
    @abstractmethod
    def increment(self, metric: str, tags: Optional[Dict[str, str]] = None) -> None:
        pass
    
    @abstractmethod
    def histogram(self, metric: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        pass
    
    @abstractmethod
    def gauge(self, metric: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        pass


class InMemoryMetrics(MetricsCollector):
    """In-memory metrics collector for development"""
    
    def __init__(self):
        self.counters: Dict[str, int] = {}
        self.histograms: Dict[str, list] = {}
        self.gauges: Dict[str, float] = {}
    
    def increment(self, metric: str, tags: Optional[Dict[str, str]] = None) -> None:
        key = self._make_key(metric, tags)
        self.counters[key] = self.counters.get(key, 0) + 1
    
    def histogram(self, metric: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        key = self._make_key(metric, tags)
        if key not in self.histograms:
            self.histograms[key] = []
        self.histograms[key].append(value)
    
    def gauge(self, metric: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        key = self._make_key(metric, tags)
        self.gauges[key] = value
    
    def _make_key(self, metric: str, tags: Optional[Dict[str, str]]) -> str:
        if tags:
            tag_str = ",".join(f"{k}={v}" for k, v in sorted(tags.items()))
            return f"{metric}[{tag_str}]"
        return metric
    
    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary"""
        summary = {
            "counters": dict(self.counters),
            "gauges": dict(self.gauges),
            "histograms": {}
        }
        
        for key, values in self.histograms.items():
            if values:
                summary["histograms"][key] = {
                    "count": len(values),
                    "sum": sum(values),
                    "avg": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values)
                }
        
        return summary


# Global metrics instance
_metrics: Optional[MetricsCollector] = None


def init_metrics(collector: MetricsCollector):
    """Initialize global metrics collector"""
    global _metrics
    _metrics = collector


def get_metrics() -> MetricsCollector:
    """Get global metrics collector"""
    global _metrics
    if _metrics is None:
        _metrics = InMemoryMetrics()
    return _metrics


@contextmanager
def trace_operation(operation_name: str, tags: Optional[Dict[str, str]] = None):
    """Context manager for tracing operations"""
    trace_id = str(uuid.uuid4())[:8]
    logger = get_logger("trace")
    metrics = get_metrics()
    
    start_time = time.time()
    
    logger.info(
        f"{operation_name} started",
        extra={"trace_id": trace_id, "operation": operation_name, **(tags or {})}
    )
    
    try:
        yield trace_id
        duration = time.time() - start_time
        
        metrics.histogram(f"{operation_name}.duration", duration, tags)
        metrics.increment(f"{operation_name}.success", tags)
        
        logger.info(
            f"{operation_name} completed",
            extra={
                "trace_id": trace_id, 
                "operation": operation_name,
                "duration": duration,
                **(tags or {})
            }
        )
        
    except Exception as e:
        duration = time.time() - start_time
        
        metrics.increment(f"{operation_name}.error", tags)
        
        logger.error(
            f"{operation_name} failed",
            extra={
                "trace_id": trace_id,
                "operation": operation_name,
                "duration": duration,
                "error": str(e),
                **(tags or {})
            }
        )
        raise
