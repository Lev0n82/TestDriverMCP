"""
Monitoring, health checks, and Prometheus metrics for production observability.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
import time
import structlog
from prometheus_client import Counter, Histogram, Gauge, Info, CollectorRegistry, generate_latest

logger = structlog.get_logger()

# Create custom registry
registry = CollectorRegistry()

# System info
system_info = Info(
    'testdriver_system',
    'TestDriver MCP Framework system information',
    registry=registry
)

# Counters
healing_attempts_total = Counter(
    'testdriver_healing_attempts_total',
    'Total number of healing attempts',
    ['strategy', 'success'],
    registry=registry
)

test_executions_total = Counter(
    'testdriver_test_executions_total',
    'Total number of test executions',
    ['status'],
    registry=registry
)

vision_api_calls_total = Counter(
    'testdriver_vision_api_calls_total',
    'Total number of vision API calls',
    ['model', 'success'],
    registry=registry
)

database_operations_total = Counter(
    'testdriver_database_operations_total',
    'Total number of database operations',
    ['operation', 'success'],
    registry=registry
)

# Histograms
healing_duration_seconds = Histogram(
    'testdriver_healing_duration_seconds',
    'Time spent on healing attempts',
    ['strategy'],
    registry=registry,
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0)
)

test_execution_duration_seconds = Histogram(
    'testdriver_test_execution_duration_seconds',
    'Time spent executing tests',
    registry=registry,
    buckets=(1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0)
)

vision_api_latency_seconds = Histogram(
    'testdriver_vision_api_latency_seconds',
    'Vision API call latency',
    ['model'],
    registry=registry,
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0)
)

database_query_duration_seconds = Histogram(
    'testdriver_database_query_duration_seconds',
    'Database query duration',
    ['operation'],
    registry=registry,
    buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0)
)

# Gauges
active_test_executions = Gauge(
    'testdriver_active_test_executions',
    'Number of currently active test executions',
    registry=registry
)

element_stability_score = Gauge(
    'testdriver_element_stability_score',
    'Current average element stability score',
    registry=registry
)

healing_success_rate = Gauge(
    'testdriver_healing_success_rate',
    'Current healing success rate',
    registry=registry
)

database_connections = Gauge(
    'testdriver_database_connections',
    'Number of active database connections',
    registry=registry
)

class MetricsCollector:
    """Collects and manages application metrics."""
    
    def __init__(self):
        """Initialize metrics collector."""
        self.start_time = time.time()
        
        # Set system info
        system_info.info({
            'version': '2.0.0',
            'component': 'testdriver-mcp',
            'environment': 'production'
        })
        
        logger.info("Metrics collector initialized")
    
    def record_healing_attempt(
        self,
        strategy: str,
        success: bool,
        duration: float
    ) -> None:
        """Record a healing attempt."""
        healing_attempts_total.labels(
            strategy=strategy,
            success=str(success).lower()
        ).inc()
        
        healing_duration_seconds.labels(strategy=strategy).observe(duration)
        
        logger.debug("Healing attempt recorded", strategy=strategy, success=success, duration=duration)
    
    def record_test_execution(
        self,
        status: str,
        duration: float
    ) -> None:
        """Record a test execution."""
        test_executions_total.labels(status=status).inc()
        test_execution_duration_seconds.observe(duration)
        
        logger.debug("Test execution recorded", status=status, duration=duration)
    
    def record_vision_api_call(
        self,
        model: str,
        success: bool,
        latency: float
    ) -> None:
        """Record a vision API call."""
        vision_api_calls_total.labels(
            model=model,
            success=str(success).lower()
        ).inc()
        
        vision_api_latency_seconds.labels(model=model).observe(latency)
        
        logger.debug("Vision API call recorded", model=model, success=success, latency=latency)
    
    def record_database_operation(
        self,
        operation: str,
        success: bool,
        duration: float
    ) -> None:
        """Record a database operation."""
        database_operations_total.labels(
            operation=operation,
            success=str(success).lower()
        ).inc()
        
        database_query_duration_seconds.labels(operation=operation).observe(duration)
        
        logger.debug("Database operation recorded", operation=operation, success=success, duration=duration)
    
    def update_active_executions(self, count: int) -> None:
        """Update active test executions gauge."""
        active_test_executions.set(count)
    
    def update_element_stability(self, score: float) -> None:
        """Update element stability score gauge."""
        element_stability_score.set(score)
    
    def update_healing_success_rate(self, rate: float) -> None:
        """Update healing success rate gauge."""
        healing_success_rate.set(rate)
    
    def update_database_connections(self, count: int) -> None:
        """Update database connections gauge."""
        database_connections.set(count)
    
    def get_metrics(self) -> bytes:
        """Get Prometheus metrics in text format."""
        return generate_latest(registry)

# Global metrics collector instance
_metrics_collector: Optional[MetricsCollector] = None

def get_metrics_collector() -> MetricsCollector:
    """Get or create global metrics collector instance."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector
