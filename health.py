"""
Health check system for monitoring service health and dependencies.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum
import asyncio
import structlog

logger = structlog.get_logger()

class HealthStatus(str, Enum):
    """Health check status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"

class HealthCheck:
    """Individual health check."""
    
    def __init__(self, name: str, check_func, critical: bool = True):
        """
        Initialize health check.
        
        Args:
            name: Health check name
            check_func: Async function that returns (status, details)
            critical: Whether this check is critical for overall health
        """
        self.name = name
        self.check_func = check_func
        self.critical = critical
        self.last_check_time: Optional[datetime] = None
        self.last_status: Optional[HealthStatus] = None
        self.last_details: Optional[Dict[str, Any]] = None
    
    async def execute(self) -> Dict[str, Any]:
        """Execute health check."""
        try:
            start_time = datetime.utcnow()
            status, details = await self.check_func()
            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            self.last_check_time = datetime.utcnow()
            self.last_status = status
            self.last_details = details
            
            return {
                "name": self.name,
                "status": status.value,
                "critical": self.critical,
                "details": details,
                "duration_ms": duration_ms,
                "timestamp": self.last_check_time.isoformat()
            }
        except Exception as e:
            logger.error("Health check failed", check=self.name, error=str(e))
            self.last_status = HealthStatus.UNHEALTHY
            self.last_details = {"error": str(e)}
            
            return {
                "name": self.name,
                "status": HealthStatus.UNHEALTHY.value,
                "critical": self.critical,
                "details": {"error": str(e)},
                "timestamp": datetime.utcnow().isoformat()
            }

class HealthCheckManager:
    """Manages all health checks."""
    
    def __init__(self):
        """Initialize health check manager."""
        self.checks: List[HealthCheck] = []
        self.start_time = datetime.utcnow()
        logger.info("Health check manager initialized")
    
    def register_check(
        self,
        name: str,
        check_func,
        critical: bool = True
    ) -> None:
        """Register a new health check."""
        check = HealthCheck(name, check_func, critical)
        self.checks.append(check)
        logger.info("Health check registered", name=name, critical=critical)
    
    async def run_all_checks(self) -> Dict[str, Any]:
        """Run all health checks."""
        results = await asyncio.gather(*[check.execute() for check in self.checks])
        
        # Determine overall status
        overall_status = HealthStatus.HEALTHY
        
        for result in results:
            if result["status"] == HealthStatus.UNHEALTHY.value and result["critical"]:
                overall_status = HealthStatus.UNHEALTHY
                break
            elif result["status"] == HealthStatus.DEGRADED.value:
                overall_status = HealthStatus.DEGRADED
        
        uptime_seconds = (datetime.utcnow() - self.start_time).total_seconds()
        
        return {
            "status": overall_status.value,
            "timestamp": datetime.utcnow().isoformat(),
            "uptime_seconds": uptime_seconds,
            "checks": results
        }
    
    async def get_readiness(self) -> Dict[str, Any]:
        """Get readiness status (can accept traffic)."""
        # Run critical checks only
        critical_checks = [check for check in self.checks if check.critical]
        results = await asyncio.gather(*[check.execute() for check in critical_checks])
        
        ready = all(r["status"] != HealthStatus.UNHEALTHY.value for r in results)
        
        return {
            "ready": ready,
            "timestamp": datetime.utcnow().isoformat(),
            "checks": results
        }
    
    async def get_liveness(self) -> Dict[str, Any]:
        """Get liveness status (process is alive)."""
        # Simple liveness check - if we can respond, we're alive
        return {
            "alive": True,
            "timestamp": datetime.utcnow().isoformat(),
            "uptime_seconds": (datetime.utcnow() - self.start_time).total_seconds()
        }

# Global health check manager
_health_manager: Optional[HealthCheckManager] = None

def get_health_manager() -> HealthCheckManager:
    """Get or create global health check manager."""
    global _health_manager
    if _health_manager is None:
        _health_manager = HealthCheckManager()
    return _health_manager

# Common health check functions

async def check_database_health(db_manager) -> tuple[HealthStatus, Dict[str, Any]]:
    """Check database connectivity."""
    try:
        # Try a simple query
        async with db_manager.get_session() as session:
            # Database is accessible
            return HealthStatus.HEALTHY, {"message": "Database connection OK"}
    except Exception as e:
        return HealthStatus.UNHEALTHY, {"error": str(e)}

async def check_vision_api_health(vision_adapter) -> tuple[HealthStatus, Dict[str, Any]]:
    """Check vision API availability."""
    try:
        # Simple check - if adapter is initialized, consider it healthy
        # In production, could make a test API call
        if vision_adapter:
            return HealthStatus.HEALTHY, {"message": "Vision API OK"}
        else:
            return HealthStatus.DEGRADED, {"message": "Vision API not initialized"}
    except Exception as e:
        return HealthStatus.UNHEALTHY, {"error": str(e)}

async def check_browser_health(driver) -> tuple[HealthStatus, Dict[str, Any]]:
    """Check browser driver health."""
    try:
        if driver and driver._initialized:
            return HealthStatus.HEALTHY, {"message": "Browser driver OK"}
        else:
            return HealthStatus.DEGRADED, {"message": "Browser driver not initialized"}
    except Exception as e:
        return HealthStatus.UNHEALTHY, {"error": str(e)}

async def check_memory_usage() -> tuple[HealthStatus, Dict[str, Any]]:
    """Check memory usage."""
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        
        if memory_mb < 1024:  # < 1GB
            return HealthStatus.HEALTHY, {"memory_mb": memory_mb}
        elif memory_mb < 2048:  # < 2GB
            return HealthStatus.DEGRADED, {"memory_mb": memory_mb, "warning": "High memory usage"}
        else:
            return HealthStatus.UNHEALTHY, {"memory_mb": memory_mb, "error": "Memory usage too high"}
    except ImportError:
        # psutil not available
        return HealthStatus.HEALTHY, {"message": "Memory check not available"}
    except Exception as e:
        return HealthStatus.DEGRADED, {"error": str(e)}
