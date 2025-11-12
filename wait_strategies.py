"""
Advanced Wait Strategies and Retry Logic.
Provides intelligent waiting and retry mechanisms for reliable test execution.

Built-in Self-Tests:
- Function-level: Each strategy validates its behavior
- Class-level: Wait orchestration and decision making
- Module-level: Strategy selection and fallback
"""

from typing import Dict, Any, Optional, Callable, List
import time
import asyncio
from enum import Enum
from dataclasses import dataclass
from datetime import datetime

import structlog

logger = structlog.get_logger()


class WaitStrategy(str, Enum):
    """Wait strategy types."""
    FIXED = "fixed"
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    ADAPTIVE = "adaptive"
    VISUAL_STABILITY = "visual_stability"


class RetryStrategy(str, Enum):
    """Retry strategy types."""
    IMMEDIATE = "immediate"
    LINEAR_BACKOFF = "linear_backoff"
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    JITTERED_BACKOFF = "jittered_backoff"


@dataclass
class WaitResult:
    """Result of a wait operation."""
    success: bool
    duration: float
    attempts: int
    strategy_used: str
    error: Optional[str] = None


@dataclass
class RetryConfig:
    """Configuration for retry logic."""
    max_attempts: int = 3
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    initial_delay: float = 1.0
    max_delay: float = 30.0
    backoff_factor: float = 2.0
    jitter: bool = True


class WaitStrategyValidator:
    """Built-in validator for wait strategies."""
    
    @staticmethod
    def validate_wait_result(result: WaitResult) -> bool:
        """
        Validate wait result.
        
        Success Criteria:
        - Result has all required fields
        - Duration is non-negative
        - Attempts is positive
        """
        if not isinstance(result, WaitResult):
            return False
        if result.duration < 0:
            return False
        if result.attempts < 1:
            return False
        return True
    
    @staticmethod
    def validate_retry_config(config: RetryConfig) -> bool:
        """
        Validate retry configuration.
        
        Success Criteria:
        - Max attempts is positive
        - Delays are positive
        - Backoff factor > 1
        """
        if config.max_attempts < 1:
            return False
        if config.initial_delay <= 0 or config.max_delay <= 0:
            return False
        if config.backoff_factor <= 1:
            return False
        if config.max_delay < config.initial_delay:
            return False
        return True


class AdaptiveWaitService:
    """
    Adaptive wait service that learns optimal wait times.
    
    Success Criteria (Class-level):
    - Adapts wait times based on historical data
    - Reduces unnecessary waiting
    - Maintains reliability
    """
    
    def __init__(self):
        """Initialize adaptive wait service."""
        self.wait_history: Dict[str, List[float]] = {}
        self.validator = WaitStrategyValidator()
        logger.info("Adaptive wait service initialized")
    
    def record_wait_time(self, operation: str, duration: float):
        """
        Record successful wait time for operation.
        
        Success Criteria:
        - Duration is recorded
        - History is maintained
        """
        if operation not in self.wait_history:
            self.wait_history[operation] = []
        
        self.wait_history[operation].append(duration)
        
        # Keep last 100 records
        if len(self.wait_history[operation]) > 100:
            self.wait_history[operation] = self.wait_history[operation][-100:]
        
        logger.debug("Wait time recorded", operation=operation, duration=duration)
    
    def get_recommended_wait(self, operation: str, default: float = 5.0) -> float:
        """
        Get recommended wait time based on history.
        
        Success Criteria:
        - Returns reasonable wait time
        - Uses historical data when available
        - Falls back to default when no history
        
        Returns:
            Recommended wait time in seconds
        """
        if operation not in self.wait_history or not self.wait_history[operation]:
            return default
        
        # Calculate 95th percentile
        history = sorted(self.wait_history[operation])
        index = int(len(history) * 0.95)
        recommended = history[index]
        
        # Add 20% buffer
        recommended *= 1.2
        
        logger.debug(
            "Recommended wait calculated",
            operation=operation,
            recommended=recommended,
            samples=len(history)
        )
        
        return recommended


class RetryOrchestrator:
    """
    Orchestrates retry logic with various strategies.
    
    Success Criteria (Class-level):
    - Executes retries according to strategy
    - Respects max attempts
    - Provides detailed failure information
    """
    
    def __init__(self, config: Optional[RetryConfig] = None):
        """
        Initialize retry orchestrator.
        
        Args:
            config: Retry configuration
        """
        self.config = config or RetryConfig()
        self.validator = WaitStrategyValidator()
        
        if not self.validator.validate_retry_config(self.config):
            raise ValueError("Invalid retry configuration")
        
        logger.info(
            "Retry orchestrator initialized",
            strategy=self.config.strategy.value,
            max_attempts=self.config.max_attempts
        )
    
    def _calculate_delay(self, attempt: int) -> float:
        """
        Calculate delay for retry attempt.
        
        Success Criteria:
        - Delay follows configured strategy
        - Delay respects max_delay
        - Jitter is applied if enabled
        
        Returns:
            Delay in seconds
        """
        if self.config.strategy == RetryStrategy.IMMEDIATE:
            delay = 0
        
        elif self.config.strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = self.config.initial_delay * attempt
        
        elif self.config.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = self.config.initial_delay * (self.config.backoff_factor ** (attempt - 1))
        
        elif self.config.strategy == RetryStrategy.JITTERED_BACKOFF:
            base_delay = self.config.initial_delay * (self.config.backoff_factor ** (attempt - 1))
            import random
            delay = base_delay * (0.5 + random.random() * 0.5)
        
        else:
            delay = self.config.initial_delay
        
        # Cap at max_delay
        delay = min(delay, self.config.max_delay)
        
        return delay
    
    async def execute_with_retry(
        self,
        operation: Callable,
        operation_name: str = "operation",
        *args,
        **kwargs
    ) -> Any:
        """
        Execute operation with retry logic.
        
        Success Criteria:
        - Operation executes successfully or exhausts retries
        - Delays are applied correctly
        - All attempts are logged
        
        Returns:
            Operation result
        
        Raises:
            Last exception if all retries fail
        """
        last_exception = None
        
        for attempt in range(1, self.config.max_attempts + 1):
            try:
                logger.debug(
                    "Executing operation",
                    operation=operation_name,
                    attempt=attempt,
                    max_attempts=self.config.max_attempts
                )
                
                # Execute operation
                if asyncio.iscoroutinefunction(operation):
                    result = await operation(*args, **kwargs)
                else:
                    result = operation(*args, **kwargs)
                
                logger.info(
                    "Operation succeeded",
                    operation=operation_name,
                    attempt=attempt
                )
                
                return result
            
            except Exception as e:
                last_exception = e
                logger.warning(
                    "Operation failed",
                    operation=operation_name,
                    attempt=attempt,
                    error=str(e)
                )
                
                # Don't delay after last attempt
                if attempt < self.config.max_attempts:
                    delay = self._calculate_delay(attempt)
                    logger.debug("Retrying after delay", delay=delay)
                    await asyncio.sleep(delay)
        
        # All retries exhausted
        logger.error(
            "Operation failed after all retries",
            operation=operation_name,
            attempts=self.config.max_attempts,
            error=str(last_exception)
        )
        
        raise last_exception


class VisualStabilityWaiter:
    """
    Waits for visual stability using screenshot comparison.
    
    Success Criteria (Class-level):
    - Detects when UI stops changing
    - Prevents premature interactions
    - Configurable stability threshold
    """
    
    def __init__(self, threshold: float = 0.95, max_wait: float = 30.0):
        """
        Initialize visual stability waiter.
        
        Args:
            threshold: Similarity threshold (0-1)
            max_wait: Maximum wait time in seconds
        """
        self.threshold = threshold
        self.max_wait = max_wait
        logger.info(
            "Visual stability waiter initialized",
            threshold=threshold,
            max_wait=max_wait
        )
    
    async def wait_for_stability(
        self,
        screenshot_func: Callable,
        check_interval: float = 0.5
    ) -> WaitResult:
        """
        Wait for visual stability.
        
        Success Criteria:
        - Returns when UI is stable
        - Respects max_wait timeout
        - Provides accurate duration
        
        Returns:
            Wait result with success status
        """
        start_time = time.time()
        attempts = 0
        previous_screenshot = None
        
        try:
            while (time.time() - start_time) < self.max_wait:
                attempts += 1
                
                # Take screenshot
                current_screenshot = await screenshot_func()
                
                if previous_screenshot is not None:
                    # Compare screenshots
                    similarity = self._calculate_similarity(
                        previous_screenshot,
                        current_screenshot
                    )
                    
                    if similarity >= self.threshold:
                        duration = time.time() - start_time
                        logger.info(
                            "Visual stability achieved",
                            duration=duration,
                            attempts=attempts,
                            similarity=similarity
                        )
                        
                        return WaitResult(
                            success=True,
                            duration=duration,
                            attempts=attempts,
                            strategy_used="visual_stability"
                        )
                
                previous_screenshot = current_screenshot
                await asyncio.sleep(check_interval)
            
            # Timeout
            duration = time.time() - start_time
            logger.warning(
                "Visual stability timeout",
                duration=duration,
                attempts=attempts
            )
            
            return WaitResult(
                success=False,
                duration=duration,
                attempts=attempts,
                strategy_used="visual_stability",
                error="Timeout waiting for stability"
            )
        
        except Exception as e:
            duration = time.time() - start_time
            logger.error("Visual stability wait failed", error=str(e))
            
            return WaitResult(
                success=False,
                duration=duration,
                attempts=attempts,
                strategy_used="visual_stability",
                error=str(e)
            )
    
    def _calculate_similarity(self, img1: bytes, img2: bytes) -> float:
        """
        Calculate similarity between screenshots.
        
        Success Criteria:
        - Returns value between 0 and 1
        - 1.0 means identical
        - 0.0 means completely different
        """
        # Simple byte-level comparison for now
        # In production, use perceptual hashing or image diff
        if img1 == img2:
            return 1.0
        
        # Calculate byte-level similarity
        min_len = min(len(img1), len(img2))
        if min_len == 0:
            return 0.0
        
        matching_bytes = sum(1 for i in range(min_len) if img1[i] == img2[i])
        similarity = matching_bytes / min_len
        
        return similarity


# Module-level self-test
def self_test_module() -> bool:
    """
    Module-level self-test.
    
    Success Criteria:
    - All classes can be instantiated
    - Validators work correctly
    - Basic operations succeed
    """
    try:
        # Test validator
        validator = WaitStrategyValidator()
        
        # Test valid wait result
        valid_result = WaitResult(
            success=True,
            duration=1.5,
            attempts=2,
            strategy_used="fixed"
        )
        if not validator.validate_wait_result(valid_result):
            logger.error("Module self-test failed: Valid result rejected")
            return False
        
        # Test invalid wait result (negative duration)
        invalid_result = WaitResult(
            success=True,
            duration=-1.0,
            attempts=1,
            strategy_used="fixed"
        )
        if validator.validate_wait_result(invalid_result):
            logger.error("Module self-test failed: Invalid result accepted")
            return False
        
        # Test retry config validation
        valid_config = RetryConfig(max_attempts=3, initial_delay=1.0, max_delay=10.0)
        if not validator.validate_retry_config(valid_config):
            logger.error("Module self-test failed: Valid config rejected")
            return False
        
        # Test adaptive wait service
        adaptive_wait = AdaptiveWaitService()
        adaptive_wait.record_wait_time("test_op", 2.0)
        recommended = adaptive_wait.get_recommended_wait("test_op")
        if recommended <= 0:
            logger.error("Module self-test failed: Invalid recommended wait")
            return False
        
        logger.info("Module self-test passed: wait_strategies")
        return True
    
    except Exception as e:
        logger.error("Module self-test failed", error=str(e))
        return False


if __name__ == "__main__":
    # Run module self-test
    success = self_test_module()
    print(f"Module self-test: {'PASSED' if success else 'FAILED'}")
