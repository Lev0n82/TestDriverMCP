"""
Comprehensive test suite for Advanced Wait Strategies and Retry Logic.
Tests all functions, classes, and module-level functionality.
"""

import pytest
import asyncio
import time

import sys
sys.path.insert(0, '/home/ubuntu/testdriver-full/testdriver-mcp-full/src')

from reliability.wait_strategies import (
    WaitStrategy,
    RetryStrategy,
    WaitResult,
    RetryConfig,
    WaitStrategyValidator,
    AdaptiveWaitService,
    RetryOrchestrator,
    VisualStabilityWaiter,
    self_test_module
)


def test_validator_wait_result():
    """Test wait result validation."""
    validator = WaitStrategyValidator()
    
    # Valid result
    valid = WaitResult(success=True, duration=1.5, attempts=2, strategy_used="fixed")
    assert validator.validate_wait_result(valid)
    
    # Invalid: negative duration
    invalid = WaitResult(success=True, duration=-1.0, attempts=1, strategy_used="fixed")
    assert not validator.validate_wait_result(invalid)
    
    # Invalid: zero attempts
    invalid = WaitResult(success=True, duration=1.0, attempts=0, strategy_used="fixed")
    assert not validator.validate_wait_result(invalid)
    
    print("✓ Wait result validation works correctly")


def test_validator_retry_config():
    """Test retry config validation."""
    validator = WaitStrategyValidator()
    
    # Valid config
    valid = RetryConfig(max_attempts=3, initial_delay=1.0, max_delay=10.0)
    assert validator.validate_retry_config(valid)
    
    # Invalid: zero attempts
    invalid = RetryConfig(max_attempts=0)
    assert not validator.validate_retry_config(invalid)
    
    # Invalid: negative delay
    invalid = RetryConfig(initial_delay=-1.0)
    assert not validator.validate_retry_config(invalid)
    
    # Invalid: max < initial
    invalid = RetryConfig(initial_delay=10.0, max_delay=1.0)
    assert not validator.validate_retry_config(invalid)
    
    print("✓ Retry config validation works correctly")


def test_adaptive_wait_service():
    """Test adaptive wait service."""
    service = AdaptiveWaitService()
    
    # Record some wait times
    service.record_wait_time("page_load", 2.0)
    service.record_wait_time("page_load", 2.5)
    service.record_wait_time("page_load", 1.8)
    
    # Get recommendation
    recommended = service.get_recommended_wait("page_load")
    assert recommended > 0
    assert recommended >= 2.0  # Should be around 95th percentile + buffer
    
    # Test with no history
    default_wait = service.get_recommended_wait("unknown_op", default=5.0)
    assert default_wait == 5.0
    
    print(f"✓ Adaptive wait service works (recommended: {recommended:.2f}s)")


@pytest.mark.asyncio
async def test_retry_orchestrator_success():
    """Test retry orchestrator with successful operation."""
    config = RetryConfig(max_attempts=3, strategy=RetryStrategy.IMMEDIATE)
    orchestrator = RetryOrchestrator(config)
    
    call_count = 0
    
    async def successful_operation():
        nonlocal call_count
        call_count += 1
        return "success"
    
    result = await orchestrator.execute_with_retry(successful_operation, "test_op")
    
    assert result == "success"
    assert call_count == 1  # Should succeed on first try
    
    print("✓ Retry orchestrator handles successful operations")


@pytest.mark.asyncio
async def test_retry_orchestrator_eventual_success():
    """Test retry orchestrator with eventual success."""
    config = RetryConfig(max_attempts=3, strategy=RetryStrategy.IMMEDIATE)
    orchestrator = RetryOrchestrator(config)
    
    call_count = 0
    
    async def eventually_successful():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ValueError("Not yet")
        return "success"
    
    result = await orchestrator.execute_with_retry(eventually_successful, "test_op")
    
    assert result == "success"
    assert call_count == 3  # Should succeed on third try
    
    print("✓ Retry orchestrator handles eventual success")


@pytest.mark.asyncio
async def test_retry_orchestrator_failure():
    """Test retry orchestrator with persistent failure."""
    config = RetryConfig(max_attempts=3, strategy=RetryStrategy.IMMEDIATE)
    orchestrator = RetryOrchestrator(config)
    
    call_count = 0
    
    async def always_fails():
        nonlocal call_count
        call_count += 1
        raise ValueError("Always fails")
    
    with pytest.raises(ValueError, match="Always fails"):
        await orchestrator.execute_with_retry(always_fails, "test_op")
    
    assert call_count == 3  # Should try all attempts
    
    print("✓ Retry orchestrator handles persistent failures")


@pytest.mark.asyncio
async def test_retry_delay_calculation():
    """Test retry delay calculation strategies."""
    # Exponential backoff
    config = RetryConfig(
        max_attempts=4,
        strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
        initial_delay=1.0,
        backoff_factor=2.0
    )
    orchestrator = RetryOrchestrator(config)
    
    delays = [orchestrator._calculate_delay(i) for i in range(1, 5)]
    
    # Should be: 1.0, 2.0, 4.0, 8.0
    assert delays[0] == 1.0
    assert delays[1] == 2.0
    assert delays[2] == 4.0
    assert delays[3] == 8.0
    
    # Linear backoff
    config = RetryConfig(
        max_attempts=3,
        strategy=RetryStrategy.LINEAR_BACKOFF,
        initial_delay=2.0
    )
    orchestrator = RetryOrchestrator(config)
    
    delays = [orchestrator._calculate_delay(i) for i in range(1, 4)]
    
    # Should be: 2.0, 4.0, 6.0
    assert delays[0] == 2.0
    assert delays[1] == 4.0
    assert delays[2] == 6.0
    
    print("✓ Retry delay calculation works correctly")


@pytest.mark.asyncio
async def test_visual_stability_waiter():
    """Test visual stability waiter."""
    waiter = VisualStabilityWaiter(threshold=0.95, max_wait=5.0)
    
    screenshot_count = 0
    stable_screenshot = b'stable_image_data'
    
    async def get_screenshot():
        nonlocal screenshot_count
        screenshot_count += 1
        # Return changing screenshots first, then stable
        if screenshot_count < 3:
            return b'changing_' + str(screenshot_count).encode()
        return stable_screenshot
    
    result = await waiter.wait_for_stability(get_screenshot, check_interval=0.1)
    
    assert result.success is True
    assert result.attempts >= 3
    assert result.duration > 0
    
    print(f"✓ Visual stability waiter works ({result.attempts} attempts, {result.duration:.2f}s)")


@pytest.mark.asyncio
async def test_visual_stability_timeout():
    """Test visual stability timeout."""
    waiter = VisualStabilityWaiter(threshold=0.95, max_wait=1.0)
    
    screenshot_count = 0
    
    async def always_changing_screenshot():
        nonlocal screenshot_count
        screenshot_count += 1
        # Always return different screenshots
        return b'changing_' + str(screenshot_count).encode()
    
    result = await waiter.wait_for_stability(always_changing_screenshot, check_interval=0.1)
    
    assert result.success is False
    assert result.error is not None
    assert "timeout" in result.error.lower() or "Timeout" in result.error
    
    print("✓ Visual stability timeout works correctly")


def test_similarity_calculation():
    """Test screenshot similarity calculation."""
    waiter = VisualStabilityWaiter()
    
    # Identical screenshots
    img1 = b'test_image_data'
    img2 = b'test_image_data'
    similarity = waiter._calculate_similarity(img1, img2)
    assert similarity == 1.0
    
    # Completely different
    img1 = b'aaaaaaaaaa'
    img2 = b'bbbbbbbbbb'
    similarity = waiter._calculate_similarity(img1, img2)
    assert similarity == 0.0
    
    # Partially similar
    img1 = b'aaaaaaaaaa'
    img2 = b'aaaaabbbbb'
    similarity = waiter._calculate_similarity(img1, img2)
    assert 0.0 < similarity < 1.0
    
    print(f"✓ Similarity calculation works correctly")


def test_module_self_test():
    """Test module-level self-test."""
    success = self_test_module()
    assert success is True
    print("✓ Module self-test passed")


async def run_all_tests():
    """Run all wait strategy tests."""
    print("\n" + "="*60)
    print("Advanced Wait Strategies - Comprehensive Test Suite")
    print("="*60)
    
    tests = [
        ("Validator - Wait Result", test_validator_wait_result),
        ("Validator - Retry Config", test_validator_retry_config),
        ("Adaptive Wait Service", test_adaptive_wait_service),
        ("Retry - Successful Operation", test_retry_orchestrator_success),
        ("Retry - Eventual Success", test_retry_orchestrator_eventual_success),
        ("Retry - Persistent Failure", test_retry_orchestrator_failure),
        ("Retry - Delay Calculation", test_retry_delay_calculation),
        ("Visual Stability - Success", test_visual_stability_waiter),
        ("Visual Stability - Timeout", test_visual_stability_timeout),
        ("Similarity Calculation", test_similarity_calculation),
        ("Module Self-Test", test_module_self_test),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            if asyncio.iscoroutinefunction(test_func):
                await test_func()
            else:
                test_func()
            passed += 1
            print(f"✓ {name}: PASSED")
        except Exception as e:
            failed += 1
            print(f"✗ {name}: FAILED - {e}")
    
    print("\n" + "="*60)
    print(f"Total: {len(tests)} tests")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Success Rate: {passed/len(tests)*100:.1f}%")
    print("="*60)
    
    return passed, failed


if __name__ == "__main__":
    asyncio.run(run_all_tests())
