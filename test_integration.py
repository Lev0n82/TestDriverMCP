"""
Comprehensive integration tests for TestDriver MCP Framework v2.0.
Tests all 6 production-critical features working together.
"""

import pytest
import asyncio
from datetime import datetime
import os

# Import all components
from storage.database import init_database, get_database_manager
from storage.persistent_store import PersistentMemoryStore
from execution.playwright_driver import PlaywrightDriver
from vision.openai_adapter import OpenAIVisionAdapter
from self_healing.healing_strategies import execute_healing_strategy, VisualSimilarityStrategy
from monitoring.metrics import get_metrics_collector
from monitoring.health import get_health_manager, check_database_health, check_browser_health
from models import HealingEvent, HealingStrategy, ValidationMethod

@pytest.mark.asyncio
async def test_database_integration():
    """Test 1: Database persistence and retrieval."""
    print("\n=== Test 1: Database Integration ===")
    
    # Initialize database (SQLite for testing)
    db = await init_database("sqlite:///./test_integration.db")
    
    # Create persistent store
    store = PersistentMemoryStore({"database_url": "sqlite:///./test_integration.db"})
    await store.initialize()
    
    # Create and store a healing event
    event = HealingEvent(
        event_id="test-event-1",
        test_id="test-001",
        execution_id="exec-001",
        element_id="button-login",
        original_locator={"css": "#old-button"},
        failure_reason="Element not found",
        healing_strategy=HealingStrategy.VISUAL_SIMILARITY,
        new_locator={"css": "#new-button"},
        confidence_score=0.92,
        healing_successful=True,
        validation_method=ValidationMethod.VISUAL_VERIFICATION
    )
    
    await store.store_healing_event(event)
    
    # Retrieve event
    retrieved = await store.get_healing_event("test-event-1")
    
    assert retrieved is not None
    assert retrieved.event_id == "test-event-1"
    assert retrieved.confidence_score == 0.92
    assert retrieved.healing_successful == True
    
    # Get statistics
    stats = await store.get_statistics()
    assert stats["total_healings"] >= 1
    
    await store.close()
    
    print("✓ Database integration test PASSED")
    print(f"  - Event stored and retrieved successfully")
    print(f"  - Statistics: {stats}")
    
    return True

@pytest.mark.asyncio
async def test_playwright_integration():
    """Test 2: Playwright browser automation."""
    print("\n=== Test 2: Playwright Integration ===")
    
    # Initialize Playwright driver
    driver = PlaywrightDriver({
        "browser": "chromium",
        "headless": True
    })
    
    await driver.initialize()
    
    # Navigate to a test page
    await driver.navigate("https://example.com")
    
    # Get current URL
    url = await driver.get_current_url()
    assert "example.com" in url
    
    # Take screenshot
    screenshot = await driver.take_screenshot()
    assert len(screenshot) > 0
    
    # Check element visibility
    h1_visible = await driver.is_visible({"css": "h1"})
    assert h1_visible == True
    
    # Get text content
    text = await driver.get_text({"css": "h1"})
    assert text is not None
    assert len(text) > 0
    
    await driver.close()
    
    print("✓ Playwright integration test PASSED")
    print(f"  - Navigated to: {url}")
    print(f"  - Screenshot size: {len(screenshot)} bytes")
    print(f"  - H1 text: {text}")
    
    return True

@pytest.mark.asyncio
async def test_vision_api_integration():
    """Test 3: OpenAI Vision API integration."""
    print("\n=== Test 3: Vision API Integration ===")
    
    # Check if API key is available
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠ SKIPPED: OPENAI_API_KEY not set")
        return True
    
    # Initialize vision adapter
    vision = OpenAIVisionAdapter({
        "api_key": api_key,
        "model": "gpt-4.1-mini"
    })
    
    # Create a simple test screenshot (using Playwright)
    driver = PlaywrightDriver({"headless": True})
    await driver.initialize()
    await driver.navigate("https://example.com")
    screenshot = await driver.take_screenshot()
    await driver.close()
    
    # Test element detection
    result = await vision.detect_element(
        screenshot=screenshot,
        element_description="The main heading on the page",
        context={"previous_locator": {"css": "h1"}}
    )
    
    assert result.get("success") == True
    assert result.get("confidence", 0) > 0.5
    assert "locator" in result
    
    print("✓ Vision API integration test PASSED")
    print(f"  - Detection success: {result.get('success')}")
    print(f"  - Confidence: {result.get('confidence', 0):.2f}")
    print(f"  - Locator: {result.get('locator')}")
    
    return True

@pytest.mark.asyncio
async def test_healing_strategy_integration():
    """Test 4: Self-healing strategy integration."""
    print("\n=== Test 4: Healing Strategy Integration ===")
    
    # Initialize components
    driver = PlaywrightDriver({"headless": True})
    await driver.initialize()
    await driver.navigate("https://example.com")
    
    # Mock vision adapter (to avoid API calls in testing)
    class MockVisionAdapter:
        async def detect_element(self, screenshot, element_description, context=None):
            return {
                "success": True,
                "locator": {"css": "h1"},
                "confidence": 0.88,
                "explanation": "Mock detection"
            }
    
    vision = MockVisionAdapter()
    
    # Test visual similarity strategy
    strategy = VisualSimilarityStrategy()
    result = await strategy.heal(
        driver=driver,
        vision_adapter=vision,
        original_locator={"css": "#nonexistent"},
        element_description="Main heading",
        context={"failure_reason": "Element not found"}
    )
    
    assert result.get("success") == True
    assert result.get("confidence", 0) > 0.8
    assert "new_locator" in result
    
    await driver.close()
    
    print("✓ Healing strategy integration test PASSED")
    print(f"  - Healing success: {result.get('success')}")
    print(f"  - New locator: {result.get('new_locator')}")
    print(f"  - Confidence: {result.get('confidence', 0):.2f}")
    
    return True

@pytest.mark.asyncio
async def test_monitoring_integration():
    """Test 5: Monitoring and metrics integration."""
    print("\n=== Test 5: Monitoring Integration ===")
    
    # Get metrics collector
    metrics = get_metrics_collector()
    
    # Record some metrics
    metrics.record_healing_attempt("visual_similarity", True, 1.5)
    metrics.record_test_execution("passed", 5.2)
    metrics.record_vision_api_call("gpt-4.1-mini", True, 0.8)
    metrics.record_database_operation("insert", True, 0.05)
    
    # Update gauges
    metrics.update_active_executions(3)
    metrics.update_element_stability(0.92)
    metrics.update_healing_success_rate(0.85)
    
    # Get metrics output
    metrics_output = metrics.get_metrics()
    assert len(metrics_output) > 0
    assert b"testdriver_healing_attempts_total" in metrics_output
    
    print("✓ Monitoring integration test PASSED")
    print(f"  - Metrics collected: {len(metrics_output)} bytes")
    print(f"  - Sample metrics recorded successfully")
    
    return True

@pytest.mark.asyncio
async def test_health_checks_integration():
    """Test 6: Health checks integration."""
    print("\n=== Test 6: Health Checks Integration ===")
    
    # Get health manager
    health = get_health_manager()
    
    # Register health checks
    db = get_database_manager("sqlite:///./test_integration.db")
    
    health.register_check(
        "database",
        lambda: check_database_health(db),
        critical=True
    )
    
    # Run all checks
    health_result = await health.run_all_checks()
    
    assert health_result["status"] in ["healthy", "degraded", "unhealthy"]
    assert "checks" in health_result
    assert len(health_result["checks"]) > 0
    
    # Test readiness
    readiness = await health.get_readiness()
    assert "ready" in readiness
    
    # Test liveness
    liveness = await health.get_liveness()
    assert liveness["alive"] == True
    
    print("✓ Health checks integration test PASSED")
    print(f"  - Overall status: {health_result['status']}")
    print(f"  - Checks run: {len(health_result['checks'])}")
    print(f"  - Ready: {readiness['ready']}")
    print(f"  - Alive: {liveness['alive']}")
    
    return True

@pytest.mark.asyncio
async def test_end_to_end_workflow():
    """Test 7: Complete end-to-end workflow."""
    print("\n=== Test 7: End-to-End Workflow ===")
    
    # Initialize all components
    db = await init_database("sqlite:///./test_e2e.db")
    store = PersistentMemoryStore({"database_url": "sqlite:///./test_e2e.db"})
    await store.initialize()
    
    driver = PlaywrightDriver({"headless": True})
    await driver.initialize()
    
    metrics = get_metrics_collector()
    
    # Simulate a test execution with healing
    print("  1. Navigating to test page...")
    await driver.navigate("https://example.com")
    
    print("  2. Taking screenshot...")
    screenshot = await driver.take_screenshot()
    
    print("  3. Simulating element failure...")
    original_locator = {"css": "#nonexistent-button"}
    is_visible = await driver.is_visible(original_locator)
    assert is_visible == False  # Element should not exist
    
    print("  4. Attempting self-healing...")
    # Mock vision adapter
    class MockVision:
        async def detect_element(self, screenshot, element_description, context=None):
            return {
                "success": True,
                "locator": {"css": "h1"},  # Use existing element
                "confidence": 0.90,
                "explanation": "Found alternative element"
            }
    
    vision = MockVision()
    strategy = VisualSimilarityStrategy()
    
    healing_result = await strategy.heal(
        driver=driver,
        vision_adapter=vision,
        original_locator=original_locator,
        element_description="Submit button",
        context={"failure_reason": "Element not found"}
    )
    
    assert healing_result["success"] == True
    
    print("  5. Recording healing event...")
    event = HealingEvent(
        event_id="e2e-test-1",
        test_id="e2e-001",
        execution_id="exec-e2e-001",
        element_id="submit-button",
        original_locator=original_locator,
        failure_reason="Element not found",
        healing_strategy=HealingStrategy.VISUAL_SIMILARITY,
        new_locator=healing_result["new_locator"],
        confidence_score=healing_result["confidence"],
        healing_successful=True,
        validation_method=ValidationMethod.VISUAL_VERIFICATION
    )
    
    await store.store_healing_event(event)
    
    print("  6. Recording metrics...")
    metrics.record_healing_attempt("visual_similarity", True, 2.1)
    metrics.record_test_execution("passed", 8.5)
    
    print("  7. Verifying persistence...")
    retrieved = await store.get_healing_event("e2e-test-1")
    assert retrieved is not None
    assert retrieved.healing_successful == True
    
    # Cleanup
    await driver.close()
    await store.close()
    
    print("✓ End-to-end workflow test PASSED")
    print(f"  - Healing confidence: {healing_result['confidence']:.2f}")
    print(f"  - New locator: {healing_result['new_locator']}")
    print(f"  - Event persisted successfully")
    
    return True

async def run_all_integration_tests():
    """Run all integration tests."""
    print("\n" + "="*60)
    print("TestDriver MCP Framework v2.0 - Integration Test Suite")
    print("="*60)
    
    tests = [
        ("Database Integration", test_database_integration),
        ("Playwright Integration", test_playwright_integration),
        ("Vision API Integration", test_vision_api_integration),
        ("Healing Strategy Integration", test_healing_strategy_integration),
        ("Monitoring Integration", test_monitoring_integration),
        ("Health Checks Integration", test_health_checks_integration),
        ("End-to-End Workflow", test_end_to_end_workflow),
    ]
    
    results = []
    
    for name, test_func in tests:
        try:
            result = await test_func()
            results.append((name, "PASSED", None))
        except Exception as e:
            results.append((name, "FAILED", str(e)))
            print(f"✗ {name} FAILED: {e}")
    
    # Print summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    passed = sum(1 for _, status, _ in results if status == "PASSED")
    failed = sum(1 for _, status, _ in results if status == "FAILED")
    
    for name, status, error in results:
        symbol = "✓" if status == "PASSED" else "✗"
        print(f"{symbol} {name}: {status}")
        if error:
            print(f"  Error: {error}")
    
    print(f"\nTotal: {len(results)} tests")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Success Rate: {passed/len(results)*100:.1f}%")
    
    return passed, failed

if __name__ == "__main__":
    asyncio.run(run_all_integration_tests())
