"""Comprehensive system tests for TestDriver MCP Framework."""

import asyncio
import sys
import os
import pytest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models import TestStep, TestPlan, HealingEvent, HealingStrategy, ValidationMethod
from mcp_server.server import MCPServer
from memory.store import TestMemoryStore
from learning.orchestrator import TestLearningOrchestrator
from testing_scope.self_test import EmbeddedValidator, HealthMonitor


class TestMCPServer:
    """Test MCP Server functionality."""
    
    @pytest.mark.asyncio
    async def test_initialize(self):
        """Test MCP server initialization."""
        config = {"name": "test-server"}
        server = MCPServer(config)
        
        request = {
            "jsonrpc": "2.0",
            "id": "1",
            "method": "initialize",
            "params": {"clientInfo": {"name": "test-client"}}
        }
        
        response = await server.handle_request(request)
        
        assert response["jsonrpc"] == "2.0"
        assert response["id"] == "1"
        assert "result" in response
        assert response["result"]["protocolVersion"] == "2025-06-18"
        print("✓ MCP Server initialization test passed")
    
    @pytest.mark.asyncio
    async def test_tools_list(self):
        """Test tools list endpoint."""
        server = MCPServer({})
        
        request = {
            "jsonrpc": "2.0",
            "id": "2",
            "method": "tools/list",
            "params": {}
        }
        
        response = await server.handle_request(request)
        
        assert "result" in response
        assert "tools" in response["result"]
        assert len(response["result"]["tools"]) > 0
        
        # Verify required tools exist
        tool_names = [t["name"] for t in response["result"]["tools"]]
        assert "generate_test_plan" in tool_names
        assert "execute_test" in tool_names
        print("✓ MCP Server tools list test passed")
    
    @pytest.mark.asyncio
    async def test_generate_test_plan(self):
        """Test test plan generation."""
        server = MCPServer({})
        
        request = {
            "jsonrpc": "2.0",
            "id": "3",
            "method": "tools/call",
            "params": {
                "name": "generate_test_plan",
                "arguments": {
                    "requirements": "Test login functionality",
                    "target_url": "https://example.com",
                    "test_type": "functional"
                }
            }
        }
        
        response = await server.handle_request(request)
        
        assert "result" in response
        assert "content" in response["result"]
        print("✓ MCP Server test plan generation test passed")


class TestMemoryStore:
    """Test memory store functionality."""
    
    @pytest.mark.asyncio
    async def test_store_and_retrieve_healing_event(self):
        """Test storing and retrieving healing events."""
        store = TestMemoryStore({"storage_type": "memory"})
        
        event = HealingEvent(
            event_id="test-001",
            test_id="test-123",
            execution_id="exec-456",
            element_id="btn-submit",
            original_locator={"css": "#old-id"},
            failure_reason="Element not found",
            healing_strategy=HealingStrategy.VISUAL_SIMILARITY,
            new_locator={"css": "#new-id"},
            confidence_score=0.92,
            healing_successful=True,
            validation_method=ValidationMethod.AUTO,
            visual_embedding=[0.1] * 512
        )
        
        await store.store_healing_event(event)
        
        retrieved = await store.get_healing_event("test-001")
        
        assert retrieved is not None
        assert retrieved.event_id == "test-001"
        assert retrieved.confidence_score == 0.92
        print("✓ Memory store and retrieve test passed")
    
    @pytest.mark.asyncio
    async def test_similarity_search(self):
        """Test similarity search functionality."""
        store = TestMemoryStore({"storage_type": "memory"})
        
        # Store multiple events with different embeddings
        event1 = HealingEvent(
            event_id="test-001",
            test_id="test-123",
            execution_id="exec-456",
            element_id="btn-1",
            original_locator={"css": "#btn1"},
            failure_reason="Not found",
            healing_strategy=HealingStrategy.VISUAL_SIMILARITY,
            new_locator={"css": "#new-btn1"},
            confidence_score=0.9,
            healing_successful=True,
            validation_method=ValidationMethod.AUTO,
            visual_embedding=[0.9] * 512
        )
        
        event2 = HealingEvent(
            event_id="test-002",
            test_id="test-124",
            execution_id="exec-457",
            element_id="btn-2",
            original_locator={"css": "#btn2"},
            failure_reason="Not found",
            healing_strategy=HealingStrategy.VISUAL_SIMILARITY,
            new_locator={"css": "#new-btn2"},
            confidence_score=0.85,
            healing_successful=True,
            validation_method=ValidationMethod.AUTO,
            visual_embedding=[0.1] * 512
        )
        
        await store.store_healing_event(event1)
        await store.store_healing_event(event2)
        
        # Search with embedding similar to event1
        query_embedding = [0.85] * 512
        results = await store.find_similar_healing_events(query_embedding, limit=5)
        
        assert len(results) > 0
        assert results[0][0].event_id == "test-001"  # Most similar
        print("✓ Memory similarity search test passed")
    
    @pytest.mark.asyncio
    async def test_element_stability_calculation(self):
        """Test element stability calculation."""
        store = TestMemoryStore({"storage_type": "memory"})
        
        # Record test executions
        for i in range(100):
            await store.record_test_execution(
                test_id=f"test-{i}",
                elements_tested=["btn-submit"]
            )
        
        # Record some healing events
        for i in range(10):
            event = HealingEvent(
                event_id=f"heal-{i}",
                test_id=f"test-{i}",
                execution_id=f"exec-{i}",
                element_id="btn-submit",
                original_locator={"css": "#old"},
                failure_reason="Not found",
                healing_strategy=HealingStrategy.VISUAL_SIMILARITY,
                new_locator={"css": "#new"},
                confidence_score=0.8,
                healing_successful=True,
                validation_method=ValidationMethod.AUTO
            )
            await store.store_healing_event(event)
        
        # Calculate stability
        stability = await store.calculate_element_stability("btn-submit")
        
        # Should be 1 - (10/100) = 0.9
        assert abs(stability - 0.9) < 0.01
        print("✓ Element stability calculation test passed")


class TestLearningOrchestrator:
    """Test learning orchestrator functionality."""
    
    @pytest.mark.asyncio
    async def test_learning_cycle(self):
        """Test learning cycle execution."""
        store = TestMemoryStore({"storage_type": "memory"})
        orchestrator = TestLearningOrchestrator(store, {})
        
        results = await orchestrator.run_learning_cycle()
        
        assert "timestamp" in results
        assert "optimizations" in results
        assert "insights" in results
        print("✓ Learning cycle test passed")
    
    @pytest.mark.asyncio
    async def test_insight_generation(self):
        """Test insight generation."""
        store = TestMemoryStore({"storage_type": "memory"})
        orchestrator = TestLearningOrchestrator(store, {})
        
        insights = await orchestrator.generate_insights()
        
        assert isinstance(insights, list)
        print("✓ Insight generation test passed")


class TestEmbeddedValidator:
    """Test embedded validator functionality."""
    
    def test_validator_initialization(self):
        """Test validator initialization."""
        class DummyComponent:
            pass
        
        component = DummyComponent()
        config = {"enabled": True, "sampling_rate": 0.5}
        
        validator = EmbeddedValidator(component, config)
        
        assert validator.enabled == True
        assert validator.sampling_rate == 0.5
        print("✓ Embedded validator initialization test passed")
    
    def test_validation_sampling(self):
        """Test validation sampling."""
        class DummyComponent:
            pass
        
        component = DummyComponent()
        config = {"enabled": True, "sampling_rate": 1.0}  # Always validate
        
        validator = EmbeddedValidator(component, config)
        
        # Should always validate with sampling_rate=1.0
        should_validate_count = sum(1 for _ in range(100) if validator._should_validate())
        assert should_validate_count == 100
        print("✓ Validation sampling test passed")


class TestHealthMonitor:
    """Test health monitor functionality."""
    
    @pytest.mark.asyncio
    async def test_health_monitor_start_stop(self):
        """Test health monitor start and stop."""
        config = {"check_interval_seconds": 1}
        monitor = HealthMonitor(config)
        
        await monitor.start()
        assert monitor._running == True
        
        await asyncio.sleep(0.5)
        
        await monitor.stop()
        assert monitor._running == False
        print("✓ Health monitor start/stop test passed")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("TestDriver MCP Framework - Comprehensive Test Suite")
    print("=" * 70 + "\n")
    
    pytest.main([__file__, "-v", "--tb=short", "-s"])
