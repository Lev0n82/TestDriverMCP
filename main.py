"""Main application entry point for TestDriver MCP Framework."""

import asyncio
import os
from typing import Dict, Any

import structlog

from .mcp_server import MCPServer
from .vision import OpenAIVisionAdapter
from .execution import ExecutionFramework
from .self_healing import AILocatorHealingEngine
from .memory import TestMemoryStore
from .learning import TestLearningOrchestrator
from .testing_scope import HealthMonitor, SyntheticTestGenerator

# Configure structured logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.JSONRenderer()
    ]
)

logger = structlog.get_logger()


class TestDriverApp:
    """Main TestDriver MCP Framework application."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize components
        logger.info("Initializing TestDriver MCP Framework")
        
        # Memory store
        self.memory_store = TestMemoryStore(config.get("memory", {}))
        
        # Vision adapter
        vision_config = config.get("vision", {})
        self.vision_adapter = OpenAIVisionAdapter(vision_config)
        
        # Execution framework
        exec_config = config.get("execution", {})
        self.execution_framework = ExecutionFramework(exec_config)
        
        # Self-healing engine
        self.healing_engine = AILocatorHealingEngine(
            vision_adapter=self.vision_adapter,
            memory_store=self.memory_store,
            config=config.get("healing", {})
        )
        
        # Learning orchestrator
        self.learning_orchestrator = TestLearningOrchestrator(
            memory_store=self.memory_store,
            config=config.get("learning", {})
        )
        
        # MCP Server
        self.mcp_server = MCPServer(config.get("mcp", {}))
        
        # Self-testing components
        self.health_monitor = HealthMonitor(config.get("self_test", {}))
        self.synthetic_test_generator = SyntheticTestGenerator(config.get("self_test", {}))
        
        logger.info("TestDriver initialized successfully")
    
    async def start(self) -> None:
        """Start the application."""
        logger.info("Starting TestDriver MCP Framework")
        
        # Start learning orchestrator
        await self.learning_orchestrator.start()
        
        # Start self-testing components
        await self.health_monitor.start()
        await self.synthetic_test_generator.start()
        
        logger.info("TestDriver started successfully")
    
    async def stop(self) -> None:
        """Stop the application."""
        logger.info("Stopping TestDriver MCP Framework")
        
        # Stop self-testing components
        await self.health_monitor.stop()
        await self.synthetic_test_generator.stop()
        
        # Stop learning orchestrator
        await self.learning_orchestrator.stop()
        
        # Cleanup execution framework
        await self.execution_framework.cleanup()
        
        logger.info("TestDriver stopped successfully")
    
    async def handle_mcp_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP request."""
        return await self.mcp_server.handle_request(request)


async def main():
    """Main entry point."""
    # Load configuration
    config = {
        "vision": {
            "api_key": os.getenv("OPENAI_API_KEY"),
            "model": "gpt-4.1-mini"
        },
        "execution": {
            "driver": "playwright",
            "headless": True
        },
        "healing": {
            "auto_commit_threshold": 0.9,
            "pr_review_threshold": 0.8,
            "manual_review_threshold": 0.7
        },
        "learning": {
            "learning_interval_hours": 24
        },
        "memory": {
            "storage_type": "memory"
        },
        "self_test": {
            "enabled": True,
            "check_interval_seconds": 60,
            "test_interval_seconds": 3600
        }
    }
    
    # Create and start application
    app = TestDriverApp(config)
    
    try:
        await app.start()
        
        # Keep running
        logger.info("TestDriver is running. Press Ctrl+C to stop.")
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Shutdown requested")
    finally:
        await app.stop()


if __name__ == "__main__":
    asyncio.run(main())
