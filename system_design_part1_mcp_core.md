# TestDriver MCP Framework: Low-Level Development System Design Specification

**Version**: 1.0  
**Date**: November 11, 2025  
**Author**: Manus AI  
**Document Type**: Technical System Design Specification

---

## Part 1: Core MCP Server Architecture and API Specifications

### 1.1 Overview and Technology Stack

The TestDriver MCP Server is implemented as a standards-compliant Model Context Protocol server that exposes testing capabilities through JSON-RPC 2.0. The server is designed for high performance, extensibility, and operational reliability.

**Primary Technology Stack**:

- **Runtime**: Python 3.11+ (for maximum compatibility with AI/ML libraries and async capabilities)
- **MCP Framework**: `mcp` Python SDK (official Model Context Protocol implementation)
- **Async Framework**: `asyncio` with `aiohttp` for concurrent request handling
- **API Layer**: JSON-RPC 2.0 over stdio, HTTP, or WebSocket transports
- **Configuration**: `pydantic` for type-safe configuration management with validation
- **Logging**: Structured logging with `structlog` for machine-readable logs
- **Containerization**: Podman for container runtime (preferred over Docker for security and rootless operation)

**Alternative Technology Stack Considerations**:

For organizations requiring JVM-based infrastructure, the core server can be implemented in Kotlin with Spring Boot and kotlinx.coroutines. For organizations requiring Node.js infrastructure, TypeScript with the official MCP SDK provides excellent performance. The pluggable architecture ensures that adapter implementations can use language-specific libraries regardless of the core server language.

### 1.2 Project Structure and Module Organization

The codebase follows a modular, layered architecture with clear separation of concerns.

```
testdriver-mcp/
├── src/
│   ├── mcp_server/
│   │   ├── __init__.py
│   │   ├── server.py              # Main MCP server implementation
│   │   ├── transport.py           # Transport layer (stdio, HTTP, WebSocket)
│   │   ├── tools.py               # MCP tool definitions and handlers
│   │   ├── resources.py           # MCP resource providers
│   │   ├── prompts.py             # MCP prompt templates
│   │   └── lifecycle.py           # Server lifecycle management
│   │
│   ├── autonomous_engine/
│   │   ├── __init__.py
│   │   ├── engine.py              # Main autonomous testing engine
│   │   ├── planner.py             # Test plan generation
│   │   ├── executor.py            # Vision-guided execution loop
│   │   ├── reporter.py            # Comprehensive report generation
│   │   └── recovery.py            # Error handling and recovery
│   │
│   ├── vision/
│   │   ├── __init__.py
│   │   ├── adapter_base.py        # Abstract VisionAdapter interface
│   │   ├── openai_adapter.py     # OpenAI GPT-4V implementation
│   │   ├── anthropic_adapter.py  # Anthropic Claude Vision implementation
│   │   ├── google_adapter.py     # Google Gemini Vision implementation
│   │   ├── local_vlm_adapter.py  # Local VLM (Ollama/HF) implementation
│   │   ├── vision_cache.py       # Response caching layer
│   │   └── embeddings.py         # Visual embedding generation
│   │
│   ├── execution/
│   │   ├── __init__.py
│   │   ├── driver_base.py         # Abstract BrowserDriver interface
│   │   ├── selenium_adapter.py   # Selenium WebDriver implementation
│   │   ├── playwright_adapter.py # Playwright implementation
│   │   ├── driver_manager.py     # Driver lifecycle and selection
│   │   └── browser_pool.py       # Browser instance pooling
│   │
│   ├── reliability/
│   │   ├── __init__.py
│   │   ├── state_store.py         # State Synchronization Store
│   │   ├── replay_engine.py      # Deterministic Replay Engine
│   │   ├── adaptive_wait.py      # Adaptive Wait Service
│   │   ├── recovery_engine.py    # Heuristic Recovery Engine
│   │   ├── module_manager.py     # Hot-Swappable Module Manager
│   │   └── chaos_controller.py   # Chaos Engineering Controller
│   │
│   ├── self_healing/
│   │   ├── __init__.py
│   │   ├── locator_healer.py     # AI Locator Healing Service
│   │   ├── anomaly_learner.py    # Anomaly Learning Engine
│   │   ├── drift_detector.py     # Environment Drift Detector
│   │   └── learning_db.py        # Learning database interface
│   │
│   ├── telemetry/
│   │   ├── __init__.py
│   │   ├── metrics_service.py    # Telemetry & Metrics Service
│   │   ├── reliability_scorer.py # Reliability Scoring Engine
│   │   ├── collectors.py         # Metric collectors
│   │   └── exporters.py          # Prometheus/OpenTelemetry exporters
│   │
│   ├── testing_scope/
│   │   ├── __init__.py
│   │   ├── accessibility.py      # Accessibility scanning (axe-core)
│   │   ├── security.py           # Security scanning (SAST/DAST/SCA)
│   │   ├── performance.py        # Performance testing (Lighthouse/k6)
│   │   ├── api_fusion.py         # API + UI Fusion Testing
│   │   └── data_generator.py    # Synthetic Data Generator
│   │
│   ├── storage/
│   │   ├── __init__.py
│   │   ├── timeseries_db.py     # Time-series database interface
│   │   ├── document_db.py       # Document database interface
│   │   ├── blob_storage.py      # Blob storage for screenshots/videos
│   │   └── cache.py             # Distributed cache interface
│   │
│   ├── config/
│   │   ├── __init__.py
│   │   ├── settings.py          # Configuration models (Pydantic)
│   │   ├── secrets.py           # Secret management integration
│   │   └── validation.py        # Configuration validation
│   │
│   └── utils/
│       ├── __init__.py
│       ├── logging.py           # Structured logging setup
│       ├── errors.py            # Custom exception hierarchy
│       ├── retry.py             # Retry decorators and utilities
│       └── async_utils.py      # Async helper functions
│
├── tests/
│   ├── unit/                    # Unit tests for each module
│   ├── integration/             # Integration tests
│   ├── e2e/                     # End-to-end tests
│   └── fixtures/                # Test fixtures and mocks
│
├── config/
│   ├── default.yaml            # Default configuration
│   ├── development.yaml        # Development environment config
│   ├── staging.yaml            # Staging environment config
│   └── production.yaml         # Production environment config
│
├── deployment/
│   ├── Containerfile           # Podman container definition
│   ├── kubernetes/             # Kubernetes manifests
│   ├── helm/                   # Helm charts
│   └── terraform/              # Infrastructure as Code
│
├── docs/
│   ├── api/                    # API documentation
│   ├── architecture/           # Architecture diagrams and docs
│   └── guides/                 # User and developer guides
│
├── scripts/
│   ├── setup.sh               # Development environment setup
│   ├── run_tests.sh           # Test execution script
│   └── deploy.sh              # Deployment automation
│
├── pyproject.toml             # Python project configuration (Poetry)
├── requirements.txt           # Python dependencies
├── README.md                  # Project overview
└── LICENSE                    # License file
```

### 1.3 Core MCP Server Implementation

The MCP Server Core implements the JSON-RPC 2.0 protocol and manages the lifecycle of all subsystems.

**File**: `src/mcp_server/server.py`

```python
"""
TestDriver MCP Server Core Implementation
Implements the Model Context Protocol server with JSON-RPC 2.0 transport
"""

import asyncio
from typing import Dict, List, Optional, Any
from mcp.server import Server
from mcp.types import (
    Tool, Resource, Prompt, TextContent, ImageContent,
    EmbeddedResource, LoggingLevel
)
import structlog

from ..config.settings import ServerConfig
from ..autonomous_engine.engine import AutonomousTestingEngine
from ..telemetry.metrics_service import MetricsService
from ..reliability.module_manager import ModuleManager
from .tools import ToolRegistry
from .resources import ResourceProvider
from .prompts import PromptProvider

logger = structlog.get_logger(__name__)


class TestDriverMCPServer:
    """
    Main TestDriver MCP Server implementation.
    
    Responsibilities:
    - Implement MCP protocol (tools, resources, prompts)
    - Manage subsystem lifecycle
    - Handle client connections and requests
    - Coordinate autonomous testing engine
    - Provide observability and health checks
    """
    
    def __init__(self, config: ServerConfig):
        """
        Initialize the MCP server with configuration.
        
        Args:
            config: Server configuration object (validated Pydantic model)
        """
        self.config = config
        self.server = Server(name="testdriver-mcp")
        
        # Initialize subsystems
        self.metrics_service = MetricsService(config.telemetry)
        self.module_manager = ModuleManager(config.modules)
        self.autonomous_engine = AutonomousTestingEngine(
            config=config.engine,
            metrics_service=self.metrics_service,
            module_manager=self.module_manager
        )
        
        # Initialize MCP components
        self.tool_registry = ToolRegistry(self.autonomous_engine)
        self.resource_provider = ResourceProvider(self.autonomous_engine)
        self.prompt_provider = PromptProvider()
        
        # Server state
        self._running = False
        self._active_tests: Dict[str, Any] = {}
        
        logger.info("TestDriver MCP Server initialized", config=config.dict())
    
    async def start(self) -> None:
        """Start the MCP server and all subsystems."""
        logger.info("Starting TestDriver MCP Server")
        
        # Start subsystems in dependency order
        await self.metrics_service.start()
        await self.module_manager.start()
        await self.autonomous_engine.start()
        
        # Register MCP handlers
        self._register_tool_handlers()
        self._register_resource_handlers()
        self._register_prompt_handlers()
        
        self._running = True
        logger.info("TestDriver MCP Server started successfully")
    
    async def stop(self) -> None:
        """Stop the MCP server and all subsystems gracefully."""
        logger.info("Stopping TestDriver MCP Server")
        self._running = False
        
        # Stop subsystems in reverse dependency order
        await self.autonomous_engine.stop()
        await self.module_manager.stop()
        await self.metrics_service.stop()
        
        logger.info("TestDriver MCP Server stopped")
    
    def _register_tool_handlers(self) -> None:
        """Register all MCP tool handlers."""
        
        @self.server.list_tools()
        async def list_tools() -> List[Tool]:
            """Return list of available tools."""
            return self.tool_registry.list_tools()
        
        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
            """Execute a tool with given arguments."""
            logger.info("Tool called", tool=name, arguments=arguments)
            
            try:
                result = await self.tool_registry.execute_tool(name, arguments)
                self.metrics_service.record_tool_call(name, success=True)
                return [TextContent(type="text", text=str(result))]
            except Exception as e:
                logger.error("Tool execution failed", tool=name, error=str(e))
                self.metrics_service.record_tool_call(name, success=False)
                raise
    
    def _register_resource_handlers(self) -> None:
        """Register all MCP resource handlers."""
        
        @self.server.list_resources()
        async def list_resources() -> List[Resource]:
            """Return list of available resources."""
            return await self.resource_provider.list_resources()
        
        @self.server.read_resource()
        async def read_resource(uri: str) -> str:
            """Read and return resource content."""
            logger.info("Resource read", uri=uri)
            return await self.resource_provider.read_resource(uri)
    
    def _register_prompt_handlers(self) -> None:
        """Register all MCP prompt handlers."""
        
        @self.server.list_prompts()
        async def list_prompts() -> List[Prompt]:
            """Return list of available prompts."""
            return self.prompt_provider.list_prompts()
        
        @self.server.get_prompt()
        async def get_prompt(name: str, arguments: Dict[str, str]) -> str:
            """Get a prompt with arguments filled in."""
            return self.prompt_provider.get_prompt(name, arguments)
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform comprehensive health check of all subsystems.
        
        Returns:
            Dictionary with health status of each subsystem
        """
        health = {
            "server": "healthy" if self._running else "stopped",
            "metrics_service": await self.metrics_service.health_check(),
            "module_manager": await self.module_manager.health_check(),
            "autonomous_engine": await self.autonomous_engine.health_check(),
            "active_tests": len(self._active_tests)
        }
        
        overall_healthy = all(
            status == "healthy" 
            for key, status in health.items() 
            if key != "active_tests"
        )
        health["overall"] = "healthy" if overall_healthy else "degraded"
        
        return health


async def main():
    """Main entry point for the MCP server."""
    # Load configuration
    config = ServerConfig.from_file("config/default.yaml")
    
    # Create and start server
    server = TestDriverMCPServer(config)
    
    try:
        await server.start()
        
        # Run until interrupted
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    finally:
        await server.stop()


if __name__ == "__main__":
    asyncio.run(main())
```

### 1.4 MCP Tool Definitions and Handlers

Tools are the primary interface through which MCP clients interact with the TestDriver server. Each tool corresponds to a specific testing action or lifecycle operation.

**File**: `src/mcp_server/tools.py`

```python
"""
MCP Tool Definitions and Execution Handlers
Defines all tools exposed by the TestDriver MCP server
"""

from typing import Dict, List, Any, Optional
from mcp.types import Tool
from pydantic import BaseModel, Field
import structlog

from ..autonomous_engine.engine import AutonomousTestingEngine

logger = structlog.get_logger(__name__)


class ToolRegistry:
    """
    Registry and executor for all MCP tools.
    
    Responsibilities:
    - Define tool schemas with input/output specifications
    - Route tool calls to appropriate handlers
    - Validate tool arguments
    - Handle tool execution errors
    """
    
    def __init__(self, engine: AutonomousTestingEngine):
        self.engine = engine
        self._tools: Dict[str, Tool] = {}
        self._register_tools()
    
    def _register_tools(self) -> None:
        """Register all available tools with their schemas."""
        
        # Core testing tools
        self._tools["testdriver.startTest"] = Tool(
            name="testdriver.startTest",
            description="Initiate autonomous testing from high-level requirements",
            inputSchema={
                "type": "object",
                "properties": {
                    "requirements": {
                        "type": "string",
                        "description": "Natural language description of what to test"
                    },
                    "application_url": {
                        "type": "string",
                        "description": "Base URL of the application under test"
                    },
                    "browser": {
                        "type": "string",
                        "enum": ["chrome", "firefox", "webkit"],
                        "default": "chrome",
                        "description": "Browser to use for testing"
                    },
                    "framework": {
                        "type": "string",
                        "enum": ["selenium", "playwright", "auto"],
                        "default": "auto",
                        "description": "Browser automation framework preference"
                    },
                    "vision_model": {
                        "type": "string",
                        "enum": ["gpt4v", "claude", "gemini", "local"],
                        "default": "auto",
                        "description": "AI vision model to use"
                    },
                    "test_scope": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["functional", "accessibility", "security", "performance"]
                        },
                        "default": ["functional"],
                        "description": "Testing dimensions to include"
                    },
                    "timeout": {
                        "type": "integer",
                        "default": 300,
                        "description": "Maximum test execution time in seconds"
                    }
                },
                "required": ["requirements", "application_url"]
            }
        )
        
        self._tools["testdriver.executeStep"] = Tool(
            name="testdriver.executeStep",
            description="Execute a single test step programmatically",
            inputSchema={
                "type": "object",
                "properties": {
                    "test_id": {
                        "type": "string",
                        "description": "ID of the active test session"
                    },
                    "action": {
                        "type": "string",
                        "enum": ["navigate", "click", "type", "assert", "wait"],
                        "description": "Action to perform"
                    },
                    "params": {
                        "type": "object",
                        "description": "Action-specific parameters"
                    }
                },
                "required": ["test_id", "action", "params"]
            }
        )
        
        self._tools["testdriver.getReport"] = Tool(
            name="testdriver.getReport",
            description="Retrieve comprehensive test report",
            inputSchema={
                "type": "object",
                "properties": {
                    "test_id": {
                        "type": "string",
                        "description": "ID of the test to get report for"
                    },
                    "format": {
                        "type": "string",
                        "enum": ["html", "json", "junit"],
                        "default": "html",
                        "description": "Report format"
                    }
                },
                "required": ["test_id"]
            }
        )
        
        self._tools["testdriver.healTest"] = Tool(
            name="testdriver.healTest",
            description="Trigger self-healing for a failed test",
            inputSchema={
                "type": "object",
                "properties": {
                    "test_id": {
                        "type": "string",
                        "description": "ID of the failed test"
                    },
                    "auto_commit": {
                        "type": "boolean",
                        "default": False,
                        "description": "Automatically commit healing changes if confidence > 0.90"
                    }
                },
                "required": ["test_id"]
            }
        )
        
        self._tools["testdriver.listTests"] = Tool(
            name="testdriver.listTests",
            description="List all test executions with status",
            inputSchema={
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "enum": ["all", "running", "passed", "failed", "healing"],
                        "default": "all",
                        "description": "Filter by test status"
                    },
                    "limit": {
                        "type": "integer",
                        "default": 50,
                        "description": "Maximum number of tests to return"
                    }
                }
            }
        )
        
        self._tools["testdriver.stopTest"] = Tool(
            name="testdriver.stopTest",
            description="Stop a running test execution",
            inputSchema={
                "type": "object",
                "properties": {
                    "test_id": {
                        "type": "string",
                        "description": "ID of the test to stop"
                    },
                    "save_state": {
                        "type": "boolean",
                        "default": True,
                        "description": "Save state for potential resume"
                    }
                },
                "required": ["test_id"]
            }
        )
        
        # Reliability and monitoring tools
        self._tools["testdriver.getMetrics"] = Tool(
            name="testdriver.getMetrics",
            description="Get telemetry metrics for analysis",
            inputSchema={
                "type": "object",
                "properties": {
                    "metric_type": {
                        "type": "string",
                        "enum": ["execution", "quality", "system", "business"],
                        "description": "Type of metrics to retrieve"
                    },
                    "time_range": {
                        "type": "string",
                        "enum": ["1h", "24h", "7d", "30d"],
                        "default": "24h",
                        "description": "Time range for metrics"
                    }
                },
                "required": ["metric_type"]
            }
        )
        
        self._tools["testdriver.getReliabilityScore"] = Tool(
            name="testdriver.getReliabilityScore",
            description="Get reliability scores for tests or modules",
            inputSchema={
                "type": "object",
                "properties": {
                    "entity_type": {
                        "type": "string",
                        "enum": ["test", "module", "adapter"],
                        "description": "Type of entity to score"
                    },
                    "entity_id": {
                        "type": "string",
                        "description": "ID of the entity (optional, returns all if omitted)"
                    }
                },
                "required": ["entity_type"]
            }
        )
        
        logger.info("Registered tools", count=len(self._tools))
    
    def list_tools(self) -> List[Tool]:
        """Return list of all registered tools."""
        return list(self._tools.values())
    
    async def execute_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """
        Execute a tool with given arguments.
        
        Args:
            name: Tool name (e.g., "testdriver.startTest")
            arguments: Tool-specific arguments
            
        Returns:
            Tool execution result
            
        Raises:
            ValueError: If tool not found or arguments invalid
        """
        if name not in self._tools:
            raise ValueError(f"Tool not found: {name}")
        
        # Route to appropriate handler
        if name == "testdriver.startTest":
            return await self._handle_start_test(arguments)
        elif name == "testdriver.executeStep":
            return await self._handle_execute_step(arguments)
        elif name == "testdriver.getReport":
            return await self._handle_get_report(arguments)
        elif name == "testdriver.healTest":
            return await self._handle_heal_test(arguments)
        elif name == "testdriver.listTests":
            return await self._handle_list_tests(arguments)
        elif name == "testdriver.stopTest":
            return await self._handle_stop_test(arguments)
        elif name == "testdriver.getMetrics":
            return await self._handle_get_metrics(arguments)
        elif name == "testdriver.getReliabilityScore":
            return await self._handle_get_reliability_score(arguments)
        else:
            raise ValueError(f"Handler not implemented for tool: {name}")
    
    async def _handle_start_test(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle testdriver.startTest tool execution."""
        logger.info("Starting autonomous test", requirements=args["requirements"])
        
        test_id = await self.engine.start_test(
            requirements=args["requirements"],
            application_url=args["application_url"],
            browser=args.get("browser", "chrome"),
            framework=args.get("framework", "auto"),
            vision_model=args.get("vision_model", "auto"),
            test_scope=args.get("test_scope", ["functional"]),
            timeout=args.get("timeout", 300)
        )
        
        return {
            "test_id": test_id,
            "status": "started",
            "message": "Test execution initiated successfully"
        }
    
    async def _handle_execute_step(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle testdriver.executeStep tool execution."""
        result = await self.engine.execute_step(
            test_id=args["test_id"],
            action=args["action"],
            params=args["params"]
        )
        return result
    
    async def _handle_get_report(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle testdriver.getReport tool execution."""
        report = await self.engine.get_report(
            test_id=args["test_id"],
            format=args.get("format", "html")
        )
        return report
    
    async def _handle_heal_test(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle testdriver.healTest tool execution."""
        result = await self.engine.heal_test(
            test_id=args["test_id"],
            auto_commit=args.get("auto_commit", False)
        )
        return result
    
    async def _handle_list_tests(self, args: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Handle testdriver.listTests tool execution."""
        tests = await self.engine.list_tests(
            status=args.get("status", "all"),
            limit=args.get("limit", 50)
        )
        return tests
    
    async def _handle_stop_test(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle testdriver.stopTest tool execution."""
        result = await self.engine.stop_test(
            test_id=args["test_id"],
            save_state=args.get("save_state", True)
        )
        return result
    
    async def _handle_get_metrics(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle testdriver.getMetrics tool execution."""
        metrics = await self.engine.get_metrics(
            metric_type=args["metric_type"],
            time_range=args.get("time_range", "24h")
        )
        return metrics
    
    async def _handle_get_reliability_score(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle testdriver.getReliabilityScore tool execution."""
        scores = await self.engine.get_reliability_score(
            entity_type=args["entity_type"],
            entity_id=args.get("entity_id")
        )
        return scores
```

### 1.5 Configuration Management System

Configuration is managed through type-safe Pydantic models with environment variable overrides and validation.

**File**: `src/config/settings.py`

```python
"""
Configuration Management System
Defines all configuration models with validation and defaults
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, validator, SecretStr
from pathlib import Path
import yaml


class TelemetryConfig(BaseModel):
    """Telemetry and metrics configuration."""
    enabled: bool = True
    prometheus_port: int = 9090
    prometheus_path: str = "/metrics"
    log_level: str = "INFO"
    structured_logging: bool = True
    trace_sampling_rate: float = 0.1


class VisionAdapterConfig(BaseModel):
    """Vision adapter configuration."""
    adapter_type: str = Field(..., description="Adapter type: openai, anthropic, google, local")
    api_key: Optional[SecretStr] = None
    api_base_url: Optional[str] = None
    model_name: Optional[str] = None
    timeout: int = 30
    max_retries: int = 3
    cache_enabled: bool = True
    cache_ttl: int = 3600


class ExecutionAdapterConfig(BaseModel):
    """Execution adapter configuration."""
    adapter_type: str = Field(..., description="Adapter type: selenium, playwright")
    browser: str = "chrome"
    headless: bool = True
    viewport_width: int = 1920
    viewport_height: int = 1080
    timeout: int = 30
    screenshot_on_failure: bool = True
    video_recording: bool = True


class StateStoreConfig(BaseModel):
    """State Synchronization Store configuration."""
    enabled: bool = True
    backend: str = Field("influxdb", description="Backend: influxdb, timescaledb, postgresql")
    connection_url: str
    retention_days: int = 30
    snapshot_interval: int = 5  # seconds
    compression_enabled: bool = True


class ReliabilityConfig(BaseModel):
    """Reliability layer configuration."""
    state_store: StateStoreConfig
    adaptive_wait_enabled: bool = True
    adaptive_wait_timeout: int = 30
    adaptive_wait_stability_threshold: float = 0.98
    recovery_enabled: bool = True
    recovery_max_attempts: int = 3
    replay_enabled: bool = True


class SelfHealingConfig(BaseModel):
    """Self-healing layer configuration."""
    enabled: bool = True
    confidence_threshold_auto: float = 0.90
    confidence_threshold_pr: float = 0.80
    confidence_threshold_manual: float = 0.70
    learning_db_backend: str = "postgresql"
    learning_db_url: str
    drift_detection_enabled: bool = True
    drift_check_interval: int = 300  # seconds


class TestingScopeConfig(BaseModel):
    """Testing scope configuration."""
    accessibility_enabled: bool = True
    accessibility_standards: List[str] = ["WCAG2.1-AA"]
    security_enabled: bool = True
    security_tools: List[str] = ["bandit", "safety", "semgrep"]
    performance_enabled: bool = False
    performance_budgets: Dict[str, int] = {
        "fcp": 2000,  # First Contentful Paint (ms)
        "lcp": 2500,  # Largest Contentful Paint (ms)
        "cls": 0.1,   # Cumulative Layout Shift
        "tti": 3500   # Time to Interactive (ms)
    }


class EngineConfig(BaseModel):
    """Autonomous Testing Engine configuration."""
    planner_model: str = "gpt-4"
    planner_temperature: float = 0.7
    max_test_steps: int = 50
    step_timeout: int = 30
    parallel_execution: bool = False
    max_parallel_tests: int = 5


class ModuleConfig(BaseModel):
    """Module Manager configuration."""
    health_check_interval: int = 60  # seconds
    adapter_switch_threshold: float = 0.05  # 5% error rate
    circuit_breaker_enabled: bool = True
    circuit_breaker_threshold: int = 5  # consecutive failures


class ServerConfig(BaseModel):
    """Main server configuration."""
    server_name: str = "testdriver-mcp"
    server_version: str = "2.0.0"
    transport: str = Field("stdio", description="Transport: stdio, http, websocket")
    http_host: str = "0.0.0.0"
    http_port: int = 8080
    
    telemetry: TelemetryConfig
    vision: VisionAdapterConfig
    execution: ExecutionAdapterConfig
    reliability: ReliabilityConfig
    self_healing: SelfHealingConfig
    testing_scope: TestingScopeConfig
    engine: EngineConfig
    modules: ModuleConfig
    
    @classmethod
    def from_file(cls, path: str) -> "ServerConfig":
        """Load configuration from YAML file."""
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    @validator("transport")
    def validate_transport(cls, v):
        """Validate transport type."""
        if v not in ["stdio", "http", "websocket"]:
            raise ValueError(f"Invalid transport: {v}")
        return v
```

This completes Part 1 of the low-level system design specification, covering the core MCP server architecture, API specifications, tool definitions, and configuration management. The next parts will cover pluggable adapters, reliability layers, data models, and deployment specifications.
