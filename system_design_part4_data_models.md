# TestDriver MCP Framework: Low-Level Development System Design Specification

## Part 4: Data Models, Schemas, and Storage Architecture

### 4.1 Data Architecture Overview

The TestDriver system requires a multi-tier storage architecture to handle different data types with appropriate performance, consistency, and retention characteristics.

**Storage Tiers**:

| Storage Tier | Technology | Data Types | Retention | Performance Requirement |
|:---|:---|:---|:---|:---|
| **Time-Series Store** | InfluxDB / TimescaleDB | State snapshots, metrics, telemetry | 30-90 days | High write throughput, range queries |
| **Document Store** | PostgreSQL / MongoDB | Test plans, reports, configurations | Indefinite | ACID transactions, complex queries |
| **Key-Value Cache** | Redis / Valkey | Session state, adapter health, locks | TTL-based | Sub-millisecond latency |
| **Blob Storage** | S3 / MinIO | Screenshots, videos, artifacts | 90-365 days | High throughput, cost-effective |
| **Learning Database** | PostgreSQL | Visual embeddings, healing history | Indefinite | Vector similarity search |

### 4.2 Core Data Models

**File**: `src/models/test_execution.py`

```python
"""
Core data models for test execution
"""

from typing import List, Dict, Optional, Any
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field
from uuid import UUID, uuid4


class TestStatus(str, Enum):
    """Test execution status."""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    HEALING = "healing"
    STOPPED = "stopped"
    ERROR = "error"


class StepStatus(str, Enum):
    """Test step execution status."""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    HEALED = "healed"


class ActionType(str, Enum):
    """Types of test actions."""
    NAVIGATE = "navigate"
    CLICK = "click"
    TYPE = "type"
    ASSERT = "assert"
    WAIT = "wait"
    SCROLL = "scroll"
    SELECT = "select"
    HOVER = "hover"
    DRAG_DROP = "drag_drop"
    UPLOAD_FILE = "upload_file"
    EXECUTE_SCRIPT = "execute_script"


class TestStep(BaseModel):
    """
    Individual test step definition.
    """
    step_id: str = Field(default_factory=lambda: str(uuid4()))
    description: str
    action: ActionType
    params: Dict[str, Any]
    expected_outcome: Optional[str] = None
    timeout: int = 30
    retry_on_failure: bool = True
    max_retries: int = 3
    critical: bool = True  # If False, failure doesn't fail entire test
    
    # Execution metadata (populated during execution)
    status: StepStatus = StepStatus.PENDING
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_ms: Optional[float] = None
    error_message: Optional[str] = None
    screenshot_path: Optional[str] = None
    confidence_score: Optional[float] = None
    retry_count: int = 0
    healed: bool = False
    healing_details: Optional[Dict[str, Any]] = None


class TestPlan(BaseModel):
    """
    Complete test plan generated from requirements.
    """
    plan_id: str = Field(default_factory=lambda: str(uuid4()))
    test_id: str
    name: str
    description: str
    requirements: str  # Original natural language requirements
    
    # Test configuration
    application_url: str
    browser: str = "chrome"
    framework: str = "auto"
    vision_model: str = "auto"
    test_scope: List[str] = ["functional"]
    
    # Test steps
    setup_steps: List[TestStep] = []
    test_steps: List[TestStep] = []
    teardown_steps: List[TestStep] = []
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: str = "autonomous_engine"
    estimated_duration_seconds: Optional[int] = None
    tags: List[str] = []
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class TestExecution(BaseModel):
    """
    Test execution instance with results.
    """
    test_id: str = Field(default_factory=lambda: str(uuid4()))
    plan_id: str
    test_plan: TestPlan
    
    # Execution state
    status: TestStatus = TestStatus.PENDING
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_ms: Optional[float] = None
    
    # Results
    total_steps: int = 0
    passed_steps: int = 0
    failed_steps: int = 0
    skipped_steps: int = 0
    healed_steps: int = 0
    
    # Artifacts
    report_path: Optional[str] = None
    video_path: Optional[str] = None
    log_path: Optional[str] = None
    
    # Quality metrics
    accessibility_issues: List[Dict[str, Any]] = []
    security_issues: List[Dict[str, Any]] = []
    performance_metrics: Dict[str, float] = {}
    
    # Metadata
    environment: Dict[str, str] = {}
    browser_version: Optional[str] = None
    framework_version: Optional[str] = None
    vision_model_version: Optional[str] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class TestReport(BaseModel):
    """
    Comprehensive test report.
    """
    report_id: str = Field(default_factory=lambda: str(uuid4()))
    test_id: str
    test_execution: TestExecution
    
    # Summary
    summary: str  # AI-generated executive summary
    pass_rate: float
    reliability_score: float
    
    # Detailed results
    step_results: List[Dict[str, Any]]
    failures: List[Dict[str, Any]]
    warnings: List[Dict[str, Any]]
    
    # AI analysis
    failure_analysis: Optional[str] = None  # AI-generated root cause analysis
    recommendations: List[str] = []  # AI-generated recommendations
    
    # Artifacts
    screenshots: List[str] = []
    video_url: Optional[str] = None
    network_har: Optional[str] = None
    console_logs: List[Dict[str, Any]] = []
    
    # Timestamps
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
```

**File**: `src/models/self_healing.py`

```python
"""
Data models for self-healing and learning
"""

from typing import List, Dict, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field
from uuid import uuid4
import numpy as np


class LocatorStrategy(BaseModel):
    """
    Element locator strategy.
    """
    strategy_id: str = Field(default_factory=lambda: str(uuid4()))
    strategy_type: str  # "vision", "xpath", "css", "hybrid"
    
    # Vision-based locator
    element_description: Optional[str] = None
    visual_embedding: Optional[List[float]] = None  # Serialized numpy array
    reference_screenshot: Optional[str] = None  # Path to reference image
    
    # DOM-based locator (fallback)
    xpath: Optional[str] = None
    css_selector: Optional[str] = None
    
    # Metadata
    confidence: float = 0.0
    success_rate: float = 0.0
    usage_count: int = 0
    last_used: Optional[datetime] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class HealingEvent(BaseModel):
    """
    Record of a self-healing operation.
    """
    healing_id: str = Field(default_factory=lambda: str(uuid4()))
    test_id: str
    step_id: str
    
    # Failure context
    failure_timestamp: datetime
    failure_reason: str
    original_locator: LocatorStrategy
    
    # Healing process
    healing_timestamp: datetime
    healing_method: str  # "visual_similarity", "semantic_search", "dom_analysis"
    new_locator: LocatorStrategy
    confidence_score: float
    
    # Outcome
    healing_successful: bool
    auto_committed: bool
    requires_review: bool
    reviewed_by: Optional[str] = None
    review_timestamp: Optional[datetime] = None
    review_approved: Optional[bool] = None
    
    # Application context
    application_version: Optional[str] = None
    ui_change_description: Optional[str] = None  # AI-generated
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class VisualBaseline(BaseModel):
    """
    Visual baseline for drift detection.
    """
    baseline_id: str = Field(default_factory=lambda: str(uuid4()))
    page_identifier: str  # URL pattern or page name
    
    # Visual data
    screenshot_path: str
    visual_embedding: List[float]
    dom_hash: str
    
    # Key UI regions
    regions: List[Dict[str, Any]] = []  # List of {name, bounding_box, embedding}
    
    # Metadata
    application_version: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class DriftDetectionResult(BaseModel):
    """
    Result of drift detection analysis.
    """
    detection_id: str = Field(default_factory=lambda: str(uuid4()))
    baseline_id: str
    
    # Comparison
    current_screenshot_path: str
    similarity_score: float
    drift_detected: bool
    drift_severity: str  # "none", "cosmetic", "structural", "functional", "critical"
    
    # Changes detected
    changed_regions: List[Dict[str, Any]] = []
    new_elements: List[Dict[str, Any]] = []
    removed_elements: List[Dict[str, Any]] = []
    
    # AI analysis
    change_description: str  # AI-generated description of changes
    impact_assessment: str  # AI assessment of impact on tests
    recommended_actions: List[str] = []
    
    # Metadata
    detected_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
```

### 4.3 Database Schemas

**PostgreSQL Schema for Document Store**:

```sql
-- Test Plans and Executions
CREATE TABLE test_plans (
    plan_id UUID PRIMARY KEY,
    test_id UUID NOT NULL,
    name TEXT NOT NULL,
    description TEXT,
    requirements TEXT NOT NULL,
    application_url TEXT NOT NULL,
    browser TEXT NOT NULL,
    framework TEXT NOT NULL,
    vision_model TEXT NOT NULL,
    test_scope TEXT[] NOT NULL,
    setup_steps JSONB NOT NULL DEFAULT '[]',
    test_steps JSONB NOT NULL DEFAULT '[]',
    teardown_steps JSONB NOT NULL DEFAULT '[]',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by TEXT NOT NULL,
    estimated_duration_seconds INTEGER,
    tags TEXT[] DEFAULT '{}',
    CONSTRAINT fk_test FOREIGN KEY (test_id) REFERENCES test_executions(test_id)
);

CREATE INDEX idx_test_plans_test_id ON test_plans(test_id);
CREATE INDEX idx_test_plans_created_at ON test_plans(created_at DESC);
CREATE INDEX idx_test_plans_tags ON test_plans USING GIN(tags);

-- Test Executions
CREATE TABLE test_executions (
    test_id UUID PRIMARY KEY,
    plan_id UUID NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('pending', 'running', 'passed', 'failed', 'healing', 'stopped', 'error')),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    duration_ms DOUBLE PRECISION,
    total_steps INTEGER NOT NULL DEFAULT 0,
    passed_steps INTEGER NOT NULL DEFAULT 0,
    failed_steps INTEGER NOT NULL DEFAULT 0,
    skipped_steps INTEGER NOT NULL DEFAULT 0,
    healed_steps INTEGER NOT NULL DEFAULT 0,
    report_path TEXT,
    video_path TEXT,
    log_path TEXT,
    accessibility_issues JSONB DEFAULT '[]',
    security_issues JSONB DEFAULT '[]',
    performance_metrics JSONB DEFAULT '{}',
    environment JSONB DEFAULT '{}',
    browser_version TEXT,
    framework_version TEXT,
    vision_model_version TEXT
);

CREATE INDEX idx_test_executions_status ON test_executions(status);
CREATE INDEX idx_test_executions_started_at ON test_executions(started_at DESC);
CREATE INDEX idx_test_executions_plan_id ON test_executions(plan_id);

-- Test Reports
CREATE TABLE test_reports (
    report_id UUID PRIMARY KEY,
    test_id UUID NOT NULL REFERENCES test_executions(test_id),
    summary TEXT NOT NULL,
    pass_rate DOUBLE PRECISION NOT NULL,
    reliability_score DOUBLE PRECISION NOT NULL,
    step_results JSONB NOT NULL,
    failures JSONB NOT NULL DEFAULT '[]',
    warnings JSONB NOT NULL DEFAULT '[]',
    failure_analysis TEXT,
    recommendations TEXT[] DEFAULT '{}',
    screenshots TEXT[] DEFAULT '{}',
    video_url TEXT,
    network_har TEXT,
    console_logs JSONB DEFAULT '[]',
    generated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_test_reports_test_id ON test_reports(test_id);
CREATE INDEX idx_test_reports_generated_at ON test_reports(generated_at DESC);

-- Locator Strategies
CREATE TABLE locator_strategies (
    strategy_id UUID PRIMARY KEY,
    strategy_type TEXT NOT NULL CHECK (strategy_type IN ('vision', 'xpath', 'css', 'hybrid')),
    element_description TEXT,
    visual_embedding VECTOR(512),  -- Using pgvector extension for similarity search
    reference_screenshot TEXT,
    xpath TEXT,
    css_selector TEXT,
    confidence DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    success_rate DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    usage_count INTEGER NOT NULL DEFAULT 0,
    last_used TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_locator_strategies_type ON locator_strategies(strategy_type);
CREATE INDEX idx_locator_strategies_embedding ON locator_strategies USING ivfflat (visual_embedding vector_cosine_ops);

-- Healing Events
CREATE TABLE healing_events (
    healing_id UUID PRIMARY KEY,
    test_id UUID NOT NULL REFERENCES test_executions(test_id),
    step_id TEXT NOT NULL,
    failure_timestamp TIMESTAMPTZ NOT NULL,
    failure_reason TEXT NOT NULL,
    original_locator_id UUID REFERENCES locator_strategies(strategy_id),
    healing_timestamp TIMESTAMPTZ NOT NULL,
    healing_method TEXT NOT NULL,
    new_locator_id UUID REFERENCES locator_strategies(strategy_id),
    confidence_score DOUBLE PRECISION NOT NULL,
    healing_successful BOOLEAN NOT NULL,
    auto_committed BOOLEAN NOT NULL DEFAULT FALSE,
    requires_review BOOLEAN NOT NULL DEFAULT TRUE,
    reviewed_by TEXT,
    review_timestamp TIMESTAMPTZ,
    review_approved BOOLEAN,
    application_version TEXT,
    ui_change_description TEXT
);

CREATE INDEX idx_healing_events_test_id ON healing_events(test_id);
CREATE INDEX idx_healing_events_timestamp ON healing_events(healing_timestamp DESC);
CREATE INDEX idx_healing_events_requires_review ON healing_events(requires_review) WHERE requires_review = TRUE;

-- Visual Baselines
CREATE TABLE visual_baselines (
    baseline_id UUID PRIMARY KEY,
    page_identifier TEXT NOT NULL,
    screenshot_path TEXT NOT NULL,
    visual_embedding VECTOR(512),
    dom_hash TEXT NOT NULL,
    regions JSONB NOT NULL DEFAULT '[]',
    application_version TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(page_identifier, application_version)
);

CREATE INDEX idx_visual_baselines_page ON visual_baselines(page_identifier);
CREATE INDEX idx_visual_baselines_version ON visual_baselines(application_version);
CREATE INDEX idx_visual_baselines_embedding ON visual_baselines USING ivfflat (visual_embedding vector_cosine_ops);

-- Drift Detection Results
CREATE TABLE drift_detection_results (
    detection_id UUID PRIMARY KEY,
    baseline_id UUID NOT NULL REFERENCES visual_baselines(baseline_id),
    current_screenshot_path TEXT NOT NULL,
    similarity_score DOUBLE PRECISION NOT NULL,
    drift_detected BOOLEAN NOT NULL,
    drift_severity TEXT NOT NULL CHECK (drift_severity IN ('none', 'cosmetic', 'structural', 'functional', 'critical')),
    changed_regions JSONB NOT NULL DEFAULT '[]',
    new_elements JSONB NOT NULL DEFAULT '[]',
    removed_elements JSONB NOT NULL DEFAULT '[]',
    change_description TEXT NOT NULL,
    impact_assessment TEXT NOT NULL,
    recommended_actions TEXT[] DEFAULT '{}',
    detected_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_drift_detection_baseline ON drift_detection_results(baseline_id);
CREATE INDEX idx_drift_detection_detected_at ON drift_detection_results(detected_at DESC);
CREATE INDEX idx_drift_detection_severity ON drift_detection_results(drift_severity);
```

**InfluxDB Schema for Time-Series Metrics**:

```
# Measurement: test_execution_metrics
# Tags: test_id, step_id, status, browser, framework
# Fields: duration_ms, confidence_score, retry_count, memory_mb, cpu_percent
# Timestamp: execution time

# Measurement: adapter_health_metrics
# Tags: adapter_name, adapter_type (vision|execution)
# Fields: latency_ms, error_rate, success_rate, availability
# Timestamp: health check time

# Measurement: system_metrics
# Tags: component (server|engine|adapter), instance_id
# Fields: cpu_percent, memory_mb, disk_io_mb, network_io_mb
# Timestamp: collection time

# Measurement: reliability_scores
# Tags: entity_type (test|module|adapter), entity_id
# Fields: score, pass_rate, stability, recovery_rate
# Timestamp: calculation time
```

### 4.4 Caching Strategy

**Redis/Valkey Cache Structure**:

```python
"""
Cache key patterns and TTL configuration
"""

CACHE_PATTERNS = {
    # Session state (short TTL)
    "session:{test_id}": {
        "ttl": 3600,  # 1 hour
        "type": "hash",
        "description": "Active test session state"
    },
    
    # Adapter health (medium TTL)
    "adapter:health:{adapter_name}": {
        "ttl": 60,  # 1 minute
        "type": "hash",
        "description": "Adapter health status"
    },
    
    # Vision model responses (long TTL)
    "vision:response:{image_hash}:{prompt_hash}": {
        "ttl": 86400,  # 24 hours
        "type": "string",
        "description": "Cached vision model response"
    },
    
    # Visual embeddings (long TTL)
    "embedding:{image_hash}": {
        "ttl": 604800,  # 7 days
        "type": "string",
        "description": "Cached visual embedding vector"
    },
    
    # Distributed locks
    "lock:{resource_name}": {
        "ttl": 30,  # 30 seconds
        "type": "string",
        "description": "Distributed lock for resource"
    },
    
    # Rate limiting
    "ratelimit:{adapter_name}:{window}": {
        "ttl": 60,  # 1 minute
        "type": "string",
        "description": "Rate limit counter"
    }
}
```

### 4.5 Blob Storage Organization

**S3/MinIO Bucket Structure**:

```
testdriver-artifacts/
├── screenshots/
│   ├── {test_id}/
│   │   ├── {step_id}_before.png
│   │   ├── {step_id}_after.png
│   │   └── {step_id}_annotated.png
│   └── baselines/
│       └── {page_identifier}_{version}.png
│
├── videos/
│   └── {test_id}/
│       ├── full_session.mp4
│       └── segments/
│           ├── {step_id}.mp4
│           └── ...
│
├── reports/
│   └── {test_id}/
│       ├── report.html
│       ├── report.json
│       └── report.pdf
│
├── logs/
│   └── {test_id}/
│       ├── execution.log
│       ├── network.har
│       └── console.log
│
└── learning/
    ├── reference_images/
    │   └── {strategy_id}.png
    └── embeddings/
        └── {embedding_id}.npy
```

**Lifecycle Policies**:

```yaml
lifecycle_rules:
  - name: "expire_old_screenshots"
    prefix: "screenshots/"
    expiration_days: 90
    
  - name: "expire_old_videos"
    prefix: "videos/"
    expiration_days: 30
    
  - name: "transition_reports_to_glacier"
    prefix: "reports/"
    transition_days: 180
    storage_class: "GLACIER"
    
  - name: "keep_learning_data_indefinitely"
    prefix: "learning/"
    expiration_days: null
```

This completes the data models, schemas, and storage architecture specifications.
