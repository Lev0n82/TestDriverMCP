"""Core data models for TestDriver MCP Framework."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class TestStepType(str, Enum):
    """Types of test steps."""
    NAVIGATE = "navigate"
    CLICK = "click"
    TYPE = "type"
    VERIFY = "verify"
    WAIT = "wait"
    SCREENSHOT = "screenshot"
    CUSTOM = "custom"


class DetectionMode(str, Enum):
    """Element detection modes."""
    DOM = "dom"
    VISION = "vision"
    HYBRID = "hybrid"


class HealingStrategy(str, Enum):
    """Healing strategies for broken locators."""
    VISUAL_SIMILARITY = "visual_similarity"
    SEMANTIC_SIMILARITY = "semantic_similarity"
    STRUCTURAL_SIMILARITY = "structural_similarity"
    BEHAVIORAL_SIMILARITY = "behavioral_similarity"
    MEMORY_LOOKUP = "memory_lookup"


class ValidationMethod(str, Enum):
    """Methods for validating healing results."""
    AUTO = "auto"
    MANUAL = "manual"
    PEER_REVIEW = "peer_review"


class TestStep(BaseModel):
    """Individual test step."""
    step_id: str
    step_type: TestStepType
    description: str
    target_element: Optional[str] = None
    action: Optional[str] = None
    input_data: Optional[str] = None
    expected_result: Optional[str] = None
    detection_mode: DetectionMode = DetectionMode.HYBRID
    timeout_seconds: int = 30
    retry_count: int = 3


class TestPlan(BaseModel):
    """Complete test plan."""
    test_id: str
    test_name: str
    description: str
    target_url: str
    steps: List[TestStep]
    tags: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: str = "testdriver"


class TestStepResult(BaseModel):
    """Result of executing a test step."""
    step_id: str
    success: bool
    execution_time_ms: float
    screenshot_path: Optional[str] = None
    error_message: Optional[str] = None
    healing_applied: bool = False
    healing_confidence: Optional[float] = None
    detection_mode_used: DetectionMode
    retry_count: int = 0


class TestExecution(BaseModel):
    """Complete test execution record."""
    execution_id: str
    test_id: str
    test_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    status: str  # running, passed, failed, error
    step_results: List[TestStepResult] = Field(default_factory=list)
    total_steps: int
    passed_steps: int = 0
    failed_steps: int = 0
    healing_events: int = 0
    environment: Dict[str, Any] = Field(default_factory=dict)


class HealingEvent(BaseModel):
    """Record of a healing event."""
    event_id: str
    test_id: str
    execution_id: str
    element_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    original_locator: Dict[str, Any]
    failure_reason: str
    failure_screenshot: Optional[str] = None
    healing_strategy: HealingStrategy
    new_locator: Dict[str, Any]
    confidence_score: float
    healing_successful: bool
    validation_method: ValidationMethod
    user_feedback: Optional[str] = None
    visual_embedding: Optional[List[float]] = None
    semantic_embedding: Optional[List[float]] = None
    context_features: Dict[str, Any] = Field(default_factory=dict)


class TestReport(BaseModel):
    """Comprehensive test report."""
    report_id: str
    test_id: str
    execution_id: str
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    summary: Dict[str, Any]
    step_details: List[Dict[str, Any]]
    healing_summary: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    recommendations: List[str] = Field(default_factory=list)


class ElementStability(BaseModel):
    """Element stability metrics."""
    element_id: str
    total_executions: int
    successful_executions: int
    healing_count: int
    stability_score: float  # 0.0 to 1.0
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    trend: str  # improving, stable, degrading


class SuccessCriterion(BaseModel):
    """Success criterion definition."""
    criterion_id: str
    name: str
    level: str  # system, module, class, function
    category: str  # functional, performance, reliability, security, maintainability
    description: str
    measurement_method: str
    threshold: Any
    measurement_frequency: str
    dependencies: List[str] = Field(default_factory=list)
    owner: str
    severity: str = "normal"  # critical, high, normal, low
    remediation_advice: str = ""


class ValidationFailure(BaseModel):
    """Self-test validation failure."""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    component: str
    criterion_id: str
    criterion_name: str
    phase: str  # precondition, postcondition, error_handling
    context: Dict[str, Any]
    recommendation: str


class EvaluationResult(BaseModel):
    """Result of criterion evaluation."""
    criterion_id: str
    passed: bool
    actual_value: Any
    threshold: Any
    gap: Optional[float] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    details: Dict[str, Any] = Field(default_factory=dict)


class Alert(BaseModel):
    """System alert."""
    alert_id: Optional[str] = None
    severity: str  # critical, high, normal, low
    title: str
    description: str
    criterion_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    resolved: bool = False
    resolved_at: Optional[datetime] = None
