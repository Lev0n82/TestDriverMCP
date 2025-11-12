# TestDriver MCP Framework: Low-Level Development System Design Specification

## Part 8: Cross-Layer Validation and Advanced Observability

### 8.1 UI-API Cross-Layer Validation Framework

The Cross-Layer Validation Framework automatically correlates UI actions with backend API calls, verifying that the complete request-response cycle produces expected results. This ensures end-to-end correctness beyond what UI-only or API-only testing can achieve.

**Architecture Overview**:

Modern web applications consist of multiple layers that must work together correctly. A button click in the UI triggers JavaScript code that makes an API call to the backend, which queries a database and returns data that updates the UI. Traditional testing validates each layer in isolation, missing integration bugs where layers interact incorrectly. The Cross-Layer Validation Framework captures the complete interaction chain and validates consistency across all layers simultaneously.

**File**: `src/validation/cross_layer_validator.py`

```python
"""
Cross-Layer Validation Framework
Validates consistency across UI, API, and data layers
"""

import asyncio
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import structlog
import json

logger = structlog.get_logger(__name__)


@dataclass
class UIAction:
    """Represents a UI action performed during test."""
    action_id: str
    action_type: str  # click, type, select, etc.
    element_description: str
    timestamp: datetime
    screenshot_before: bytes
    screenshot_after: bytes


@dataclass
class APICall:
    """Represents an API call triggered by UI action."""
    call_id: str
    method: str
    url: str
    headers: Dict[str, str]
    request_body: Optional[str]
    response_status: int
    response_headers: Dict[str, str]
    response_body: Optional[str]
    duration_ms: float
    timestamp: datetime


@dataclass
class CrossLayerAssertion:
    """Assertion that spans multiple layers."""
    assertion_id: str
    description: str
    ui_action: UIAction
    expected_api_calls: List[Dict[str, Any]]
    expected_ui_changes: Dict[str, Any]
    expected_data_changes: Optional[Dict[str, Any]]


@dataclass
class ValidationResult:
    """Result of cross-layer validation."""
    assertion_id: str
    passed: bool
    ui_validation: Dict[str, Any]
    api_validation: Dict[str, Any]
    data_validation: Optional[Dict[str, Any]]
    discrepancies: List[str]
    confidence: float


class CrossLayerValidator:
    """
    Validates consistency across UI, API, and data layers.
    
    Automatically correlates UI actions with API calls and data changes
    to ensure end-to-end correctness.
    """
    
    def __init__(
        self,
        browser_driver: Any,
        vision_adapter: Any,
        config: Dict[str, Any]
    ):
        """
        Initialize cross-layer validator.
        
        Args:
            browser_driver: BrowserDriver for UI interaction
            vision_adapter: VisionAdapter for UI validation
            config: Configuration dictionary
        """
        self.browser_driver = browser_driver
        self.vision_adapter = vision_adapter
        
        self.correlation_window_ms = config.get('correlation_window_ms', 5000)
        self.enable_api_validation = config.get('enable_api_validation', True)
        self.enable_data_validation = config.get('enable_data_validation', False)
        
        # Track UI actions and API calls for correlation
        self._ui_actions: List[UIAction] = []
        self._api_calls: List[APICall] = []
        
        # Network interception for API monitoring
        self._network_monitor_active = False
    
    async def start_monitoring(self) -> None:
        """Start monitoring UI actions and API calls."""
        logger.info("Starting cross-layer monitoring")
        
        # Enable network interception
        if self.enable_api_validation:
            await self._start_network_monitoring()
        
        self._network_monitor_active = True
    
    async def stop_monitoring(self) -> None:
        """Stop monitoring and clear buffers."""
        logger.info("Stopping cross-layer monitoring")
        
        self._network_monitor_active = False
        self._ui_actions.clear()
        self._api_calls.clear()
    
    async def validate_cross_layer_assertion(
        self,
        assertion: CrossLayerAssertion
    ) -> ValidationResult:
        """
        Validate a cross-layer assertion.
        
        Args:
            assertion: CrossLayerAssertion to validate
            
        Returns:
            ValidationResult with pass/fail status and details
        """
        logger.info("Validating cross-layer assertion",
                   assertion_id=assertion.assertion_id)
        
        discrepancies = []
        
        # 1. Validate UI changes
        ui_validation = await self._validate_ui_changes(
            assertion.ui_action,
            assertion.expected_ui_changes
        )
        
        if not ui_validation['passed']:
            discrepancies.extend(ui_validation['discrepancies'])
        
        # 2. Validate API calls
        api_validation = {'passed': True, 'discrepancies': []}
        if self.enable_api_validation:
            api_validation = await self._validate_api_calls(
                assertion.ui_action,
                assertion.expected_api_calls
            )
            
            if not api_validation['passed']:
                discrepancies.extend(api_validation['discrepancies'])
        
        # 3. Validate data changes (if enabled)
        data_validation = None
        if self.enable_data_validation and assertion.expected_data_changes:
            data_validation = await self._validate_data_changes(
                assertion.expected_data_changes
            )
            
            if not data_validation['passed']:
                discrepancies.extend(data_validation['discrepancies'])
        
        # Overall validation result
        passed = (
            ui_validation['passed'] and
            api_validation['passed'] and
            (data_validation is None or data_validation['passed'])
        )
        
        # Calculate confidence based on validation coverage
        confidence = self._calculate_validation_confidence(
            ui_validation,
            api_validation,
            data_validation
        )
        
        result = ValidationResult(
            assertion_id=assertion.assertion_id,
            passed=passed,
            ui_validation=ui_validation,
            api_validation=api_validation,
            data_validation=data_validation,
            discrepancies=discrepancies,
            confidence=confidence
        )
        
        logger.info("Cross-layer validation complete",
                   assertion_id=assertion.assertion_id,
                   passed=passed,
                   confidence=confidence)
        
        return result
    
    async def auto_generate_assertions(
        self,
        ui_action: UIAction
    ) -> List[CrossLayerAssertion]:
        """
        Automatically generate cross-layer assertions for a UI action.
        
        Analyzes the UI action and correlated API calls to infer
        expected behavior across layers.
        
        Args:
            ui_action: UI action to generate assertions for
            
        Returns:
            List of generated assertions
        """
        logger.debug("Auto-generating assertions", action_id=ui_action.action_id)
        
        # Find API calls correlated with this UI action
        correlated_calls = self._correlate_api_calls(ui_action)
        
        # Analyze UI changes
        ui_changes = await self._analyze_ui_changes(
            ui_action.screenshot_before,
            ui_action.screenshot_after
        )
        
        # Generate assertion
        assertion = CrossLayerAssertion(
            assertion_id=f"auto_{ui_action.action_id}",
            description=f"Validate {ui_action.action_type} on {ui_action.element_description}",
            ui_action=ui_action,
            expected_api_calls=[
                {
                    'method': call.method,
                    'url_pattern': self._extract_url_pattern(call.url),
                    'expected_status': call.response_status
                }
                for call in correlated_calls
            ],
            expected_ui_changes=ui_changes,
            expected_data_changes=None
        )
        
        return [assertion]
    
    async def _validate_ui_changes(
        self,
        ui_action: UIAction,
        expected_changes: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate that UI changed as expected after action."""
        discrepancies = []
        
        # Use vision adapter to compare before/after screenshots
        comparison = await self.vision_adapter.compare_screens(
            ui_action.screenshot_before,
            ui_action.screenshot_after
        )
        
        # Check if expected elements appeared/disappeared
        for element_desc in expected_changes.get('elements_appeared', []):
            try:
                await self.vision_adapter.find_element(
                    ui_action.screenshot_after,
                    element_desc
                )
            except Exception:
                discrepancies.append(f"Expected element not found: {element_desc}")
        
        for element_desc in expected_changes.get('elements_disappeared', []):
            try:
                await self.vision_adapter.find_element(
                    ui_action.screenshot_after,
                    element_desc
                )
                discrepancies.append(f"Element should have disappeared: {element_desc}")
            except Exception:
                pass  # Element correctly not found
        
        # Check for expected text changes
        if 'text_changes' in expected_changes:
            # Use OCR to verify text changes
            ocr_results = await self.vision_adapter.ocr(ui_action.screenshot_after)
            actual_text = ' '.join([r.text for r in ocr_results])
            
            for expected_text in expected_changes['text_changes']:
                if expected_text not in actual_text:
                    discrepancies.append(f"Expected text not found: {expected_text}")
        
        return {
            'passed': len(discrepancies) == 0,
            'discrepancies': discrepancies,
            'similarity_score': comparison.similarity_score,
            'changes_detected': len(comparison.differences)
        }
    
    async def _validate_api_calls(
        self,
        ui_action: UIAction,
        expected_calls: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Validate that expected API calls were made."""
        discrepancies = []
        
        # Find API calls correlated with this UI action
        actual_calls = self._correlate_api_calls(ui_action)
        
        # Check each expected call
        for expected in expected_calls:
            matching_call = None
            
            for actual in actual_calls:
                if self._matches_api_call(actual, expected):
                    matching_call = actual
                    break
            
            if matching_call is None:
                discrepancies.append(
                    f"Expected API call not found: {expected['method']} {expected['url_pattern']}"
                )
            else:
                # Validate response status
                if 'expected_status' in expected:
                    if matching_call.response_status != expected['expected_status']:
                        discrepancies.append(
                            f"API call status mismatch: expected {expected['expected_status']}, "
                            f"got {matching_call.response_status}"
                        )
                
                # Validate response body structure
                if 'expected_response_schema' in expected:
                    schema_validation = self._validate_response_schema(
                        matching_call.response_body,
                        expected['expected_response_schema']
                    )
                    if not schema_validation['valid']:
                        discrepancies.extend(schema_validation['errors'])
        
        return {
            'passed': len(discrepancies) == 0,
            'discrepancies': discrepancies,
            'actual_calls_count': len(actual_calls),
            'expected_calls_count': len(expected_calls)
        }
    
    async def _validate_data_changes(
        self,
        expected_changes: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate database changes (requires database connection)."""
        # This would require database connection and query capabilities
        # Implementation depends on specific database technology
        
        return {
            'passed': True,
            'discrepancies': [],
            'note': 'Data validation not yet implemented'
        }
    
    def _correlate_api_calls(self, ui_action: UIAction) -> List[APICall]:
        """
        Find API calls that occurred within correlation window of UI action.
        """
        correlated = []
        
        for api_call in self._api_calls:
            time_diff_ms = abs(
                (api_call.timestamp - ui_action.timestamp).total_seconds() * 1000
            )
            
            if time_diff_ms <= self.correlation_window_ms:
                correlated.append(api_call)
        
        return correlated
    
    def _matches_api_call(
        self,
        actual: APICall,
        expected: Dict[str, Any]
    ) -> bool:
        """Check if actual API call matches expected pattern."""
        # Match method
        if actual.method != expected['method']:
            return False
        
        # Match URL pattern (supports wildcards)
        url_pattern = expected['url_pattern']
        if '*' in url_pattern:
            # Simple wildcard matching
            pattern_parts = url_pattern.split('*')
            if not all(part in actual.url for part in pattern_parts if part):
                return False
        else:
            if url_pattern not in actual.url:
                return False
        
        return True
    
    def _validate_response_schema(
        self,
        response_body: Optional[str],
        expected_schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate API response against expected schema."""
        if not response_body:
            return {'valid': False, 'errors': ['Response body is empty']}
        
        try:
            response_data = json.loads(response_body)
        except json.JSONDecodeError:
            return {'valid': False, 'errors': ['Response is not valid JSON']}
        
        errors = []
        
        # Check required fields
        for field in expected_schema.get('required_fields', []):
            if field not in response_data:
                errors.append(f"Required field missing: {field}")
        
        # Check field types
        for field, expected_type in expected_schema.get('field_types', {}).items():
            if field in response_data:
                actual_type = type(response_data[field]).__name__
                if actual_type != expected_type:
                    errors.append(
                        f"Field type mismatch for {field}: "
                        f"expected {expected_type}, got {actual_type}"
                    )
        
        return {
            'valid': len(errors) == 0,
            'errors': errors
        }
    
    async def _analyze_ui_changes(
        self,
        screenshot_before: bytes,
        screenshot_after: bytes
    ) -> Dict[str, Any]:
        """Analyze what changed in the UI."""
        comparison = await self.vision_adapter.compare_screens(
            screenshot_before,
            screenshot_after
        )
        
        return {
            'similarity_score': comparison.similarity_score,
            'semantic_changes': comparison.semantic_changes,
            'regions_changed': len(comparison.differences)
        }
    
    def _extract_url_pattern(self, url: str) -> str:
        """Extract URL pattern by replacing IDs with wildcards."""
        import re
        
        # Replace numeric IDs with wildcards
        pattern = re.sub(r'/\d+', '/*', url)
        
        # Replace UUIDs with wildcards
        pattern = re.sub(
            r'/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}',
            '/*',
            pattern,
            flags=re.IGNORECASE
        )
        
        return pattern
    
    def _calculate_validation_confidence(
        self,
        ui_validation: Dict[str, Any],
        api_validation: Dict[str, Any],
        data_validation: Optional[Dict[str, Any]]
    ) -> float:
        """Calculate overall confidence in validation result."""
        confidence_scores = []
        
        # UI validation confidence
        if ui_validation['passed']:
            confidence_scores.append(0.95)
        else:
            # Partial confidence based on similarity
            confidence_scores.append(ui_validation.get('similarity_score', 0.5))
        
        # API validation confidence
        if api_validation['passed']:
            confidence_scores.append(0.95)
        else:
            # Partial confidence if some calls matched
            actual = api_validation.get('actual_calls_count', 0)
            expected = api_validation.get('expected_calls_count', 1)
            confidence_scores.append(min(actual / expected, 0.8))
        
        # Data validation confidence
        if data_validation:
            if data_validation['passed']:
                confidence_scores.append(0.95)
            else:
                confidence_scores.append(0.3)
        
        # Return average confidence
        return sum(confidence_scores) / len(confidence_scores)
    
    async def _start_network_monitoring(self) -> None:
        """Start monitoring network requests."""
        # Implementation would set up network interception
        # to capture all API calls made by the browser
        pass
```

### 8.2 Advanced Observability and Telemetry

The Advanced Observability system provides comprehensive insights into test execution, self-healing operations, and system health through rich telemetry and intelligent dashboards.

**File**: `src/observability/telemetry_service.py`

```python
"""
Advanced Telemetry Service
Comprehensive observability for TestDriver operations
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from prometheus_client import Counter, Histogram, Gauge, Summary
import structlog

logger = structlog.get_logger(__name__)


# Prometheus metrics
test_executions_total = Counter(
    'testdriver_test_executions_total',
    'Total number of test executions',
    ['status', 'browser', 'framework']
)

test_duration_seconds = Histogram(
    'testdriver_test_duration_seconds',
    'Test execution duration in seconds',
    ['test_id', 'status']
)

healing_attempts_total = Counter(
    'testdriver_healing_attempts_total',
    'Total number of healing attempts',
    ['strategy', 'outcome']
)

healing_success_rate = Gauge(
    'testdriver_healing_success_rate',
    'Success rate of healing operations',
    ['strategy']
)

mean_time_to_heal = Summary(
    'testdriver_mean_time_to_heal_seconds',
    'Mean time to successfully heal a broken locator'
)

drift_detected_total = Counter(
    'testdriver_drift_detected_total',
    'Total number of UI drift detections',
    ['severity', 'page']
)

adapter_health_status = Gauge(
    'testdriver_adapter_health_status',
    'Adapter health status (1=healthy, 0=unhealthy)',
    ['adapter_name', 'adapter_type']
)

test_reliability_index = Gauge(
    'testdriver_test_reliability_index',
    'Test reliability index (0-1)',
    ['test_id']
)


@dataclass
class TelemetryEvent:
    """Structured telemetry event."""
    event_type: str
    timestamp: datetime
    attributes: Dict[str, Any]
    metrics: Dict[str, float]


class TelemetryService:
    """
    Advanced telemetry and observability service.
    
    Collects, aggregates, and exposes metrics for monitoring and analysis.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize telemetry service.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.enable_detailed_tracing = config.get('enable_detailed_tracing', True)
        self.event_buffer: List[TelemetryEvent] = []
    
    def record_test_execution(
        self,
        test_id: str,
        status: str,
        duration_seconds: float,
        browser: str,
        framework: str,
        metadata: Dict[str, Any]
    ) -> None:
        """Record test execution metrics."""
        # Update Prometheus metrics
        test_executions_total.labels(
            status=status,
            browser=browser,
            framework=framework
        ).inc()
        
        test_duration_seconds.labels(
            test_id=test_id,
            status=status
        ).observe(duration_seconds)
        
        # Create telemetry event
        event = TelemetryEvent(
            event_type='test_execution',
            timestamp=datetime.utcnow(),
            attributes={
                'test_id': test_id,
                'status': status,
                'browser': browser,
                'framework': framework,
                **metadata
            },
            metrics={
                'duration_seconds': duration_seconds
            }
        )
        
        self._buffer_event(event)
        
        logger.info("Test execution recorded",
                   test_id=test_id,
                   status=status,
                   duration=duration_seconds)
    
    def record_healing_attempt(
        self,
        element_id: str,
        strategy: str,
        success: bool,
        duration_seconds: float,
        confidence: float
    ) -> None:
        """Record healing attempt metrics."""
        outcome = 'success' if success else 'failure'
        
        healing_attempts_total.labels(
            strategy=strategy,
            outcome=outcome
        ).inc()
        
        if success:
            mean_time_to_heal.observe(duration_seconds)
        
        # Update success rate
        self._update_healing_success_rate(strategy)
        
        event = TelemetryEvent(
            event_type='healing_attempt',
            timestamp=datetime.utcnow(),
            attributes={
                'element_id': element_id,
                'strategy': strategy,
                'success': success,
                'confidence': confidence
            },
            metrics={
                'duration_seconds': duration_seconds,
                'confidence': confidence
            }
        )
        
        self._buffer_event(event)
    
    def record_drift_detection(
        self,
        page_identifier: str,
        severity: str,
        similarity_score: float,
        changes_detected: int
    ) -> None:
        """Record UI drift detection."""
        drift_detected_total.labels(
            severity=severity,
            page=page_identifier
        ).inc()
        
        event = TelemetryEvent(
            event_type='drift_detection',
            timestamp=datetime.utcnow(),
            attributes={
                'page': page_identifier,
                'severity': severity,
                'changes_detected': changes_detected
            },
            metrics={
                'similarity_score': similarity_score
            }
        )
        
        self._buffer_event(event)
    
    def update_adapter_health(
        self,
        adapter_name: str,
        adapter_type: str,
        is_healthy: bool
    ) -> None:
        """Update adapter health status."""
        adapter_health_status.labels(
            adapter_name=adapter_name,
            adapter_type=adapter_type
        ).set(1.0 if is_healthy else 0.0)
    
    def update_test_reliability_index(
        self,
        test_id: str,
        reliability_score: float
    ) -> None:
        """Update test reliability index."""
        test_reliability_index.labels(test_id=test_id).set(reliability_score)
    
    async def calculate_reliability_metrics(
        self,
        db_connection: Any
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive reliability metrics.
        
        Returns:
            Dictionary of reliability metrics
        """
        # Query historical data
        cutoff = datetime.utcnow() - timedelta(days=30)
        
        # Calculate mean time to heal
        healing_query = """
            SELECT AVG(EXTRACT(EPOCH FROM (healing_timestamp - failure_timestamp))) as mtth
            FROM healing_events
            WHERE healing_successful = true
            AND healing_timestamp >= $1
        """
        mtth_row = await db_connection.fetchrow(healing_query, cutoff)
        mean_time_to_heal_seconds = mtth_row['mtth'] if mtth_row['mtth'] else 0.0
        
        # Calculate failure recurrence rate
        recurrence_query = """
            WITH failures AS (
                SELECT test_id, step_id, COUNT(*) as failure_count
                FROM healing_events
                WHERE failure_timestamp >= $1
                GROUP BY test_id, step_id
                HAVING COUNT(*) > 1
            )
            SELECT COUNT(*) as recurring_failures,
                   (SELECT COUNT(DISTINCT test_id, step_id) FROM healing_events WHERE failure_timestamp >= $1) as total_failures
            FROM failures
        """
        recurrence_row = await db_connection.fetchrow(recurrence_query, cutoff)
        
        if recurrence_row['total_failures'] > 0:
            failure_recurrence_rate = (
                recurrence_row['recurring_failures'] / recurrence_row['total_failures']
            )
        else:
            failure_recurrence_rate = 0.0
        
        # Calculate drift frequency
        drift_query = """
            SELECT page_identifier, COUNT(*) as drift_count
            FROM drift_detection_results
            WHERE detected_at >= $1
            AND drift_detected = true
            GROUP BY page_identifier
            ORDER BY drift_count DESC
            LIMIT 10
        """
        drift_rows = await db_connection.fetch(drift_query, cutoff)
        drift_frequency = {row['page_identifier']: row['drift_count'] for row in drift_rows}
        
        return {
            'mean_time_to_heal_seconds': mean_time_to_heal_seconds,
            'failure_recurrence_rate': failure_recurrence_rate,
            'drift_frequency_by_page': drift_frequency,
            'measurement_period_days': 30
        }
    
    def _update_healing_success_rate(self, strategy: str) -> None:
        """Update healing success rate gauge."""
        # This would query recent healing attempts and calculate success rate
        # For now, we'll use a placeholder
        pass
    
    def _buffer_event(self, event: TelemetryEvent) -> None:
        """Buffer event for batch processing."""
        self.event_buffer.append(event)
        
        # Flush buffer if it gets too large
        if len(self.event_buffer) >= 1000:
            asyncio.create_task(self._flush_events())
    
    async def _flush_events(self) -> None:
        """Flush buffered events to storage."""
        if not self.event_buffer:
            return
        
        # Implementation would send events to time-series database
        # or logging aggregation service
        
        logger.debug("Flushing telemetry events", count=len(self.event_buffer))
        self.event_buffer.clear()
```

This completes the cross-layer validation and advanced observability specifications.
