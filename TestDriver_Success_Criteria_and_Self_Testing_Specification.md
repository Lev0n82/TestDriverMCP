# TestDriver MCP Framework: Success Criteria and Self-Testing Specification

**Version**: 4.0  
**Date**: November 2025  
**Author**: Manus AI  
**Status**: Success Criteria and Self-Testing Framework

---

## Executive Summary

This document defines comprehensive, measurable success criteria for every component of the TestDriver MCP Framework at function, class, module, and system levels. It also specifies a built-in self-testing architecture that continuously validates system functionality, performance, and reliability without external test infrastructure.

The success criteria framework follows a hierarchical structure where system-level criteria decompose into module-level criteria, which further decompose into class-level and function-level criteria. Each criterion is measurable, testable, and aligned with business objectives. The built-in self-testing functionality ensures that the system can validate its own correctness continuously, providing real-time health monitoring and early detection of regressions.

---

## Table of Contents

1. [Success Criteria Hierarchy](#1-success-criteria-hierarchy)
2. [Success Criteria Framework](#2-success-criteria-framework)
3. [System-Level Success Criteria](#3-system-level-success-criteria)
4. [Module-Level Success Criteria](#4-module-level-success-criteria)
5. [Class-Level Success Criteria](#5-class-level-success-criteria)
6. [Function-Level Success Criteria](#6-function-level-success-criteria)
7. [Built-In Self-Testing Architecture](#7-built-in-self-testing-architecture)
8. [Self-Testing Implementation](#8-self-testing-implementation)
9. [Continuous Validation Framework](#9-continuous-validation-framework)
10. [Success Metrics Dashboard](#10-success-metrics-dashboard)

---

## 1. Success Criteria Hierarchy

### 1.1 Hierarchical Structure

The success criteria framework follows a four-level hierarchy that ensures alignment from individual functions up to system-wide objectives. This hierarchical approach enables traceability where every function-level criterion contributes to class-level criteria, which aggregate into module-level criteria, ultimately supporting system-level success.

**Level 1: System-Level Criteria** define overall system success from a business and operational perspective. These criteria measure end-to-end capabilities such as test execution throughput, maintenance burden reduction, defect detection effectiveness, and operational availability. System-level criteria are measured over extended periods (weeks to months) and represent the ultimate value delivered to users.

**Level 2: Module-Level Criteria** define success for major architectural components such as the MCP Server, Vision Adapters, Execution Framework, Self-Healing Engine, and Telemetry Service. Module-level criteria measure component-specific capabilities such as adapter health, healing success rates, and metric collection completeness. These criteria are measured over medium periods (days to weeks) and ensure each major component fulfills its architectural responsibilities.

**Level 3: Class-Level Criteria** define success for individual classes within modules. Class-level criteria measure specific behaviors such as correct state management, proper error handling, resource cleanup, and performance characteristics. These criteria are measured per operation or over short periods (minutes to hours) and ensure each class implements its contract correctly.

**Level 4: Function-Level Criteria** define success for individual functions and methods. Function-level criteria measure specific behaviors such as correct input validation, expected output generation, error handling, and performance bounds. These criteria are measured per invocation and ensure each function behaves correctly under all conditions.

### 1.2 Criteria Categories

Success criteria are organized into five categories that span functional correctness, performance, reliability, security, and maintainability.

**Functional Correctness Criteria** verify that components produce correct outputs for given inputs, handle edge cases appropriately, maintain correct state transitions, and integrate correctly with dependencies. These criteria ensure the system does what it is supposed to do.

**Performance Criteria** verify that components meet latency requirements, achieve throughput targets, use resources efficiently, and scale appropriately under load. These criteria ensure the system performs adequately under expected and peak loads.

**Reliability Criteria** verify that components handle failures gracefully, recover from errors automatically, maintain data consistency, and provide high availability. These criteria ensure the system continues operating correctly even when failures occur.

**Security Criteria** verify that components authenticate and authorize correctly, protect sensitive data, prevent injection attacks, and maintain audit trails. These criteria ensure the system resists security threats and maintains data privacy.

**Maintainability Criteria** verify that components have clear interfaces, comprehensive logging, complete documentation, and testable designs. These criteria ensure the system can be understood, debugged, and evolved over time.

### 1.3 Measurement Approach

Each success criterion includes a specific measurement method that defines how to determine whether the criterion is met. Measurement methods fall into several categories.

**Automated Testing** uses unit tests, integration tests, and end-to-end tests to verify functional correctness. Tests are executed continuously in CI/CD pipelines and on production systems through built-in self-testing.

**Performance Monitoring** uses instrumentation and metrics collection to measure latency, throughput, and resource utilization. Metrics are compared against defined thresholds to determine success.

**Statistical Analysis** uses historical data analysis to measure reliability metrics such as mean time between failures, error rates, and availability percentages.

**Security Scanning** uses automated security tools to detect vulnerabilities, verify encryption, and validate access controls.

**Code Analysis** uses static analysis tools to measure code quality, complexity, and adherence to standards.

---

## 2. Success Criteria Framework

### 2.1 Criterion Definition Template

Each success criterion is defined using a standardized template that ensures completeness and measurability.

| Field | Description |
|:------|:------------|
| **Criterion ID** | Unique identifier for traceability (e.g., SYS-001, MOD-MCP-001, CLS-VISION-001) |
| **Criterion Name** | Concise descriptive name |
| **Level** | System, Module, Class, or Function |
| **Category** | Functional, Performance, Reliability, Security, or Maintainability |
| **Description** | Detailed description of what success means |
| **Measurement Method** | How to measure whether criterion is met |
| **Success Threshold** | Quantitative threshold defining success |
| **Measurement Frequency** | How often to measure (per invocation, hourly, daily, etc.) |
| **Dependencies** | Other criteria that must be met for this criterion to be valid |
| **Owner** | Module or component responsible for meeting this criterion |

### 2.2 Example Criterion Definition

To illustrate the framework, consider a criterion for the AI Locator Healing Engine.

| Field | Value |
|:------|:------|
| **Criterion ID** | CLS-HEAL-001 |
| **Criterion Name** | Healing Success Rate |
| **Level** | Class |
| **Category** | Functional Correctness |
| **Description** | The AI Locator Healing Engine successfully heals broken locators with high accuracy, measured as the percentage of healing attempts that result in correct element location |
| **Measurement Method** | Track healing attempts and outcomes; calculate success_rate = successful_healings / total_healing_attempts over rolling 7-day window |
| **Success Threshold** | ≥ 80% success rate for healing attempts with confidence ≥ 0.7 |
| **Measurement Frequency** | Continuous tracking with daily aggregation |
| **Dependencies** | Vision adapter availability (MOD-VISION-001), Test Memory Store availability (MOD-MEMORY-001) |
| **Owner** | Self-Healing Module |

### 2.3 Traceability Matrix

The framework maintains a traceability matrix that maps function-level criteria to class-level criteria, class-level to module-level, and module-level to system-level. This ensures that every low-level criterion contributes to higher-level success and that all system-level criteria are supported by concrete implementation-level criteria.

For example, the system-level criterion "Reduce test maintenance effort by 60-80%" traces down through module-level criteria for healing success rates and learning effectiveness, to class-level criteria for healing accuracy and memory retention, to function-level criteria for similarity search performance and embedding quality.

---

## 3. System-Level Success Criteria

System-level criteria define overall success from business and operational perspectives. These criteria are measured over extended periods and represent the ultimate value delivered to users.

### 3.1 Test Maintenance Reduction

**Criterion ID**: SYS-001  
**Category**: Functional Correctness

**Description**: The system reduces test maintenance effort through autonomous self-healing and adaptive learning. Maintenance effort is measured as hours spent by QA engineers manually fixing broken tests per month.

**Measurement Method**: Track time spent on manual test fixes before and after TestDriver deployment. Calculate reduction percentage as (baseline_hours - current_hours) / baseline_hours × 100.

**Success Thresholds**:
- Phase 1 (0-6 months): ≥ 30% reduction in maintenance hours
- Phase 2 (6-12 months): ≥ 60% reduction in maintenance hours
- Phase 3 (12-18 months): ≥ 80% reduction in maintenance hours

**Measurement Frequency**: Monthly with quarterly trend analysis

**Dependencies**: Healing success rate (MOD-HEAL-001), Learning effectiveness (MOD-LEARN-001)

### 3.2 Test Reliability Improvement

**Criterion ID**: SYS-002  
**Category**: Reliability

**Description**: The system improves test reliability by reducing flaky tests through adaptive waiting, resilient execution, and predictive failure detection. Test reliability is measured as the percentage of test executions that produce consistent results across multiple runs.

**Measurement Method**: Execute each test three times in identical conditions. Calculate reliability as percentage of tests where all three executions produce the same result (all pass or all fail).

**Success Thresholds**:
- Phase 1: ≥ 90% test reliability (≤ 10% flaky tests)
- Phase 2: ≥ 95% test reliability (≤ 5% flaky tests)
- Phase 3: ≥ 98% test reliability (≤ 2% flaky tests)

**Measurement Frequency**: Daily with weekly trend analysis

**Dependencies**: Adaptive wait effectiveness (MOD-WAIT-001), State synchronization (MOD-STATE-001)

### 3.3 Defect Detection Effectiveness

**Criterion ID**: SYS-003  
**Category**: Functional Correctness

**Description**: The system detects defects effectively across multiple testing dimensions including functional, accessibility, security, performance, and visual regression. Effectiveness is measured as the percentage of production defects that were detected in testing.

**Measurement Method**: Track defects found in production and correlate with testing coverage. Calculate detection_rate = defects_found_in_testing / total_defects × 100.

**Success Thresholds**:
- Phase 1: ≥ 70% defect detection rate
- Phase 2: ≥ 85% defect detection rate (with expanded testing scope)
- Phase 3: ≥ 95% defect detection rate (with predictive analytics)

**Measurement Frequency**: Per release with monthly aggregation

**Dependencies**: Multi-layer verification (MOD-MLV-001), Cross-layer validation (MOD-CROSS-001)

### 3.4 System Availability

**Criterion ID**: SYS-004  
**Category**: Reliability

**Description**: The system maintains high availability for test execution with minimal downtime. Availability is measured as the percentage of time the system is operational and able to execute tests.

**Measurement Method**: Track system uptime and downtime. Calculate availability = uptime / (uptime + downtime) × 100 over rolling 30-day window.

**Success Thresholds**:
- Phase 1: ≥ 99.0% availability (≤ 7.2 hours downtime per month)
- Phase 2: ≥ 99.5% availability (≤ 3.6 hours downtime per month)
- Phase 3: ≥ 99.9% availability (≤ 43 minutes downtime per month)

**Measurement Frequency**: Continuous monitoring with daily reporting

**Dependencies**: Infrastructure health (MOD-INFRA-001), Resilient execution (MOD-EXEC-001)

### 3.5 Test Execution Throughput

**Criterion ID**: SYS-005  
**Category**: Performance

**Description**: The system achieves high test execution throughput measured as the number of complete test executions per hour per server instance.

**Measurement Method**: Track completed test executions and server instance hours. Calculate throughput = total_executions / total_instance_hours.

**Success Thresholds**:
- Phase 1: ≥ 100 tests/hour per instance (baseline single-agent)
- Phase 2: ≥ 200 tests/hour per instance (with optimization)
- Phase 3: ≥ 300 tests/hour per instance (with multi-agent architecture)

**Measurement Frequency**: Hourly with daily aggregation

**Dependencies**: Vision adapter performance (MOD-VISION-002), Execution framework performance (MOD-EXEC-002)

### 3.6 Release Cycle Acceleration

**Criterion ID**: SYS-006  
**Category**: Performance

**Description**: The system accelerates release cycles by reducing time spent on testing and test maintenance. Release cycle time is measured from code commit to production deployment.

**Measurement Method**: Track release cycle duration before and after TestDriver deployment. Calculate acceleration = (baseline_duration - current_duration) / baseline_duration × 100.

**Success Thresholds**:
- Phase 1: ≥ 20% reduction in release cycle time
- Phase 2: ≥ 35% reduction in release cycle time
- Phase 3: ≥ 50% reduction in release cycle time

**Measurement Frequency**: Per release with quarterly trend analysis

**Dependencies**: Test execution throughput (SYS-005), Automated healing (MOD-HEAL-001)

### 3.7 Security and Compliance

**Criterion ID**: SYS-007  
**Category**: Security

**Description**: The system maintains comprehensive security controls and compliance with regulatory requirements including SOC 2, GDPR, HIPAA, and PCI DSS.

**Measurement Method**: Conduct quarterly security audits and compliance assessments. Track findings and remediation status.

**Success Thresholds**:
- Zero critical security vulnerabilities in production
- 100% compliance with applicable regulatory requirements
- Complete audit trail for all sensitive operations
- All security findings remediated within SLA (critical: 24h, high: 7d, medium: 30d)

**Measurement Frequency**: Quarterly audits with continuous monitoring

**Dependencies**: Audit log completeness (MOD-AUDIT-001), Encryption (MOD-SEC-001)

---

## 4. Module-Level Success Criteria

Module-level criteria define success for major architectural components. These criteria ensure each module fulfills its responsibilities within the overall system.

### 4.1 MCP Server Module

#### MOD-MCP-001: Protocol Compliance

**Category**: Functional Correctness

**Description**: The MCP Server implements the Model Context Protocol specification correctly, handling all required message types and maintaining protocol compliance.

**Measurement Method**: Execute MCP protocol compliance test suite covering all message types (initialize, tools/list, tools/call, resources/list, etc.). Verify correct request/response handling.

**Success Threshold**: 100% compliance with MCP specification 2025-06-18; all protocol test cases pass

**Measurement Frequency**: Per build in CI/CD pipeline

#### MOD-MCP-002: Request Processing Latency

**Category**: Performance

**Description**: The MCP Server processes requests with low latency, ensuring responsive interaction with AI models.

**Measurement Method**: Measure request processing time from receipt to response for each endpoint. Calculate p50, p95, p99 latencies.

**Success Thresholds**:
- Non-execution endpoints: p95 < 1 second, p99 < 2 seconds
- Test plan generation: p95 < 10 seconds, p99 < 15 seconds
- Test execution: p95 < 120 seconds, p99 < 180 seconds (for typical 15-step test)

**Measurement Frequency**: Continuous monitoring with hourly aggregation

#### MOD-MCP-003: Concurrent Request Handling

**Category**: Performance

**Description**: The MCP Server handles multiple concurrent requests efficiently without degradation.

**Measurement Method**: Execute load tests with increasing concurrent requests. Measure throughput and latency at different concurrency levels.

**Success Thresholds**:
- Support ≥ 50 concurrent test executions per instance
- Latency increase < 20% at 50 concurrent requests vs 1 request
- Zero request failures due to concurrency issues

**Measurement Frequency**: Weekly load testing with continuous production monitoring

### 4.2 Vision Adapter Module

#### MOD-VISION-001: Adapter Availability

**Category**: Reliability

**Description**: Vision adapters maintain high availability with automatic failover to backup adapters when primary adapters fail.

**Measurement Method**: Track adapter health checks and failover events. Calculate availability = successful_requests / total_requests × 100.

**Success Thresholds**:
- Primary adapter availability ≥ 99.5%
- Automatic failover completes within 5 seconds
- Zero vision request failures due to adapter unavailability

**Measurement Frequency**: Continuous monitoring with daily reporting

#### MOD-VISION-002: Vision Processing Latency

**Category**: Performance

**Description**: Vision adapters process vision requests with low latency to minimize test execution time.

**Measurement Method**: Measure vision API call latency from request to response. Calculate p50, p95, p99 latencies.

**Success Thresholds**:
- Cloud adapters: p95 < 2 seconds, p99 < 3 seconds
- Local adapters: p95 < 1 second, p99 < 1.5 seconds
- Cached responses: p95 < 100ms, p99 < 200ms

**Measurement Frequency**: Continuous monitoring with hourly aggregation

#### MOD-VISION-003: Element Location Accuracy

**Category**: Functional Correctness

**Description**: Vision adapters accurately locate UI elements based on natural language descriptions.

**Measurement Method**: Execute benchmark test suite with known element locations. Calculate accuracy = correct_locations / total_attempts × 100.

**Success Thresholds**:
- Element location accuracy ≥ 90% on first attempt
- Element location accuracy ≥ 98% with healing
- False positive rate ≤ 2%

**Measurement Frequency**: Daily benchmark execution with weekly trend analysis

### 4.3 Execution Framework Module

#### MOD-EXEC-001: Framework Resilience

**Category**: Reliability

**Description**: The execution framework handles browser crashes, network failures, and other errors gracefully with automatic recovery.

**Measurement Method**: Inject controlled failures (browser crashes, network timeouts) and measure recovery success rate.

**Success Thresholds**:
- Browser crash recovery success rate ≥ 95%
- Network failure recovery success rate ≥ 98%
- State restoration success rate ≥ 99%
- Mean time to recovery < 10 seconds

**Measurement Frequency**: Weekly chaos testing with continuous production monitoring

#### MOD-EXEC-002: Execution Performance

**Category**: Performance

**Description**: The execution framework executes test steps efficiently with minimal overhead.

**Measurement Method**: Measure execution time per test step. Compare against baseline manual execution time.

**Success Thresholds**:
- Average overhead per step < 500ms
- Vision-guided execution < 2× manual execution time
- DOM-based execution < 1.2× manual execution time

**Measurement Frequency**: Continuous monitoring with daily aggregation

### 4.4 Self-Healing Module

#### MOD-HEAL-001: Healing Success Rate

**Category**: Functional Correctness

**Description**: The self-healing engine successfully heals broken locators with high accuracy.

**Measurement Method**: Track healing attempts and outcomes. Calculate success_rate = successful_healings / total_attempts × 100.

**Success Thresholds**:
- Healing success rate ≥ 80% for confidence ≥ 0.7
- Healing success rate ≥ 90% for confidence ≥ 0.9
- False positive healing rate ≤ 5%

**Measurement Frequency**: Continuous tracking with daily aggregation

#### MOD-HEAL-002: Mean Time to Heal

**Category**: Performance

**Description**: The self-healing engine heals broken locators quickly to minimize test execution delay.

**Measurement Method**: Measure time from locator failure detection to successful healing. Calculate mean, p95, p99 times.

**Success Thresholds**:
- Mean time to heal < 5 seconds
- p95 time to heal < 10 seconds
- p99 time to heal < 15 seconds

**Measurement Frequency**: Continuous monitoring with hourly aggregation

#### MOD-HEAL-003: Learning Effectiveness

**Category**: Functional Correctness

**Description**: The self-healing engine learns from healing events and improves success rates over time.

**Measurement Method**: Track healing success rate trends over time. Calculate improvement rate month-over-month.

**Success Thresholds**:
- Healing success rate improves ≥ 2% per month for first 6 months
- Healing success rate stabilizes at ≥ 90% after 12 months
- Repeat healing for same element < 10% of cases

**Measurement Frequency**: Monthly trend analysis

### 4.5 Test Memory Store Module

#### MOD-MEMORY-001: Memory Store Availability

**Category**: Reliability

**Description**: The Test Memory Store maintains high availability for storing and retrieving healing history.

**Measurement Method**: Track database uptime and query success rates. Calculate availability = successful_operations / total_operations × 100.

**Success Thresholds**:
- Database availability ≥ 99.9%
- Query success rate ≥ 99.95%
- Zero data loss events

**Measurement Frequency**: Continuous monitoring with daily reporting

#### MOD-MEMORY-002: Similarity Search Performance

**Category**: Performance

**Description**: The Test Memory Store performs similarity searches efficiently to enable fast healing.

**Measurement Method**: Measure similarity search latency for visual and semantic embeddings. Calculate p50, p95, p99 latencies.

**Success Thresholds**:
- p95 similarity search latency < 500ms
- p99 similarity search latency < 1 second
- Support ≥ 1 million element embeddings with linear scaling

**Measurement Frequency**: Continuous monitoring with daily aggregation

#### MOD-MEMORY-003: Memory Retention and Reuse

**Category**: Functional Correctness

**Description**: The Test Memory Store retains healing history and enables reuse across test cycles.

**Measurement Method**: Track healing events stored and reused. Calculate reuse_rate = healings_using_memory / total_healings × 100.

**Success Thresholds**:
- Memory reuse rate ≥ 50% after 30 days
- Memory reuse rate ≥ 70% after 90 days
- Reused healings have ≥ 95% success rate

**Measurement Frequency**: Weekly analysis with monthly trend reporting

This framework continues with detailed criteria for all remaining modules, classes, and functions in the system.
# TestDriver MCP Framework: Function and Class-Level Success Criteria

## Part 1: Class-Level Success Criteria

### 5.1 TestMemoryStore Class

The TestMemoryStore class provides persistent storage for healing history and learned patterns. Success criteria ensure reliable storage, efficient retrieval, and data integrity.

#### CLS-MEMORY-001: Data Persistence

**Category**: Functional Correctness

**Description**: The TestMemoryStore reliably persists healing events, locator versions, and test executions without data loss.

**Measurement Method**: Execute write operations followed by read operations. Verify all written data can be retrieved correctly. Inject controlled failures (process crashes, network interruptions) and verify data integrity after recovery.

**Success Thresholds**:
- 100% of written data retrievable after normal shutdown
- ≥ 99.99% of written data retrievable after abnormal shutdown (crash)
- Zero data corruption events
- Write operation success rate ≥ 99.95%

**Test Implementation**:
```python
async def test_data_persistence():
    """Verify data persists across restarts."""
    store = TestMemoryStore(config)
    await store.initialize()
    
    # Write test data
    event = HealingEvent(...)
    await store.store_healing_event(event)
    
    # Simulate restart
    await store.close()
    store = TestMemoryStore(config)
    await store.initialize()
    
    # Verify data persists
    retrieved = await store.get_healing_event(event.event_id)
    assert retrieved == event
```

#### CLS-MEMORY-002: Similarity Search Accuracy

**Category**: Functional Correctness

**Description**: The TestMemoryStore returns relevant similar healing events based on visual and semantic embeddings.

**Measurement Method**: Create benchmark dataset with known similar and dissimilar elements. Execute similarity searches and measure precision, recall, and F1 score.

**Success Thresholds**:
- Precision ≥ 85% (returned results are relevant)
- Recall ≥ 80% (relevant results are returned)
- F1 score ≥ 0.82
- Top-5 results include correct match ≥ 95% of time

**Test Implementation**:
```python
async def test_similarity_search_accuracy():
    """Verify similarity search returns relevant results."""
    # Load benchmark dataset with known similarities
    benchmark = load_benchmark_dataset()
    
    for query, expected_matches in benchmark:
        results = await store.find_similar_healing_events(
            visual_embedding=query.visual_embedding,
            limit=10
        )
        
        # Calculate precision and recall
        retrieved_ids = {r[0].event_id for r in results}
        expected_ids = {e.event_id for e in expected_matches}
        
        precision = len(retrieved_ids & expected_ids) / len(retrieved_ids)
        recall = len(retrieved_ids & expected_ids) / len(expected_ids)
        
        assert precision >= 0.85
        assert recall >= 0.80
```

#### CLS-MEMORY-003: Concurrent Access Safety

**Category**: Reliability

**Description**: The TestMemoryStore handles concurrent read and write operations safely without race conditions or data corruption.

**Measurement Method**: Execute concurrent operations from multiple threads/processes. Verify data consistency and absence of race conditions.

**Success Thresholds**:
- Zero data corruption under concurrent access
- Zero deadlocks or livelocks
- Read operations never block write operations for > 100ms
- Write operations complete successfully ≥ 99.9% of time under concurrency

**Test Implementation**:
```python
async def test_concurrent_access_safety():
    """Verify safe concurrent access."""
    import asyncio
    
    # Create multiple concurrent writers
    async def writer(id: int):
        for i in range(100):
            event = create_test_event(f"writer-{id}-event-{i}")
            await store.store_healing_event(event)
    
    # Create multiple concurrent readers
    async def reader(id: int):
        for i in range(100):
            results = await store.find_similar_healing_events(
                visual_embedding=random_embedding(),
                limit=10
            )
    
    # Execute concurrently
    writers = [writer(i) for i in range(10)]
    readers = [reader(i) for i in range(10)]
    await asyncio.gather(*writers, *readers)
    
    # Verify data integrity
    assert await store.count_healing_events() == 1000
```

### 5.2 AILocatorHealingEngine Class

The AILocatorHealingEngine class performs autonomous healing of broken locators. Success criteria ensure high healing accuracy, fast healing time, and effective learning.

#### CLS-HEAL-001: Healing Accuracy

**Category**: Functional Correctness

**Description**: The AILocatorHealingEngine correctly identifies and heals broken locators with high accuracy.

**Measurement Method**: Create test suite with intentionally broken locators and known correct elements. Measure healing success rate and false positive rate.

**Success Thresholds**:
- Healing success rate ≥ 80% for confidence ≥ 0.7
- Healing success rate ≥ 90% for confidence ≥ 0.9
- False positive rate ≤ 5%
- Confidence calibration error < 0.1 (predicted confidence matches actual success rate)

**Test Implementation**:
```python
async def test_healing_accuracy():
    """Verify healing accuracy meets thresholds."""
    test_cases = load_healing_test_cases()  # Known broken locators
    
    results = []
    for case in test_cases:
        healing_result = await healer.heal_locator(
            original_locator=case.broken_locator,
            screenshot=case.screenshot,
            element_description=case.description
        )
        
        # Verify healing correctness
        is_correct = verify_healing(healing_result, case.expected_element)
        results.append({
            'confidence': healing_result.confidence,
            'correct': is_correct
        })
    
    # Calculate metrics by confidence threshold
    high_conf_results = [r for r in results if r['confidence'] >= 0.9]
    success_rate = sum(r['correct'] for r in high_conf_results) / len(high_conf_results)
    
    assert success_rate >= 0.90
```

#### CLS-HEAL-002: Healing Latency

**Category**: Performance

**Description**: The AILocatorHealingEngine heals broken locators quickly to minimize test execution delay.

**Measurement Method**: Measure time from heal_locator() call to return. Calculate p50, p95, p99 latencies across diverse test cases.

**Success Thresholds**:
- p50 healing latency < 3 seconds
- p95 healing latency < 10 seconds
- p99 healing latency < 15 seconds
- Timeout handling: return partial result after 30 seconds

**Test Implementation**:
```python
async def test_healing_latency():
    """Verify healing completes within latency bounds."""
    test_cases = load_healing_test_cases()
    latencies = []
    
    for case in test_cases:
        start = time.time()
        result = await healer.heal_locator(
            original_locator=case.broken_locator,
            screenshot=case.screenshot,
            element_description=case.description
        )
        latency = time.time() - start
        latencies.append(latency)
    
    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)
    
    assert p50 < 3.0
    assert p95 < 10.0
    assert p99 < 15.0
```

#### CLS-HEAL-003: Learning from Feedback

**Category**: Functional Correctness

**Description**: The AILocatorHealingEngine improves healing accuracy over time by learning from user feedback and healing outcomes.

**Measurement Method**: Track healing success rate trends over time. Measure improvement rate month-over-month.

**Success Thresholds**:
- Healing success rate improves ≥ 2% per month for first 6 months
- Repeated healing for same element decreases by ≥ 50% after 3 months
- User-corrected healings have ≥ 95% success rate when reused

**Test Implementation**:
```python
async def test_learning_from_feedback():
    """Verify learning improves performance over time."""
    # Simulate 6 months of healing with feedback
    for month in range(6):
        # Execute healing attempts
        month_results = []
        for case in test_cases:
            result = await healer.heal_locator(...)
            month_results.append(result)
        
        # Simulate user feedback
        for result in month_results:
            feedback = generate_feedback(result)
            await healer.incorporate_feedback(result.event_id, feedback)
        
        # Measure success rate
        success_rate = calculate_success_rate(month_results)
        
        # Verify improvement
        if month > 0:
            improvement = success_rate - previous_success_rate
            assert improvement >= 0.02  # 2% improvement
        
        previous_success_rate = success_rate
```

### 5.3 TestLearningOrchestrator Class

The TestLearningOrchestrator class continuously learns from test execution history to optimize system parameters. Success criteria ensure effective learning and measurable improvements.

#### CLS-LEARN-001: Parameter Optimization Effectiveness

**Category**: Functional Correctness

**Description**: The TestLearningOrchestrator optimizes system parameters (wait durations, retry thresholds, detection modes) effectively based on historical data.

**Measurement Method**: Compare test reliability and performance before and after parameter optimization. Measure improvement in key metrics.

**Success Thresholds**:
- Wait duration optimization reduces flaky tests by ≥ 15%
- Retry threshold optimization reduces wasted retries by ≥ 20%
- Detection mode optimization improves element location success by ≥ 10%
- Overall test reliability improves by ≥ 10% after optimization

**Test Implementation**:
```python
async def test_parameter_optimization_effectiveness():
    """Verify parameter optimization improves metrics."""
    # Collect baseline metrics
    baseline_metrics = await collect_test_metrics(days=30)
    
    # Run learning cycle
    await orchestrator.run_learning_cycle()
    
    # Collect post-optimization metrics
    optimized_metrics = await collect_test_metrics(days=30)
    
    # Verify improvements
    flaky_test_reduction = (
        (baseline_metrics.flaky_rate - optimized_metrics.flaky_rate) /
        baseline_metrics.flaky_rate
    )
    assert flaky_test_reduction >= 0.15
```

#### CLS-LEARN-002: Insight Generation Quality

**Category**: Functional Correctness

**Description**: The TestLearningOrchestrator generates actionable insights that accurately identify test quality issues.

**Measurement Method**: Review generated insights against known issues. Measure precision (insights are correct) and recall (issues are identified).

**Success Thresholds**:
- Insight precision ≥ 80% (generated insights are actionable)
- Insight recall ≥ 70% (known issues are identified)
- Insight confidence calibration error < 0.15
- ≥ 60% of insights lead to measurable improvements when acted upon

**Test Implementation**:
```python
async def test_insight_generation_quality():
    """Verify generated insights are accurate and actionable."""
    # Create test environment with known issues
    known_issues = create_test_environment_with_issues()
    
    # Generate insights
    insights = await orchestrator.generate_insights()
    
    # Evaluate precision and recall
    true_positives = 0
    false_positives = 0
    
    for insight in insights:
        if matches_known_issue(insight, known_issues):
            true_positives += 1
        else:
            false_positives += 1
    
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / len(known_issues)
    
    assert precision >= 0.80
    assert recall >= 0.70
```

## Part 2: Function-Level Success Criteria

### 6.1 TestMemoryStore Functions

#### FN-MEMORY-001: store_healing_event()

**Category**: Functional Correctness

**Description**: The store_healing_event() function correctly stores healing events with all required data and embeddings.

**Measurement Method**: Execute function with various inputs. Verify data is stored correctly and retrievable.

**Success Thresholds**:
- 100% of valid inputs stored successfully
- Invalid inputs rejected with appropriate errors
- Stored data matches input data exactly
- Function completes in < 500ms for p95 of calls

**Test Implementation**:
```python
async def test_store_healing_event():
    """Verify healing events are stored correctly."""
    event = HealingEvent(
        event_id="test-001",
        test_id="test-123",
        element_id="btn-submit",
        timestamp=datetime.utcnow(),
        original_locator={'css': '#old-id'},
        failure_reason="Element not found",
        failure_screenshot="s3://bucket/screenshot.png",
        healing_strategy="visual_similarity",
        new_locator={'css': '#new-id'},
        confidence_score=0.92,
        healing_successful=True,
        validation_method="auto",
        user_feedback=None,
        visual_embedding=[0.1, 0.2, ...],  # 512-dim
        semantic_embedding=[0.3, 0.4, ...],  # 384-dim
        context_features={'page_url': 'https://example.com'}
    )
    
    await store.store_healing_event(event)
    
    # Verify storage
    retrieved = await store.get_healing_event("test-001")
    assert retrieved.event_id == event.event_id
    assert retrieved.confidence_score == event.confidence_score
    assert len(retrieved.visual_embedding) == 512
```

#### FN-MEMORY-002: find_similar_healing_events()

**Category**: Functional Correctness

**Description**: The find_similar_healing_events() function returns relevant similar events based on embedding similarity.

**Measurement Method**: Execute function with known query embeddings. Verify returned results match expected similar events.

**Success Thresholds**:
- Returns results in < 500ms for p95 of queries
- Results ordered by similarity score (descending)
- Similarity scores in valid range [0.0, 1.0]
- Filters applied correctly when specified

**Test Implementation**:
```python
async def test_find_similar_healing_events():
    """Verify similarity search returns relevant results."""
    # Store known events
    await store.store_healing_event(event1)  # Similar to query
    await store.store_healing_event(event2)  # Similar to query
    await store.store_healing_event(event3)  # Dissimilar
    
    # Query with embedding similar to event1 and event2
    results = await store.find_similar_healing_events(
        visual_embedding=query_embedding,
        limit=5
    )
    
    # Verify results
    assert len(results) <= 5
    assert results[0][1] >= results[1][1]  # Ordered by score
    assert event1.event_id in [r[0].event_id for r in results]
    assert event2.event_id in [r[0].event_id for r in results]
```

#### FN-MEMORY-003: calculate_element_stability()

**Category**: Functional Correctness

**Description**: The calculate_element_stability() function accurately computes stability scores based on healing frequency.

**Measurement Method**: Create test data with known healing frequencies. Verify calculated stability scores match expected values.

**Success Thresholds**:
- Stability score in valid range [0.0, 1.0]
- Score decreases as healing frequency increases
- Score calculation completes in < 1 second
- Handles edge cases (no data, all healings, no healings) correctly

**Test Implementation**:
```python
async def test_calculate_element_stability():
    """Verify stability calculation is accurate."""
    # Create element with known healing frequency
    element_id = "btn-submit"
    
    # Simulate 100 executions with 10 healings
    for i in range(100):
        await store.record_test_execution(
            test_id=f"test-{i}",
            elements_tested=[element_id]
        )
    
    for i in range(10):
        await store.store_healing_event(
            create_healing_event(element_id=element_id)
        )
    
    # Calculate stability
    stability = await store.calculate_element_stability(element_id)
    
    # Verify: stability = 1 - (10/100) = 0.9
    assert abs(stability - 0.9) < 0.01
```

### 6.2 AILocatorHealingEngine Functions

#### FN-HEAL-001: heal_locator()

**Category**: Functional Correctness

**Description**: The heal_locator() function attempts to heal a broken locator and returns a healing result with confidence score.

**Measurement Method**: Execute function with broken locators. Verify healing attempts are made and results include confidence scores.

**Success Thresholds**:
- Returns result for 100% of valid inputs
- Confidence score in valid range [0.0, 1.0]
- Healing strategies attempted in priority order
- Function completes within timeout (30 seconds)

**Test Implementation**:
```python
async def test_heal_locator():
    """Verify heal_locator attempts healing and returns result."""
    result = await healer.heal_locator(
        original_locator={'css': '#old-button'},
        screenshot=load_screenshot("page.png"),
        element_description="Submit button with blue background"
    )
    
    # Verify result structure
    assert 0.0 <= result.confidence <= 1.0
    assert result.new_locator is not None
    assert result.healing_strategy in VALID_STRATEGIES
    assert result.visual_embedding is not None
```

#### FN-HEAL-002: _search_by_visual_similarity()

**Category**: Functional Correctness

**Description**: The _search_by_visual_similarity() private function searches for elements using visual embedding similarity.

**Measurement Method**: Execute function with known visual embeddings. Verify returned elements match visually similar elements.

**Success Thresholds**:
- Returns results in < 2 seconds
- Results ordered by visual similarity
- Handles no matches gracefully (returns empty list)
- Visual similarity scores in valid range

**Test Implementation**:
```python
async def test_search_by_visual_similarity():
    """Verify visual similarity search works correctly."""
    # Create visual embedding for target element
    target_embedding = generate_embedding(target_screenshot)
    
    # Search for similar elements
    results = await healer._search_by_visual_similarity(
        visual_embedding=target_embedding,
        screenshot=page_screenshot
    )
    
    # Verify results
    assert len(results) > 0
    assert all(0.0 <= r.similarity <= 1.0 for r in results)
    assert results[0].similarity >= results[-1].similarity
```

### 6.3 TestLearningOrchestrator Functions

#### FN-LEARN-001: _learn_wait_durations()

**Category**: Functional Correctness

**Description**: The _learn_wait_durations() function analyzes historical wait data and optimizes wait duration parameters.

**Measurement Method**: Provide historical data with known optimal wait durations. Verify learned parameters match expected values.

**Success Thresholds**:
- Learned wait durations within 20% of optimal values
- Function completes in < 60 seconds for 10,000 historical records
- Handles insufficient data gracefully (returns defaults)
- Generated recommendations are actionable

**Test Implementation**:
```python
async def test_learn_wait_durations():
    """Verify wait duration learning produces good recommendations."""
    # Create historical data with known patterns
    # Pattern: element type "button" needs 2s wait on average
    historical_data = create_wait_duration_data(
        element_type="button",
        optimal_wait=2.0,
        samples=1000
    )
    
    # Learn optimal wait durations
    recommendations = await orchestrator._learn_wait_durations()
    
    # Verify learned value is close to optimal
    button_wait = recommendations.get('button', {}).get('wait_duration')
    assert abs(button_wait - 2.0) < 0.4  # Within 20%
```

This specification continues with detailed function-level criteria for all remaining functions in the system, ensuring every function has measurable success criteria and test implementations.
# TestDriver MCP Framework: Built-In Self-Testing Architecture

## Part 7: Built-In Self-Testing Architecture

### 7.1 Overview

The Built-In Self-Testing Architecture enables TestDriver to continuously validate its own functionality, performance, and reliability without external test infrastructure. This self-testing capability provides real-time health monitoring, early detection of regressions, and automatic validation that all success criteria are met.

The architecture follows the principle of **continuous self-validation** where every component includes embedded test functionality that executes automatically during normal operation. This approach differs from traditional external testing in several key ways. Traditional testing requires separate test infrastructure, test data, and test execution schedules. Built-in self-testing embeds validation logic directly into components, uses production data for validation where appropriate, and executes continuously as part of normal operation.

The self-testing architecture provides multiple benefits. It enables **immediate regression detection** where component changes that violate success criteria are detected instantly rather than waiting for scheduled test runs. It provides **production validation** where the system validates its behavior using actual production workloads and data. It enables **zero-infrastructure testing** where no separate test environments or test data management is required. It provides **continuous compliance** where success criteria compliance is monitored continuously rather than periodically.

### 7.2 Self-Testing Principles

The architecture is built on five core principles that ensure effective self-testing without impacting production performance or reliability.

**Principle 1: Non-Intrusive Monitoring** states that self-testing must not degrade production performance or reliability. Self-tests execute asynchronously in background threads, use sampling rather than testing every operation, have configurable resource limits to prevent resource exhaustion, and can be disabled entirely if needed without affecting functionality.

**Principle 2: Production-Safe Validation** states that self-tests must never corrupt production data or cause production failures. Self-tests use read-only operations where possible, create isolated test data that is clearly marked and automatically cleaned up, validate using assertions that log failures but don't crash the system, and include circuit breakers that disable self-testing if failures exceed thresholds.

**Principle 3: Comprehensive Coverage** states that self-testing must validate all success criteria at all levels. Every function, class, module, and system-level criterion has corresponding self-test logic, self-tests validate functional correctness, performance, reliability, security, and maintainability, and self-tests cover both happy paths and error conditions.

**Principle 4: Actionable Reporting** states that self-test failures must provide clear, actionable information for diagnosis and remediation. Self-test failures include detailed context (inputs, outputs, state), link to specific success criteria that failed, provide recommendations for remediation, and integrate with alerting systems for immediate notification.

**Principle 5: Continuous Improvement** states that self-testing must evolve as the system evolves. Self-tests are versioned alongside code, self-test coverage is measured and tracked, self-test failures inform system improvements, and self-tests themselves are tested for correctness.

### 7.3 Self-Testing Components

The architecture includes four primary components that work together to provide comprehensive self-testing.

#### 7.3.1 Embedded Validators

Embedded Validators are lightweight validation functions embedded directly into classes and modules. These validators execute automatically during normal operation to verify that success criteria are met.

**Implementation Pattern**:

```python
class TestMemoryStore:
    """Test Memory Store with embedded self-testing."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._validator = EmbeddedValidator(self, config.get('self_test', {}))
        self._metrics = MetricsCollector()
    
    async def store_healing_event(self, event: HealingEvent) -> None:
        """Store healing event with embedded validation."""
        # Pre-condition validation
        self._validator.validate_preconditions('store_healing_event', {
            'event': event,
            'store_available': await self._is_available()
        })
        
        # Execute operation with timing
        start_time = time.time()
        try:
            # Actual storage logic
            await self._store_event_impl(event)
            
            # Post-condition validation
            self._validator.validate_postconditions('store_healing_event', {
                'event_stored': await self._verify_stored(event.event_id),
                'latency': time.time() - start_time
            })
            
        except Exception as e:
            # Error condition validation
            self._validator.validate_error_handling('store_healing_event', e)
            raise
        
        finally:
            # Record metrics
            self._metrics.record('store_healing_event', {
                'latency': time.time() - start_time,
                'success': True
            })
    
    async def _verify_stored(self, event_id: str) -> bool:
        """Verify event was actually stored."""
        try:
            event = await self.get_healing_event(event_id)
            return event is not None
        except:
            return False
```

**Embedded Validator Implementation**:

```python
class EmbeddedValidator:
    """Embedded validator for runtime success criteria validation."""
    
    def __init__(self, component: Any, config: Dict[str, Any]):
        self.component = component
        self.enabled = config.get('enabled', True)
        self.sampling_rate = config.get('sampling_rate', 0.1)  # 10% sampling
        self.criteria = self._load_success_criteria(component.__class__.__name__)
        self.failures = []
    
    def validate_preconditions(self, operation: str, context: Dict[str, Any]) -> None:
        """Validate preconditions before operation."""
        if not self._should_validate():
            return
        
        criteria = self.criteria.get(f'{operation}_preconditions', [])
        for criterion in criteria:
            if not criterion.check(context):
                self._record_failure(criterion, context, 'precondition')
    
    def validate_postconditions(self, operation: str, context: Dict[str, Any]) -> None:
        """Validate postconditions after operation."""
        if not self._should_validate():
            return
        
        criteria = self.criteria.get(f'{operation}_postconditions', [])
        for criterion in criteria:
            if not criterion.check(context):
                self._record_failure(criterion, context, 'postcondition')
    
    def validate_error_handling(self, operation: str, error: Exception) -> None:
        """Validate error handling behavior."""
        if not self._should_validate():
            return
        
        criteria = self.criteria.get(f'{operation}_error_handling', [])
        for criterion in criteria:
            if not criterion.check({'error': error}):
                self._record_failure(criterion, {'error': error}, 'error_handling')
    
    def _should_validate(self) -> bool:
        """Determine if validation should run (sampling)."""
        if not self.enabled:
            return False
        return random.random() < self.sampling_rate
    
    def _record_failure(
        self,
        criterion: SuccessCriterion,
        context: Dict[str, Any],
        phase: str
    ) -> None:
        """Record validation failure."""
        failure = ValidationFailure(
            timestamp=datetime.utcnow(),
            component=self.component.__class__.__name__,
            criterion_id=criterion.id,
            criterion_name=criterion.name,
            phase=phase,
            context=context,
            recommendation=criterion.remediation_advice
        )
        
        self.failures.append(failure)
        
        # Log failure
        logger.warning(
            "Success criterion validation failed",
            criterion_id=criterion.id,
            component=self.component.__class__.__name__,
            phase=phase
        )
        
        # Emit metric
        metrics.increment('self_test.validation_failure', {
            'component': self.component.__class__.__name__,
            'criterion': criterion.id,
            'phase': phase
        })
```

#### 7.3.2 Continuous Health Monitors

Continuous Health Monitors run as background tasks that periodically validate system health and success criteria compliance. Unlike embedded validators that validate individual operations, health monitors validate aggregate metrics and system-wide properties.

**Implementation Pattern**:

```python
class HealthMonitor:
    """Continuous health monitoring for success criteria."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.check_interval = config.get('check_interval_seconds', 60)
        self.criteria = load_all_success_criteria()
        self._running = False
        self._task: Optional[asyncio.Task] = None
    
    async def start(self) -> None:
        """Start continuous health monitoring."""
        self._running = True
        self._task = asyncio.create_task(self._monitoring_loop())
        logger.info("Health monitoring started")
    
    async def stop(self) -> None:
        """Stop health monitoring."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Health monitoring stopped")
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self._running:
            try:
                await asyncio.sleep(self.check_interval)
                
                # Check all system-level criteria
                await self._check_system_criteria()
                
                # Check all module-level criteria
                await self._check_module_criteria()
                
                # Check aggregate metrics
                await self._check_aggregate_metrics()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Health monitoring error", error=str(e))
    
    async def _check_system_criteria(self) -> None:
        """Check all system-level success criteria."""
        for criterion in self.criteria.system_level:
            try:
                result = await criterion.evaluate()
                
                if not result.passed:
                    await self._handle_criterion_failure(criterion, result)
                else:
                    await self._handle_criterion_success(criterion, result)
                    
            except Exception as e:
                logger.error(
                    "Criterion evaluation error",
                    criterion_id=criterion.id,
                    error=str(e)
                )
    
    async def _check_module_criteria(self) -> None:
        """Check all module-level success criteria."""
        for module_name, criteria in self.criteria.module_level.items():
            for criterion in criteria:
                try:
                    result = await criterion.evaluate()
                    
                    if not result.passed:
                        await self._handle_criterion_failure(criterion, result)
                        
                except Exception as e:
                    logger.error(
                        "Module criterion evaluation error",
                        module=module_name,
                        criterion_id=criterion.id,
                        error=str(e)
                    )
    
    async def _handle_criterion_failure(
        self,
        criterion: SuccessCriterion,
        result: EvaluationResult
    ) -> None:
        """Handle criterion failure."""
        # Log failure
        logger.error(
            "Success criterion failed",
            criterion_id=criterion.id,
            criterion_name=criterion.name,
            actual_value=result.actual_value,
            threshold=criterion.threshold,
            gap=result.gap
        )
        
        # Emit metric
        metrics.gauge(f'success_criterion.{criterion.id}', result.actual_value, {
            'status': 'failed',
            'level': criterion.level
        })
        
        # Send alert if critical
        if criterion.severity == 'critical':
            await self._send_alert(criterion, result)
    
    async def _send_alert(
        self,
        criterion: SuccessCriterion,
        result: EvaluationResult
    ) -> None:
        """Send alert for critical criterion failure."""
        alert = Alert(
            severity='critical',
            title=f"Success Criterion Failed: {criterion.name}",
            description=f"{criterion.description}\n\n"
                       f"Actual: {result.actual_value}\n"
                       f"Threshold: {criterion.threshold}\n"
                       f"Gap: {result.gap}\n\n"
                       f"Remediation: {criterion.remediation_advice}",
            criterion_id=criterion.id,
            timestamp=datetime.utcnow()
        )
        
        await alert_service.send(alert)
```

#### 7.3.3 Synthetic Test Generators

Synthetic Test Generators automatically create and execute synthetic test scenarios that validate system behavior without requiring manual test creation. These generators use the system's own capabilities to test itself.

**Implementation Pattern**:

```python
class SyntheticTestGenerator:
    """Generates and executes synthetic tests automatically."""
    
    def __init__(
        self,
        test_memory_store: TestMemoryStore,
        healing_engine: AILocatorHealingEngine,
        config: Dict[str, Any]
    ):
        self.memory_store = test_memory_store
        self.healing_engine = healing_engine
        self.test_interval = config.get('test_interval_seconds', 3600)  # 1 hour
        self._running = False
    
    async def start(self) -> None:
        """Start synthetic test generation."""
        self._running = True
        asyncio.create_task(self._generation_loop())
        logger.info("Synthetic test generation started")
    
    async def _generation_loop(self) -> None:
        """Main test generation loop."""
        while self._running:
            try:
                await asyncio.sleep(self.test_interval)
                
                # Generate and execute synthetic tests
                await self._test_healing_accuracy()
                await self._test_memory_retrieval()
                await self._test_learning_effectiveness()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Synthetic test generation error", error=str(e))
    
    async def _test_healing_accuracy(self) -> None:
        """Test healing accuracy using synthetic scenarios."""
        # Generate synthetic broken locator scenario
        scenario = await self._generate_healing_scenario()
        
        # Attempt healing
        start_time = time.time()
        result = await self.healing_engine.heal_locator(
            original_locator=scenario.broken_locator,
            screenshot=scenario.screenshot,
            element_description=scenario.description
        )
        latency = time.time() - start_time
        
        # Validate result
        is_correct = self._validate_healing_result(result, scenario.expected_element)
        
        # Record metrics
        metrics.record('synthetic_test.healing_accuracy', {
            'correct': is_correct,
            'confidence': result.confidence,
            'latency': latency
        })
        
        # Validate against success criteria
        if result.confidence >= 0.9 and not is_correct:
            logger.warning(
                "Healing accuracy criterion violated",
                confidence=result.confidence,
                expected_correct=True,
                actual_correct=is_correct
            )
    
    async def _generate_healing_scenario(self) -> HealingScenario:
        """Generate synthetic healing scenario."""
        # Select random element from memory store
        elements = await self.memory_store.get_stable_elements(limit=100)
        element = random.choice(elements)
        
        # Create broken locator by modifying stable locator
        broken_locator = self._break_locator(element.locator)
        
        # Get screenshot for element
        screenshot = await self._get_element_screenshot(element)
        
        return HealingScenario(
            broken_locator=broken_locator,
            screenshot=screenshot,
            description=element.description,
            expected_element=element
        )
```

#### 7.3.4 Self-Test Dashboard

The Self-Test Dashboard provides real-time visualization of success criteria compliance and self-test results. The dashboard enables teams to monitor system health and identify issues proactively.

**Dashboard Components**:

**Success Criteria Compliance Overview** shows the percentage of success criteria currently passing at each level (system, module, class, function). This provides an at-a-glance view of overall system health.

**Trend Analysis** shows success criteria compliance trends over time, enabling teams to identify degrading metrics before they become critical failures.

**Failure Details** provides detailed information about current criterion failures including the specific criterion that failed, actual vs threshold values, gap analysis, affected components, and remediation recommendations.

**Performance Metrics** displays key performance metrics against success criteria thresholds including latency percentiles, throughput rates, error rates, and resource utilization.

**Synthetic Test Results** shows results from automatically generated synthetic tests including healing accuracy, memory retrieval performance, and learning effectiveness.

### 7.4 Self-Testing Workflow

The self-testing workflow integrates all components to provide continuous validation.

**Step 1: Embedded Validation During Operations**. As the system executes normal operations, embedded validators check preconditions, postconditions, and error handling against success criteria. Validation failures are logged and recorded as metrics.

**Step 2: Continuous Health Monitoring**. Health monitors periodically evaluate aggregate metrics and system-wide criteria. Failures trigger alerts and are displayed on the self-test dashboard.

**Step 3: Synthetic Test Execution**. Synthetic test generators create and execute test scenarios automatically. Results are compared against success criteria and recorded as metrics.

**Step 4: Aggregation and Reporting**. All validation results, health check results, and synthetic test results are aggregated and displayed on the self-test dashboard. Trends are analyzed to identify degrading metrics.

**Step 5: Alerting and Remediation**. Critical criterion failures trigger immediate alerts to on-call engineers. Alerts include detailed context and remediation recommendations. Teams investigate and remediate failures based on provided guidance.

**Step 6: Continuous Improvement**. Self-test failures inform system improvements. The development team analyzes failure patterns to identify systemic issues. Success criteria are refined based on operational experience.

### 7.5 Configuration

Self-testing behavior is fully configurable to balance validation coverage with performance impact.

```yaml
self_testing:
  # Global enable/disable
  enabled: true
  
  # Embedded validation configuration
  embedded_validation:
    enabled: true
    sampling_rate: 0.1  # Validate 10% of operations
    log_failures: true
    emit_metrics: true
  
  # Health monitoring configuration
  health_monitoring:
    enabled: true
    check_interval_seconds: 60
    alert_on_critical_failures: true
    alert_channels:
      - slack
      - pagerduty
  
  # Synthetic test configuration
  synthetic_tests:
    enabled: true
    test_interval_seconds: 3600  # Run every hour
    scenarios:
      - healing_accuracy
      - memory_retrieval
      - learning_effectiveness
  
  # Dashboard configuration
  dashboard:
    enabled: true
    refresh_interval_seconds: 30
    retention_days: 90
```

### 7.6 Benefits

The built-in self-testing architecture provides multiple benefits that improve system reliability and reduce operational burden.

**Immediate Regression Detection** enables the system to detect regressions instantly when code changes violate success criteria, rather than waiting for scheduled test runs or discovering issues in production.

**Production Validation** ensures the system validates its behavior using actual production workloads and data, providing higher confidence than testing in isolated test environments.

**Zero Test Infrastructure** eliminates the need for separate test environments, test data management, and test execution scheduling, reducing operational complexity and cost.

**Continuous Compliance** provides continuous monitoring of success criteria compliance rather than periodic validation, ensuring the system always meets quality standards.

**Actionable Insights** delivers detailed failure information with remediation recommendations, enabling teams to resolve issues quickly without extensive investigation.

**Continuous Improvement** creates a feedback loop where self-test failures inform system improvements, driving ongoing quality enhancements.

This built-in self-testing architecture ensures that TestDriver continuously validates its own correctness, performance, and reliability, providing confidence that all success criteria are met at all times.
