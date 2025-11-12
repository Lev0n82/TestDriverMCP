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
