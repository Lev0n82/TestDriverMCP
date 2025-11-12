#!/usr/bin/env python3.11
"""
TestDriver MCP Framework v2.0 - Performance and Integration Tests
Tests system performance, scalability, and integration scenarios
"""

import asyncio
import time
import json
import statistics
from datetime import datetime
from typing import List
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from models import HealingEvent, HealingStrategy, ValidationMethod
from memory.store import TestMemoryStore
from learning.orchestrator import TestLearningOrchestrator

class PerformanceResults:
    """Track performance test results."""
    
    def __init__(self):
        self.tests = []
        
    def add_test(self, name: str, metrics: dict):
        """Add performance test result."""
        self.tests.append({
            "name": name,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat()
        })

async def test_memory_store_performance(results: PerformanceResults):
    """Performance Test 1: Memory Store Throughput."""
    print("\n" + "="*70)
    print("PERFORMANCE TEST 1: Memory Store Throughput")
    print("="*70)
    
    store = TestMemoryStore({"storage_type": "memory"})
    
    # Test write performance
    write_times = []
    num_writes = 100
    
    print(f"  Testing write performance ({num_writes} events)...")
    for i in range(num_writes):
        event = HealingEvent(
            event_id=f"perf-{i:04d}",
            test_id=f"test-{i}",
            execution_id=f"exec-{i}",
            element_id=f"element-{i}",
            original_locator={"css": f"#old-{i}"},
            failure_reason="Not found",
            healing_strategy=HealingStrategy.VISUAL_SIMILARITY,
            new_locator={"css": f"#new-{i}"},
            confidence_score=0.85,
            healing_successful=True,
            validation_method=ValidationMethod.AUTO,
            visual_embedding=[float(i)/num_writes] * 512
        )
        
        start = time.time()
        await store.store_healing_event(event)
        write_times.append(time.time() - start)
    
    avg_write = statistics.mean(write_times) * 1000  # Convert to ms
    p95_write = statistics.quantiles(write_times, n=20)[18] * 1000
    p99_write = statistics.quantiles(write_times, n=100)[98] * 1000
    
    print(f"  ✓ Write performance:")
    print(f"    - Average: {avg_write:.2f}ms")
    print(f"    - P95: {p95_write:.2f}ms")
    print(f"    - P99: {p99_write:.2f}ms")
    
    # Test read performance
    read_times = []
    num_reads = 100
    
    print(f"\n  Testing read performance ({num_reads} queries)...")
    for i in range(num_reads):
        start = time.time()
        await store.get_healing_event(f"perf-{i:04d}")
        read_times.append(time.time() - start)
    
    avg_read = statistics.mean(read_times) * 1000
    p95_read = statistics.quantiles(read_times, n=20)[18] * 1000
    p99_read = statistics.quantiles(read_times, n=100)[98] * 1000
    
    print(f"  ✓ Read performance:")
    print(f"    - Average: {avg_read:.2f}ms")
    print(f"    - P95: {p95_read:.2f}ms")
    print(f"    - P99: {p99_read:.2f}ms")
    
    # Test similarity search performance
    search_times = []
    num_searches = 50
    
    print(f"\n  Testing similarity search performance ({num_searches} queries)...")
    for i in range(num_searches):
        query_embedding = [float(i)/num_searches] * 512
        start = time.time()
        await store.find_similar_healing_events(query_embedding, limit=10)
        search_times.append(time.time() - start)
    
    avg_search = statistics.mean(search_times) * 1000
    p95_search = statistics.quantiles(search_times, n=20)[18] * 1000
    p99_search = statistics.quantiles(search_times, n=100)[98] * 1000
    
    print(f"  ✓ Similarity search performance:")
    print(f"    - Average: {avg_search:.2f}ms")
    print(f"    - P95: {p95_search:.2f}ms")
    print(f"    - P99: {p99_search:.2f}ms")
    
    results.add_test("Memory Store Performance", {
        "write_avg_ms": round(avg_write, 2),
        "write_p95_ms": round(p95_write, 2),
        "write_p99_ms": round(p99_write, 2),
        "read_avg_ms": round(avg_read, 2),
        "read_p95_ms": round(p95_read, 2),
        "read_p99_ms": round(p99_read, 2),
        "search_avg_ms": round(avg_search, 2),
        "search_p95_ms": round(p95_search, 2),
        "search_p99_ms": round(p99_search, 2),
        "target_met": avg_search < 500  # Target: < 500ms
    })
    
    print(f"\n✅ PERFORMANCE TEST 1 COMPLETE")
    print(f"   Target (< 500ms search): {'MET' if avg_search < 500 else 'NOT MET'}")

async def test_learning_scalability(results: PerformanceResults):
    """Performance Test 2: Learning Orchestrator Scalability."""
    print("\n" + "="*70)
    print("PERFORMANCE TEST 2: Learning Orchestrator Scalability")
    print("="*70)
    
    store = TestMemoryStore({"storage_type": "memory"})
    orchestrator = TestLearningOrchestrator(store, {})
    
    # Populate with test data
    data_sizes = [10, 50, 100, 500, 1000]
    learning_times = []
    
    for size in data_sizes:
        # Clear and populate
        store = TestMemoryStore({"storage_type": "memory"})
        orchestrator = TestLearningOrchestrator(store, {})
        
        for i in range(size):
            event = HealingEvent(
                event_id=f"scale-{i:04d}",
                test_id=f"test-{i}",
                execution_id=f"exec-{i}",
                element_id=f"element-{i % 10}",  # Reuse elements
                original_locator={"css": f"#old"},
                failure_reason="Not found",
                healing_strategy=HealingStrategy.VISUAL_SIMILARITY,
                new_locator={"css": f"#new"},
                confidence_score=0.85,
                healing_successful=True,
                validation_method=ValidationMethod.AUTO
            )
            await store.store_healing_event(event)
        
        # Measure learning cycle time
        start = time.time()
        await orchestrator.run_learning_cycle()
        duration = (time.time() - start) * 1000  # Convert to ms
        learning_times.append(duration)
        
        print(f"  ✓ {size:4d} events: {duration:6.2f}ms")
    
    # Calculate scaling factor
    if len(learning_times) > 1:
        scaling_factor = learning_times[-1] / learning_times[0]
        data_factor = data_sizes[-1] / data_sizes[0]
        efficiency = (data_factor / scaling_factor) * 100
        
        print(f"\n  ✓ Scaling analysis:")
        print(f"    - Data increased: {data_factor}x")
        print(f"    - Time increased: {scaling_factor:.2f}x")
        print(f"    - Efficiency: {efficiency:.1f}%")
    
    results.add_test("Learning Scalability", {
        "data_sizes": data_sizes,
        "learning_times_ms": [round(t, 2) for t in learning_times],
        "max_time_ms": round(max(learning_times), 2),
        "target_met": max(learning_times) < 5000  # Target: < 5 seconds
    })
    
    print(f"\n✅ PERFORMANCE TEST 2 COMPLETE")
    print(f"   Target (< 5s for 1000 events): {'MET' if max(learning_times) < 5000 else 'NOT MET'}")

async def test_concurrent_operations(results: PerformanceResults):
    """Performance Test 3: Concurrent Operations."""
    print("\n" + "="*70)
    print("PERFORMANCE TEST 3: Concurrent Operations")
    print("="*70)
    
    store = TestMemoryStore({"storage_type": "memory"})
    
    # Test concurrent writes
    num_concurrent = 50
    print(f"  Testing {num_concurrent} concurrent writes...")
    
    async def write_event(i):
        event = HealingEvent(
            event_id=f"concurrent-{i:04d}",
            test_id=f"test-{i}",
            execution_id=f"exec-{i}",
            element_id=f"element-{i}",
            original_locator={"css": f"#old-{i}"},
            failure_reason="Not found",
            healing_strategy=HealingStrategy.VISUAL_SIMILARITY,
            new_locator={"css": f"#new-{i}"},
            confidence_score=0.85,
            healing_successful=True,
            validation_method=ValidationMethod.AUTO,
            visual_embedding=[float(i)/num_concurrent] * 512
        )
        await store.store_healing_event(event)
    
    start = time.time()
    await asyncio.gather(*[write_event(i) for i in range(num_concurrent)])
    concurrent_write_time = (time.time() - start) * 1000
    
    print(f"  ✓ Concurrent writes: {concurrent_write_time:.2f}ms")
    print(f"    - Throughput: {num_concurrent / (concurrent_write_time/1000):.0f} ops/sec")
    
    # Test concurrent reads
    print(f"\n  Testing {num_concurrent} concurrent reads...")
    
    async def read_event(i):
        await store.get_healing_event(f"concurrent-{i:04d}")
    
    start = time.time()
    await asyncio.gather(*[read_event(i) for i in range(num_concurrent)])
    concurrent_read_time = (time.time() - start) * 1000
    
    print(f"  ✓ Concurrent reads: {concurrent_read_time:.2f}ms")
    print(f"    - Throughput: {num_concurrent / (concurrent_read_time/1000):.0f} ops/sec")
    
    results.add_test("Concurrent Operations", {
        "concurrent_writes_ms": round(concurrent_write_time, 2),
        "write_throughput_ops_sec": round(num_concurrent / (concurrent_write_time/1000), 0),
        "concurrent_reads_ms": round(concurrent_read_time, 2),
        "read_throughput_ops_sec": round(num_concurrent / (concurrent_read_time/1000), 0)
    })
    
    print(f"\n✅ PERFORMANCE TEST 3 COMPLETE")

async def test_stability_accuracy(results: PerformanceResults):
    """Integration Test: Stability Calculation Accuracy."""
    print("\n" + "="*70)
    print("INTEGRATION TEST: Stability Calculation Accuracy")
    print("="*70)
    
    store = TestMemoryStore({"storage_type": "memory"})
    
    test_scenarios = [
        {"executions": 100, "failures": 0, "expected": 1.0},
        {"executions": 100, "failures": 10, "expected": 0.9},
        {"executions": 100, "failures": 25, "expected": 0.75},
        {"executions": 100, "failures": 50, "expected": 0.5},
    ]
    
    accuracy_results = []
    
    for scenario in test_scenarios:
        store = TestMemoryStore({"storage_type": "memory"})
        
        # Record executions
        for i in range(scenario["executions"]):
            await store.record_test_execution(
                test_id=f"test-{i}",
                elements_tested=["test-element"]
            )
        
        # Record failures
        for i in range(scenario["failures"]):
            event = HealingEvent(
                event_id=f"fail-{i:04d}",
                test_id=f"test-{i}",
                execution_id=f"exec-{i}",
                element_id="test-element",
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
        calculated = await store.calculate_element_stability("test-element")
        expected = scenario["expected"]
        error = abs(calculated - expected)
        
        accuracy_results.append({
            "executions": scenario["executions"],
            "failures": scenario["failures"],
            "expected": expected,
            "calculated": round(calculated, 3),
            "error": round(error, 4)
        })
        
        status = "✓" if error < 0.01 else "✗"
        print(f"  {status} {scenario['failures']:2d} failures / {scenario['executions']:3d} executions:")
        print(f"    Expected: {expected:.2f}, Calculated: {calculated:.3f}, Error: {error:.4f}")
    
    max_error = max(r["error"] for r in accuracy_results)
    
    results.add_test("Stability Accuracy", {
        "test_scenarios": len(test_scenarios),
        "max_error": round(max_error, 4),
        "accuracy_threshold_met": max_error < 0.01,
        "results": accuracy_results
    })
    
    print(f"\n✅ INTEGRATION TEST COMPLETE")
    print(f"   Maximum error: {max_error:.4f} (target: < 0.01)")

async def run_all_performance_tests():
    """Run all performance and integration tests."""
    results = PerformanceResults()
    
    print("\n" + "="*70)
    print("TestDriver MCP Framework v2.0 - Performance & Integration Tests")
    print("="*70)
    print(f"Start Time: {datetime.now().isoformat()}")
    print("="*70)
    
    # Run performance tests
    await test_memory_store_performance(results)
    await test_learning_scalability(results)
    await test_concurrent_operations(results)
    await test_stability_accuracy(results)
    
    # Print summary
    print("\n" + "="*70)
    print("PERFORMANCE TEST SUMMARY")
    print("="*70)
    
    for test in results.tests:
        print(f"\n{test['name']}:")
        for key, value in test['metrics'].items():
            if isinstance(value, (list, dict)):
                continue  # Skip complex structures in summary
            print(f"  {key}: {value}")
    
    # Save results
    with open('performance_results.json', 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "tests": results.tests
        }, f, indent=2)
    
    print("\n" + "="*70)
    print("Results saved to: performance_results.json")
    print("="*70)
    print("\n🎉 ALL PERFORMANCE TESTS COMPLETE!")

if __name__ == "__main__":
    asyncio.run(run_all_performance_tests())
