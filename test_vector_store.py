"""
Comprehensive test suite for Qdrant Vector Store.
Tests all functions, classes, and module-level functionality.
"""

import pytest
import asyncio
from typing import List, Dict, Any

# Mock Qdrant if not available
try:
    from qdrant_client import QdrantClient
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False

import sys
sys.path.insert(0, '/home/ubuntu/testdriver-full/testdriver-mcp-full/src')

from storage.vector_store import (
    QdrantVectorStore,
    VectorStoreValidator,
    self_test_module
)


class MockQdrantClient:
    """Mock Qdrant client for testing without server."""
    
    def __init__(self, *args, **kwargs):
        self.collections = {}
        self.points = {}
    
    def get_collections(self):
        class Collections:
            def __init__(self, colls):
                self.collections = colls
        return Collections([])
    
    def create_collection(self, collection_name, vectors_config):
        self.collections[collection_name] = {
            'vectors_config': vectors_config
        }
    
    def get_collection(self, collection_name):
        class CollectionInfo:
            class Config:
                class Params:
                    class Vectors:
                        size = 384
                        distance = "COSINE"
                    vectors = Vectors()
                params = Params()
            config = Config()
            points_count = 0
        return CollectionInfo()
    
    def upsert(self, collection_name, points):
        if collection_name not in self.points:
            self.points[collection_name] = []
        self.points[collection_name].extend(points)
    
    def retrieve(self, collection_name, ids):
        class Point:
            def __init__(self, id, payload):
                self.id = id
                self.payload = payload
        
        return [Point(id, {'test': 'data'}) for id in ids]
    
    def search(self, collection_name, query_vector, limit, score_threshold=0.0):
        class Hit:
            def __init__(self, id, score, payload):
                self.id = id
                self.score = score
                self.payload = payload
        
        # Return mock results
        return [
            Hit('1', 0.95, {'element_description': 'test', 'confidence': 0.9}),
            Hit('2', 0.85, {'element_description': 'test2', 'confidence': 0.8})
        ]
    
    def close(self):
        pass


@pytest.fixture
def mock_qdrant(monkeypatch):
    """Mock Qdrant client."""
    if not QDRANT_AVAILABLE:
        monkeypatch.setattr('storage.vector_store.QdrantClient', MockQdrantClient)


def test_validator_valid_embedding():
    """Test validator accepts valid embeddings."""
    validator = VectorStoreValidator()
    
    # Test valid embedding
    valid_embedding = [0.1, 0.2, 0.3] * 128  # 384 dimensions
    assert validator.validate_embedding(valid_embedding, 384)
    
    print("✓ Validator accepts valid embeddings")


def test_validator_invalid_embedding():
    """Test validator rejects invalid embeddings."""
    validator = VectorStoreValidator()
    
    # Wrong dimension
    wrong_dim = [0.1] * 100
    assert not validator.validate_embedding(wrong_dim, 384)
    
    # Not a list
    assert not validator.validate_embedding("not a list", 384)
    
    # Invalid values
    invalid_values = [float('inf')] * 384
    assert not validator.validate_embedding(invalid_values, 384)
    
    print("✓ Validator rejects invalid embeddings")


def test_validator_search_results():
    """Test validator validates search results."""
    validator = VectorStoreValidator()
    
    # Valid results
    valid_results = [
        {'id': '1', 'score': 0.9, 'payload': {}},
        {'id': '2', 'score': 0.8, 'payload': {}}
    ]
    assert validator.validate_search_results(valid_results)
    
    # Invalid: wrong order
    wrong_order = [
        {'id': '1', 'score': 0.7, 'payload': {}},
        {'id': '2', 'score': 0.9, 'payload': {}}
    ]
    assert not validator.validate_search_results(wrong_order)
    
    # Invalid: missing fields
    missing_fields = [
        {'id': '1', 'score': 0.9}
    ]
    assert not validator.validate_search_results(missing_fields)
    
    print("✓ Validator correctly validates search results")


@pytest.mark.asyncio
async def test_vector_store_initialization(mock_qdrant):
    """Test vector store initialization."""
    config = {
        'host': 'localhost',
        'port': 6333,
        'collection_name': 'test_collection',
        'embedding_model': 'all-MiniLM-L6-v2',
        'embedding_dim': 384
    }
    
    store = QdrantVectorStore(config)
    
    assert store.host == 'localhost'
    assert store.port == 6333
    assert store.collection_name == 'test_collection'
    assert store.embedding_dim == 384
    
    print("✓ Vector store initializes correctly")


@pytest.mark.asyncio
async def test_vector_store_collection_init(mock_qdrant):
    """Test collection initialization."""
    config = {
        'host': 'localhost',
        'port': 6333,
        'collection_name': 'test_collection'
    }
    
    store = QdrantVectorStore(config)
    success = await store.initialize()
    
    assert success is True
    
    print("✓ Collection initializes successfully")


@pytest.mark.asyncio
async def test_embedding_generation(mock_qdrant):
    """Test embedding generation."""
    config = {'host': 'localhost', 'port': 6333}
    store = QdrantVectorStore(config)
    
    text = "Click the submit button"
    embedding = store.generate_embedding(text)
    
    assert isinstance(embedding, list)
    assert len(embedding) == 384
    assert all(isinstance(x, float) for x in embedding)
    
    print("✓ Embedding generation works correctly")


@pytest.mark.asyncio
async def test_store_healing_memory(mock_qdrant):
    """Test storing healing memory."""
    config = {'host': 'localhost', 'port': 6333}
    store = QdrantVectorStore(config)
    await store.initialize()
    
    success = await store.store_healing_memory(
        event_id='test-001',
        element_description='Submit button',
        original_locator={'css': '#old-button'},
        new_locator={'css': '#new-button'},
        confidence=0.92,
        context={'page': 'login'}
    )
    
    assert success is True
    
    print("✓ Healing memory stored successfully")


@pytest.mark.asyncio
async def test_find_similar_healings(mock_qdrant):
    """Test finding similar healings."""
    config = {'host': 'localhost', 'port': 6333}
    store = QdrantVectorStore(config)
    await store.initialize()
    
    # Store some memories first
    await store.store_healing_memory(
        event_id='test-001',
        element_description='Submit button',
        original_locator={'css': '#old-button'},
        new_locator={'css': '#new-button'},
        confidence=0.92,
        context={}
    )
    
    # Search for similar
    results = await store.find_similar_healings(
        element_description='Click submit',
        limit=5,
        min_confidence=0.7
    )
    
    assert isinstance(results, list)
    assert len(results) > 0
    assert all('score' in r for r in results)
    assert all('payload' in r for r in results)
    
    print(f"✓ Found {len(results)} similar healings")


@pytest.mark.asyncio
async def test_get_statistics(mock_qdrant):
    """Test getting statistics."""
    config = {'host': 'localhost', 'port': 6333}
    store = QdrantVectorStore(config)
    await store.initialize()
    
    stats = await store.get_statistics()
    
    assert isinstance(stats, dict)
    assert 'total_vectors' in stats
    assert 'vector_dimension' in stats
    assert 'collection_name' in stats
    
    print("✓ Statistics retrieved successfully")


def test_module_self_test():
    """Test module-level self-test."""
    success = self_test_module()
    assert success is True
    print("✓ Module self-test passed")


async def run_all_tests():
    """Run all vector store tests."""
    print("\n" + "="*60)
    print("Qdrant Vector Store - Comprehensive Test Suite")
    print("="*60)
    
    tests = [
        ("Validator - Valid Embedding", test_validator_valid_embedding),
        ("Validator - Invalid Embedding", test_validator_invalid_embedding),
        ("Validator - Search Results", test_validator_search_results),
        ("Vector Store - Initialization", test_vector_store_initialization),
        ("Vector Store - Collection Init", test_vector_store_collection_init),
        ("Embedding Generation", test_embedding_generation),
        ("Store Healing Memory", test_store_healing_memory),
        ("Find Similar Healings", test_find_similar_healings),
        ("Get Statistics", test_get_statistics),
        ("Module Self-Test", test_module_self_test),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            if asyncio.iscoroutinefunction(test_func):
                await test_func(None)  # Pass mock fixture
            else:
                test_func()
            passed += 1
            print(f"✓ {name}: PASSED")
        except Exception as e:
            failed += 1
            print(f"✗ {name}: FAILED - {e}")
    
    print("\n" + "="*60)
    print(f"Total: {len(tests)} tests")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Success Rate: {passed/len(tests)*100:.1f}%")
    print("="*60)
    
    return passed, failed


if __name__ == "__main__":
    asyncio.run(run_all_tests())
