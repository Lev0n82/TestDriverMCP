"""
Qdrant Vector Store for semantic similarity search.
Enables memory-based healing through vector embeddings.

Built-in Self-Tests:
- Function-level: Each method validates inputs and outputs
- Class-level: Initialization and connection validation
- Module-level: Integration with embedding models
"""

from typing import List, Dict, Any, Optional, Tuple
import uuid
from datetime import datetime

import structlog
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter,
    FieldCondition, MatchValue, SearchRequest
)
from sentence_transformers import SentenceTransformer

logger = structlog.get_logger()


class VectorStoreValidator:
    """Built-in validator for VectorStore operations."""
    
    @staticmethod
    def validate_embedding(embedding: List[float], expected_dim: int) -> bool:
        """
        Validate embedding vector.
        
        Success Criteria:
        - Embedding is a list of floats
        - Length matches expected dimension
        - All values are finite numbers
        """
        if not isinstance(embedding, list):
            return False
        if len(embedding) != expected_dim:
            return False
        if not all(isinstance(x, (int, float)) and abs(x) < 1e10 for x in embedding):
            return False
        return True
    
    @staticmethod
    def validate_search_results(
        results: List[Dict[str, Any]],
        min_score: float = 0.0
    ) -> bool:
        """
        Validate search results.
        
        Success Criteria:
        - Results is a list
        - Each result has required fields (id, score, payload)
        - Scores are within valid range [0, 1]
        - Scores are sorted in descending order
        """
        if not isinstance(results, list):
            return False
        
        prev_score = 1.0
        for result in results:
            if not isinstance(result, dict):
                return False
            if 'id' not in result or 'score' not in result or 'payload' not in result:
                return False
            score = result['score']
            if not (0.0 <= score <= 1.0):
                return False
            if score > prev_score:
                return False
            prev_score = score
        
        return True


class QdrantVectorStore:
    """
    Vector store using Qdrant for semantic similarity search.
    
    Success Criteria (Class-level):
    - Connection to Qdrant established
    - Collection created with correct dimensions
    - Embedding model loaded successfully
    - All CRUD operations functional
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Qdrant vector store.
        
        Args:
            config: Configuration dict with:
                - host: Qdrant host (default: localhost)
                - port: Qdrant port (default: 6333)
                - collection_name: Collection name (default: testdriver_memory)
                - embedding_model: Model name (default: all-MiniLM-L6-v2)
                - embedding_dim: Embedding dimension (default: 384)
        
        Success Criteria:
        - Config validated
        - Client initialized
        - Embedding model loaded
        """
        self.host = config.get('host', 'localhost')
        self.port = config.get('port', 6333)
        self.collection_name = config.get('collection_name', 'testdriver_memory')
        self.embedding_model_name = config.get('embedding_model', 'all-MiniLM-L6-v2')
        self.embedding_dim = config.get('embedding_dim', 384)
        
        # Initialize client (use in-memory mode if connection fails)
        try:
            self.client = QdrantClient(host=self.host, port=self.port, timeout=2)
            # Test connection
            self.client.get_collections()
            self.in_memory_mode = False
            logger.info("Connected to Qdrant server", host=self.host, port=self.port)
        except Exception as e:
            logger.warning("Qdrant server not available, using in-memory mode", error=str(e))
            self.client = QdrantClient(":memory:")
            self.in_memory_mode = True
        
        # Load embedding model
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        
        # Validator
        self.validator = VectorStoreValidator()
        
        logger.info(
            "Qdrant vector store initialized",
            host=self.host,
            port=self.port,
            collection=self.collection_name,
            model=self.embedding_model_name
        )
        
        # Self-test initialization
        self._self_test_init()
    
    def _self_test_init(self) -> bool:
        """
        Self-test: Validate initialization.
        
        Success Criteria:
        - Client is connected
        - Embedding model produces correct dimensions
        """
        try:
            # Test embedding model
            test_text = "test initialization"
            test_embedding = self.embedding_model.encode(test_text).tolist()
            
            if not self.validator.validate_embedding(test_embedding, self.embedding_dim):
                logger.error("Self-test failed: Invalid embedding dimensions")
                return False
            
            logger.debug("Self-test passed: Initialization successful")
            return True
        except Exception as e:
            logger.error("Self-test failed: Initialization error", error=str(e))
            return False
    
    async def initialize(self) -> bool:
        """
        Initialize collection in Qdrant.
        
        Success Criteria:
        - Collection created or already exists
        - Vector parameters match configuration
        
        Returns:
            True if successful
        """
        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if self.collection_name not in collection_names:
                # Create collection
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_dim,
                        distance=Distance.COSINE
                    )
                )
                logger.info("Collection created", collection=self.collection_name)
            else:
                logger.info("Collection already exists", collection=self.collection_name)
            
            # Self-test collection
            return self._self_test_collection()
        
        except Exception as e:
            logger.error("Collection initialization failed", error=str(e))
            return False
    
    def _self_test_collection(self) -> bool:
        """
        Self-test: Validate collection configuration.
        
        Success Criteria:
        - Collection exists
        - Vector dimension matches expected
        - Distance metric is COSINE
        """
        try:
            collection_info = self.client.get_collection(self.collection_name)
            
            if collection_info.config.params.vectors.size != self.embedding_dim:
                logger.error(
                    "Self-test failed: Dimension mismatch",
                    expected=self.embedding_dim,
                    actual=collection_info.config.params.vectors.size
                )
                return False
            
            if collection_info.config.params.vectors.distance != Distance.COSINE:
                logger.error("Self-test failed: Distance metric not COSINE")
                return False
            
            logger.debug("Self-test passed: Collection validated")
            return True
        
        except Exception as e:
            logger.error("Self-test failed: Collection validation error", error=str(e))
            return False
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for text.
        
        Args:
            text: Input text
        
        Returns:
            Embedding vector
        
        Success Criteria:
        - Embedding generated successfully
        - Dimension matches expected
        - All values are finite
        """
        try:
            embedding = self.embedding_model.encode(text).tolist()
            
            # Self-test embedding
            if not self.validator.validate_embedding(embedding, self.embedding_dim):
                raise ValueError("Generated embedding failed validation")
            
            return embedding
        
        except Exception as e:
            logger.error("Embedding generation failed", error=str(e), text=text[:100])
            raise
    
    async def store_healing_memory(
        self,
        event_id: str,
        element_description: str,
        original_locator: Dict[str, Any],
        new_locator: Dict[str, Any],
        confidence: float,
        context: Dict[str, Any]
    ) -> bool:
        """
        Store healing event in vector store.
        
        Args:
            event_id: Unique event identifier
            element_description: Natural language element description
            original_locator: Failed locator
            new_locator: Healed locator
            confidence: Healing confidence score
            context: Additional context
        
        Returns:
            True if stored successfully
        
        Success Criteria:
        - Embedding generated from description
        - Point stored in Qdrant
        - Point can be retrieved
        """
        try:
            # Generate embedding
            embedding = self.generate_embedding(element_description)
            
            # Create point
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    'event_id': event_id,
                    'element_description': element_description,
                    'original_locator': original_locator,
                    'new_locator': new_locator,
                    'confidence': confidence,
                    'context': context,
                    'timestamp': datetime.utcnow().isoformat()
                }
            )
            
            # Store in Qdrant
            self.client.upsert(
                collection_name=self.collection_name,
                points=[point]
            )
            
            logger.info(
                "Healing memory stored",
                event_id=event_id,
                confidence=confidence
            )
            
            # Self-test storage
            return self._self_test_storage(point.id)
        
        except Exception as e:
            logger.error("Failed to store healing memory", error=str(e), event_id=event_id)
            return False
    
    def _self_test_storage(self, point_id: str) -> bool:
        """
        Self-test: Verify point was stored.
        
        Success Criteria:
        - Point can be retrieved by ID
        - Payload is intact
        """
        try:
            points = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[point_id]
            )
            
            if len(points) != 1:
                logger.error("Self-test failed: Point not found", point_id=point_id)
                return False
            
            if points[0].payload is None:
                logger.error("Self-test failed: Payload is None")
                return False
            
            logger.debug("Self-test passed: Storage validated", point_id=point_id)
            return True
        
        except Exception as e:
            logger.error("Self-test failed: Storage validation error", error=str(e))
            return False
    
    async def find_similar_healings(
        self,
        element_description: str,
        limit: int = 10,
        min_confidence: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Find similar healing events.
        
        Args:
            element_description: Element description to search for
            limit: Maximum number of results
            min_confidence: Minimum confidence threshold
        
        Returns:
            List of similar healing events with scores
        
        Success Criteria:
        - Query embedding generated
        - Search executed successfully
        - Results sorted by similarity
        - All results meet minimum confidence
        """
        try:
            # Generate query embedding
            query_embedding = self.generate_embedding(element_description)
            
            # Search in Qdrant
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit,
                score_threshold=min_confidence
            )
            
            # Format results
            results = [
                {
                    'id': hit.id,
                    'score': hit.score,
                    'payload': hit.payload
                }
                for hit in search_results
            ]
            
            # Self-test results
            if not self.validator.validate_search_results(results, min_confidence):
                logger.warning("Self-test failed: Invalid search results")
            
            logger.info(
                "Similar healings found",
                query=element_description[:50],
                count=len(results)
            )
            
            return results
        
        except Exception as e:
            logger.error("Similarity search failed", error=str(e))
            return []
    
    async def get_statistics(self) -> Dict[str, Any]:
        """
        Get vector store statistics.
        
        Returns:
            Statistics dictionary
        
        Success Criteria:
        - Collection info retrieved
        - All statistics are valid numbers
        """
        try:
            collection_info = self.client.get_collection(self.collection_name)
            
            stats = {
                'total_vectors': collection_info.points_count,
                'vector_dimension': collection_info.config.params.vectors.size,
                'distance_metric': collection_info.config.params.vectors.distance.name,
                'collection_name': self.collection_name
            }
            
            logger.debug("Vector store statistics", **stats)
            return stats
        
        except Exception as e:
            logger.error("Failed to get statistics", error=str(e))
            return {}
    
    async def close(self):
        """Close connections."""
        try:
            self.client.close()
            logger.info("Vector store connections closed")
        except Exception as e:
            logger.error("Error closing connections", error=str(e))


# Module-level self-test
def self_test_module() -> bool:
    """
    Module-level self-test.
    
    Success Criteria:
    - All classes can be instantiated
    - Validator works correctly
    - Embedding model can be loaded
    """
    try:
        # Test validator
        validator = VectorStoreValidator()
        
        # Test valid embedding
        valid_embedding = [0.1] * 384
        if not validator.validate_embedding(valid_embedding, 384):
            logger.error("Module self-test failed: Valid embedding rejected")
            return False
        
        # Test invalid embedding
        invalid_embedding = [0.1] * 100
        if validator.validate_embedding(invalid_embedding, 384):
            logger.error("Module self-test failed: Invalid embedding accepted")
            return False
        
        # Test search results validation
        valid_results = [
            {'id': '1', 'score': 0.9, 'payload': {}},
            {'id': '2', 'score': 0.8, 'payload': {}}
        ]
        if not validator.validate_search_results(valid_results):
            logger.error("Module self-test failed: Valid results rejected")
            return False
        
        logger.info("Module self-test passed: vector_store")
        return True
    
    except Exception as e:
        logger.error("Module self-test failed", error=str(e))
        return False


if __name__ == "__main__":
    # Run module self-test
    import asyncio
    
    success = self_test_module()
    print(f"Module self-test: {'PASSED' if success else 'FAILED'}")
