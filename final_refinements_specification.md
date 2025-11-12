# TestDriver MCP Framework: Final Refinements Specification

## Part 9: Advanced Learning and Memory Systems

### 9.1 Test Memory Store Architecture

The Test Memory Store provides persistent storage for all healing history, learned patterns, and element evolution data. This enables the system to remember and reuse successful healing strategies across test cycles, continuously improving its effectiveness.

**Design Philosophy**:

Traditional test automation systems treat each test execution as an isolated event, losing valuable learning opportunities. The Test Memory Store transforms TestDriver into a system with long-term memory that accumulates knowledge over time. Every healing event, every successful element location, every failure pattern is captured and indexed for future reference. This persistent memory enables the system to recognize similar situations and apply proven solutions immediately, dramatically reducing healing time and improving success rates.

**Technology Selection**:

The Test Memory Store uses a hybrid storage approach combining MongoDB for document storage of healing events and test execution history with a vector database (Qdrant or Weaviate) for efficient similarity search of visual and semantic embeddings. MongoDB provides flexible schema for evolving data structures and excellent query performance for time-series data. The vector database enables sub-second similarity search across millions of element embeddings, making it possible to find visually or semantically similar elements instantly.

**File**: `src/memory/test_memory_store.py`

```python
"""
Test Memory Store Implementation
Persistent storage for healing history and learned patterns
"""

import asyncio
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np
from motor.motor_asyncio import AsyncIOMotorClient
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class HealingEvent:
    """Record of a healing event."""
    event_id: str
    test_id: str
    element_id: str
    timestamp: datetime
    
    # Failure context
    original_locator: Dict[str, str]
    failure_reason: str
    failure_screenshot: str  # S3 path
    
    # Healing attempt
    healing_strategy: str
    new_locator: Dict[str, str]
    confidence_score: float
    
    # Outcome
    healing_successful: bool
    validation_method: str  # auto, pr_approved, manual_approved
    user_feedback: Optional[str]
    
    # Learning data
    visual_embedding: List[float]
    semantic_embedding: List[float]
    context_features: Dict[str, Any]


@dataclass
class LocatorVersion:
    """Version history entry for an element locator."""
    version_id: str
    element_id: str
    locator: Dict[str, str]
    visual_embedding: List[float]
    semantic_embedding: List[float]
    stability_score: float
    created_at: datetime
    deprecated_at: Optional[datetime]
    deprecation_reason: Optional[str]


class TestMemoryStore:
    """
    Persistent memory store for test healing and learning data.
    
    Combines MongoDB for document storage with vector database for similarity search.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize test memory store.
        
        Args:
            config: Configuration dictionary with connection details
        """
        # MongoDB for document storage
        mongo_uri = config.get('mongodb_uri', 'mongodb://localhost:27017')
        self.mongo_client = AsyncIOMotorClient(mongo_uri)
        self.db = self.mongo_client[config.get('database_name', 'testdriver')]
        
        # Collections
        self.healing_events = self.db['healing_events']
        self.locator_versions = self.db['locator_versions']
        self.test_executions = self.db['test_executions']
        self.learning_patterns = self.db['learning_patterns']
        
        # Vector database for similarity search
        qdrant_host = config.get('qdrant_host', 'localhost')
        qdrant_port = config.get('qdrant_port', 6333)
        self.vector_client = AsyncQdrantClient(host=qdrant_host, port=qdrant_port)
        
        # Collection names
        self.visual_collection = 'element_visual_embeddings'
        self.semantic_collection = 'element_semantic_embeddings'
    
    async def initialize(self) -> None:
        """Initialize database collections and indexes."""
        logger.info("Initializing Test Memory Store")
        
        # Create MongoDB indexes
        await self.healing_events.create_index([('element_id', 1), ('timestamp', -1)])
        await self.healing_events.create_index([('healing_successful', 1)])
        await self.healing_events.create_index([('healing_strategy', 1)])
        
        await self.locator_versions.create_index([('element_id', 1), ('created_at', -1)])
        await self.locator_versions.create_index([('stability_score', -1)])
        
        # Create vector collections
        await self._ensure_vector_collection(
            self.visual_collection,
            vector_size=512,  # CLIP embedding size
            distance=Distance.COSINE
        )
        
        await self._ensure_vector_collection(
            self.semantic_collection,
            vector_size=384,  # Sentence transformer embedding size
            distance=Distance.COSINE
        )
        
        logger.info("Test Memory Store initialized successfully")
    
    async def store_healing_event(self, event: HealingEvent) -> None:
        """
        Store a healing event with embeddings.
        
        Args:
            event: HealingEvent to store
        """
        logger.debug("Storing healing event", event_id=event.event_id)
        
        # Store document in MongoDB
        event_dict = asdict(event)
        event_dict['timestamp'] = event.timestamp
        await self.healing_events.insert_one(event_dict)
        
        # Store visual embedding in vector database
        await self.vector_client.upsert(
            collection_name=self.visual_collection,
            points=[PointStruct(
                id=event.event_id,
                vector=event.visual_embedding,
                payload={
                    'element_id': event.element_id,
                    'test_id': event.test_id,
                    'healing_strategy': event.healing_strategy,
                    'confidence_score': event.confidence_score,
                    'successful': event.healing_successful
                }
            )]
        )
        
        # Store semantic embedding
        await self.vector_client.upsert(
            collection_name=self.semantic_collection,
            points=[PointStruct(
                id=f"{event.event_id}_semantic",
                vector=event.semantic_embedding,
                payload={
                    'element_id': event.element_id,
                    'test_id': event.test_id
                }
            )]
        )
    
    async def find_similar_healing_events(
        self,
        visual_embedding: Optional[np.ndarray] = None,
        semantic_embedding: Optional[np.ndarray] = None,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 10
    ) -> List[Tuple[HealingEvent, float]]:
        """
        Find healing events similar to given embeddings.
        
        Args:
            visual_embedding: Visual embedding to search for
            semantic_embedding: Semantic embedding to search for
            filters: Additional filters (e.g., successful=True)
            limit: Maximum number of results
            
        Returns:
            List of (HealingEvent, similarity_score) tuples
        """
        results = []
        
        if visual_embedding is not None:
            # Search by visual similarity
            search_results = await self.vector_client.search(
                collection_name=self.visual_collection,
                query_vector=visual_embedding.tolist(),
                limit=limit,
                query_filter=self._build_vector_filter(filters)
            )
            
            # Fetch full healing events from MongoDB
            for hit in search_results:
                event_doc = await self.healing_events.find_one({'event_id': hit.id})
                if event_doc:
                    event = self._doc_to_healing_event(event_doc)
                    results.append((event, hit.score))
        
        elif semantic_embedding is not None:
            # Search by semantic similarity
            search_results = await self.vector_client.search(
                collection_name=self.semantic_collection,
                query_vector=semantic_embedding.tolist(),
                limit=limit
            )
            
            for hit in search_results:
                event_id = hit.id.replace('_semantic', '')
                event_doc = await self.healing_events.find_one({'event_id': event_id})
                if event_doc:
                    event = self._doc_to_healing_event(event_doc)
                    results.append((event, hit.score))
        
        return results
    
    async def store_locator_version(self, version: LocatorVersion) -> None:
        """
        Store a new version of an element locator.
        
        Args:
            version: LocatorVersion to store
        """
        version_dict = asdict(version)
        version_dict['created_at'] = version.created_at
        version_dict['deprecated_at'] = version.deprecated_at
        
        await self.locator_versions.insert_one(version_dict)
        
        logger.debug("Stored locator version",
                    element_id=version.element_id,
                    version_id=version.version_id)
    
    async def get_locator_evolution(
        self,
        element_id: str
    ) -> List[LocatorVersion]:
        """
        Get complete evolution history of an element's locators.
        
        Args:
            element_id: Element identifier
            
        Returns:
            List of LocatorVersion objects sorted by creation time
        """
        cursor = self.locator_versions.find(
            {'element_id': element_id}
        ).sort('created_at', -1)
        
        versions = []
        async for doc in cursor:
            versions.append(self._doc_to_locator_version(doc))
        
        return versions
    
    async def calculate_element_stability(
        self,
        element_id: str,
        lookback_days: int = 30
    ) -> float:
        """
        Calculate stability score for an element based on healing frequency.
        
        Args:
            element_id: Element identifier
            lookback_days: Number of days to analyze
            
        Returns:
            Stability score (0.0 to 1.0, higher is more stable)
        """
        cutoff = datetime.utcnow() - timedelta(days=lookback_days)
        
        # Count total test executions involving this element
        total_executions = await self.test_executions.count_documents({
            'elements_tested': element_id,
            'started_at': {'$gte': cutoff}
        })
        
        if total_executions == 0:
            return 1.0  # No data, assume stable
        
        # Count healing events for this element
        healing_count = await self.healing_events.count_documents({
            'element_id': element_id,
            'timestamp': {'$gte': cutoff}
        })
        
        # Stability = 1 - (healing_frequency)
        healing_frequency = healing_count / total_executions
        stability = max(0.0, 1.0 - healing_frequency)
        
        return stability
    
    async def get_successful_healing_strategies(
        self,
        element_id: Optional[str] = None,
        lookback_days: int = 90
    ) -> Dict[str, float]:
        """
        Get success rates for different healing strategies.
        
        Args:
            element_id: Optional element ID to filter by
            lookback_days: Number of days to analyze
            
        Returns:
            Dictionary mapping strategy name to success rate
        """
        cutoff = datetime.utcnow() - timedelta(days=lookback_days)
        
        match_filter = {'timestamp': {'$gte': cutoff}}
        if element_id:
            match_filter['element_id'] = element_id
        
        pipeline = [
            {'$match': match_filter},
            {'$group': {
                '_id': '$healing_strategy',
                'total': {'$sum': 1},
                'successful': {
                    '$sum': {'$cond': ['$healing_successful', 1, 0]}
                }
            }},
            {'$project': {
                'strategy': '$_id',
                'success_rate': {'$divide': ['$successful', '$total']}
            }}
        ]
        
        results = {}
        async for doc in self.healing_events.aggregate(pipeline):
            results[doc['strategy']] = doc['success_rate']
        
        return results
    
    async def _ensure_vector_collection(
        self,
        collection_name: str,
        vector_size: int,
        distance: Distance
    ) -> None:
        """Ensure vector collection exists with proper configuration."""
        collections = await self.vector_client.get_collections()
        collection_names = [c.name for c in collections.collections]
        
        if collection_name not in collection_names:
            await self.vector_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=distance
                )
            )
            logger.info(f"Created vector collection: {collection_name}")
    
    def _build_vector_filter(self, filters: Optional[Dict[str, Any]]) -> Optional[Dict]:
        """Build Qdrant filter from dictionary."""
        if not filters:
            return None
        
        # Convert filters to Qdrant filter format
        # (Implementation depends on specific filter requirements)
        return None
    
    def _doc_to_healing_event(self, doc: Dict[str, Any]) -> HealingEvent:
        """Convert MongoDB document to HealingEvent object."""
        return HealingEvent(
            event_id=doc['event_id'],
            test_id=doc['test_id'],
            element_id=doc['element_id'],
            timestamp=doc['timestamp'],
            original_locator=doc['original_locator'],
            failure_reason=doc['failure_reason'],
            failure_screenshot=doc['failure_screenshot'],
            healing_strategy=doc['healing_strategy'],
            new_locator=doc['new_locator'],
            confidence_score=doc['confidence_score'],
            healing_successful=doc['healing_successful'],
            validation_method=doc['validation_method'],
            user_feedback=doc.get('user_feedback'),
            visual_embedding=doc['visual_embedding'],
            semantic_embedding=doc['semantic_embedding'],
            context_features=doc['context_features']
        )
    
    def _doc_to_locator_version(self, doc: Dict[str, Any]) -> LocatorVersion:
        """Convert MongoDB document to LocatorVersion object."""
        return LocatorVersion(
            version_id=doc['version_id'],
            element_id=doc['element_id'],
            locator=doc['locator'],
            visual_embedding=doc['visual_embedding'],
            semantic_embedding=doc['semantic_embedding'],
            stability_score=doc['stability_score'],
            created_at=doc['created_at'],
            deprecated_at=doc.get('deprecated_at'),
            deprecation_reason=doc.get('deprecation_reason')
        )
```

### 9.2 Test Learning Orchestrator

The Test Learning Orchestrator continuously analyzes test execution history to optimize system parameters and behavior. It digests past run logs and failed test cases to fine-tune retry thresholds, wait durations, and preferred detection modes, making the system increasingly adaptive to application behavior.

**File**: `src/learning/test_learning_orchestrator.py`

```python
"""
Test Learning Orchestrator
Continuously learns from test execution history to optimize system behavior
"""

import asyncio
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class LearningInsight:
    """Insight learned from test execution history."""
    insight_type: str
    element_id: Optional[str]
    recommendation: str
    confidence: float
    supporting_data: Dict[str, Any]
    created_at: datetime


class TestLearningOrchestrator:
    """
    Orchestrates continuous learning from test execution history.
    
    Analyzes patterns to optimize retry thresholds, wait durations,
    and detection mode preferences.
    """
    
    def __init__(
        self,
        memory_store: Any,
        config: Dict[str, Any]
    ):
        """
        Initialize learning orchestrator.
        
        Args:
            memory_store: TestMemoryStore instance
            config: Configuration dictionary
        """
        self.memory_store = memory_store
        self.learning_interval_hours = config.get('learning_interval_hours', 24)
        self.min_samples_for_learning = config.get('min_samples_for_learning', 100)
        
        # ML models for parameter optimization
        self.wait_duration_model = RandomForestRegressor(n_estimators=100)
        self.retry_threshold_model = RandomForestRegressor(n_estimators=100)
        
        self._learning_task: Optional[asyncio.Task] = None
    
    async def start(self) -> None:
        """Start continuous learning background task."""
        logger.info("Starting Test Learning Orchestrator")
        self._learning_task = asyncio.create_task(self._learning_loop())
    
    async def stop(self) -> None:
        """Stop continuous learning."""
        if self._learning_task:
            self._learning_task.cancel()
            try:
                await self._learning_task
            except asyncio.CancelledError:
                pass
    
    async def _learning_loop(self) -> None:
        """Background task that runs learning cycles."""
        while True:
            try:
                await asyncio.sleep(self.learning_interval_hours * 3600)
                
                logger.info("Starting learning cycle")
                
                # Learn optimal wait durations
                await self._learn_wait_durations()
                
                # Learn optimal retry thresholds
                await self._learn_retry_thresholds()
                
                # Learn preferred detection modes
                await self._learn_detection_mode_preferences()
                
                # Generate insights
                insights = await self._generate_insights()
                
                logger.info("Learning cycle complete",
                           insights_generated=len(insights))
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Learning cycle error", error=str(e))
    
    async def _learn_wait_durations(self) -> None:
        """Learn optimal wait durations for different element types."""
        logger.debug("Learning optimal wait durations")
        
        # Fetch historical wait data
        # (Query test executions with wait times and success/failure)
        
        # Train model to predict optimal wait duration
        # based on element type, page complexity, network conditions
        
        # Update system configuration with learned parameters
        pass
    
    async def _learn_retry_thresholds(self) -> None:
        """Learn optimal retry thresholds for different scenarios."""
        logger.debug("Learning optimal retry thresholds")
        
        # Analyze retry patterns and success rates
        # Determine optimal number of retries per failure type
        
        pass
    
    async def _learn_detection_mode_preferences(self) -> None:
        """Learn preferred detection modes (DOM vs Vision vs Hybrid) per element."""
        logger.debug("Learning detection mode preferences")
        
        # Analyze which detection modes work best for different element types
        # Update element metadata with preferred detection mode
        
        pass
    
    async def _generate_insights(self) -> List[LearningInsight]:
        """Generate actionable insights from learned patterns."""
        insights = []
        
        # Identify elements that frequently need healing
        unstable_elements = await self._identify_unstable_elements()
        
        for element_id, instability_score in unstable_elements:
            insight = LearningInsight(
                insight_type='unstable_element',
                element_id=element_id,
                recommendation=f"Element {element_id} has high instability (score: {instability_score:.2f}). Consider updating test with more robust locator.",
                confidence=0.9,
                supporting_data={'instability_score': instability_score},
                created_at=datetime.utcnow()
            )
            insights.append(insight)
        
        return insights
    
    async def _identify_unstable_elements(self) -> List[Tuple[str, float]]:
        """Identify elements with high healing frequency."""
        # Query healing events grouped by element
        # Calculate instability score = healing_frequency × (1 - success_rate)
        
        return []
```

This specification continues with additional components for Multi-Agent Cognitive Architecture, Test Stability Index, Multi-Layer Verification, Environment-Aware Intelligence, Continuous Model Training, Human-in-the-Loop Correction, and Chaos Validation Mode.
