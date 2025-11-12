"""
Persistent Test Memory Store using PostgreSQL/SQLite.
Replaces in-memory storage with database-backed persistence.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
import structlog
from sqlalchemy import select, update, func
from sqlalchemy.ext.asyncio import AsyncSession

from models import HealingEvent, ValidationMethod, HealingStrategy
from storage.database import (
    DatabaseManager,
    HealingEventDB,
    TestExecutionDB,
    ElementStabilityDB,
    get_database_manager
)

logger = structlog.get_logger()

class PersistentMemoryStore:
    """
    Persistent memory store for healing events and test executions.
    Uses PostgreSQL for production, SQLite for testing.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize persistent memory store.
        
        Args:
            config: Configuration dict with 'database_url' key
        """
        database_url = config.get("database_url", "sqlite:///./testdriver.db")
        self.db = get_database_manager(database_url)
        self.config = config
        logger.info("Persistent Memory Store initialized", database_url=database_url)
    
    async def initialize(self):
        """Initialize database tables."""
        await self.db.create_tables()
    
    async def store_healing_event(self, event: HealingEvent) -> None:
        """Store a healing event in the database."""
        async with self.db.get_session() as session:
            db_event = HealingEventDB(
                event_id=event.event_id,
                test_id=event.test_id,
                execution_id=event.execution_id,
                element_id=event.element_id,
                original_locator=event.original_locator,
                failure_reason=event.failure_reason,
                healing_strategy=event.healing_strategy.value,
                new_locator=event.new_locator,
                confidence_score=event.confidence_score,
                healing_successful=event.healing_successful,
                validation_method=event.validation_method.value,
                created_at=event.timestamp,
                metadata_json=event.metadata
            )
            session.add(db_event)
        
        logger.info("Healing event stored", event_id=event.event_id, confidence=event.confidence_score)
    
    async def get_healing_event(self, event_id: str) -> Optional[HealingEvent]:
        """Retrieve a healing event by ID."""
        async with self.db.get_session() as session:
            if self.db.is_async:
                result = await session.execute(
                    select(HealingEventDB).where(HealingEventDB.event_id == event_id)
                )
                db_event = result.scalar_one_or_none()
            else:
                result = session.execute(
                    select(HealingEventDB).where(HealingEventDB.event_id == event_id)
                )
                db_event = result.scalar_one_or_none()
            
            if db_event is None:
                return None
            
            return HealingEvent(
                event_id=db_event.event_id,
                test_id=db_event.test_id,
                execution_id=db_event.execution_id,
                element_id=db_event.element_id,
                original_locator=db_event.original_locator,
                failure_reason=db_event.failure_reason,
                healing_strategy=HealingStrategy(db_event.healing_strategy),
                new_locator=db_event.new_locator,
                confidence_score=db_event.confidence_score,
                healing_successful=db_event.healing_successful,
                validation_method=ValidationMethod(db_event.validation_method),
                timestamp=db_event.created_at,
                metadata=db_event.metadata_json or {}
            )
    
    async def find_similar_healing_events(
        self,
        query_embedding: List[float],
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Find similar healing events using cosine similarity.
        Note: For production, use Qdrant for vector similarity search.
        This is a simplified implementation for SQLite/PostgreSQL.
        """
        # For now, return recent healing events
        # In production, this would query Qdrant vector database
        async with self.db.get_session() as session:
            result = await session.execute(
                select(HealingEventDB)
                .order_by(HealingEventDB.created_at.desc())
                .limit(limit)
            )
            db_events = result.scalars().all()
            
            return [
                {
                    "event_id": event.event_id,
                    "element_id": event.element_id,
                    "confidence_score": event.confidence_score,
                    "healing_strategy": event.healing_strategy,
                    "similarity": 0.85  # Placeholder similarity score
                }
                for event in db_events
            ]
    
    async def record_test_execution(
        self,
        test_id: str,
        elements_tested: List[str],
        execution_id: Optional[str] = None
    ) -> str:
        """Record a test execution."""
        if execution_id is None:
            execution_id = f"exec-{test_id}-{datetime.utcnow().timestamp()}"
        
        async with self.db.get_session() as session:
            db_execution = TestExecutionDB(
                execution_id=execution_id,
                test_id=test_id,
                status="completed",
                started_at=datetime.utcnow(),
                completed_at=datetime.utcnow(),
                elements_tested=elements_tested
            )
            session.add(db_execution)
            
            # Update element stability
            for element_id in elements_tested:
                result = await session.execute(
                    select(ElementStabilityDB).where(
                        ElementStabilityDB.element_id == element_id
                    )
                )
                stability = result.scalar_one_or_none()
                
                if stability is None:
                    # Create new stability record
                    stability = ElementStabilityDB(
                        element_id=element_id,
                        total_executions=1,
                        total_healings=0,
                        stability_score=1.0
                    )
                    session.add(stability)
                else:
                    # Update existing record
                    stability.total_executions += 1
                    stability.last_updated = datetime.utcnow()
                    # Recalculate stability
                    if stability.total_executions > 0:
                        stability.stability_score = (
                            stability.total_executions - stability.total_healings
                        ) / stability.total_executions
        
        return execution_id
    
    async def calculate_element_stability(self, element_id: str) -> float:
        """Calculate stability score for an element."""
        async with self.db.get_session() as session:
            result = await session.execute(
                select(ElementStabilityDB).where(
                    ElementStabilityDB.element_id == element_id
                )
            )
            stability = result.scalar_one_or_none()
            
            if stability is None:
                return 1.0  # Perfect stability if no data
            
            return stability.stability_score
    
    async def get_healing_history(
        self,
        element_id: Optional[str] = None,
        test_id: Optional[str] = None,
        limit: int = 100
    ) -> List[HealingEvent]:
        """Get healing history with optional filters."""
        async with self.db.get_session() as session:
            query = select(HealingEventDB).order_by(HealingEventDB.created_at.desc())
            
            if element_id:
                query = query.where(HealingEventDB.element_id == element_id)
            if test_id:
                query = query.where(HealingEventDB.test_id == test_id)
            
            query = query.limit(limit)
            
            result = await session.execute(query)
            db_events = result.scalars().all()
            
            return [
                HealingEvent(
                    event_id=event.event_id,
                    test_id=event.test_id,
                    execution_id=event.execution_id,
                    element_id=event.element_id,
                    original_locator=event.original_locator,
                    failure_reason=event.failure_reason,
                    healing_strategy=HealingStrategy(event.healing_strategy),
                    new_locator=event.new_locator,
                    confidence_score=event.confidence_score,
                    healing_successful=event.healing_successful,
                    validation_method=ValidationMethod(event.validation_method),
                    timestamp=event.created_at,
                    metadata=event.metadata_json or {}
                )
                for event in db_events
            ]
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get overall statistics."""
        async with self.db.get_session() as session:
            # Count total healing events
            result = await session.execute(select(func.count()).select_from(HealingEventDB))
            total_healings = result.scalar()
            
            # Count successful healings
            result = await session.execute(
                select(func.count()).select_from(HealingEventDB).where(
                    HealingEventDB.healing_successful == True
                )
            )
            successful_healings = result.scalar()
            
            # Count total executions
            result = await session.execute(select(func.count()).select_from(TestExecutionDB))
            total_executions = result.scalar()
            
            # Calculate average stability
            result = await session.execute(select(func.avg(ElementStabilityDB.stability_score)))
            avg_stability = result.scalar() or 1.0
            
            return {
                "total_healings": total_healings,
                "successful_healings": successful_healings,
                "healing_success_rate": successful_healings / total_healings if total_healings > 0 else 0.0,
                "total_executions": total_executions,
                "average_stability": float(avg_stability)
            }
    
    async def close(self):
        """Close database connections."""
        await self.db.close()
