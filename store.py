"""Test memory store for healing history and learned patterns."""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import structlog

from ..models import HealingEvent, ElementStability

logger = structlog.get_logger()


class TestMemoryStore:
    """
    Persistent storage for test healing history and learned patterns.
    
    Stores healing events with visual and semantic embeddings,
    enabling similarity search for future healing attempts.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.storage_type = config.get("storage_type", "memory")
        
        # In-memory storage for prototype
        self._healing_events: Dict[str, HealingEvent] = {}
        self._element_stability: Dict[str, ElementStability] = {}
        self._test_executions: List[Dict[str, Any]] = []
        
        logger.info("Test Memory Store initialized", storage=self.storage_type)
    
    async def store_healing_event(self, event: HealingEvent) -> None:
        """
        Store healing event with embeddings.
        
        Args:
            event: Healing event to store
        """
        try:
            self._healing_events[event.event_id] = event
            
            # Update element stability
            await self._update_element_stability(event)
            
            logger.info(
                "Healing event stored",
                event_id=event.event_id,
                confidence=event.confidence_score
            )
            
        except Exception as e:
            logger.error("Error storing healing event", error=str(e))
            raise
    
    async def get_healing_event(self, event_id: str) -> Optional[HealingEvent]:
        """Get healing event by ID."""
        return self._healing_events.get(event_id)
    
    async def find_similar_healing_events(
        self,
        visual_embedding: List[float],
        limit: int = 10,
        min_confidence: float = 0.7
    ) -> List[Tuple[HealingEvent, float]]:
        """
        Find similar healing events based on visual embedding.
        
        Args:
            visual_embedding: Visual embedding to search for
            limit: Maximum number of results
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of (HealingEvent, similarity_score) tuples
        """
        try:
            results = []
            
            for event in self._healing_events.values():
                if not event.visual_embedding:
                    continue
                
                if not event.healing_successful:
                    continue
                
                if event.confidence_score < min_confidence:
                    continue
                
                # Calculate cosine similarity
                similarity = self._cosine_similarity(
                    visual_embedding,
                    event.visual_embedding
                )
                
                results.append((event, similarity))
            
            # Sort by similarity descending
            results.sort(key=lambda x: x[1], reverse=True)
            
            return results[:limit]
            
        except Exception as e:
            logger.error("Error finding similar events", error=str(e))
            return []
    
    async def calculate_element_stability(self, element_id: str) -> float:
        """
        Calculate stability score for an element.
        
        Args:
            element_id: Element identifier
            
        Returns:
            Stability score (0.0 to 1.0)
        """
        stability = self._element_stability.get(element_id)
        
        if not stability:
            return 1.0  # Unknown elements are considered stable
        
        return stability.stability_score
    
    async def get_stable_elements(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get most stable elements."""
        stable_elements = sorted(
            self._element_stability.values(),
            key=lambda x: x.stability_score,
            reverse=True
        )
        
        return [
            {
                "element_id": elem.element_id,
                "stability_score": elem.stability_score,
                "total_executions": elem.total_executions,
                "healing_count": elem.healing_count
            }
            for elem in stable_elements[:limit]
        ]
    
    async def record_test_execution(
        self,
        test_id: str,
        elements_tested: List[str]
    ) -> None:
        """Record test execution for stability tracking."""
        execution = {
            "test_id": test_id,
            "timestamp": datetime.utcnow(),
            "elements_tested": elements_tested
        }
        
        self._test_executions.append(execution)
        
        # Update element stability
        for element_id in elements_tested:
            if element_id not in self._element_stability:
                self._element_stability[element_id] = ElementStability(
                    element_id=element_id,
                    total_executions=0,
                    successful_executions=0,
                    healing_count=0,
                    stability_score=1.0,
                    trend="stable"
                )
            
            stability = self._element_stability[element_id]
            stability.total_executions += 1
            stability.successful_executions += 1
            stability.last_updated = datetime.utcnow()
            
            # Recalculate stability score
            if stability.total_executions > 0:
                stability.stability_score = (
                    1.0 - (stability.healing_count / stability.total_executions)
                )
    
    async def update_healing_feedback(
        self,
        event_id: str,
        feedback: Dict[str, Any]
    ) -> None:
        """Update healing event with user feedback."""
        event = self._healing_events.get(event_id)
        
        if event:
            event.user_feedback = feedback.get("comment")
            
            # Update success status based on feedback
            if "correct" in feedback:
                event.healing_successful = feedback["correct"]
            
            logger.info("Healing feedback updated", event_id=event_id)
    
    async def count_healing_events(self) -> int:
        """Count total healing events."""
        return len(self._healing_events)
    
    async def _update_element_stability(self, event: HealingEvent) -> None:
        """Update element stability based on healing event."""
        element_id = event.element_id
        
        if element_id not in self._element_stability:
            self._element_stability[element_id] = ElementStability(
                element_id=element_id,
                total_executions=0,
                successful_executions=0,
                healing_count=0,
                stability_score=1.0,
                trend="stable"
            )
        
        stability = self._element_stability[element_id]
        stability.healing_count += 1
        stability.last_updated = datetime.utcnow()
        
        # Recalculate stability score
        if stability.total_executions > 0:
            stability.stability_score = (
                1.0 - (stability.healing_count / stability.total_executions)
            )
        
        # Determine trend
        if stability.healing_count > stability.total_executions * 0.3:
            stability.trend = "degrading"
        elif stability.healing_count > stability.total_executions * 0.1:
            stability.trend = "stable"
        else:
            stability.trend = "improving"
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if len(vec1) != len(vec2):
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = sum(a * a for a in vec1) ** 0.5
        magnitude2 = sum(b * b for b in vec2) ** 0.5
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
