"""Self-healing engine for autonomous test maintenance."""

import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import structlog
from PIL import Image

from ..models import HealingEvent, HealingStrategy, ValidationMethod
from ..vision.adapters import VisionAdapter

logger = structlog.get_logger()


class AILocatorHealingEngine:
    """
    AI-powered locator healing engine.
    
    Automatically heals broken locators using vision-based element detection,
    semantic similarity, and historical healing patterns.
    """
    
    def __init__(
        self,
        vision_adapter: VisionAdapter,
        memory_store: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        self.vision_adapter = vision_adapter
        self.memory_store = memory_store
        self.config = config or {}
        
        self.confidence_thresholds = {
            "auto_commit": self.config.get("auto_commit_threshold", 0.9),
            "pr_review": self.config.get("pr_review_threshold", 0.8),
            "manual_review": self.config.get("manual_review_threshold", 0.7)
        }
        
        logger.info("AI Locator Healing Engine initialized", thresholds=self.confidence_thresholds)
    
    async def heal_locator(
        self,
        original_locator: Dict[str, Any],
        screenshot: Image.Image,
        element_description: str,
        test_id: str = "unknown",
        execution_id: str = "unknown",
        context: Optional[Dict[str, Any]] = None
    ) -> HealingEvent:
        """
        Attempt to heal a broken locator.
        
        Args:
            original_locator: Original broken locator
            screenshot: Current page screenshot
            element_description: Natural language description of element
            test_id: Test ID
            execution_id: Execution ID
            context: Additional context
            
        Returns:
            HealingEvent with healing result
        """
        start_time = time.time()
        event_id = f"heal-{uuid.uuid4().hex[:8]}"
        
        logger.info(
            "Attempting locator healing",
            event_id=event_id,
            element=element_description
        )
        
        try:
            # Strategy 1: Check memory store for similar healing
            if self.memory_store:
                memory_result = await self._search_memory(
                    element_description,
                    screenshot,
                    context
                )
                
                if memory_result and memory_result["confidence"] >= 0.85:
                    logger.info(
                        "Healing found in memory",
                        confidence=memory_result["confidence"]
                    )
                    
                    return self._create_healing_event(
                        event_id=event_id,
                        test_id=test_id,
                        execution_id=execution_id,
                        element_description=element_description,
                        original_locator=original_locator,
                        new_locator=memory_result["locator"],
                        confidence=memory_result["confidence"],
                        strategy=HealingStrategy.MEMORY_LOOKUP,
                        screenshot=screenshot,
                        healing_time=time.time() - start_time
                    )
            
            # Strategy 2: Visual similarity search
            vision_result = await self._search_by_visual_similarity(
                element_description,
                screenshot,
                context
            )
            
            if vision_result and vision_result["confidence"] >= self.confidence_thresholds["manual_review"]:
                logger.info(
                    "Healing found via vision",
                    confidence=vision_result["confidence"]
                )
                
                healing_event = self._create_healing_event(
                    event_id=event_id,
                    test_id=test_id,
                    execution_id=execution_id,
                    element_description=element_description,
                    original_locator=original_locator,
                    new_locator=vision_result["locator"],
                    confidence=vision_result["confidence"],
                    strategy=HealingStrategy.VISUAL_SIMILARITY,
                    screenshot=screenshot,
                    healing_time=time.time() - start_time
                )
                
                # Store in memory for future use
                if self.memory_store:
                    await self.memory_store.store_healing_event(healing_event)
                
                return healing_event
            
            # No healing found
            logger.warning("Healing failed", element=element_description)
            
            return self._create_healing_event(
                event_id=event_id,
                test_id=test_id,
                execution_id=execution_id,
                element_description=element_description,
                original_locator=original_locator,
                new_locator=original_locator,
                confidence=0.0,
                strategy=HealingStrategy.VISUAL_SIMILARITY,
                screenshot=screenshot,
                healing_time=time.time() - start_time,
                successful=False
            )
            
        except Exception as e:
            logger.error("Healing error", error=str(e))
            
            return self._create_healing_event(
                event_id=event_id,
                test_id=test_id,
                execution_id=execution_id,
                element_description=element_description,
                original_locator=original_locator,
                new_locator=original_locator,
                confidence=0.0,
                strategy=HealingStrategy.VISUAL_SIMILARITY,
                screenshot=screenshot,
                healing_time=time.time() - start_time,
                successful=False,
                error=str(e)
            )
    
    async def _search_memory(
        self,
        element_description: str,
        screenshot: Image.Image,
        context: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Search memory store for similar healing events."""
        if not self.memory_store:
            return None
        
        try:
            # Generate embedding for current screenshot
            embedding = await self.vision_adapter.generate_embedding(screenshot)
            
            # Search for similar healings
            similar_events = await self.memory_store.find_similar_healing_events(
                visual_embedding=embedding,
                limit=5
            )
            
            if not similar_events:
                return None
            
            # Return best match
            best_match, similarity_score = similar_events[0]
            
            if best_match.healing_successful:
                return {
                    "locator": best_match.new_locator,
                    "confidence": similarity_score * 0.95,  # Slightly reduce confidence
                    "source": "memory"
                }
            
            return None
            
        except Exception as e:
            logger.error("Memory search error", error=str(e))
            return None
    
    async def _search_by_visual_similarity(
        self,
        element_description: str,
        screenshot: Image.Image,
        context: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Search for element using vision model."""
        try:
            result = await self.vision_adapter.locate_element(
                screenshot=screenshot,
                description=element_description,
                context=context
            )
            
            if not result.get("found"):
                return None
            
            # Convert percentage coordinates to CSS selector
            coords = result.get("coordinates", {})
            suggested_selectors = result.get("suggested_selectors", [])
            
            # Use first suggested selector if available
            if suggested_selectors:
                locator = {"css": suggested_selectors[0]}
            else:
                # Fallback to coordinate-based locator
                locator = {
                    "coordinates": coords,
                    "description": element_description
                }
            
            return {
                "locator": locator,
                "confidence": result.get("confidence", 0.0),
                "source": "vision"
            }
            
        except Exception as e:
            logger.error("Vision search error", error=str(e))
            return None
    
    def _create_healing_event(
        self,
        event_id: str,
        test_id: str,
        execution_id: str,
        element_description: str,
        original_locator: Dict[str, Any],
        new_locator: Dict[str, Any],
        confidence: float,
        strategy: HealingStrategy,
        screenshot: Image.Image,
        healing_time: float,
        successful: bool = True,
        error: Optional[str] = None
    ) -> HealingEvent:
        """Create healing event record."""
        # Generate embeddings
        visual_embedding = None
        try:
            import asyncio
            visual_embedding = asyncio.run(
                self.vision_adapter.generate_embedding(screenshot)
            )
        except:
            pass
        
        # Determine validation method based on confidence
        if confidence >= self.confidence_thresholds["auto_commit"]:
            validation_method = ValidationMethod.AUTO
        elif confidence >= self.confidence_thresholds["pr_review"]:
            validation_method = ValidationMethod.PEER_REVIEW
        else:
            validation_method = ValidationMethod.MANUAL
        
        return HealingEvent(
            event_id=event_id,
            test_id=test_id,
            execution_id=execution_id,
            element_id=element_description,
            timestamp=datetime.utcnow(),
            original_locator=original_locator,
            failure_reason=error or "Element not found",
            failure_screenshot=None,  # Would save to storage
            healing_strategy=strategy,
            new_locator=new_locator,
            confidence_score=confidence,
            healing_successful=successful,
            validation_method=validation_method,
            user_feedback=None,
            visual_embedding=visual_embedding,
            semantic_embedding=None,  # Would use semantic model
            context_features={
                "healing_time_ms": healing_time * 1000,
                "error": error
            }
        )
    
    async def incorporate_feedback(
        self,
        event_id: str,
        feedback: Dict[str, Any]
    ) -> None:
        """
        Incorporate user feedback on healing result.
        
        Args:
            event_id: Healing event ID
            feedback: User feedback (correct/incorrect, suggested fix, etc.)
        """
        logger.info("Incorporating feedback", event_id=event_id, feedback=feedback)
        
        if self.memory_store:
            await self.memory_store.update_healing_feedback(event_id, feedback)
        
        # In production, this would trigger model retraining
    
    def get_validation_recommendation(self, confidence: float) -> str:
        """Get validation recommendation based on confidence."""
        if confidence >= self.confidence_thresholds["auto_commit"]:
            return "auto_commit"
        elif confidence >= self.confidence_thresholds["pr_review"]:
            return "pr_review"
        else:
            return "manual_review"
