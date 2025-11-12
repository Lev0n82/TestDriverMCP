"""
Healing strategies for automatically fixing broken test locators.
Implements visual similarity, semantic matching, structural analysis, and behavioral pattern strategies.
"""

from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod
import structlog

from models import HealingStrategy
from vision.adapters import VisionAdapter
from execution.framework import BrowserDriver

logger = structlog.get_logger()

class HealingStrategyBase(ABC):
    """Base class for healing strategies."""
    
    @abstractmethod
    async def heal(
        self,
        driver: BrowserDriver,
        vision_adapter: VisionAdapter,
        original_locator: Dict[str, str],
        element_description: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Attempt to heal broken locator.
        
        Args:
            driver: Browser driver
            vision_adapter: Vision adapter for AI analysis
            original_locator: Original (broken) locator
            element_description: Description of the element
            context: Additional context
        
        Returns:
            Healing result with new locator and confidence
        """
        pass

class VisualSimilarityStrategy(HealingStrategyBase):
    """
    Visual similarity healing strategy.
    Uses AI vision to find elements that look similar to the target.
    """
    
    async def heal(
        self,
        driver: BrowserDriver,
        vision_adapter: VisionAdapter,
        original_locator: Dict[str, str],
        element_description: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Heal using visual similarity."""
        logger.info("Attempting visual similarity healing", element=element_description)
        
        try:
            # Take screenshot of current page
            screenshot = await driver.take_screenshot(full_page=False)
            
            # Build context for vision API
            vision_context = {
                "previous_locator": original_locator,
                "failure_reason": context.get("failure_reason", "Element not found") if context else "Element not found"
            }
            
            # Use AI vision to detect element
            detection_result = await vision_adapter.detect_element(
                screenshot=screenshot,
                element_description=element_description,
                context=vision_context
            )
            
            if detection_result.get("success"):
                new_locator = detection_result["locator"]
                confidence = detection_result.get("confidence", 0.0)
                
                # Verify the new locator works
                is_visible = await driver.is_visible(new_locator)
                
                if is_visible:
                    logger.info(
                        "Visual similarity healing successful",
                        new_locator=new_locator,
                        confidence=confidence
                    )
                    return {
                        "success": True,
                        "strategy": HealingStrategy.VISUAL_SIMILARITY,
                        "new_locator": new_locator,
                        "confidence": confidence,
                        "explanation": detection_result.get("explanation", "AI vision detected element"),
                        "alternatives": detection_result.get("alternatives", [])
                    }
                else:
                    logger.warning("New locator found but element not visible", locator=new_locator)
                    return {
                        "success": False,
                        "strategy": HealingStrategy.VISUAL_SIMILARITY,
                        "confidence": 0.0,
                        "error": "New locator found but element not visible"
                    }
            else:
                logger.warning("Visual similarity detection failed", error=detection_result.get("error"))
                return {
                    "success": False,
                    "strategy": HealingStrategy.VISUAL_SIMILARITY,
                    "confidence": 0.0,
                    "error": detection_result.get("error", "Detection failed")
                }
                
        except Exception as e:
            logger.error("Visual similarity healing error", error=str(e))
            return {
                "success": False,
                "strategy": HealingStrategy.VISUAL_SIMILARITY,
                "confidence": 0.0,
                "error": str(e)
            }

class SemanticMatchingStrategy(HealingStrategyBase):
    """
    Semantic matching healing strategy.
    Finds elements by text content, aria-label, title, etc.
    """
    
    async def heal(
        self,
        driver: BrowserDriver,
        vision_adapter: VisionAdapter,
        original_locator: Dict[str, str],
        element_description: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Heal using semantic matching."""
        logger.info("Attempting semantic matching healing", element=element_description)
        
        try:
            # Extract keywords from element description
            keywords = self._extract_keywords(element_description)
            
            # Try different semantic locators
            semantic_locators = []
            
            # Try text matching
            for keyword in keywords:
                semantic_locators.append({"text": keyword})
                semantic_locators.append({"label": keyword})
                semantic_locators.append({"placeholder": keyword})
            
            # Try role-based matching
            if "button" in element_description.lower():
                semantic_locators.append({"role": "button"})
            elif "link" in element_description.lower():
                semantic_locators.append({"role": "link"})
            elif "input" in element_description.lower():
                semantic_locators.append({"role": "textbox"})
            
            # Test each locator
            for locator in semantic_locators:
                is_visible = await driver.is_visible(locator)
                if is_visible:
                    logger.info("Semantic matching successful", new_locator=locator)
                    return {
                        "success": True,
                        "strategy": HealingStrategy.SEMANTIC_SIMILARITY,
                        "new_locator": locator,
                        "confidence": 0.75,  # Moderate confidence for semantic matching
                        "explanation": f"Found element using semantic locator: {locator}"
                    }
            
            # No working locator found
            return {
                "success": False,
                "strategy": HealingStrategy.SEMANTIC_SIMILARITY,
                "confidence": 0.0,
                "error": "No semantic locator found"
            }
            
        except Exception as e:
            logger.error("Semantic matching healing error", error=str(e))
            return {
                "success": False,
                "strategy": HealingStrategy.SEMANTIC_SIMILARITY,
                "confidence": 0.0,
                "error": str(e)
            }
    
    def _extract_keywords(self, description: str) -> List[str]:
        """Extract keywords from element description."""
        # Simple keyword extraction
        # In production, use NLP for better extraction
        words = description.lower().split()
        
        # Filter out common words
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for"}
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        
        return keywords[:5]  # Return top 5 keywords

class StructuralAnalysisStrategy(HealingStrategyBase):
    """
    Structural analysis healing strategy.
    Analyzes DOM tree structure and relationships.
    """
    
    async def heal(
        self,
        driver: BrowserDriver,
        vision_adapter: VisionAdapter,
        original_locator: Dict[str, str],
        element_description: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Heal using structural analysis."""
        logger.info("Attempting structural analysis healing", element=element_description)
        
        try:
            # Get original locator parts
            if "css" in original_locator:
                original_selector = original_locator["css"]
                
                # Try variations of the selector
                variations = self._generate_selector_variations(original_selector)
                
                for variation in variations:
                    is_visible = await driver.is_visible({"css": variation})
                    if is_visible:
                        logger.info("Structural analysis successful", new_selector=variation)
                        return {
                            "success": True,
                            "strategy": HealingStrategy.STRUCTURAL_SIMILARITY,
                            "new_locator": {"css": variation},
                            "confidence": 0.70,
                            "explanation": f"Found element using selector variation: {variation}"
                        }
            
            return {
                "success": False,
                "strategy": HealingStrategy.STRUCTURAL_SIMILARITY,
                "confidence": 0.0,
                "error": "No structural variation found"
            }
            
        except Exception as e:
            logger.error("Structural analysis healing error", error=str(e))
            return {
                "success": False,
                "strategy": HealingStrategy.STRUCTURAL_SIMILARITY,
                "confidence": 0.0,
                "error": str(e)
            }
    
    def _generate_selector_variations(self, selector: str) -> List[str]:
        """Generate variations of CSS selector."""
        variations = []
        
        # Remove nth-child selectors
        import re
        variations.append(re.sub(r':nth-child\(\d+\)', '', selector))
        
        # Try without IDs (in case ID changed)
        variations.append(re.sub(r'#[\w-]+', '', selector))
        
        # Try with just classes
        class_match = re.findall(r'\.[\w-]+', selector)
        if class_match:
            variations.append(''.join(class_match))
        
        # Try with just tag name
        tag_match = re.match(r'^(\w+)', selector)
        if tag_match:
            variations.append(tag_match.group(1))
        
        return [v for v in variations if v and v != selector]

class BehavioralPatternStrategy(HealingStrategyBase):
    """
    Behavioral pattern healing strategy.
    Matches elements by behavior (clickable, editable, etc.).
    """
    
    async def heal(
        self,
        driver: BrowserDriver,
        vision_adapter: VisionAdapter,
        original_locator: Dict[str, str],
        element_description: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Heal using behavioral pattern matching."""
        logger.info("Attempting behavioral pattern healing", element=element_description)
        
        try:
            # Determine expected behavior from description
            expected_behavior = self._infer_behavior(element_description)
            
            # Build locator based on behavior
            if expected_behavior == "clickable":
                # Try common clickable elements
                clickable_locators = [
                    {"css": "button"},
                    {"css": "a"},
                    {"css": "[role='button']"},
                    {"css": "[onclick]"}
                ]
                
                for locator in clickable_locators:
                    is_visible = await driver.is_visible(locator)
                    if is_visible:
                        return {
                            "success": True,
                            "strategy": HealingStrategy.BEHAVIORAL_SIMILARITY,
                            "new_locator": locator,
                            "confidence": 0.65,
                            "explanation": f"Found clickable element: {locator}"
                        }
            
            elif expected_behavior == "editable":
                # Try common input elements
                input_locators = [
                    {"css": "input"},
                    {"css": "textarea"},
                    {"css": "[contenteditable='true']"}
                ]
                
                for locator in input_locators:
                    is_visible = await driver.is_visible(locator)
                    if is_visible:
                        return {
                            "success": True,
                            "strategy": HealingStrategy.BEHAVIORAL_SIMILARITY,
                            "new_locator": locator,
                            "confidence": 0.65,
                            "explanation": f"Found editable element: {locator}"
                        }
            
            return {
                "success": False,
                "strategy": HealingStrategy.BEHAVIORAL_SIMILARITY,
                "confidence": 0.0,
                "error": "No behavioral pattern match found"
            }
            
        except Exception as e:
            logger.error("Behavioral pattern healing error", error=str(e))
            return {
                "success": False,
                "strategy": HealingStrategy.BEHAVIORAL_SIMILARITY,
                "confidence": 0.0,
                "error": str(e)
            }
    
    def _infer_behavior(self, description: str) -> str:
        """Infer expected behavior from element description."""
        description_lower = description.lower()
        
        if any(word in description_lower for word in ["button", "click", "submit"]):
            return "clickable"
        elif any(word in description_lower for word in ["input", "field", "text", "enter"]):
            return "editable"
        elif any(word in description_lower for word in ["link", "navigate"]):
            return "clickable"
        else:
            return "unknown"

# Strategy registry
HEALING_STRATEGIES = {
    HealingStrategy.VISUAL_SIMILARITY: VisualSimilarityStrategy(),
    HealingStrategy.SEMANTIC_SIMILARITY: SemanticMatchingStrategy(),
    HealingStrategy.STRUCTURAL_SIMILARITY: StructuralAnalysisStrategy(),
    HealingStrategy.BEHAVIORAL_SIMILARITY: BehavioralPatternStrategy(),
}

async def execute_healing_strategy(
    strategy: HealingStrategy,
    driver: BrowserDriver,
    vision_adapter: VisionAdapter,
    original_locator: Dict[str, str],
    element_description: str,
    context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Execute a specific healing strategy.
    
    Args:
        strategy: Healing strategy to use
        driver: Browser driver
        vision_adapter: Vision adapter
        original_locator: Original broken locator
        element_description: Element description
        context: Additional context
    
    Returns:
        Healing result
    """
    strategy_impl = HEALING_STRATEGIES.get(strategy)
    if strategy_impl is None:
        return {
            "success": False,
            "confidence": 0.0,
            "error": f"Unknown strategy: {strategy}"
        }
    
    return await strategy_impl.heal(
        driver=driver,
        vision_adapter=vision_adapter,
        original_locator=original_locator,
        element_description=element_description,
        context=context
    )
