# TestDriver MCP Framework: Low-Level Development System Design Specification

## Part 6: Advanced Self-Healing and Learning Systems

### 6.1 AI Locator Healing Engine Architecture

The AI Locator Healing Engine represents the core intelligence layer that enables TestDriver to automatically adapt to UI changes without manual intervention. This engine maintains a comprehensive memory of element locators across multiple modalities and uses machine learning to select optimal recovery strategies when elements cannot be found.

**Design Philosophy**:

Traditional test automation fails when UI elements change because locators (XPath, CSS selectors, IDs) become stale. The AI Locator Healing Engine solves this by maintaining rich, multi-modal representations of elements that go beyond simple DOM attributes. Each element is represented by visual embeddings capturing its appearance, semantic embeddings capturing its meaning and context, structural information about its position in the DOM hierarchy, and behavioral patterns describing how users interact with it. When a locator fails, the engine searches this multi-dimensional space to find the element that best matches the original intent.

**File**: `src/self_healing/locator_healing_engine.py`

```python
"""
AI Locator Healing Engine Implementation
Automatically repairs broken element locators using multi-modal search
"""

import asyncio
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import structlog
from enum import Enum

logger = structlog.get_logger(__name__)


class HealingStrategy(str, Enum):
    """Available healing strategies."""
    VISUAL_SIMILARITY = "visual_similarity"
    SEMANTIC_SEARCH = "semantic_search"
    STRUCTURAL_ANALYSIS = "structural_analysis"
    BEHAVIORAL_PATTERN = "behavioral_pattern"
    HYBRID = "hybrid"


class HealingConfidence(str, Enum):
    """Confidence levels for healing decisions."""
    HIGH = "high"  # >= 0.90 - Auto-commit
    MEDIUM = "medium"  # 0.80-0.89 - Create PR for review
    LOW = "low"  # 0.70-0.79 - Require manual review
    INSUFFICIENT = "insufficient"  # < 0.70 - Cannot heal automatically


@dataclass
class ElementMemory:
    """
    Multi-modal representation of a UI element stored in memory.
    """
    element_id: str
    page_url_pattern: str
    
    # Visual representation
    visual_embedding: np.ndarray  # CLIP or similar embedding
    reference_screenshot: bytes  # Cropped image of element
    bounding_box_history: List[Dict[str, int]]  # Historical positions
    
    # Semantic representation
    semantic_embedding: np.ndarray  # Text embedding of description
    element_description: str  # Natural language description
    element_role: str  # button, input, link, etc.
    element_labels: List[str]  # Associated text labels
    
    # Structural representation
    dom_path: str  # Simplified DOM path
    dom_attributes: Dict[str, str]  # Key attributes
    parent_context: str  # Description of parent elements
    sibling_context: List[str]  # Descriptions of siblings
    
    # Behavioral representation
    interaction_type: str  # click, type, select, etc.
    typical_user_flow: List[str]  # Common sequences involving this element
    interaction_frequency: int  # How often this element is used
    
    # Locator strategies
    primary_locator: Dict[str, str]  # Current working locator
    fallback_locators: List[Dict[str, str]]  # Alternative locators
    
    # Metadata
    created_at: datetime
    last_successful_use: datetime
    success_count: int
    failure_count: int
    last_healing_timestamp: Optional[datetime] = None


class LocatorMemoryStore:
    """
    Persistent storage for element memories with efficient similarity search.
    """
    
    def __init__(self, db_connection: Any):
        """
        Initialize memory store.
        
        Args:
            db_connection: PostgreSQL connection with pgvector extension
        """
        self.db = db_connection
        self._cache: Dict[str, ElementMemory] = {}
    
    async def store_element(self, memory: ElementMemory) -> None:
        """
        Store or update element memory.
        
        Args:
            memory: ElementMemory to persist
        """
        logger.debug("Storing element memory", element_id=memory.element_id)
        
        # Update cache
        self._cache[memory.element_id] = memory
        
        # Persist to database
        await self.db.execute("""
            INSERT INTO element_memories (
                element_id, page_url_pattern, visual_embedding, reference_screenshot,
                bounding_box_history, semantic_embedding, element_description, element_role,
                element_labels, dom_path, dom_attributes, parent_context, sibling_context,
                interaction_type, typical_user_flow, interaction_frequency,
                primary_locator, fallback_locators, created_at, last_successful_use,
                success_count, failure_count, last_healing_timestamp
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23)
            ON CONFLICT (element_id) DO UPDATE SET
                visual_embedding = EXCLUDED.visual_embedding,
                semantic_embedding = EXCLUDED.semantic_embedding,
                bounding_box_history = EXCLUDED.bounding_box_history,
                primary_locator = EXCLUDED.primary_locator,
                fallback_locators = EXCLUDED.fallback_locators,
                last_successful_use = EXCLUDED.last_successful_use,
                success_count = EXCLUDED.success_count,
                failure_count = EXCLUDED.failure_count,
                last_healing_timestamp = EXCLUDED.last_healing_timestamp
        """, 
        memory.element_id, memory.page_url_pattern, memory.visual_embedding.tolist(),
        memory.reference_screenshot, memory.bounding_box_history, 
        memory.semantic_embedding.tolist(), memory.element_description, memory.element_role,
        memory.element_labels, memory.dom_path, memory.dom_attributes, memory.parent_context,
        memory.sibling_context, memory.interaction_type, memory.typical_user_flow,
        memory.interaction_frequency, memory.primary_locator, memory.fallback_locators,
        memory.created_at, memory.last_successful_use, memory.success_count,
        memory.failure_count, memory.last_healing_timestamp
        )
    
    async def find_similar_elements(
        self,
        visual_embedding: Optional[np.ndarray] = None,
        semantic_embedding: Optional[np.ndarray] = None,
        page_url_pattern: Optional[str] = None,
        top_k: int = 5
    ) -> List[Tuple[ElementMemory, float]]:
        """
        Find elements similar to given embeddings.
        
        Args:
            visual_embedding: Visual embedding to search for
            semantic_embedding: Semantic embedding to search for
            page_url_pattern: Optional page URL filter
            top_k: Number of results to return
            
        Returns:
            List of (ElementMemory, similarity_score) tuples
        """
        if visual_embedding is not None:
            # Visual similarity search using pgvector
            query = """
                SELECT *, 
                    1 - (visual_embedding <=> $1::vector) as similarity
                FROM element_memories
            """
            params = [visual_embedding.tolist()]
            
            if page_url_pattern:
                query += " WHERE page_url_pattern LIKE $2"
                params.append(f"%{page_url_pattern}%")
            
            query += " ORDER BY visual_embedding <=> $1::vector LIMIT $" + str(len(params) + 1)
            params.append(top_k)
            
            rows = await self.db.fetch(query, *params)
            
            results = []
            for row in rows:
                memory = self._row_to_memory(row)
                results.append((memory, row['similarity']))
            
            return results
        
        elif semantic_embedding is not None:
            # Semantic similarity search
            query = """
                SELECT *,
                    1 - (semantic_embedding <=> $1::vector) as similarity
                FROM element_memories
            """
            params = [semantic_embedding.tolist()]
            
            if page_url_pattern:
                query += " WHERE page_url_pattern LIKE $2"
                params.append(f"%{page_url_pattern}%")
            
            query += " ORDER BY semantic_embedding <=> $1::vector LIMIT $" + str(len(params) + 1)
            params.append(top_k)
            
            rows = await self.db.fetch(query, *params)
            
            results = []
            for row in rows:
                memory = self._row_to_memory(row)
                results.append((memory, row['similarity']))
            
            return results
        
        return []
    
    def _row_to_memory(self, row: Dict[str, Any]) -> ElementMemory:
        """Convert database row to ElementMemory object."""
        return ElementMemory(
            element_id=row['element_id'],
            page_url_pattern=row['page_url_pattern'],
            visual_embedding=np.array(row['visual_embedding']),
            reference_screenshot=row['reference_screenshot'],
            bounding_box_history=row['bounding_box_history'],
            semantic_embedding=np.array(row['semantic_embedding']),
            element_description=row['element_description'],
            element_role=row['element_role'],
            element_labels=row['element_labels'],
            dom_path=row['dom_path'],
            dom_attributes=row['dom_attributes'],
            parent_context=row['parent_context'],
            sibling_context=row['sibling_context'],
            interaction_type=row['interaction_type'],
            typical_user_flow=row['typical_user_flow'],
            interaction_frequency=row['interaction_frequency'],
            primary_locator=row['primary_locator'],
            fallback_locators=row['fallback_locators'],
            created_at=row['created_at'],
            last_successful_use=row['last_successful_use'],
            success_count=row['success_count'],
            failure_count=row['failure_count'],
            last_healing_timestamp=row['last_healing_timestamp']
        )


class AILocatorHealingEngine:
    """
    AI-powered locator healing engine that automatically repairs broken element locators.
    """
    
    def __init__(
        self,
        memory_store: LocatorMemoryStore,
        vision_adapter: Any,
        browser_driver: Any,
        config: Dict[str, Any]
    ):
        """
        Initialize healing engine.
        
        Args:
            memory_store: LocatorMemoryStore for element memories
            vision_adapter: VisionAdapter for visual analysis
            browser_driver: BrowserDriver for page interaction
            config: Configuration dictionary
        """
        self.memory_store = memory_store
        self.vision_adapter = vision_adapter
        self.browser_driver = browser_driver
        
        # Confidence thresholds
        self.auto_commit_threshold = config.get('auto_commit_threshold', 0.90)
        self.pr_threshold = config.get('pr_threshold', 0.80)
        self.manual_review_threshold = config.get('manual_review_threshold', 0.70)
        
        # Healing strategy weights
        self.strategy_weights = config.get('strategy_weights', {
            'visual_similarity': 0.4,
            'semantic_search': 0.3,
            'structural_analysis': 0.2,
            'behavioral_pattern': 0.1
        })
    
    async def heal_locator(
        self,
        element_id: str,
        original_locator: Dict[str, str],
        failure_context: Dict[str, Any]
    ) -> Tuple[Optional[Dict[str, str]], float, HealingStrategy]:
        """
        Attempt to heal a broken element locator.
        
        Args:
            element_id: Identifier of the element that failed
            original_locator: The locator that failed
            failure_context: Context about the failure (page URL, screenshot, etc.)
            
        Returns:
            Tuple of (new_locator, confidence_score, strategy_used)
            Returns (None, 0.0, None) if healing fails
        """
        logger.info("Attempting to heal locator", 
                   element_id=element_id,
                   original_locator=original_locator)
        
        # Retrieve element memory
        memory = await self._get_element_memory(element_id)
        if not memory:
            logger.warning("No memory found for element", element_id=element_id)
            return None, 0.0, None
        
        # Capture current page state
        screenshot = await self.browser_driver.screenshot()
        current_url = await self.browser_driver.get_current_url()
        
        # Try multiple healing strategies in parallel
        healing_tasks = [
            self._heal_by_visual_similarity(memory, screenshot),
            self._heal_by_semantic_search(memory, screenshot),
            self._heal_by_structural_analysis(memory),
            self._heal_by_behavioral_pattern(memory, failure_context)
        ]
        
        results = await asyncio.gather(*healing_tasks, return_exceptions=True)
        
        # Aggregate results using weighted voting
        best_locator, best_confidence, best_strategy = await self._aggregate_healing_results(
            results,
            memory
        )
        
        if best_locator and best_confidence >= self.manual_review_threshold:
            logger.info("Locator healed successfully",
                       element_id=element_id,
                       confidence=best_confidence,
                       strategy=best_strategy)
            
            # Update memory with new locator
            await self._update_memory_with_healing(memory, best_locator, best_confidence)
            
            return best_locator, best_confidence, best_strategy
        else:
            logger.warning("Locator healing failed",
                          element_id=element_id,
                          best_confidence=best_confidence)
            return None, 0.0, None
    
    async def _heal_by_visual_similarity(
        self,
        memory: ElementMemory,
        screenshot: bytes
    ) -> Tuple[Optional[Dict[str, str]], float]:
        """
        Heal locator by finding visually similar element on current page.
        """
        logger.debug("Attempting visual similarity healing", element_id=memory.element_id)
        
        # Use vision adapter to find element matching reference screenshot
        try:
            element_location = await self.vision_adapter.find_element(
                screenshot,
                memory.element_description,
                context=f"Looking for {memory.element_role} that looks like the reference image"
            )
            
            if element_location.confidence >= 0.7:
                # Generate new locator based on found location
                new_locator = await self._generate_locator_from_location(element_location)
                return new_locator, element_location.confidence
            
        except Exception as e:
            logger.debug("Visual similarity healing failed", error=str(e))
        
        return None, 0.0
    
    async def _heal_by_semantic_search(
        self,
        memory: ElementMemory,
        screenshot: bytes
    ) -> Tuple[Optional[Dict[str, str]], float]:
        """
        Heal locator by semantic understanding of element purpose.
        """
        logger.debug("Attempting semantic search healing", element_id=memory.element_id)
        
        # Use vision adapter with enhanced semantic context
        try:
            semantic_prompt = f"""
            Find the {memory.element_role} that serves the following purpose:
            {memory.element_description}
            
            It is typically used to: {memory.interaction_type}
            Context: {memory.parent_context}
            Labels: {', '.join(memory.element_labels)}
            """
            
            element_location = await self.vision_adapter.find_element(
                screenshot,
                semantic_prompt
            )
            
            if element_location.confidence >= 0.7:
                new_locator = await self._generate_locator_from_location(element_location)
                return new_locator, element_location.confidence
            
        except Exception as e:
            logger.debug("Semantic search healing failed", error=str(e))
        
        return None, 0.0
    
    async def _heal_by_structural_analysis(
        self,
        memory: ElementMemory
    ) -> Tuple[Optional[Dict[str, str]], float]:
        """
        Heal locator by analyzing DOM structure changes.
        """
        logger.debug("Attempting structural analysis healing", element_id=memory.element_id)
        
        # Get current page DOM
        page_source = await self.browser_driver.get_page_source()
        
        # Try fallback locators first
        for fallback_locator in memory.fallback_locators:
            try:
                # Test if fallback locator works
                script = f"""
                return document.evaluate(
                    "{fallback_locator.get('xpath', '')}",
                    document,
                    null,
                    XPathResult.FIRST_ORDERED_NODE_TYPE,
                    null
                ).singleNodeValue !== null;
                """
                
                if await self.browser_driver.execute_script(script):
                    logger.info("Fallback locator successful", locator=fallback_locator)
                    return fallback_locator, 0.85
                    
            except Exception as e:
                continue
        
        # Analyze DOM structure to find similar patterns
        # (Implementation would use DOM parsing and pattern matching)
        
        return None, 0.0
    
    async def _heal_by_behavioral_pattern(
        self,
        memory: ElementMemory,
        failure_context: Dict[str, Any]
    ) -> Tuple[Optional[Dict[str, str]], float]:
        """
        Heal locator by analyzing typical user interaction patterns.
        """
        logger.debug("Attempting behavioral pattern healing", element_id=memory.element_id)
        
        # Analyze typical user flow to infer element location
        # For example, if element is typically clicked after another element,
        # use that relationship to locate it
        
        # (Implementation would analyze typical_user_flow and interaction patterns)
        
        return None, 0.0
    
    async def _aggregate_healing_results(
        self,
        results: List[Tuple[Optional[Dict[str, str]], float]],
        memory: ElementMemory
    ) -> Tuple[Optional[Dict[str, str]], float, HealingStrategy]:
        """
        Aggregate results from multiple healing strategies using weighted voting.
        """
        strategies = [
            HealingStrategy.VISUAL_SIMILARITY,
            HealingStrategy.SEMANTIC_SEARCH,
            HealingStrategy.STRUCTURAL_ANALYSIS,
            HealingStrategy.BEHAVIORAL_PATTERN
        ]
        
        # Weight each result by strategy weight and confidence
        weighted_results = []
        for i, (locator, confidence) in enumerate(results):
            if isinstance(locator, Exception) or locator is None:
                continue
            
            strategy = strategies[i]
            weight = self.strategy_weights.get(strategy.value, 0.25)
            weighted_confidence = confidence * weight
            
            weighted_results.append((locator, weighted_confidence, strategy))
        
        if not weighted_results:
            return None, 0.0, None
        
        # Select result with highest weighted confidence
        best_result = max(weighted_results, key=lambda x: x[1])
        return best_result
    
    async def _generate_locator_from_location(
        self,
        element_location: Any
    ) -> Dict[str, str]:
        """
        Generate robust locator from element location.
        """
        # Generate multiple locator strategies
        locator = {
            'type': 'coordinates',
            'x': element_location.coordinates[0],
            'y': element_location.coordinates[1],
            'bounding_box': element_location.bounding_box.to_dict()
        }
        
        # Try to generate DOM-based locators as well
        # (Implementation would inspect DOM at coordinates)
        
        return locator
    
    async def _get_element_memory(self, element_id: str) -> Optional[ElementMemory]:
        """Retrieve element memory from store."""
        # Implementation would query memory store
        pass
    
    async def _update_memory_with_healing(
        self,
        memory: ElementMemory,
        new_locator: Dict[str, str],
        confidence: float
    ) -> None:
        """Update element memory with successful healing result."""
        memory.primary_locator = new_locator
        memory.last_healing_timestamp = datetime.utcnow()
        await self.memory_store.store_element(memory)
    
    def get_healing_confidence_level(self, confidence: float) -> HealingConfidence:
        """Determine confidence level for healing decision."""
        if confidence >= self.auto_commit_threshold:
            return HealingConfidence.HIGH
        elif confidence >= self.pr_threshold:
            return HealingConfidence.MEDIUM
        elif confidence >= self.manual_review_threshold:
            return HealingConfidence.LOW
        else:
            return HealingConfidence.INSUFFICIENT
```

This implementation provides the foundation for intelligent, multi-modal element location healing. The next section will cover the Reinforcement Learning Agent that learns optimal healing strategies over time.

### 6.2 Reinforcement Learning Agent for Strategy Optimization

The Reinforcement Learning Agent continuously improves healing strategies by learning from outcomes. It treats locator healing as a sequential decision-making problem where the agent must choose the best healing strategy given the current context.

**File**: `src/self_healing/rl_agent.py`

```python
"""
Reinforcement Learning Agent for Healing Strategy Optimization
"""

import numpy as np
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from collections import deque
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class HealingState:
    """State representation for RL agent."""
    element_type: str
    failure_reason: str
    page_complexity: float
    dom_stability: float
    visual_similarity_to_baseline: float
    previous_healing_attempts: int
    time_since_last_success: float


@dataclass
class HealingAction:
    """Action representation for RL agent."""
    strategy: str
    parameters: Dict[str, Any]


@dataclass
class HealingExperience:
    """Experience tuple for replay buffer."""
    state: HealingState
    action: HealingAction
    reward: float
    next_state: HealingState
    done: bool


class ReinforcementLearningAgent:
    """
    RL agent that learns optimal healing strategies through experience.
    
    Uses Q-learning with experience replay to optimize strategy selection.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize RL agent."""
        self.learning_rate = config.get('learning_rate', 0.001)
        self.discount_factor = config.get('discount_factor', 0.95)
        self.epsilon = config.get('epsilon', 0.1)  # Exploration rate
        self.epsilon_decay = config.get('epsilon_decay', 0.995)
        self.epsilon_min = config.get('epsilon_min', 0.01)
        
        # Experience replay buffer
        self.replay_buffer = deque(maxlen=config.get('buffer_size', 10000))
        
        # Q-table or neural network for Q-values
        self.q_network = self._initialize_q_network(config)
    
    def select_strategy(
        self,
        state: HealingState,
        available_strategies: List[str]
    ) -> HealingAction:
        """
        Select best healing strategy for given state.
        
        Uses epsilon-greedy policy for exploration/exploitation balance.
        """
        if np.random.random() < self.epsilon:
            # Explore: random strategy
            strategy = np.random.choice(available_strategies)
            logger.debug("Exploring random strategy", strategy=strategy)
        else:
            # Exploit: best known strategy
            q_values = self._compute_q_values(state, available_strategies)
            strategy = available_strategies[np.argmax(q_values)]
            logger.debug("Exploiting best strategy", strategy=strategy, q_value=max(q_values))
        
        return HealingAction(strategy=strategy, parameters={})
    
    def learn_from_experience(
        self,
        state: HealingState,
        action: HealingAction,
        reward: float,
        next_state: HealingState,
        done: bool
    ) -> None:
        """
        Learn from healing attempt outcome.
        
        Args:
            state: Initial state
            action: Action taken
            reward: Reward received (1.0 for success, -1.0 for failure, 0.0 for partial)
            next_state: Resulting state
            done: Whether episode is complete
        """
        # Store experience
        experience = HealingExperience(state, action, reward, next_state, done)
        self.replay_buffer.append(experience)
        
        # Train on batch from replay buffer
        if len(self.replay_buffer) >= 32:
            self._train_on_batch()
        
        # Decay exploration rate
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def _compute_q_values(
        self,
        state: HealingState,
        strategies: List[str]
    ) -> np.ndarray:
        """Compute Q-values for all strategies in given state."""
        # Implementation would use Q-network to compute values
        # For now, return uniform values
        return np.ones(len(strategies))
    
    def _train_on_batch(self) -> None:
        """Train Q-network on batch of experiences."""
        # Sample batch from replay buffer
        batch_size = 32
        batch = np.random.choice(len(self.replay_buffer), batch_size, replace=False)
        experiences = [self.replay_buffer[i] for i in batch]
        
        # Compute Q-learning updates
        # (Implementation would update Q-network weights)
        pass
    
    def _initialize_q_network(self, config: Dict[str, Any]) -> Any:
        """Initialize Q-network architecture."""
        # Could use simple Q-table or deep Q-network
        # Implementation depends on state/action space size
        pass
```

This completes the self-healing and adaptive learning components specification.
