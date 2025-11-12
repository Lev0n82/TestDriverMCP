# TestDriver MCP Framework: Low-Level Development System Design Specification

## Part 3: Reliability and Self-Healing Layer Specifications

### 3.1 State Synchronization Store Implementation

The State Synchronization Store maintains a comprehensive, versioned record of application state throughout test execution to enable graceful recovery and deterministic replay.

**File**: `src/reliability/state_store.py`

```python
"""
State Synchronization Store Implementation
Provides versioned state management with time-travel capabilities
"""

import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import hashlib
import json
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class StateSnapshot:
    """
    Immutable state snapshot capturing complete test context at a point in time.
    """
    snapshot_id: str
    test_id: str
    step_id: str
    timestamp: datetime
    
    # Browser state
    url: str
    page_title: str
    dom_hash: str
    viewport: Dict[str, int]  # {width, height}
    scroll_position: Dict[str, int]  # {x, y}
    
    # Application state
    cookies: List[Dict[str, Any]]
    local_storage: Dict[str, str]
    session_storage: Dict[str, str]
    
    # Network state
    pending_requests: List[str]
    recent_responses: List[Dict[str, Any]]
    
    # Console state
    console_logs: List[Dict[str, Any]]
    js_errors: List[Dict[str, Any]]
    
    # Test execution state
    executed_steps: List[str]
    test_data: Dict[str, Any]
    
    # Metadata
    screenshot_path: Optional[str] = None
    video_segment_path: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StateSnapshot':
        """Create from dictionary."""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)
    
    def compute_hash(self) -> str:
        """Compute hash of snapshot for deduplication."""
        # Hash key state components
        hash_data = {
            'url': self.url,
            'dom_hash': self.dom_hash,
            'cookies': self.cookies,
            'local_storage': self.local_storage
        }
        hash_str = json.dumps(hash_data, sort_keys=True)
        return hashlib.sha256(hash_str.encode()).hexdigest()


class StateStore:
    """
    State Synchronization Store with time-series backend.
    
    Responsibilities:
    - Capture and persist state snapshots
    - Provide time-travel query capabilities
    - Support state restoration for recovery
    - Manage retention and cleanup
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize state store.
        
        Args:
            config: Configuration including backend type and connection details
        """
        self.config = config
        self.backend_type = config.get('backend', 'influxdb')
        self.retention_days = config.get('retention_days', 30)
        self.compression_enabled = config.get('compression_enabled', True)
        
        # Backend client (initialized in start())
        self.client = None
        
        # In-memory cache for recent snapshots
        self._cache: Dict[str, StateSnapshot] = {}
        self._cache_size = config.get('cache_size', 1000)
    
    async def start(self) -> None:
        """Initialize backend connection."""
        logger.info("Starting State Store", backend=self.backend_type)
        
        if self.backend_type == 'influxdb':
            await self._init_influxdb()
        elif self.backend_type == 'timescaledb':
            await self._init_timescaledb()
        elif self.backend_type == 'postgresql':
            await self._init_postgresql()
        else:
            raise ValueError(f"Unsupported backend: {self.backend_type}")
        
        # Start background cleanup task
        asyncio.create_task(self._cleanup_task())
        
        logger.info("State Store started successfully")
    
    async def stop(self) -> None:
        """Shutdown backend connection."""
        if self.client:
            await self.client.close()
        logger.info("State Store stopped")
    
    async def _init_influxdb(self) -> None:
        """Initialize InfluxDB backend."""
        from influxdb_client.client.influxdb_client_async import InfluxDBClientAsync
        
        url = self.config.get('connection_url')
        token = self.config.get('token')
        org = self.config.get('org', 'testdriver')
        bucket = self.config.get('bucket', 'state_snapshots')
        
        self.client = InfluxDBClientAsync(url=url, token=token, org=org)
        self.write_api = self.client.write_api()
        self.query_api = self.client.query_api()
        self.bucket = bucket
        
        # Verify connection
        health = await self.client.health()
        if health.status != "pass":
            raise ConnectionError("InfluxDB health check failed")
    
    async def _init_timescaledb(self) -> None:
        """Initialize TimescaleDB backend."""
        import asyncpg
        
        self.client = await asyncpg.create_pool(
            self.config.get('connection_url'),
            min_size=2,
            max_size=10
        )
        
        # Create hypertable if not exists
        async with self.client.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS state_snapshots (
                    snapshot_id TEXT PRIMARY KEY,
                    test_id TEXT NOT NULL,
                    step_id TEXT NOT NULL,
                    timestamp TIMESTAMPTZ NOT NULL,
                    data JSONB NOT NULL,
                    hash TEXT NOT NULL
                );
                
                SELECT create_hypertable('state_snapshots', 'timestamp', 
                    if_not_exists => TRUE);
                
                CREATE INDEX IF NOT EXISTS idx_test_id ON state_snapshots(test_id);
                CREATE INDEX IF NOT EXISTS idx_timestamp ON state_snapshots(timestamp DESC);
            """)
    
    async def _init_postgresql(self) -> None:
        """Initialize PostgreSQL backend (similar to TimescaleDB but without hypertable)."""
        import asyncpg
        
        self.client = await asyncpg.create_pool(
            self.config.get('connection_url'),
            min_size=2,
            max_size=10
        )
        
        async with self.client.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS state_snapshots (
                    snapshot_id TEXT PRIMARY KEY,
                    test_id TEXT NOT NULL,
                    step_id TEXT NOT NULL,
                    timestamp TIMESTAMPTZ NOT NULL,
                    data JSONB NOT NULL,
                    hash TEXT NOT NULL
                );
                
                CREATE INDEX IF NOT EXISTS idx_test_id ON state_snapshots(test_id);
                CREATE INDEX IF NOT EXISTS idx_timestamp ON state_snapshots(timestamp DESC);
            """)
    
    async def save_snapshot(self, snapshot: StateSnapshot) -> None:
        """
        Save a state snapshot.
        
        Args:
            snapshot: StateSnapshot to persist
        """
        logger.debug("Saving state snapshot", 
                    snapshot_id=snapshot.snapshot_id,
                    test_id=snapshot.test_id)
        
        # Add to cache
        self._cache[snapshot.snapshot_id] = snapshot
        if len(self._cache) > self._cache_size:
            # Remove oldest
            oldest_key = min(self._cache.keys(), 
                           key=lambda k: self._cache[k].timestamp)
            del self._cache[oldest_key]
        
        # Persist to backend
        if self.backend_type == 'influxdb':
            await self._save_snapshot_influxdb(snapshot)
        else:  # timescaledb or postgresql
            await self._save_snapshot_sql(snapshot)
    
    async def _save_snapshot_influxdb(self, snapshot: StateSnapshot) -> None:
        """Save snapshot to InfluxDB."""
        from influxdb_client import Point
        
        point = Point("state_snapshot") \
            .tag("test_id", snapshot.test_id) \
            .tag("step_id", snapshot.step_id) \
            .field("url", snapshot.url) \
            .field("page_title", snapshot.page_title) \
            .field("dom_hash", snapshot.dom_hash) \
            .field("data", json.dumps(snapshot.to_dict())) \
            .time(snapshot.timestamp)
        
        await self.write_api.write(bucket=self.bucket, record=point)
    
    async def _save_snapshot_sql(self, snapshot: StateSnapshot) -> None:
        """Save snapshot to SQL backend."""
        async with self.client.acquire() as conn:
            await conn.execute("""
                INSERT INTO state_snapshots 
                (snapshot_id, test_id, step_id, timestamp, data, hash)
                VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT (snapshot_id) DO UPDATE
                SET data = EXCLUDED.data
            """, 
            snapshot.snapshot_id,
            snapshot.test_id,
            snapshot.step_id,
            snapshot.timestamp,
            json.dumps(snapshot.to_dict()),
            snapshot.compute_hash()
            )
    
    async def get_snapshot(self, snapshot_id: str) -> Optional[StateSnapshot]:
        """
        Retrieve a specific snapshot by ID.
        
        Args:
            snapshot_id: Snapshot identifier
            
        Returns:
            StateSnapshot if found, None otherwise
        """
        # Check cache first
        if snapshot_id in self._cache:
            return self._cache[snapshot_id]
        
        # Query backend
        if self.backend_type == 'influxdb':
            return await self._get_snapshot_influxdb(snapshot_id)
        else:
            return await self._get_snapshot_sql(snapshot_id)
    
    async def _get_snapshot_sql(self, snapshot_id: str) -> Optional[StateSnapshot]:
        """Get snapshot from SQL backend."""
        async with self.client.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT data FROM state_snapshots
                WHERE snapshot_id = $1
            """, snapshot_id)
            
            if row:
                data = json.loads(row['data'])
                return StateSnapshot.from_dict(data)
            return None
    
    async def get_test_snapshots(
        self, 
        test_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[StateSnapshot]:
        """
        Retrieve all snapshots for a test within time range.
        
        Args:
            test_id: Test identifier
            start_time: Optional start time filter
            end_time: Optional end time filter
            
        Returns:
            List of StateSnapshot objects ordered by timestamp
        """
        if self.backend_type == 'influxdb':
            return await self._get_test_snapshots_influxdb(test_id, start_time, end_time)
        else:
            return await self._get_test_snapshots_sql(test_id, start_time, end_time)
    
    async def _get_test_snapshots_sql(
        self,
        test_id: str,
        start_time: Optional[datetime],
        end_time: Optional[datetime]
    ) -> List[StateSnapshot]:
        """Get test snapshots from SQL backend."""
        query = """
            SELECT data FROM state_snapshots
            WHERE test_id = $1
        """
        params = [test_id]
        
        if start_time:
            query += " AND timestamp >= $2"
            params.append(start_time)
        if end_time:
            query += f" AND timestamp <= ${len(params) + 1}"
            params.append(end_time)
        
        query += " ORDER BY timestamp ASC"
        
        async with self.client.acquire() as conn:
            rows = await conn.fetch(query, *params)
            return [StateSnapshot.from_dict(json.loads(row['data'])) for row in rows]
    
    async def get_last_good_state(self, test_id: str, before: datetime) -> Optional[StateSnapshot]:
        """
        Get the last known good state before a specific time.
        
        Useful for recovery after failures.
        
        Args:
            test_id: Test identifier
            before: Timestamp to search before
            
        Returns:
            Last good StateSnapshot or None
        """
        async with self.client.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT data FROM state_snapshots
                WHERE test_id = $1 AND timestamp < $2
                ORDER BY timestamp DESC
                LIMIT 1
            """, test_id, before)
            
            if row:
                return StateSnapshot.from_dict(json.loads(row['data']))
            return None
    
    async def restore_state(
        self,
        snapshot: StateSnapshot,
        browser_driver: Any
    ) -> None:
        """
        Restore browser state from a snapshot.
        
        Args:
            snapshot: StateSnapshot to restore
            browser_driver: BrowserDriver instance to restore state to
        """
        logger.info("Restoring state from snapshot",
                   snapshot_id=snapshot.snapshot_id,
                   test_id=snapshot.test_id)
        
        # Navigate to URL
        await browser_driver.navigate(snapshot.url)
        
        # Restore cookies
        for cookie in snapshot.cookies:
            await browser_driver.set_cookie(cookie)
        
        # Restore local storage
        if snapshot.local_storage:
            script = """
            for (let [key, value] of Object.entries(arguments[0])) {
                localStorage.setItem(key, value);
            }
            """
            await browser_driver.execute_script(script, snapshot.local_storage)
        
        # Restore session storage
        if snapshot.session_storage:
            script = """
            for (let [key, value] of Object.entries(arguments[0])) {
                sessionStorage.setItem(key, value);
            }
            """
            await browser_driver.execute_script(script, snapshot.session_storage)
        
        # Restore viewport
        await browser_driver.set_viewport(
            snapshot.viewport['width'],
            snapshot.viewport['height']
        )
        
        # Restore scroll position
        await browser_driver.scroll_to(
            snapshot.scroll_position['x'],
            snapshot.scroll_position['y']
        )
        
        logger.info("State restored successfully")
    
    async def _cleanup_task(self) -> None:
        """Background task to clean up old snapshots."""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                
                cutoff = datetime.utcnow() - timedelta(days=self.retention_days)
                
                if self.backend_type == 'influxdb':
                    # InfluxDB handles retention automatically
                    pass
                else:
                    async with self.client.acquire() as conn:
                        deleted = await conn.execute("""
                            DELETE FROM state_snapshots
                            WHERE timestamp < $1
                        """, cutoff)
                        
                        logger.info("Cleaned up old snapshots", deleted=deleted)
                        
            except Exception as e:
                logger.error("Cleanup task failed", error=str(e))
```

### 3.2 Adaptive Wait Service Implementation

The Adaptive Wait Service eliminates flaky tests through AI-powered visual stability detection.

**File**: `src/reliability/adaptive_wait.py`

```python
"""
Adaptive Wait Service Implementation
Provides intelligent waiting strategies with visual stability detection
"""

import asyncio
from typing import Optional, Callable, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import structlog
import numpy as np

logger = structlog.get_logger(__name__)


@dataclass
class WaitResult:
    """Result of a wait operation."""
    success: bool
    duration_ms: float
    attempts: int
    final_state: Any
    reason: str


class AdaptiveWaitService:
    """
    Adaptive Wait Service with multiple intelligent wait strategies.
    
    Strategies:
    - Visual stability detection using computer vision
    - Network idle detection
    - Element visibility and interactivity
    - Custom condition evaluation
    """
    
    def __init__(
        self,
        vision_adapter: Any,
        browser_driver: Any,
        config: Dict[str, Any]
    ):
        """
        Initialize adaptive wait service.
        
        Args:
            vision_adapter: VisionAdapter instance for visual analysis
            browser_driver: BrowserDriver instance
            config: Configuration dictionary
        """
        self.vision_adapter = vision_adapter
        self.browser_driver = browser_driver
        
        self.default_timeout = config.get('timeout', 30)
        self.stability_threshold = config.get('stability_threshold', 0.98)
        self.poll_interval_ms = config.get('poll_interval_ms', 100)
        self.stability_duration_ms = config.get('stability_duration_ms', 500)
        
        # Learning: track optimal wait times for different scenarios
        self._wait_history: Dict[str, List[float]] = {}
    
    async def wait_for_visual_stability(
        self,
        timeout: Optional[int] = None,
        region: Optional[Any] = None
    ) -> WaitResult:
        """
        Wait until the UI reaches a visually stable state.
        
        Uses perceptual hashing and visual embeddings to detect when
        animations complete and content finishes loading.
        
        Args:
            timeout: Maximum wait time in seconds (uses default if None)
            region: Optional BoundingBox to limit stability check to specific region
            
        Returns:
            WaitResult with success status and timing information
        """
        timeout = timeout or self.default_timeout
        start_time = datetime.utcnow()
        attempts = 0
        
        logger.debug("Waiting for visual stability", timeout=timeout)
        
        # Capture initial screenshot
        prev_screenshot = await self.browser_driver.screenshot()
        prev_embedding = await self.vision_adapter.generate_embedding(
            prev_screenshot, 
            region=region
        )
        
        stable_since = None
        
        while (datetime.utcnow() - start_time).total_seconds() < timeout:
            attempts += 1
            await asyncio.sleep(self.poll_interval_ms / 1000.0)
            
            # Capture current screenshot
            curr_screenshot = await self.browser_driver.screenshot()
            curr_embedding = await self.vision_adapter.generate_embedding(
                curr_screenshot,
                region=region
            )
            
            # Calculate similarity
            similarity = self._cosine_similarity(prev_embedding, curr_embedding)
            
            logger.debug("Visual stability check", 
                        attempt=attempts,
                        similarity=similarity)
            
            if similarity >= self.stability_threshold:
                # Visual state is stable
                if stable_since is None:
                    stable_since = datetime.utcnow()
                
                # Check if stable for required duration
                stable_duration = (datetime.utcnow() - stable_since).total_seconds() * 1000
                if stable_duration >= self.stability_duration_ms:
                    duration = (datetime.utcnow() - start_time).total_seconds() * 1000
                    
                    logger.info("Visual stability achieved",
                               duration_ms=duration,
                               attempts=attempts)
                    
                    return WaitResult(
                        success=True,
                        duration_ms=duration,
                        attempts=attempts,
                        final_state=curr_screenshot,
                        reason="Visual stability threshold met"
                    )
            else:
                # State changed, reset stability timer
                stable_since = None
                prev_embedding = curr_embedding
        
        # Timeout reached
        duration = (datetime.utcnow() - start_time).total_seconds() * 1000
        logger.warning("Visual stability timeout",
                      duration_ms=duration,
                      attempts=attempts)
        
        return WaitResult(
            success=False,
            duration_ms=duration,
            attempts=attempts,
            final_state=None,
            reason="Timeout waiting for visual stability"
        )
    
    async def wait_for_network_idle(
        self,
        timeout: Optional[int] = None,
        idle_duration_ms: int = 500,
        max_connections: int = 0
    ) -> WaitResult:
        """
        Wait until network activity becomes idle.
        
        Args:
            timeout: Maximum wait time in seconds
            idle_duration_ms: Duration of idle state required (ms)
            max_connections: Maximum number of active connections to consider idle
            
        Returns:
            WaitResult with success status
        """
        timeout = timeout or self.default_timeout
        start_time = datetime.utcnow()
        attempts = 0
        
        logger.debug("Waiting for network idle", timeout=timeout)
        
        idle_since = None
        
        while (datetime.utcnow() - start_time).total_seconds() < timeout:
            attempts += 1
            await asyncio.sleep(self.poll_interval_ms / 1000.0)
            
            # Get pending network requests
            network_logs = await self.browser_driver.get_network_logs()
            pending = [req for req, resp in network_logs if resp is None]
            
            if len(pending) <= max_connections:
                if idle_since is None:
                    idle_since = datetime.utcnow()
                
                idle_duration = (datetime.utcnow() - idle_since).total_seconds() * 1000
                if idle_duration >= idle_duration_ms:
                    duration = (datetime.utcnow() - start_time).total_seconds() * 1000
                    
                    logger.info("Network idle achieved",
                               duration_ms=duration,
                               attempts=attempts)
                    
                    return WaitResult(
                        success=True,
                        duration_ms=duration,
                        attempts=attempts,
                        final_state=None,
                        reason="Network idle threshold met"
                    )
            else:
                idle_since = None
        
        duration = (datetime.utcnow() - start_time).total_seconds() * 1000
        return WaitResult(
            success=False,
            duration_ms=duration,
            attempts=attempts,
            final_state=None,
            reason="Timeout waiting for network idle"
        )
    
    async def wait_for_element_visible(
        self,
        element_prompt: str,
        timeout: Optional[int] = None
    ) -> WaitResult:
        """
        Wait until a specific element is visible and stable.
        
        Args:
            element_prompt: Natural language description of element
            timeout: Maximum wait time in seconds
            
        Returns:
            WaitResult with element location if successful
        """
        timeout = timeout or self.default_timeout
        start_time = datetime.utcnow()
        attempts = 0
        
        logger.debug("Waiting for element visible", 
                    element=element_prompt,
                    timeout=timeout)
        
        while (datetime.utcnow() - start_time).total_seconds() < timeout:
            attempts += 1
            await asyncio.sleep(self.poll_interval_ms / 1000.0)
            
            try:
                screenshot = await self.browser_driver.screenshot()
                element_location = await self.vision_adapter.find_element(
                    screenshot,
                    element_prompt
                )
                
                # Element found with sufficient confidence
                if element_location.confidence >= 0.85:
                    duration = (datetime.utcnow() - start_time).total_seconds() * 1000
                    
                    logger.info("Element visible",
                               element=element_prompt,
                               duration_ms=duration,
                               confidence=element_location.confidence)
                    
                    return WaitResult(
                        success=True,
                        duration_ms=duration,
                        attempts=attempts,
                        final_state=element_location,
                        reason="Element found and visible"
                    )
                    
            except Exception as e:
                logger.debug("Element not found yet", 
                           element=element_prompt,
                           attempt=attempts)
        
        duration = (datetime.utcnow() - start_time).total_seconds() * 1000
        return WaitResult(
            success=False,
            duration_ms=duration,
            attempts=attempts,
            final_state=None,
            reason=f"Timeout waiting for element: {element_prompt}"
        )
    
    async def wait_for_condition(
        self,
        condition: Callable[[], bool],
        timeout: Optional[int] = None,
        condition_name: str = "custom condition"
    ) -> WaitResult:
        """
        Wait for a custom condition to become true.
        
        Args:
            condition: Async callable that returns bool
            timeout: Maximum wait time in seconds
            condition_name: Description of condition for logging
            
        Returns:
            WaitResult with success status
        """
        timeout = timeout or self.default_timeout
        start_time = datetime.utcnow()
        attempts = 0
        
        logger.debug("Waiting for condition", 
                    condition=condition_name,
                    timeout=timeout)
        
        while (datetime.utcnow() - start_time).total_seconds() < timeout:
            attempts += 1
            await asyncio.sleep(self.poll_interval_ms / 1000.0)
            
            try:
                if await condition():
                    duration = (datetime.utcnow() - start_time).total_seconds() * 1000
                    
                    logger.info("Condition met",
                               condition=condition_name,
                               duration_ms=duration)
                    
                    return WaitResult(
                        success=True,
                        duration_ms=duration,
                        attempts=attempts,
                        final_state=None,
                        reason=f"Condition met: {condition_name}"
                    )
            except Exception as e:
                logger.debug("Condition check failed",
                           condition=condition_name,
                           error=str(e))
        
        duration = (datetime.utcnow() - start_time).total_seconds() * 1000
        return WaitResult(
            success=False,
            duration_ms=duration,
            attempts=attempts,
            final_state=None,
            reason=f"Timeout waiting for condition: {condition_name}"
        )
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    async def learn_optimal_wait(
        self,
        scenario: str,
        actual_duration_ms: float
    ) -> None:
        """
        Record actual wait duration for learning.
        
        Over time, the service learns optimal wait times for different
        scenarios and can adjust timeouts accordingly.
        
        Args:
            scenario: Scenario identifier (e.g., "login_page_load")
            actual_duration_ms: Actual duration that was needed
        """
        if scenario not in self._wait_history:
            self._wait_history[scenario] = []
        
        self._wait_history[scenario].append(actual_duration_ms)
        
        # Keep only recent history (last 100 observations)
        if len(self._wait_history[scenario]) > 100:
            self._wait_history[scenario] = self._wait_history[scenario][-100:]
    
    def get_recommended_timeout(self, scenario: str) -> int:
        """
        Get recommended timeout based on historical data.
        
        Args:
            scenario: Scenario identifier
            
        Returns:
            Recommended timeout in seconds
        """
        if scenario not in self._wait_history or len(self._wait_history[scenario]) < 5:
            return self.default_timeout
        
        # Use 95th percentile of historical durations
        durations = np.array(self._wait_history[scenario])
        p95 = np.percentile(durations, 95)
        
        # Add 20% buffer
        recommended = int((p95 / 1000.0) * 1.2)
        
        # Clamp to reasonable range
        return max(5, min(recommended, 60))
```

This completes the core reliability layer specifications. The remaining components (Heuristic Recovery Engine, Hot-Swappable Module Manager, Self-Healing services) follow similar implementation patterns with detailed state management, error handling, and observability.
