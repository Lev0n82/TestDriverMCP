"""
Cross-Layer Validation Module.
Validates consistency across UI, API, and Database layers.

Built-in Self-Tests:
- Function-level: Each validator tests its logic
- Class-level: Multi-layer validation workflows
- Module-level: End-to-end consistency checks
"""

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import json

import structlog
import requests
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

logger = structlog.get_logger()


class CrossLayerValidator:
    """Built-in validator for cross-layer operations."""
    
    @staticmethod
    def validate_consistency_result(result: Dict[str, Any]) -> bool:
        """
        Validate consistency check result.
        
        Success Criteria:
        - Result has consistent field
        - Has details for each layer
        - Discrepancies listed if inconsistent
        """
        if 'consistent' not in result:
            return False
        if 'layers' not in result:
            return False
        if not result['consistent'] and 'discrepancies' not in result:
            return False
        return True


class UIValidator:
    """
    UI layer validator.
    
    Success Criteria (Class-level):
    - Extracts data from UI screenshots
    - Validates UI state
    - Detects UI changes
    """
    
    def __init__(self, vision_adapter):
        """
        Initialize UI validator.
        
        Args:
            vision_adapter: Vision adapter for UI analysis
        """
        self.vision_adapter = vision_adapter
        logger.info("UI validator initialized")
    
    async def extract_data(self, screenshot: bytes, fields: List[str]) -> Dict[str, Any]:
        """
        Extract data from UI screenshot.
        
        Args:
            screenshot: Screenshot bytes
            fields: Fields to extract
        
        Returns:
            Extracted data
        
        Success Criteria:
        - All requested fields extracted
        - Data types inferred correctly
        - Confidence scores provided
        """
        prompt = f"""Analyze this screenshot and extract the following fields: {', '.join(fields)}

Return a JSON object with:
- For each field: the extracted value
- confidence: float (0.0 to 1.0) for extraction accuracy

Return ONLY the JSON object, no other text."""
        
        try:
            result = await self.vision_adapter.analyze_screenshot(screenshot, prompt)
            
            if not result.get('success', True):
                return {'error': 'Vision analysis failed', 'fields': {}}
            
            # Parse JSON from response
            analysis = result.get('analysis', '{}')
            try:
                import re
                json_match = re.search(r'\{.*\}', analysis, re.DOTALL)
                if json_match:
                    extracted = json.loads(json_match.group())
                else:
                    extracted = {'error': 'No JSON found'}
            except json.JSONDecodeError:
                extracted = {'error': 'Invalid JSON'}
            
            logger.debug("UI data extracted", fields=fields, confidence=extracted.get('confidence', 0.0))
            return extracted
        
        except Exception as e:
            logger.error("UI data extraction failed", error=str(e))
            return {'error': str(e), 'fields': {}}


class APIValidator:
    """
    API layer validator.
    
    Success Criteria (Class-level):
    - Fetches data from API endpoints
    - Validates API responses
    - Handles authentication
    """
    
    def __init__(self, base_url: str, headers: Optional[Dict[str, str]] = None):
        """
        Initialize API validator.
        
        Args:
            base_url: API base URL
            headers: Optional headers (auth, etc.)
        """
        self.base_url = base_url.rstrip('/')
        self.headers = headers or {}
        logger.info("API validator initialized", base_url=self.base_url)
    
    def fetch_data(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Fetch data from API.
        
        Args:
            endpoint: API endpoint path
            params: Optional query parameters
        
        Returns:
            API response data
        
        Success Criteria:
        - HTTP 200 response
        - Valid JSON returned
        - Data structure as expected
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        try:
            response = requests.get(
                url,
                params=params,
                headers=self.headers,
                timeout=10
            )
            
            response.raise_for_status()
            data = response.json()
            
            logger.debug("API data fetched", endpoint=endpoint, status=response.status_code)
            return {
                'success': True,
                'status_code': response.status_code,
                'data': data
            }
        
        except requests.exceptions.RequestException as e:
            logger.error("API fetch failed", endpoint=endpoint, error=str(e))
            return {
                'success': False,
                'error': str(e),
                'data': None
            }
        except json.JSONDecodeError as e:
            logger.error("API response not JSON", endpoint=endpoint, error=str(e))
            return {
                'success': False,
                'error': 'Invalid JSON response',
                'data': None
            }


class DatabaseValidator:
    """
    Database layer validator.
    
    Success Criteria (Class-level):
    - Queries database directly
    - Validates data integrity
    - Handles connections safely
    """
    
    def __init__(self, database_url: str):
        """
        Initialize database validator.
        
        Args:
            database_url: Database connection URL
        """
        self.database_url = database_url
        self.engine: Optional[Engine] = None
        
        try:
            self.engine = create_engine(database_url)
            logger.info("Database validator initialized")
        except Exception as e:
            logger.error("Database connection failed", error=str(e))
            self.engine = None
    
    def query_data(self, query: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Query database.
        
        Args:
            query: SQL query
            params: Optional query parameters
        
        Returns:
            Query results
        
        Success Criteria:
        - Query executes successfully
        - Results returned as list of dicts
        - Connection handled safely
        """
        if self.engine is None:
            return {
                'success': False,
                'error': 'Database not connected',
                'data': []
            }
        
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(query), params or {})
                
                # Convert to list of dicts
                rows = []
                for row in result:
                    rows.append(dict(row._mapping))
                
                logger.debug("Database query executed", row_count=len(rows))
                return {
                    'success': True,
                    'data': rows
                }
        
        except Exception as e:
            logger.error("Database query failed", error=str(e))
            return {
                'success': False,
                'error': str(e),
                'data': []
            }


class CrossLayerValidationEngine:
    """
    Cross-layer validation engine.
    
    Success Criteria (Class-level):
    - Validates consistency across all layers
    - Detects discrepancies
    - Provides detailed reports
    """
    
    def __init__(
        self,
        ui_validator: Optional[UIValidator] = None,
        api_validator: Optional[APIValidator] = None,
        db_validator: Optional[DatabaseValidator] = None
    ):
        """
        Initialize validation engine.
        
        Args:
            ui_validator: UI layer validator
            api_validator: API layer validator
            db_validator: Database layer validator
        """
        self.ui_validator = ui_validator
        self.api_validator = api_validator
        self.db_validator = db_validator
        
        self.validator = CrossLayerValidator()
        
        logger.info(
            "Cross-layer validation engine initialized",
            has_ui=ui_validator is not None,
            has_api=api_validator is not None,
            has_db=db_validator is not None
        )
    
    async def validate_consistency(
        self,
        entity_id: str,
        ui_screenshot: Optional[bytes] = None,
        api_endpoint: Optional[str] = None,
        db_query: Optional[str] = None,
        fields_to_check: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Validate consistency across layers.
        
        Args:
            entity_id: Entity identifier
            ui_screenshot: Optional UI screenshot
            api_endpoint: Optional API endpoint
            db_query: Optional database query
            fields_to_check: Fields to validate
        
        Returns:
            Validation result
        
        Success Criteria:
        - Data fetched from all available layers
        - Consistency checked across layers
        - Discrepancies identified and reported
        """
        layers = {}
        
        # Fetch UI data
        if ui_screenshot and self.ui_validator and fields_to_check:
            ui_data = await self.ui_validator.extract_data(ui_screenshot, fields_to_check)
            layers['ui'] = ui_data
        
        # Fetch API data
        if api_endpoint and self.api_validator:
            api_result = self.api_validator.fetch_data(api_endpoint)
            if api_result['success']:
                layers['api'] = api_result['data']
            else:
                layers['api'] = {'error': api_result['error']}
        
        # Fetch DB data
        if db_query and self.db_validator:
            db_result = self.db_validator.query_data(db_query, {'entity_id': entity_id})
            if db_result['success'] and db_result['data']:
                layers['database'] = db_result['data'][0]
            else:
                layers['database'] = {'error': db_result.get('error', 'No data')}
        
        # Check consistency
        consistent, discrepancies = self._check_consistency(layers, fields_to_check or [])
        
        result = {
            'entity_id': entity_id,
            'timestamp': datetime.now().isoformat(),
            'layers': layers,
            'consistent': consistent,
            'discrepancies': discrepancies,
            'fields_checked': fields_to_check or []
        }
        
        logger.info(
            "Cross-layer validation completed",
            entity_id=entity_id,
            consistent=consistent,
            discrepancy_count=len(discrepancies)
        )
        
        return result
    
    def _check_consistency(
        self,
        layers: Dict[str, Any],
        fields: List[str]
    ) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        Check consistency across layers.
        
        Args:
            layers: Data from each layer
            fields: Fields to check
        
        Returns:
            Tuple of (consistent, discrepancies)
        
        Success Criteria:
        - All fields compared across layers
        - Discrepancies identified with details
        - Type coercion handled
        """
        discrepancies = []
        
        if len(layers) < 2:
            # Need at least 2 layers to compare
            return True, []
        
        layer_names = list(layers.keys())
        
        for field in fields:
            values = {}
            
            # Collect values from each layer
            for layer_name, layer_data in layers.items():
                if isinstance(layer_data, dict) and 'error' not in layer_data:
                    value = layer_data.get(field)
                    if value is not None:
                        # Normalize value (convert to string for comparison)
                        values[layer_name] = str(value).strip()
            
            # Check if all values match
            if len(values) > 1:
                unique_values = set(values.values())
                if len(unique_values) > 1:
                    discrepancies.append({
                        'field': field,
                        'values': values,
                        'layers_affected': list(values.keys())
                    })
        
        consistent = len(discrepancies) == 0
        return consistent, discrepancies
    
    async def validate_transaction(
        self,
        transaction_id: str,
        ui_screenshot: Optional[bytes] = None,
        api_endpoint: Optional[str] = None,
        db_query: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Validate transaction across layers.
        
        Success Criteria:
        - Transaction state consistent
        - Amounts match across layers
        - Timestamps within acceptable range
        """
        fields = ['status', 'amount', 'timestamp', 'user_id']
        
        result = await self.validate_consistency(
            entity_id=transaction_id,
            ui_screenshot=ui_screenshot,
            api_endpoint=api_endpoint,
            db_query=db_query,
            fields_to_check=fields
        )
        
        # Additional transaction-specific checks
        if result['consistent']:
            # Check timestamp consistency (within 5 seconds)
            timestamps = []
            for layer_data in result['layers'].values():
                if isinstance(layer_data, dict) and 'timestamp' in layer_data:
                    try:
                        ts = datetime.fromisoformat(layer_data['timestamp'])
                        timestamps.append(ts)
                    except:
                        pass
            
            if len(timestamps) > 1:
                time_diffs = [abs((timestamps[i] - timestamps[0]).total_seconds()) for i in range(1, len(timestamps))]
                if any(diff > 5.0 for diff in time_diffs):
                    result['consistent'] = False
                    result['discrepancies'].append({
                        'field': 'timestamp',
                        'issue': 'Timestamps differ by more than 5 seconds',
                        'time_diffs': time_diffs
                    })
        
        logger.info(
            "Transaction validation completed",
            transaction_id=transaction_id,
            consistent=result['consistent']
        )
        
        return result


# Module-level self-test
def self_test_module() -> bool:
    """
    Module-level self-test.
    
    Success Criteria:
    - Validators can be instantiated
    - Mock validation works
    - Consistency checking works
    """
    try:
        # Test validator
        validator = CrossLayerValidator()
        
        # Test consistency result validation
        valid_result = {
            'consistent': True,
            'layers': {'ui': {}, 'api': {}},
            'discrepancies': []
        }
        if not validator.validate_consistency_result(valid_result):
            logger.error("Module self-test failed: Valid result rejected")
            return False
        
        # Test API validator with mock
        api_validator = APIValidator("http://localhost:8000")
        if api_validator.base_url != "http://localhost:8000":
            logger.error("Module self-test failed: API validator initialization")
            return False
        
        # Test consistency checking
        engine = CrossLayerValidationEngine()
        layers = {
            'ui': {'name': 'John Doe', 'email': 'john@example.com'},
            'api': {'name': 'John Doe', 'email': 'john@example.com'}
        }
        consistent, discrepancies = engine._check_consistency(layers, ['name', 'email'])
        
        if not consistent:
            logger.error("Module self-test failed: Consistency check false positive")
            return False
        
        # Test discrepancy detection
        layers_inconsistent = {
            'ui': {'name': 'John Doe'},
            'api': {'name': 'Jane Doe'}
        }
        consistent, discrepancies = engine._check_consistency(layers_inconsistent, ['name'])
        
        if consistent or len(discrepancies) == 0:
            logger.error("Module self-test failed: Discrepancy not detected")
            return False
        
        logger.info("Module self-test passed: validation.cross_layer")
        return True
    
    except Exception as e:
        logger.error("Module self-test failed", error=str(e))
        return False


if __name__ == "__main__":
    # Run module self-test
    success = self_test_module()
    print(f"Module self-test: {'PASSED' if success else 'FAILED'}")
