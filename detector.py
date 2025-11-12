"""
Environment Drift Detection Module.
Built-in Self-Tests at function, class, and module levels.
"""

from typing import Dict, Any, List
from datetime import datetime
import structlog

logger = structlog.get_logger()


class DriftDetector:
    """Environment drift detector."""
    
    def __init__(self):
        self.baseline: Dict[str, Any] = {}
        logger.info("Drift detector initialized")
    
    def capture_baseline(self, environment: Dict[str, Any]):
        """Capture environment baseline."""
        self.baseline = environment.copy()
        logger.info("Baseline captured", keys=list(environment.keys()))
    
    def detect_drift(self, current: Dict[str, Any]) -> Dict[str, Any]:
        """Detect environment drift."""
        drifts = []
        
        for key, baseline_value in self.baseline.items():
            current_value = current.get(key)
            if current_value != baseline_value:
                drifts.append({
                    'key': key,
                    'baseline': baseline_value,
                    'current': current_value
                })
        
        return {
            'has_drift': len(drifts) > 0,
            'drift_count': len(drifts),
            'drifts': drifts,
            'timestamp': datetime.now().isoformat()
        }


def self_test_module() -> bool:
    """Module self-test."""
    try:
        detector = DriftDetector()
        detector.capture_baseline({'version': '1.0', 'env': 'staging'})
        result = detector.detect_drift({'version': '1.1', 'env': 'staging'})
        
        if not result['has_drift'] or result['drift_count'] != 1:
            return False
        
        logger.info("Module self-test passed: drift.detector")
        return True
    except Exception as e:
        logger.error("Module self-test failed", error=str(e))
        return False
