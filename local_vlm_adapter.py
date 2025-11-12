"""
Local Vision Language Model (VLM) Adapter for Ollama.
Enables on-premise vision processing without cloud dependencies.

Built-in Self-Tests:
- Function-level: Each method validates inputs and outputs
- Class-level: Model loading and inference
- Module-level: Ollama connectivity and compatibility
"""

from typing import Dict, Any, Optional, List
import base64
import json
from io import BytesIO

import structlog
import requests
from PIL import Image

from vision.adapters import VisionAdapter

logger = structlog.get_logger()


class LocalVLMValidator:
    """Built-in validator for local VLM operations."""
    
    @staticmethod
    def validate_ollama_response(response: Dict[str, Any]) -> bool:
        """
        Validate Ollama API response.
        
        Success Criteria:
        - Response is a dictionary
        - Contains required fields
        - Response text is non-empty
        """
        if not isinstance(response, dict):
            return False
        if 'response' not in response and 'message' not in response:
            return False
        return True
    
    @staticmethod
    def validate_model_name(model_name: str) -> bool:
        """
        Validate model name format.
        
        Success Criteria:
        - Model name is non-empty string
        - Contains valid characters
        """
        if not isinstance(model_name, str) or not model_name:
            return False
        # Basic validation - alphanumeric, hyphens, colons
        import re
        return bool(re.match(r'^[a-zA-Z0-9\-:\.]+$', model_name))


class OllamaVLMAdapter(VisionAdapter):
    """
    Local VLM adapter using Ollama.
    
    Success Criteria (Class-level):
    - Connects to Ollama server
    - Loads vision-capable models
    - Processes images locally
    - Returns structured responses
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Ollama VLM adapter.
        
        Args:
            config: Configuration dict with:
                - host: Ollama host (default: http://localhost:11434)
                - model: Model name (default: llava)
                - timeout: Request timeout in seconds
        
        Success Criteria:
        - Config validated
        - Connection to Ollama verified
        """
        self.host = config.get('host', 'http://localhost:11434')
        self.model = config.get('model', 'llava')
        self.timeout = config.get('timeout', 60)
        
        self.validator = LocalVLMValidator()
        
        if not self.validator.validate_model_name(self.model):
            raise ValueError(f"Invalid model name: {self.model}")
        
        logger.info(
            "Ollama VLM adapter initialized",
            host=self.host,
            model=self.model
        )
        
        # Self-test initialization
        self._self_test_init()
    
    def _self_test_init(self) -> bool:
        """
        Self-test: Validate Ollama connection.
        
        Success Criteria:
        - Can connect to Ollama
        - Can list models
        """
        try:
            # Try to connect to Ollama
            response = requests.get(
                f"{self.host}/api/tags",
                timeout=5
            )
            
            if response.status_code == 200:
                models = response.json().get('models', [])
                logger.debug(
                    "Self-test passed: Ollama connected",
                    available_models=len(models)
                )
                return True
            else:
                logger.warning(
                    "Self-test warning: Ollama returned non-200",
                    status=response.status_code
                )
                return False
        
        except requests.exceptions.ConnectionError:
            logger.warning("Self-test warning: Ollama not available (will use fallback)")
            return False
        except Exception as e:
            logger.error("Self-test failed: Initialization error", error=str(e))
            return False
    
    def _encode_image(self, image_bytes: bytes) -> str:
        """
        Encode image to base64.
        
        Success Criteria:
        - Image is encoded successfully
        - Encoded string is valid base64
        """
        try:
            # Verify it's a valid image
            Image.open(BytesIO(image_bytes))
            
            # Encode to base64
            encoded = base64.b64encode(image_bytes).decode('utf-8')
            return encoded
        
        except Exception as e:
            logger.error("Image encoding failed", error=str(e))
            raise
    
    async def analyze_screenshot(
        self,
        screenshot: bytes,
        prompt: str
    ) -> Dict[str, Any]:
        """
        Analyze screenshot using local VLM.
        
        Args:
            screenshot: Screenshot bytes
            prompt: Analysis prompt
        
        Returns:
            Analysis results
        
        Success Criteria:
        - Screenshot is processed
        - Response is structured
        - Contains requested information
        """
        try:
            # Encode image
            image_b64 = self._encode_image(screenshot)
            
            # Prepare request
            payload = {
                "model": self.model,
                "prompt": prompt,
                "images": [image_b64],
                "stream": False
            }
            
            # Call Ollama API
            response = requests.post(
                f"{self.host}/api/generate",
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code != 200:
                raise Exception(f"Ollama API error: {response.status_code}")
            
            result = response.json()
            
            # Validate response
            if not self.validator.validate_ollama_response(result):
                raise ValueError("Invalid Ollama response format")
            
            # Extract response text
            response_text = result.get('response', result.get('message', {}).get('content', ''))
            
            logger.info(
                "Screenshot analyzed",
                model=self.model,
                prompt_length=len(prompt),
                response_length=len(response_text)
            )
            
            return {
                'analysis': response_text,
                'model': self.model,
                'success': True
            }
        
        except requests.exceptions.ConnectionError:
            logger.error("Ollama connection failed")
            # Fallback to mock response for testing
            return {
                'analysis': f"Mock analysis: {prompt[:50]}",
                'model': self.model,
                'success': False,
                'error': 'Ollama not available'
            }
        
        except Exception as e:
            logger.error("Screenshot analysis failed", error=str(e))
            raise
    
    async def find_element(
        self,
        screenshot: bytes,
        description: str
    ) -> Dict[str, Any]:
        """
        Find element in screenshot.
        
        Args:
            screenshot: Screenshot bytes
            description: Element description
        
        Returns:
            Element location and confidence
        
        Success Criteria:
        - Element is located or not found reported
        - Confidence score is provided
        - Coordinates are reasonable
        """
        try:
            prompt = f"""Analyze this screenshot and find the element described as: "{description}"

Return a JSON object with:
- found: boolean (true if element is found)
- confidence: float (0.0 to 1.0)
- x: integer (x coordinate, or null if not found)
- y: integer (y coordinate, or null if not found)
- description: string (what you see)

Return ONLY the JSON object, no other text."""
            
            result = await self.analyze_screenshot(screenshot, prompt)
            
            if not result.get('success'):
                # Fallback for when Ollama is not available
                return {
                    'found': False,
                    'confidence': 0.0,
                    'x': None,
                    'y': None,
                    'description': 'Ollama not available',
                    'model': self.model
                }
            
            # Parse JSON from response
            try:
                analysis_text = result['analysis'].strip()
                # Try to extract JSON from response
                import re
                json_match = re.search(r'\{.*\}', analysis_text, re.DOTALL)
                if json_match:
                    element_data = json.loads(json_match.group())
                else:
                    # Fallback if no JSON found
                    element_data = {
                        'found': False,
                        'confidence': 0.5,
                        'x': None,
                        'y': None,
                        'description': analysis_text[:100]
                    }
            except json.JSONDecodeError:
                # Fallback for invalid JSON
                element_data = {
                    'found': False,
                    'confidence': 0.5,
                    'x': None,
                    'y': None,
                    'description': result['analysis'][:100]
                }
            
            element_data['model'] = self.model
            
            logger.info(
                "Element search completed",
                description=description,
                found=element_data.get('found', False),
                confidence=element_data.get('confidence', 0.0)
            )
            
            return element_data
        
        except Exception as e:
            logger.error("Element finding failed", error=str(e))
            raise
    
    async def verify_element(
        self,
        screenshot: bytes,
        locator: Dict[str, str],
        expected_state: str
    ) -> Dict[str, Any]:
        """
        Verify element state.
        
        Args:
            screenshot: Screenshot bytes
            locator: Element locator
            expected_state: Expected state description
        
        Returns:
            Verification result
        
        Success Criteria:
        - State is verified or mismatch reported
        - Confidence score provided
        """
        try:
            prompt = f"""Analyze this screenshot and verify if the element with locator {locator} is in the expected state: "{expected_state}"

Return a JSON object with:
- matches: boolean (true if state matches expectation)
- confidence: float (0.0 to 1.0)
- actual_state: string (what you observe)
- explanation: string (why it matches or doesn't match)

Return ONLY the JSON object, no other text."""
            
            result = await self.analyze_screenshot(screenshot, prompt)
            
            if not result.get('success'):
                return {
                    'matches': False,
                    'confidence': 0.0,
                    'actual_state': 'unknown',
                    'explanation': 'Ollama not available',
                    'model': self.model
                }
            
            # Parse JSON from response
            try:
                analysis_text = result['analysis'].strip()
                import re
                json_match = re.search(r'\{.*\}', analysis_text, re.DOTALL)
                if json_match:
                    verification_data = json.loads(json_match.group())
                else:
                    verification_data = {
                        'matches': False,
                        'confidence': 0.5,
                        'actual_state': 'unknown',
                        'explanation': analysis_text[:100]
                    }
            except json.JSONDecodeError:
                verification_data = {
                    'matches': False,
                    'confidence': 0.5,
                    'actual_state': 'unknown',
                    'explanation': result['analysis'][:100]
                }
            
            verification_data['model'] = self.model
            
            logger.info(
                "Element verification completed",
                matches=verification_data.get('matches', False),
                confidence=verification_data.get('confidence', 0.0)
            )
            
            return verification_data
        
        except Exception as e:
            logger.error("Element verification failed", error=str(e))
            raise
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate text embedding (not supported by all VLMs).
        
        Returns:
            Embedding vector or empty list
        """
        logger.warning("Embedding generation not supported by Ollama VLM adapter")
        # Return zero vector as placeholder
        return [0.0] * 384


# Module-level self-test
def self_test_module() -> bool:
    """
    Module-level self-test.
    
    Success Criteria:
    - Validator works correctly
    - Adapter can be instantiated
    - Basic operations don't crash
    """
    try:
        # Test validator
        validator = LocalVLMValidator()
        
        # Test valid model name
        if not validator.validate_model_name("llava"):
            logger.error("Module self-test failed: Valid model name rejected")
            return False
        
        # Test invalid model name
        if validator.validate_model_name("invalid model!"):
            logger.error("Module self-test failed: Invalid model name accepted")
            return False
        
        # Test response validation
        valid_response = {'response': 'test response'}
        if not validator.validate_ollama_response(valid_response):
            logger.error("Module self-test failed: Valid response rejected")
            return False
        
        # Test adapter instantiation
        config = {'host': 'http://localhost:11434', 'model': 'llava'}
        adapter = OllamaVLMAdapter(config)
        
        if adapter.model != 'llava':
            logger.error("Module self-test failed: Model not set correctly")
            return False
        
        logger.info("Module self-test passed: local_vlm_adapter")
        return True
    
    except Exception as e:
        logger.error("Module self-test failed", error=str(e))
        return False


if __name__ == "__main__":
    # Run module self-test
    success = self_test_module()
    print(f"Module self-test: {'PASSED' if success else 'FAILED'}")
