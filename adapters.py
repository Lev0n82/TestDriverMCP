"""Vision adapters for AI-powered element detection."""

import base64
import os
from abc import ABC, abstractmethod
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import structlog
from PIL import Image

logger = structlog.get_logger()


class VisionAdapter(ABC):
    """Base class for vision adapters."""
    
    @abstractmethod
    async def locate_element(
        self,
        screenshot: Image.Image,
        description: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Locate element using vision model.
        
        Args:
            screenshot: Screenshot image
            description: Natural language description of element
            context: Additional context
            
        Returns:
            Dict with coordinates, confidence, and locator suggestions
        """
        pass
    
    @abstractmethod
    async def verify_element(
        self,
        screenshot: Image.Image,
        element_description: str,
        coordinates: Tuple[int, int, int, int]
    ) -> Dict[str, Any]:
        """
        Verify element at given coordinates matches description.
        
        Args:
            screenshot: Screenshot image
            element_description: Expected element description
            coordinates: (x, y, width, height)
            
        Returns:
            Dict with verification result and confidence
        """
        pass
    
    @abstractmethod
    async def generate_embedding(
        self,
        screenshot: Image.Image,
        region: Optional[Tuple[int, int, int, int]] = None
    ) -> List[float]:
        """
        Generate visual embedding for screenshot or region.
        
        Args:
            screenshot: Screenshot image
            region: Optional region (x, y, width, height)
            
        Returns:
            Visual embedding vector
        """
        pass


class OpenAIVisionAdapter(VisionAdapter):
    """Vision adapter using OpenAI GPT-4 Vision."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.api_key = config.get("api_key") or os.getenv("OPENAI_API_KEY")
        self.model = config.get("model", "gpt-4.1-mini")
        
        if not self.api_key:
            raise ValueError("OpenAI API key not provided")
        
        logger.info("OpenAI Vision Adapter initialized", model=self.model)
    
    async def locate_element(
        self,
        screenshot: Image.Image,
        description: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Locate element using GPT-4 Vision."""
        try:
            from openai import OpenAI
            
            client = OpenAI(api_key=self.api_key)
            
            # Convert screenshot to base64
            buffered = BytesIO()
            screenshot.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            
            # Create prompt
            prompt = f"""Analyze this screenshot and locate the element: {description}

Please provide:
1. The approximate coordinates (x, y, width, height) as percentages of image dimensions
2. A confidence score (0.0 to 1.0)
3. Suggested CSS selector or XPath if visible
4. Visual characteristics of the element

Respond in JSON format:
{{
    "found": true/false,
    "coordinates": {{"x": 0.5, "y": 0.3, "width": 0.1, "height": 0.05}},
    "confidence": 0.95,
    "suggested_selectors": ["#submit-btn", "button.primary"],
    "characteristics": "Blue button with white text 'Submit'"
}}"""
            
            # Call OpenAI API
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{img_base64}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=500
            )
            
            # Parse response
            import json
            result_text = response.choices[0].message.content
            
            # Extract JSON from response
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0]
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0]
            
            result = json.loads(result_text.strip())
            
            logger.info("Element located", description=description, confidence=result.get("confidence"))
            
            return result
            
        except Exception as e:
            logger.error("Error locating element", error=str(e))
            return {
                "found": False,
                "confidence": 0.0,
                "error": str(e)
            }
    
    async def verify_element(
        self,
        screenshot: Image.Image,
        element_description: str,
        coordinates: Tuple[int, int, int, int]
    ) -> Dict[str, Any]:
        """Verify element at coordinates."""
        try:
            from openai import OpenAI
            
            client = OpenAI(api_key=self.api_key)
            
            # Crop to element region
            x, y, w, h = coordinates
            element_img = screenshot.crop((x, y, x + w, y + h))
            
            # Convert to base64
            buffered = BytesIO()
            element_img.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            
            prompt = f"""Does this image show: {element_description}?

Respond in JSON format:
{{
    "matches": true/false,
    "confidence": 0.95,
    "actual_description": "What you actually see",
    "reasoning": "Why it matches or doesn't match"
}}"""
            
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{img_base64}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=300
            )
            
            import json
            result_text = response.choices[0].message.content
            
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0]
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0]
            
            result = json.loads(result_text.strip())
            
            return result
            
        except Exception as e:
            logger.error("Error verifying element", error=str(e))
            return {
                "matches": False,
                "confidence": 0.0,
                "error": str(e)
            }
    
    async def generate_embedding(
        self,
        screenshot: Image.Image,
        region: Optional[Tuple[int, int, int, int]] = None
    ) -> List[float]:
        """Generate visual embedding (simplified)."""
        # In production, this would use a proper embedding model
        # For now, return a placeholder embedding
        import numpy as np
        
        if region:
            x, y, w, h = region
            img = screenshot.crop((x, y, x + w, y + h))
        else:
            img = screenshot
        
        # Resize to standard size
        img = img.resize((224, 224))
        
        # Convert to array and flatten (simplified embedding)
        arr = np.array(img)
        embedding = arr.flatten()[:512].tolist()  # Take first 512 values
        
        return embedding


class LocalVisionAdapter(VisionAdapter):
    """Vision adapter using local vision model."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_name = config.get("model_name", "llava")
        
        logger.info("Local Vision Adapter initialized", model=self.model_name)
    
    async def locate_element(
        self,
        screenshot: Image.Image,
        description: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Locate element using local model."""
        # Placeholder implementation
        logger.warning("Local vision adapter not fully implemented")
        return {
            "found": False,
            "confidence": 0.0,
            "error": "Local vision adapter not implemented"
        }
    
    async def verify_element(
        self,
        screenshot: Image.Image,
        element_description: str,
        coordinates: Tuple[int, int, int, int]
    ) -> Dict[str, Any]:
        """Verify element using local model."""
        logger.warning("Local vision adapter not fully implemented")
        return {
            "matches": False,
            "confidence": 0.0,
            "error": "Local vision adapter not implemented"
        }
    
    async def generate_embedding(
        self,
        screenshot: Image.Image,
        region: Optional[Tuple[int, int, int, int]] = None
    ) -> List[float]:
        """Generate embedding using local model."""
        import numpy as np
        
        if region:
            x, y, w, h = region
            img = screenshot.crop((x, y, x + w, y + h))
        else:
            img = screenshot
        
        img = img.resize((224, 224))
        arr = np.array(img)
        embedding = arr.flatten()[:512].tolist()
        
        return embedding
