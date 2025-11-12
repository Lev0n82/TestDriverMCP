"""
OpenAI Vision API adapter for real AI-powered visual element detection.
Uses GPT-4V, GPT-4.1-mini, or GPT-4.1-nano for computer vision tasks.
"""

import os
import base64
from typing import Optional, Dict, Any, List
from io import BytesIO
from PIL import Image
import structlog
from openai import OpenAI, AsyncOpenAI

from vision.adapters import VisionAdapter

logger = structlog.get_logger()

class OpenAIVisionAdapter(VisionAdapter):
    """
    OpenAI Vision API adapter using GPT-4V or compatible models.
    Provides real AI-powered visual element detection and analysis.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize OpenAI Vision adapter.
        
        Args:
            config: Configuration dict with api_key and model settings
        """
        self.config = config or {}
        
        # Get API key from config or environment
        api_key = self.config.get("api_key") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY environment variable.")
        
        # Initialize OpenAI client
        self.client = AsyncOpenAI(api_key=api_key)
        
        # Model selection
        self.model = self.config.get("model", "gpt-4.1-mini")
        self.max_tokens = self.config.get("max_tokens", 1000)
        self.temperature = self.config.get("temperature", 0.1)
        
        # Image preprocessing settings
        self.max_image_size = self.config.get("max_image_size", (1920, 1080))
        self.image_quality = self.config.get("image_quality", 85)
        
        logger.info("OpenAI Vision Adapter initialized", model=self.model)
    
    def _preprocess_image(self, image_bytes: bytes) -> str:
        """
        Preprocess image and convert to base64.
        
        Args:
            image_bytes: Raw image bytes
        
        Returns:
            Base64-encoded image string
        """
        # Open image
        image = Image.open(BytesIO(image_bytes))
        
        # Resize if too large
        if image.size[0] > self.max_image_size[0] or image.size[1] > self.max_image_size[1]:
            image.thumbnail(self.max_image_size, Image.Resampling.LANCZOS)
            logger.info("Image resized", original_size=image.size, new_size=image.size)
        
        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Save to bytes
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=self.image_quality, optimize=True)
        image_bytes = buffer.getvalue()
        
        # Encode to base64
        base64_image = base64.b64encode(image_bytes).decode("utf-8")
        
        return base64_image
    
    async def detect_element(
        self,
        screenshot: bytes,
        element_description: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Detect element in screenshot using AI vision.
        
        Args:
            screenshot: Screenshot bytes
            element_description: Natural language description of element
            context: Additional context (e.g., previous locator, failure reason)
        
        Returns:
            Detection result with locator and confidence
        """
        # Preprocess image
        base64_image = self._preprocess_image(screenshot)
        
        # Build prompt
        prompt = self._build_detection_prompt(element_description, context)
        
        # Call OpenAI Vision API
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at analyzing web UI screenshots and identifying elements. Provide precise CSS selectors or XPath expressions to locate elements."
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            # Parse response
            result_text = response.choices[0].message.content
            result = self._parse_detection_result(result_text, element_description)
            
            logger.info(
                "Element detected",
                description=element_description,
                confidence=result.get("confidence", 0.0),
                locator_type=result.get("locator_type")
            )
            
            return result
            
        except Exception as e:
            logger.error("OpenAI Vision API error", error=str(e))
            return {
                "success": False,
                "confidence": 0.0,
                "error": str(e)
            }
    
    def _build_detection_prompt(
        self,
        element_description: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Build prompt for element detection."""
        prompt = f"""Analyze this web page screenshot and locate the following element:

Element Description: {element_description}
"""
        
        if context:
            if "previous_locator" in context:
                prompt += f"\nPrevious Locator (now broken): {context['previous_locator']}"
            if "failure_reason" in context:
                prompt += f"\nFailure Reason: {context['failure_reason']}"
        
        prompt += """

Please provide:
1. The best CSS selector or XPath to locate this element
2. A confidence score (0.0 to 1.0) indicating how certain you are
3. Alternative locators if available
4. A brief explanation of your choice

Format your response as JSON:
{
    "locator_type": "css" or "xpath",
    "locator_value": "selector or xpath expression",
    "confidence": 0.0 to 1.0,
    "alternatives": [{"type": "...", "value": "..."}],
    "explanation": "brief explanation"
}
"""
        
        return prompt
    
    def _parse_detection_result(self, result_text: str, element_description: str) -> Dict[str, Any]:
        """Parse AI response into structured result."""
        import json
        import re
        
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                result_json = json.loads(json_match.group())
                
                # Build locator dict
                locator_type = result_json.get("locator_type", "css")
                locator_value = result_json.get("locator_value", "")
                
                return {
                    "success": True,
                    "locator": {locator_type: locator_value},
                    "locator_type": locator_type,
                    "confidence": float(result_json.get("confidence", 0.7)),
                    "alternatives": result_json.get("alternatives", []),
                    "explanation": result_json.get("explanation", ""),
                    "element_description": element_description
                }
        except Exception as e:
            logger.warning("Failed to parse JSON response", error=str(e))
        
        # Fallback: try to extract CSS selector or XPath
        css_match = re.search(r'[#.]\w+[\w\-#.:\[\]="\']*', result_text)
        xpath_match = re.search(r'//[\w\[\]@="\'/.]*', result_text)
        
        if css_match:
            return {
                "success": True,
                "locator": {"css": css_match.group()},
                "locator_type": "css",
                "confidence": 0.6,
                "explanation": "Extracted CSS selector from text response"
            }
        elif xpath_match:
            return {
                "success": True,
                "locator": {"xpath": xpath_match.group()},
                "locator_type": "xpath",
                "confidence": 0.6,
                "explanation": "Extracted XPath from text response"
            }
        
        # No locator found
        return {
            "success": False,
            "confidence": 0.0,
            "error": "Could not extract locator from AI response",
            "raw_response": result_text
        }
    
    async def compare_elements(
        self,
        screenshot1: bytes,
        screenshot2: bytes,
        element_description: str
    ) -> Dict[str, Any]:
        """
        Compare element appearance across two screenshots.
        
        Args:
            screenshot1: First screenshot
            screenshot2: Second screenshot
            element_description: Description of element to compare
        
        Returns:
            Comparison result with similarity score
        """
        base64_img1 = self._preprocess_image(screenshot1)
        base64_img2 = self._preprocess_image(screenshot2)
        
        prompt = f"""Compare the appearance of the following element in these two screenshots:

Element: {element_description}

Analyze:
1. Is the element present in both screenshots?
2. How similar is the element's appearance (0.0 to 1.0)?
3. What differences do you observe?

Respond in JSON format:
{{
    "present_in_both": true/false,
    "similarity_score": 0.0 to 1.0,
    "differences": ["list", "of", "differences"]
}}
"""
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at comparing visual elements in web UI screenshots."
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img1}"}},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img2}"}}
                        ]
                    }
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            result_text = response.choices[0].message.content
            
            # Parse JSON response
            import json
            import re
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            
            return {"similarity_score": 0.5, "error": "Could not parse response"}
            
        except Exception as e:
            logger.error("Element comparison error", error=str(e))
            return {"similarity_score": 0.0, "error": str(e)}
    
    async def extract_text(self, screenshot: bytes) -> List[str]:
        """
        Extract all visible text from screenshot (OCR).
        
        Args:
            screenshot: Screenshot bytes
        
        Returns:
            List of extracted text strings
        """
        base64_image = self._preprocess_image(screenshot)
        
        prompt = "Extract all visible text from this screenshot. Return as a JSON array of strings."
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                        ]
                    }
                ],
                max_tokens=self.max_tokens,
                temperature=0.0
            )
            
            result_text = response.choices[0].message.content
            
            # Parse JSON array
            import json
            import re
            json_match = re.search(r'\[.*\]', result_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            
            return []
            
        except Exception as e:
            logger.error("Text extraction error", error=str(e))
            return []
    
    async def get_embedding(self, image_bytes: bytes) -> List[float]:
        """
        Get visual embedding for image.
        Note: OpenAI doesn't provide direct image embeddings via API.
        This is a placeholder - use CLIP or similar for production.
        
        Args:
            image_bytes: Image bytes
        
        Returns:
            512-dimensional embedding vector
        """
        # Placeholder: return zero vector
        # In production, use CLIP model or similar
        logger.warning("get_embedding not fully implemented, returning placeholder")
        return [0.0] * 512
