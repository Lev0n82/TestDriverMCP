# TestDriver MCP Framework: Low-Level Development System Design Specification

## Part 2: Pluggable Adapter System and Interface Specifications

### 2.1 Adapter Architecture Overview

The pluggable adapter system is the cornerstone of TestDriver's flexibility and extensibility. It enables seamless integration with multiple AI vision models and browser automation frameworks through standardized interfaces and dynamic loading mechanisms.

**Design Principles**:

The adapter system follows the **Strategy Pattern** where each adapter implements a common interface but provides different implementations. Adapters are **dynamically loaded** at runtime based on configuration, enabling hot-swapping without code changes. The system maintains **adapter isolation** where failures in one adapter do not affect others, and **graceful degradation** allows the system to continue operating with reduced capabilities when adapters fail.

**Adapter Lifecycle States**:

Adapters progress through well-defined lifecycle states: **UNINITIALIZED** (adapter class loaded but not configured), **INITIALIZING** (adapter performing setup and validation), **READY** (adapter healthy and available for use), **DEGRADED** (adapter experiencing issues but partially functional), **FAILED** (adapter non-functional, requires restart), and **STOPPED** (adapter cleanly shut down).

### 2.2 Vision Adapter Interface and Implementations

The Vision Adapter interface defines the contract for all computer vision operations required for autonomous testing.

**File**: `src/vision/adapter_base.py`

```python
"""
Abstract Vision Adapter Interface
Defines the contract for all vision model adapters
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np


class ElementType(Enum):
    """Enumeration of UI element types."""
    BUTTON = "button"
    INPUT = "input"
    LINK = "link"
    TEXT = "text"
    IMAGE = "image"
    SELECT = "select"
    CHECKBOX = "checkbox"
    RADIO = "radio"
    UNKNOWN = "unknown"


@dataclass
class BoundingBox:
    """Bounding box coordinates."""
    x: int
    y: int
    width: int
    height: int
    
    @property
    def center(self) -> Tuple[int, int]:
        """Calculate center coordinates."""
        return (self.x + self.width // 2, self.y + self.height // 2)
    
    def to_dict(self) -> Dict[str, int]:
        """Convert to dictionary."""
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height
        }


@dataclass
class ElementLocation:
    """Location and metadata for a UI element."""
    coordinates: Tuple[int, int]
    bounding_box: BoundingBox
    confidence: float
    element_type: ElementType
    attributes: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "coordinates": self.coordinates,
            "bounding_box": self.bounding_box.to_dict(),
            "confidence": self.confidence,
            "element_type": self.element_type.value,
            "attributes": self.attributes
        }


@dataclass
class OCRResult:
    """OCR extraction result."""
    text: str
    bounding_box: BoundingBox
    confidence: float
    language: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "bounding_box": self.bounding_box.to_dict(),
            "confidence": self.confidence,
            "language": self.language
        }


@dataclass
class ScreenComparison:
    """Result of comparing two screenshots."""
    similarity_score: float
    differences: List[BoundingBox]
    semantic_changes: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "similarity_score": self.similarity_score,
            "differences": [d.to_dict() for d in self.differences],
            "semantic_changes": self.semantic_changes
        }


@dataclass
class VisualStateVerification:
    """Result of visual state verification."""
    matches: bool
    confidence: float
    discrepancies: List[str]
    explanation: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "matches": self.matches,
            "confidence": self.confidence,
            "discrepancies": self.discrepancies,
            "explanation": self.explanation
        }


class VisionAdapter(ABC):
    """
    Abstract base class for all vision model adapters.
    
    All vision adapters must implement this interface to ensure
    compatibility with the Autonomous Testing Engine.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the adapter with configuration.
        
        Args:
            config: Adapter-specific configuration dictionary
        """
        self.config = config
        self._initialized = False
    
    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialize the adapter and establish connections.
        
        This method should:
        - Validate configuration
        - Establish API connections
        - Load models (for local adapters)
        - Perform health checks
        
        Raises:
            ConnectionError: If unable to connect to service
            ValueError: If configuration is invalid
        """
        pass
    
    @abstractmethod
    async def shutdown(self) -> None:
        """
        Gracefully shutdown the adapter and release resources.
        """
        pass
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check and return status.
        
        Returns:
            Dictionary with health status information:
            {
                "status": "healthy" | "degraded" | "failed",
                "latency_ms": float,
                "error_rate": float,
                "details": str
            }
        """
        pass
    
    @abstractmethod
    async def describe_screen(
        self, 
        image: bytes, 
        context: Optional[str] = None
    ) -> str:
        """
        Generate a comprehensive textual description of the screen.
        
        Args:
            image: Screenshot as raw bytes (PNG or JPEG format)
            context: Optional context about what to focus on
            
        Returns:
            Natural language description of screen contents
            
        Example:
            "The screen shows a login form with two input fields labeled 
            'Email' and 'Password', and a blue 'Sign In' button below them.
            There is also a 'Forgot Password?' link in the bottom right."
        """
        pass
    
    @abstractmethod
    async def find_element(
        self, 
        image: bytes, 
        prompt: str,
        context: Optional[str] = None
    ) -> ElementLocation:
        """
        Find a specific UI element based on natural language description.
        
        Args:
            image: Screenshot as raw bytes
            prompt: Natural language description of target element
                   (e.g., "the sign up button", "the email input field")
            context: Optional context about the page or workflow
            
        Returns:
            ElementLocation with coordinates, bounding box, and metadata
            
        Raises:
            ElementNotFoundError: If element cannot be located with sufficient confidence
        """
        pass
    
    @abstractmethod
    async def find_multiple_elements(
        self,
        image: bytes,
        prompt: str,
        max_results: int = 10
    ) -> List[ElementLocation]:
        """
        Find multiple matching UI elements.
        
        Args:
            image: Screenshot as raw bytes
            prompt: Natural language description of target elements
            max_results: Maximum number of results to return
            
        Returns:
            List of ElementLocation objects sorted by confidence (descending)
        """
        pass
    
    @abstractmethod
    async def ocr(
        self, 
        image: bytes, 
        region: Optional[BoundingBox] = None,
        language: Optional[str] = None
    ) -> List[OCRResult]:
        """
        Perform optical character recognition on the image.
        
        Args:
            image: Screenshot as raw bytes
            region: Optional bounding box to limit OCR to specific area
            language: Optional language hint (ISO 639-1 code, e.g., "en", "es")
            
        Returns:
            List of OCRResult objects with extracted text and locations
        """
        pass
    
    @abstractmethod
    async def compare_screens(
        self, 
        image1: bytes, 
        image2: bytes,
        ignore_regions: Optional[List[BoundingBox]] = None
    ) -> ScreenComparison:
        """
        Compare two screenshots to detect visual differences.
        
        Args:
            image1: First screenshot as raw bytes
            image2: Second screenshot as raw bytes
            ignore_regions: Optional list of regions to ignore in comparison
            
        Returns:
            ScreenComparison with similarity score and detected differences
        """
        pass
    
    @abstractmethod
    async def verify_visual_state(
        self, 
        image: bytes, 
        expected_state: str
    ) -> VisualStateVerification:
        """
        Verify that the screen matches an expected visual state.
        
        Args:
            image: Screenshot as raw bytes
            expected_state: Natural language description of expected state
                          (e.g., "the user should see a success message")
            
        Returns:
            VisualStateVerification with match status and explanation
        """
        pass
    
    @abstractmethod
    async def generate_embedding(
        self,
        image: bytes,
        region: Optional[BoundingBox] = None
    ) -> np.ndarray:
        """
        Generate a visual embedding vector for the image or region.
        
        This is used for similarity search and drift detection.
        
        Args:
            image: Screenshot as raw bytes
            region: Optional bounding box to generate embedding for specific region
            
        Returns:
            NumPy array containing the embedding vector
        """
        pass
    
    @property
    def adapter_name(self) -> str:
        """Return the adapter name for identification."""
        return self.__class__.__name__
    
    @property
    def adapter_version(self) -> str:
        """Return the adapter version."""
        return "1.0.0"
    
    @property
    def supports_streaming(self) -> bool:
        """Whether this adapter supports streaming responses."""
        return False
```

**File**: `src/vision/openai_adapter.py`

```python
"""
OpenAI GPT-4 Vision Adapter Implementation
"""

import asyncio
from typing import Dict, List, Optional, Any
import numpy as np
from openai import AsyncOpenAI
import structlog

from .adapter_base import (
    VisionAdapter, ElementLocation, OCRResult, ScreenComparison,
    VisualStateVerification, BoundingBox, ElementType
)
from ..utils.errors import ElementNotFoundError, AdapterError

logger = structlog.get_logger(__name__)


class OpenAIVisionAdapter(VisionAdapter):
    """
    OpenAI GPT-4 Vision adapter implementation.
    
    Uses GPT-4V API for computer vision tasks.
    Supports both gpt-4-vision-preview and gpt-4-turbo-vision models.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.client: Optional[AsyncOpenAI] = None
        self.model = config.get("model_name", "gpt-4-vision-preview")
        self.max_tokens = config.get("max_tokens", 1000)
        self.temperature = config.get("temperature", 0.7)
    
    async def initialize(self) -> None:
        """Initialize OpenAI client."""
        logger.info("Initializing OpenAI Vision adapter", model=self.model)
        
        api_key = self.config.get("api_key")
        if not api_key:
            raise ValueError("OpenAI API key not provided in configuration")
        
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=self.config.get("api_base_url"),
            timeout=self.config.get("timeout", 30.0),
            max_retries=self.config.get("max_retries", 3)
        )
        
        # Perform health check
        await self.health_check()
        self._initialized = True
        
        logger.info("OpenAI Vision adapter initialized successfully")
    
    async def shutdown(self) -> None:
        """Shutdown the adapter."""
        if self.client:
            await self.client.close()
        self._initialized = False
        logger.info("OpenAI Vision adapter shut down")
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        try:
            # Simple test request
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=5
            )
            
            return {
                "status": "healthy",
                "model": self.model,
                "details": "API connection successful"
            }
        except Exception as e:
            logger.error("Health check failed", error=str(e))
            return {
                "status": "failed",
                "model": self.model,
                "details": str(e)
            }
    
    async def describe_screen(
        self, 
        image: bytes, 
        context: Optional[str] = None
    ) -> str:
        """Generate screen description using GPT-4V."""
        import base64
        
        # Encode image to base64
        image_b64 = base64.b64encode(image).decode('utf-8')
        
        # Construct prompt
        prompt = "Describe this screenshot in detail, focusing on UI elements, layout, and content."
        if context:
            prompt += f" Context: {context}"
        
        # Call API
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_b64}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=self.max_tokens
        )
        
        return response.choices[0].message.content
    
    async def find_element(
        self, 
        image: bytes, 
        prompt: str,
        context: Optional[str] = None
    ) -> ElementLocation:
        """Find element using GPT-4V with structured output."""
        import base64
        import json
        
        image_b64 = base64.b64encode(image).decode('utf-8')
        
        # Construct detailed prompt for element location
        system_prompt = """You are a UI element locator. Given a screenshot and element description,
        return the element's location in JSON format:
        {
            "found": true/false,
            "coordinates": [x, y],
            "bounding_box": {"x": int, "y": int, "width": int, "height": int},
            "confidence": float (0-1),
            "element_type": "button|input|link|text|image|select|checkbox|radio|unknown",
            "attributes": {"text": "...", "label": "...", etc}
        }
        """
        
        user_prompt = f"Find the element: {prompt}"
        if context:
            user_prompt += f"\nContext: {context}"
        
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_b64}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=500,
            temperature=0.3  # Lower temperature for more deterministic output
        )
        
        # Parse JSON response
        result_text = response.choices[0].message.content
        try:
            result = json.loads(result_text)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code block
            import re
            json_match = re.search(r'```json\n(.*?)\n```', result_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(1))
            else:
                raise AdapterError(f"Failed to parse element location response: {result_text}")
        
        if not result.get("found"):
            raise ElementNotFoundError(f"Element not found: {prompt}")
        
        # Construct ElementLocation
        bbox_data = result["bounding_box"]
        bbox = BoundingBox(
            x=bbox_data["x"],
            y=bbox_data["y"],
            width=bbox_data["width"],
            height=bbox_data["height"]
        )
        
        return ElementLocation(
            coordinates=tuple(result["coordinates"]),
            bounding_box=bbox,
            confidence=result["confidence"],
            element_type=ElementType(result["element_type"]),
            attributes=result.get("attributes", {})
        )
    
    async def find_multiple_elements(
        self,
        image: bytes,
        prompt: str,
        max_results: int = 10
    ) -> List[ElementLocation]:
        """Find multiple matching elements."""
        # Similar implementation to find_element but requests multiple results
        # Implementation details omitted for brevity
        pass
    
    async def ocr(
        self, 
        image: bytes, 
        region: Optional[BoundingBox] = None,
        language: Optional[str] = None
    ) -> List[OCRResult]:
        """Perform OCR using GPT-4V."""
        import base64
        import json
        
        # If region specified, crop image first
        if region:
            from PIL import Image
            import io
            img = Image.open(io.BytesIO(image))
            cropped = img.crop((region.x, region.y, region.x + region.width, region.y + region.height))
            buffer = io.BytesIO()
            cropped.save(buffer, format='PNG')
            image = buffer.getvalue()
        
        image_b64 = base64.b64encode(image).decode('utf-8')
        
        prompt = """Extract all visible text from this image. Return as JSON array:
        [{"text": "...", "bounding_box": {"x": int, "y": int, "width": int, "height": int}, "confidence": float}]
        """
        
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_b64}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=2000
        )
        
        # Parse and convert to OCRResult objects
        result_text = response.choices[0].message.content
        results_data = json.loads(result_text)
        
        return [
            OCRResult(
                text=item["text"],
                bounding_box=BoundingBox(**item["bounding_box"]),
                confidence=item["confidence"],
                language=language
            )
            for item in results_data
        ]
    
    async def compare_screens(
        self, 
        image1: bytes, 
        image2: bytes,
        ignore_regions: Optional[List[BoundingBox]] = None
    ) -> ScreenComparison:
        """Compare two screenshots."""
        # Implementation using GPT-4V to analyze both images
        # Details omitted for brevity
        pass
    
    async def verify_visual_state(
        self, 
        image: bytes, 
        expected_state: str
    ) -> VisualStateVerification:
        """Verify visual state."""
        # Implementation using GPT-4V to verify state
        # Details omitted for brevity
        pass
    
    async def generate_embedding(
        self,
        image: bytes,
        region: Optional[BoundingBox] = None
    ) -> np.ndarray:
        """
        Generate visual embedding.
        
        Note: OpenAI doesn't provide direct embedding API for images.
        This implementation uses CLIP embeddings via a separate service
        or falls back to perceptual hashing.
        """
        # Implementation using CLIP or similar
        # Details omitted for brevity
        pass
```

**File**: `src/vision/local_vlm_adapter.py`

```python
"""
Local Vision Language Model Adapter
Supports Ollama, Hugging Face Transformers, and vLLM backends
"""

from typing import Dict, List, Optional, Any
import numpy as np
import structlog

from .adapter_base import (
    VisionAdapter, ElementLocation, OCRResult, ScreenComparison,
    VisualStateVerification, BoundingBox, ElementType
)

logger = structlog.get_logger(__name__)


class LocalVLMAdapter(VisionAdapter):
    """
    Local Vision Language Model adapter.
    
    Supports multiple backends:
    - Ollama (llava, bakllava models)
    - Hugging Face Transformers (Qwen2-VL, Llama-3.2-Vision)
    - vLLM (for production deployment)
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.backend = config.get("backend", "ollama")  # ollama, huggingface, vllm
        self.model_name = config.get("model_name", "llava:13b")
        self.device = config.get("device", "cuda")  # cuda, cpu, mps
        self.model = None
        self.processor = None
    
    async def initialize(self) -> None:
        """Initialize local VLM based on backend."""
        logger.info("Initializing Local VLM adapter", 
                   backend=self.backend, 
                   model=self.model_name)
        
        if self.backend == "ollama":
            await self._initialize_ollama()
        elif self.backend == "huggingface":
            await self._initialize_huggingface()
        elif self.backend == "vllm":
            await self._initialize_vllm()
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")
        
        self._initialized = True
        logger.info("Local VLM adapter initialized successfully")
    
    async def _initialize_ollama(self) -> None:
        """Initialize Ollama backend."""
        import aiohttp
        
        base_url = self.config.get("base_url", "http://localhost:11434")
        self.ollama_url = f"{base_url}/api/generate"
        
        # Verify Ollama is running and model is available
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{base_url}/api/tags") as response:
                if response.status != 200:
                    raise ConnectionError("Ollama server not accessible")
                
                data = await response.json()
                models = [m["name"] for m in data.get("models", [])]
                
                if self.model_name not in models:
                    logger.warning("Model not found, attempting to pull", 
                                 model=self.model_name)
                    # Trigger model pull
                    await self._pull_ollama_model()
    
    async def _pull_ollama_model(self) -> None:
        """Pull model from Ollama registry."""
        import aiohttp
        
        base_url = self.config.get("base_url", "http://localhost:11434")
        pull_url = f"{base_url}/api/pull"
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                pull_url,
                json={"name": self.model_name}
            ) as response:
                if response.status != 200:
                    raise RuntimeError(f"Failed to pull model: {self.model_name}")
                
                # Stream pull progress
                async for line in response.content:
                    logger.debug("Model pull progress", line=line.decode())
    
    async def _initialize_huggingface(self) -> None:
        """Initialize Hugging Face Transformers backend."""
        from transformers import AutoProcessor, AutoModelForVision2Seq
        import torch
        
        logger.info("Loading Hugging Face model", model=self.model_name)
        
        # Load processor and model
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map=self.device
        )
        
        logger.info("Hugging Face model loaded successfully")
    
    async def _initialize_vllm(self) -> None:
        """Initialize vLLM backend for production deployment."""
        from vllm import LLM, SamplingParams
        
        self.model = LLM(
            model=self.model_name,
            tensor_parallel_size=self.config.get("tensor_parallel_size", 1),
            dtype=self.config.get("dtype", "float16")
        )
        
        self.sampling_params = SamplingParams(
            temperature=self.config.get("temperature", 0.7),
            max_tokens=self.config.get("max_tokens", 1000)
        )
    
    async def shutdown(self) -> None:
        """Shutdown the adapter."""
        if self.model:
            del self.model
            self.model = None
        if self.processor:
            del self.processor
            self.processor = None
        
        # Force garbage collection
        import gc
        gc.collect()
        
        if self.device == "cuda":
            import torch
            torch.cuda.empty_cache()
        
        self._initialized = False
        logger.info("Local VLM adapter shut down")
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        if not self._initialized:
            return {
                "status": "failed",
                "details": "Adapter not initialized"
            }
        
        try:
            # Simple inference test
            test_prompt = "Describe this image briefly."
            # Implementation would test actual inference
            
            return {
                "status": "healthy",
                "backend": self.backend,
                "model": self.model_name,
                "device": self.device
            }
        except Exception as e:
            return {
                "status": "failed",
                "details": str(e)
            }
    
    async def describe_screen(
        self, 
        image: bytes, 
        context: Optional[str] = None
    ) -> str:
        """Generate screen description using local VLM."""
        if self.backend == "ollama":
            return await self._describe_screen_ollama(image, context)
        elif self.backend == "huggingface":
            return await self._describe_screen_huggingface(image, context)
        elif self.backend == "vllm":
            return await self._describe_screen_vllm(image, context)
    
    async def _describe_screen_ollama(
        self,
        image: bytes,
        context: Optional[str]
    ) -> str:
        """Describe screen using Ollama."""
        import aiohttp
        import base64
        
        image_b64 = base64.b64encode(image).decode('utf-8')
        
        prompt = "Describe this screenshot in detail, focusing on UI elements."
        if context:
            prompt += f" Context: {context}"
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.ollama_url,
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "images": [image_b64],
                    "stream": False
                }
            ) as response:
                if response.status != 200:
                    raise RuntimeError(f"Ollama request failed: {response.status}")
                
                data = await response.json()
                return data["response"]
    
    async def _describe_screen_huggingface(
        self,
        image: bytes,
        context: Optional[str]
    ) -> str:
        """Describe screen using Hugging Face."""
        from PIL import Image
        import io
        import torch
        
        # Load image
        img = Image.open(io.BytesIO(image))
        
        # Prepare prompt
        prompt = "Describe this screenshot in detail."
        if context:
            prompt += f" {context}"
        
        # Process inputs
        inputs = self.processor(
            text=prompt,
            images=img,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=500,
                do_sample=True,
                temperature=0.7
            )
        
        # Decode
        description = self.processor.decode(outputs[0], skip_special_tokens=True)
        return description
    
    async def _describe_screen_vllm(
        self,
        image: bytes,
        context: Optional[str]
    ) -> str:
        """Describe screen using vLLM."""
        # vLLM implementation
        # Details omitted for brevity
        pass
    
    async def find_element(
        self, 
        image: bytes, 
        prompt: str,
        context: Optional[str] = None
    ) -> ElementLocation:
        """Find element using local VLM with structured output."""
        # Implementation similar to OpenAI adapter but using local model
        # Details omitted for brevity
        pass
    
    # Additional method implementations follow same pattern...
    
    async def generate_embedding(
        self,
        image: bytes,
        region: Optional[BoundingBox] = None
    ) -> np.ndarray:
        """
        Generate visual embedding using CLIP or similar.
        
        For local deployment, uses open-source CLIP models.
        """
        from transformers import CLIPProcessor, CLIPModel
        from PIL import Image
        import io
        import torch
        
        # Load CLIP model (cached after first load)
        if not hasattr(self, 'clip_model'):
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_model.to(self.device)
        
        # Load and optionally crop image
        img = Image.open(io.BytesIO(image))
        if region:
            img = img.crop((region.x, region.y, region.x + region.width, region.y + region.height))
        
        # Generate embedding
        inputs = self.clip_processor(images=img, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            image_features = self.clip_model.get_image_features(**inputs)
        
        # Convert to numpy and normalize
        embedding = image_features.cpu().numpy()[0]
        embedding = embedding / np.linalg.norm(embedding)
        
        return embedding
```

This completes the vision adapter specifications. The next section will cover browser automation adapters.

### 2.3 Browser Automation Adapter Interface and Implementations

The Browser Driver interface provides a unified abstraction over Selenium and Playwright.

**File**: `src/execution/driver_base.py`

```python
"""
Abstract Browser Driver Interface
Defines the contract for all browser automation adapters
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum


class BrowserType(Enum):
    """Supported browser types."""
    CHROME = "chrome"
    FIREFOX = "firefox"
    WEBKIT = "webkit"
    EDGE = "edge"


class MouseButton(Enum):
    """Mouse button types."""
    LEFT = "left"
    RIGHT = "right"
    MIDDLE = "middle"


@dataclass
class NetworkRequest:
    """Network request information."""
    url: str
    method: str
    headers: Dict[str, str]
    body: Optional[str]
    timestamp: float


@dataclass
class NetworkResponse:
    """Network response information."""
    url: str
    status: int
    headers: Dict[str, str]
    body: Optional[str]
    timestamp: float
    duration_ms: float


@dataclass
class ConsoleLog:
    """Browser console log entry."""
    level: str  # log, warn, error, debug
    message: str
    timestamp: float
    source: Optional[str]


class BrowserDriver(ABC):
    """
    Abstract base class for all browser automation adapters.
    
    Provides a unified interface for Selenium and Playwright.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the driver with configuration."""
        self.config = config
        self._initialized = False
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the browser driver."""
        pass
    
    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the browser driver."""
        pass
    
    @abstractmethod
    async def navigate(self, url: str, wait_until: str = "load") -> None:
        """
        Navigate to a URL.
        
        Args:
            url: Target URL
            wait_until: Wait condition - "load", "domcontentloaded", "networkidle"
        """
        pass
    
    @abstractmethod
    async def click(
        self, 
        coordinates: Tuple[int, int], 
        button: MouseButton = MouseButton.LEFT,
        click_count: int = 1
    ) -> None:
        """
        Click at specific coordinates.
        
        Args:
            coordinates: (x, y) coordinates
            button: Mouse button to use
            click_count: Number of clicks (1=single, 2=double)
        """
        pass
    
    @abstractmethod
    async def type_text(
        self, 
        coordinates: Tuple[int, int], 
        text: str, 
        clear_first: bool = True,
        delay_ms: int = 0
    ) -> None:
        """
        Type text at specific coordinates.
        
        Args:
            coordinates: (x, y) coordinates of input field
            text: Text to type
            clear_first: Clear existing content first
            delay_ms: Delay between keystrokes (for human-like typing)
        """
        pass
    
    @abstractmethod
    async def screenshot(self, full_page: bool = False) -> bytes:
        """
        Capture screenshot.
        
        Args:
            full_page: Capture entire page (scrolling) vs viewport only
            
        Returns:
            Screenshot as PNG bytes
        """
        pass
    
    @abstractmethod
    async def execute_script(self, script: str, *args) -> Any:
        """
        Execute JavaScript in browser context.
        
        Args:
            script: JavaScript code to execute
            *args: Arguments to pass to script
            
        Returns:
            Script return value
        """
        pass
    
    @abstractmethod
    async def get_network_logs(self) -> List[Tuple[NetworkRequest, NetworkResponse]]:
        """
        Retrieve network request/response logs.
        
        Returns:
            List of (request, response) tuples
        """
        pass
    
    @abstractmethod
    async def get_console_logs(self) -> List[ConsoleLog]:
        """
        Retrieve browser console logs.
        
        Returns:
            List of console log entries
        """
        pass
    
    @abstractmethod
    async def set_viewport(self, width: int, height: int) -> None:
        """Set browser viewport dimensions."""
        pass
    
    @abstractmethod
    async def get_cookies(self) -> List[Dict[str, Any]]:
        """Retrieve all cookies for current domain."""
        pass
    
    @abstractmethod
    async def set_cookie(self, cookie: Dict[str, Any]) -> None:
        """Set a cookie in current browser context."""
        pass
    
    @abstractmethod
    async def scroll_to(self, x: int, y: int) -> None:
        """Scroll to specific coordinates."""
        pass
    
    @abstractmethod
    async def wait_for_navigation(self, timeout: int = 30) -> None:
        """Wait for navigation to complete."""
        pass
    
    @abstractmethod
    async def get_page_source(self) -> str:
        """Get current page HTML source."""
        pass
    
    @abstractmethod
    async def get_current_url(self) -> str:
        """Get current page URL."""
        pass
    
    @property
    @abstractmethod
    def supports_network_interception(self) -> bool:
        """Whether this driver supports network request interception."""
        pass
    
    @property
    @abstractmethod
    def supports_video_recording(self) -> bool:
        """Whether this driver supports video recording."""
        pass
```

The Selenium and Playwright adapter implementations would follow similar patterns to the vision adapters, implementing all abstract methods with framework-specific code.

This completes Part 2 of the low-level system design specification covering pluggable adapter systems.
