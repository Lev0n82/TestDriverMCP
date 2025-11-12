"""Execution framework for browser automation."""

import asyncio
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import structlog
from PIL import Image

logger = structlog.get_logger()


class BrowserDriver(ABC):
    """Base class for browser drivers."""
    
    @abstractmethod
    async def navigate(self, url: str) -> None:
        """Navigate to URL."""
        pass
    
    @abstractmethod
    async def click(self, locator: Dict[str, str]) -> None:
        """Click element."""
        pass
    
    @abstractmethod
    async def type_text(self, locator: Dict[str, str], text: str) -> None:
        """Type text into element."""
        pass
    
    @abstractmethod
    async def get_screenshot(self) -> Image.Image:
        """Get screenshot of current page."""
        pass
    
    @abstractmethod
    async def close(self) -> None:
        """Close browser."""
        pass


class PlaywrightDriver(BrowserDriver):
    """Playwright-based browser driver."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.browser = None
        self.context = None
        self.page = None
        self._initialized = False
        
        logger.info("Playwright driver initialized")
    
    async def initialize(self) -> None:
        """Initialize Playwright browser."""
        if self._initialized:
            return
        
        try:
            from playwright.async_api import async_playwright
            
            self.playwright = await async_playwright().start()
            
            browser_type = self.config.get("browser", "chromium")
            headless = self.config.get("headless", True)
            
            if browser_type == "chromium":
                self.browser = await self.playwright.chromium.launch(headless=headless)
            elif browser_type == "firefox":
                self.browser = await self.playwright.firefox.launch(headless=headless)
            elif browser_type == "webkit":
                self.browser = await self.playwright.webkit.launch(headless=headless)
            else:
                raise ValueError(f"Unknown browser type: {browser_type}")
            
            self.context = await self.browser.new_context(
                viewport={"width": 1920, "height": 1080}
            )
            self.page = await self.context.new_page()
            
            self._initialized = True
            logger.info("Playwright browser launched", browser=browser_type)
            
        except Exception as e:
            logger.error("Failed to initialize Playwright", error=str(e))
            raise
    
    async def navigate(self, url: str) -> None:
        """Navigate to URL."""
        if not self._initialized:
            await self.initialize()
        
        logger.info("Navigating to URL", url=url)
        await self.page.goto(url, wait_until="networkidle")
    
    async def click(self, locator: Dict[str, str]) -> None:
        """Click element."""
        if not self._initialized:
            await self.initialize()
        
        selector = self._build_selector(locator)
        logger.info("Clicking element", selector=selector)
        
        await self.page.click(selector)
    
    async def type_text(self, locator: Dict[str, str], text: str) -> None:
        """Type text into element."""
        if not self._initialized:
            await self.initialize()
        
        selector = self._build_selector(locator)
        logger.info("Typing text", selector=selector, text_length=len(text))
        
        await self.page.fill(selector, text)
    
    async def get_screenshot(self) -> Image.Image:
        """Get screenshot of current page."""
        if not self._initialized:
            await self.initialize()
        
        screenshot_bytes = await self.page.screenshot(full_page=False)
        
        from io import BytesIO
        return Image.open(BytesIO(screenshot_bytes))
    
    async def close(self) -> None:
        """Close browser."""
        if self.context:
            await self.context.close()
        if self.browser:
            await self.browser.close()
        if hasattr(self, 'playwright'):
            await self.playwright.stop()
        
        self._initialized = False
        logger.info("Playwright browser closed")
    
    def _build_selector(self, locator: Dict[str, str]) -> str:
        """Build Playwright selector from locator dict."""
        if "css" in locator:
            return locator["css"]
        elif "xpath" in locator:
            return f"xpath={locator['xpath']}"
        elif "text" in locator:
            return f"text={locator['text']}"
        elif "id" in locator:
            return f"#{locator['id']}"
        else:
            raise ValueError(f"Unsupported locator type: {locator}")


class SeleniumDriver(BrowserDriver):
    """Selenium-based browser driver."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.driver = None
        self._initialized = False
        
        logger.info("Selenium driver initialized")
    
    async def initialize(self) -> None:
        """Initialize Selenium browser."""
        if self._initialized:
            return
        
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options
            
            options = Options()
            if self.config.get("headless", True):
                options.add_argument("--headless")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("--window-size=1920,1080")
            
            self.driver = webdriver.Chrome(options=options)
            self._initialized = True
            
            logger.info("Selenium browser launched")
            
        except Exception as e:
            logger.error("Failed to initialize Selenium", error=str(e))
            raise
    
    async def navigate(self, url: str) -> None:
        """Navigate to URL."""
        if not self._initialized:
            await self.initialize()
        
        logger.info("Navigating to URL", url=url)
        self.driver.get(url)
        await asyncio.sleep(2)  # Wait for page load
    
    async def click(self, locator: Dict[str, str]) -> None:
        """Click element."""
        if not self._initialized:
            await self.initialize()
        
        from selenium.webdriver.common.by import By
        
        element = self._find_element(locator)
        logger.info("Clicking element", locator=locator)
        element.click()
    
    async def type_text(self, locator: Dict[str, str], text: str) -> None:
        """Type text into element."""
        if not self._initialized:
            await self.initialize()
        
        element = self._find_element(locator)
        logger.info("Typing text", locator=locator, text_length=len(text))
        element.clear()
        element.send_keys(text)
    
    async def get_screenshot(self) -> Image.Image:
        """Get screenshot of current page."""
        if not self._initialized:
            await self.initialize()
        
        screenshot_bytes = self.driver.get_screenshot_as_png()
        
        from io import BytesIO
        return Image.open(BytesIO(screenshot_bytes))
    
    async def close(self) -> None:
        """Close browser."""
        if self.driver:
            self.driver.quit()
        
        self._initialized = False
        logger.info("Selenium browser closed")
    
    def _find_element(self, locator: Dict[str, str]):
        """Find element using Selenium."""
        from selenium.webdriver.common.by import By
        
        if "css" in locator:
            return self.driver.find_element(By.CSS_SELECTOR, locator["css"])
        elif "xpath" in locator:
            return self.driver.find_element(By.XPATH, locator["xpath"])
        elif "id" in locator:
            return self.driver.find_element(By.ID, locator["id"])
        elif "name" in locator:
            return self.driver.find_element(By.NAME, locator["name"])
        else:
            raise ValueError(f"Unsupported locator type: {locator}")


class ExecutionFramework:
    """
    Unified execution framework supporting multiple drivers.
    
    Provides resilient test execution with automatic retry,
    state management, and error recovery.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.driver_type = config.get("driver", "playwright")
        self.driver: Optional[BrowserDriver] = None
        
        logger.info("Execution framework initialized", driver=self.driver_type)
    
    async def initialize(self) -> None:
        """Initialize execution framework."""
        if self.driver_type == "playwright":
            self.driver = PlaywrightDriver(self.config)
        elif self.driver_type == "selenium":
            self.driver = SeleniumDriver(self.config)
        else:
            raise ValueError(f"Unknown driver type: {self.driver_type}")
        
        await self.driver.initialize()
    
    async def execute_step(
        self,
        step: Dict[str, Any],
        retry_count: int = 3
    ) -> Dict[str, Any]:
        """
        Execute a single test step with retry logic.
        
        Args:
            step: Test step to execute
            retry_count: Number of retries on failure
            
        Returns:
            Step execution result
        """
        for attempt in range(retry_count):
            try:
                start_time = time.time()
                
                step_type = step["step_type"]
                
                if step_type == "navigate":
                    await self.driver.navigate(step["input_data"])
                elif step_type == "click":
                    await self.driver.click(step["target_element"])
                elif step_type == "type":
                    await self.driver.type_text(
                        step["target_element"],
                        step["input_data"]
                    )
                elif step_type == "screenshot":
                    screenshot = await self.driver.get_screenshot()
                    # Save screenshot logic here
                else:
                    raise ValueError(f"Unknown step type: {step_type}")
                
                execution_time = (time.time() - start_time) * 1000
                
                return {
                    "step_id": step["step_id"],
                    "success": True,
                    "execution_time_ms": execution_time,
                    "retry_count": attempt
                }
                
            except Exception as e:
                logger.warning(
                    "Step execution failed",
                    step_id=step["step_id"],
                    attempt=attempt + 1,
                    error=str(e)
                )
                
                if attempt == retry_count - 1:
                    return {
                        "step_id": step["step_id"],
                        "success": False,
                        "error_message": str(e),
                        "retry_count": attempt + 1
                    }
                
                await asyncio.sleep(1)  # Wait before retry
    
    async def cleanup(self) -> None:
        """Cleanup resources."""
        if self.driver:
            await self.driver.close()
