"""
Real Playwright browser automation driver.
Provides actual browser control for test execution.
"""

from typing import Optional, Dict, Any, List
import asyncio
import base64
from pathlib import Path
from playwright.async_api import async_playwright, Browser, BrowserContext, Page, Playwright
import structlog

from execution.framework import BrowserDriver

logger = structlog.get_logger()

class PlaywrightDriver(BrowserDriver):
    """
    Real Playwright browser automation driver.
    Supports Chromium, Firefox, and WebKit browsers.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Playwright driver.
        
        Args:
            config: Configuration dict with browser settings
        """
        self.config = config or {}
        self.browser_type = self.config.get("browser", "chromium")
        self.headless = self.config.get("headless", True)
        self.slow_mo = self.config.get("slow_mo", 0)
        
        self.playwright: Optional[Playwright] = None
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None
        
        self._initialized = False
        logger.info("Playwright driver created", browser=self.browser_type, headless=self.headless)
    
    async def initialize(self) -> None:
        """Initialize Playwright and launch browser."""
        if self._initialized:
            return
        
        self.playwright = await async_playwright().start()
        
        # Select browser type
        if self.browser_type == "chromium":
            browser_launcher = self.playwright.chromium
        elif self.browser_type == "firefox":
            browser_launcher = self.playwright.firefox
        elif self.browser_type == "webkit":
            browser_launcher = self.playwright.webkit
        else:
            raise ValueError(f"Unsupported browser: {self.browser_type}")
        
        # Launch browser
        self.browser = await browser_launcher.launch(
            headless=self.headless,
            slow_mo=self.slow_mo
        )
        
        # Create browser context
        self.context = await self.browser.new_context(
            viewport={"width": 1920, "height": 1080},
            user_agent="TestDriver/2.0 (Autonomous Testing)"
        )
        
        # Create initial page
        self.page = await self.context.new_page()
        
        self._initialized = True
        logger.info("Playwright browser launched", browser=self.browser_type)
    
    async def navigate(self, url: str, wait_until: str = "networkidle") -> None:
        """
        Navigate to URL.
        
        Args:
            url: URL to navigate to
            wait_until: Wait strategy (load, domcontentloaded, networkidle)
        """
        if not self._initialized:
            await self.initialize()
        
        await self.page.goto(url, wait_until=wait_until)
        logger.info("Navigated to URL", url=url)
    
    async def find_element(self, locator: Dict[str, str]) -> Optional[Any]:
        """
        Find element using locator.
        
        Args:
            locator: Locator dict (e.g., {"css": "#button"})
        
        Returns:
            Playwright Locator object or None
        """
        if not self._initialized:
            await self.initialize()
        
        # Extract locator strategy and value
        if "css" in locator:
            return self.page.locator(locator["css"])
        elif "xpath" in locator:
            return self.page.locator(f"xpath={locator['xpath']}")
        elif "text" in locator:
            return self.page.get_by_text(locator["text"])
        elif "role" in locator:
            return self.page.get_by_role(locator["role"])
        elif "label" in locator:
            return self.page.get_by_label(locator["label"])
        elif "placeholder" in locator:
            return self.page.get_by_placeholder(locator["placeholder"])
        else:
            raise ValueError(f"Unsupported locator strategy: {locator}")
    
    async def click(self, locator: Dict[str, str]) -> None:
        """Click element."""
        element = await self.find_element(locator)
        if element:
            await element.click()
            logger.info("Clicked element", locator=locator)
    
    async def type_text(self, locator: Dict[str, str], text: str) -> None:
        """Type text into element."""
        element = await self.find_element(locator)
        if element:
            await element.fill(text)
            logger.info("Typed text", locator=locator, text_length=len(text))
    
    async def get_text(self, locator: Dict[str, str]) -> Optional[str]:
        """Get element text content."""
        element = await self.find_element(locator)
        if element:
            text = await element.text_content()
            return text
        return None
    
    async def is_visible(self, locator: Dict[str, str]) -> bool:
        """Check if element is visible."""
        try:
            element = await self.find_element(locator)
            if element:
                return await element.is_visible()
        except Exception:
            pass
        return False
    
    async def wait_for_element(
        self,
        locator: Dict[str, str],
        timeout: int = 30000,
        state: str = "visible"
    ) -> bool:
        """
        Wait for element to reach desired state.
        
        Args:
            locator: Element locator
            timeout: Timeout in milliseconds
            state: Desired state (visible, hidden, attached, detached)
        
        Returns:
            True if element reached state, False otherwise
        """
        try:
            element = await self.find_element(locator)
            if element:
                await element.wait_for(state=state, timeout=timeout)
                return True
        except Exception as e:
            logger.warning("Wait for element failed", locator=locator, error=str(e))
        return False
    
    async def take_screenshot(
        self,
        full_page: bool = False,
        element_locator: Optional[Dict[str, str]] = None
    ) -> bytes:
        """
        Take screenshot.
        
        Args:
            full_page: Capture full scrollable page
            element_locator: Capture specific element only
        
        Returns:
            Screenshot bytes (PNG format)
        """
        if not self._initialized:
            await self.initialize()
        
        if element_locator:
            element = await self.find_element(element_locator)
            screenshot = await element.screenshot()
        else:
            screenshot = await self.page.screenshot(full_page=full_page)
        
        logger.info("Screenshot captured", full_page=full_page, has_element=bool(element_locator))
        return screenshot
    
    async def execute_script(self, script: str) -> Any:
        """Execute JavaScript in browser."""
        if not self._initialized:
            await self.initialize()
        
        result = await self.page.evaluate(script)
        return result
    
    async def get_page_source(self) -> str:
        """Get page HTML source."""
        if not self._initialized:
            await self.initialize()
        
        return await self.page.content()
    
    async def get_current_url(self) -> str:
        """Get current page URL."""
        if not self._initialized:
            await self.initialize()
        
        return self.page.url
    
    async def close(self) -> None:
        """Close browser and cleanup."""
        if self.page:
            await self.page.close()
        if self.context:
            await self.context.close()
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()
        
        self._initialized = False
        logger.info("Playwright browser closed")
    
    async def new_page(self) -> Page:
        """Create a new page in the same context."""
        if not self._initialized:
            await self.initialize()
        
        new_page = await self.context.new_page()
        logger.info("New page created")
        return new_page
    
    async def switch_to_page(self, page_index: int = -1) -> None:
        """Switch to a different page."""
        if not self._initialized:
            await self.initialize()
        
        pages = self.context.pages
        if 0 <= page_index < len(pages):
            self.page = pages[page_index]
        elif page_index == -1:
            self.page = pages[-1]
        
        logger.info("Switched to page", index=page_index)
    
    async def wait_for_network_idle(self, timeout: int = 30000) -> None:
        """Wait for network to be idle."""
        if not self._initialized:
            await self.initialize()
        
        await self.page.wait_for_load_state("networkidle", timeout=timeout)
        logger.info("Network idle")
    
    async def set_viewport(self, width: int, height: int) -> None:
        """Set viewport size."""
        if not self._initialized:
            await self.initialize()
        
        await self.page.set_viewport_size({"width": width, "height": height})
        logger.info("Viewport set", width=width, height=height)
    
    async def emulate_mobile(self, device_name: str = "iPhone 12") -> None:
        """Emulate mobile device."""
        if not self._initialized:
            await self.initialize()
        
        device = self.playwright.devices.get(device_name)
        if device:
            await self.context.close()
            self.context = await self.browser.new_context(**device)
            self.page = await self.context.new_page()
            logger.info("Mobile emulation enabled", device=device_name)
    
    async def get_element_attributes(self, locator: Dict[str, str]) -> Dict[str, Any]:
        """Get all attributes of an element."""
        element = await self.find_element(locator)
        if element:
            # Get common attributes
            attributes = {}
            for attr in ["id", "class", "name", "type", "value", "href", "src", "alt", "title"]:
                value = await element.get_attribute(attr)
                if value:
                    attributes[attr] = value
            
            # Get computed styles
            bounding_box = await element.bounding_box()
            if bounding_box:
                attributes["position"] = bounding_box
            
            return attributes
        return {}
    
    async def is_element_stable(self, locator: Dict[str, str], duration_ms: int = 1000) -> bool:
        """
        Check if element is visually stable (not animating/moving).
        
        Args:
            locator: Element locator
            duration_ms: Duration to observe stability
        
        Returns:
            True if element position didn't change during observation
        """
        element = await self.find_element(locator)
        if not element:
            return False
        
        # Get initial position
        initial_box = await element.bounding_box()
        if not initial_box:
            return False
        
        # Wait and check again
        await asyncio.sleep(duration_ms / 1000)
        
        final_box = await element.bounding_box()
        if not final_box:
            return False
        
        # Compare positions
        stable = (
            abs(initial_box["x"] - final_box["x"]) < 1 and
            abs(initial_box["y"] - final_box["y"]) < 1 and
            abs(initial_box["width"] - final_box["width"]) < 1 and
            abs(initial_box["height"] - final_box["height"]) < 1
        )
        
        return stable
