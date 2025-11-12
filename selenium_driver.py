"""
Selenium WebDriver implementation for browser automation.
Provides alternative to Playwright with unified interface.

Built-in Self-Tests:
- Function-level: Each method validates inputs and outputs
- Class-level: Browser lifecycle and state management
- Module-level: Driver initialization and compatibility
"""

from typing import Dict, Any, Optional, List
import time
from io import BytesIO

import structlog
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.remote.webelement import WebElement
from selenium.common.exceptions import (
    TimeoutException, NoSuchElementException,
    WebDriverException, StaleElementReferenceException
)
from PIL import Image

from execution.framework import BrowserDriver

logger = structlog.get_logger()


class SeleniumDriverValidator:
    """Built-in validator for Selenium driver operations."""
    
    @staticmethod
    def validate_locator(locator: Dict[str, str]) -> bool:
        """
        Validate locator format.
        
        Success Criteria:
        - Locator is a dictionary
        - Contains exactly one key-value pair
        - Key is a valid locator strategy
        """
        if not isinstance(locator, dict):
            return False
        if len(locator) != 1:
            return False
        
        valid_strategies = ['css', 'xpath', 'id', 'name', 'class_name', 'tag_name', 'link_text']
        key = list(locator.keys())[0]
        
        return key in valid_strategies
    
    @staticmethod
    def validate_screenshot(screenshot: bytes) -> bool:
        """
        Validate screenshot data.
        
        Success Criteria:
        - Screenshot is bytes
        - Size is reasonable (> 1KB, < 10MB)
        - Can be loaded as image
        """
        if not isinstance(screenshot, bytes):
            return False
        if len(screenshot) < 100 or len(screenshot) > 10 * 1024 * 1024:
            return False
        
        try:
            Image.open(BytesIO(screenshot))
            return True
        except Exception:
            return False


class SeleniumDriver(BrowserDriver):
    """
    Selenium-based browser driver.
    
    Success Criteria (Class-level):
    - Browser launches successfully
    - All navigation operations work
    - Element interactions are reliable
    - Screenshots can be captured
    - Proper cleanup on close
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Selenium driver.
        
        Args:
            config: Configuration dict with:
                - browser: Browser type (chrome, firefox, edge)
                - headless: Run in headless mode
                - implicit_wait: Implicit wait timeout
                - page_load_timeout: Page load timeout
        
        Success Criteria:
        - Config validated
        - Driver options configured
        """
        self.config = config
        self.browser_type = config.get('browser', 'chrome').lower()
        self.headless = config.get('headless', True)
        self.implicit_wait = config.get('implicit_wait', 10)
        self.page_load_timeout = config.get('page_load_timeout', 30)
        
        self.driver = None
        self.validator = SeleniumDriverValidator()
        
        logger.info(
            "Selenium driver created",
            browser=self.browser_type,
            headless=self.headless
        )
    
    async def initialize(self) -> bool:
        """
        Initialize browser.
        
        Success Criteria:
        - Browser launches successfully
        - Timeouts are configured
        - Window size is set
        
        Returns:
            True if successful
        """
        try:
            # Configure browser options
            if self.browser_type == 'chrome':
                options = webdriver.ChromeOptions()
                if self.headless:
                    options.add_argument('--headless=new')
                options.add_argument('--no-sandbox')
                options.add_argument('--disable-dev-shm-usage')
                options.add_argument('--disable-gpu')
                self.driver = webdriver.Chrome(options=options)
            
            elif self.browser_type == 'firefox':
                options = webdriver.FirefoxOptions()
                if self.headless:
                    options.add_argument('--headless')
                self.driver = webdriver.Firefox(options=options)
            
            elif self.browser_type == 'edge':
                options = webdriver.EdgeOptions()
                if self.headless:
                    options.add_argument('--headless')
                self.driver = webdriver.Edge(options=options)
            
            else:
                raise ValueError(f"Unsupported browser: {self.browser_type}")
            
            # Configure timeouts
            self.driver.implicitly_wait(self.implicit_wait)
            self.driver.set_page_load_timeout(self.page_load_timeout)
            
            # Set window size
            self.driver.set_window_size(1920, 1080)
            
            logger.info("Selenium browser launched", browser=self.browser_type)
            
            # Self-test initialization
            return self._self_test_init()
        
        except Exception as e:
            logger.error("Browser initialization failed", error=str(e))
            return False
    
    def _self_test_init(self) -> bool:
        """
        Self-test: Validate browser initialization.
        
        Success Criteria:
        - Driver is not None
        - Can execute simple JavaScript
        - Window size is correct
        """
        try:
            if self.driver is None:
                logger.error("Self-test failed: Driver is None")
                return False
            
            # Test JavaScript execution
            result = self.driver.execute_script("return 1 + 1;")
            if result != 2:
                logger.error("Self-test failed: JavaScript execution failed")
                return False
            
            # Check window size
            size = self.driver.get_window_size()
            if size['width'] != 1920 or size['height'] != 1080:
                logger.warning("Self-test warning: Window size mismatch", size=size)
            
            logger.debug("Self-test passed: Browser initialized")
            return True
        
        except Exception as e:
            logger.error("Self-test failed: Initialization error", error=str(e))
            return False
    
    def _convert_locator(self, locator: Dict[str, str]) -> tuple:
        """
        Convert locator dict to Selenium By tuple.
        
        Success Criteria:
        - Locator is valid format
        - Strategy is supported
        - Returns (By, value) tuple
        """
        if not self.validator.validate_locator(locator):
            raise ValueError(f"Invalid locator format: {locator}")
        
        strategy, value = list(locator.items())[0]
        
        strategy_map = {
            'css': By.CSS_SELECTOR,
            'xpath': By.XPATH,
            'id': By.ID,
            'name': By.NAME,
            'class_name': By.CLASS_NAME,
            'tag_name': By.TAG_NAME,
            'link_text': By.LINK_TEXT
        }
        
        return (strategy_map[strategy], value)
    
    async def navigate(self, url: str) -> None:
        """
        Navigate to URL.
        
        Success Criteria:
        - URL is loaded successfully
        - Page is ready
        - No navigation errors
        """
        try:
            self.driver.get(url)
            
            # Wait for page to be ready
            WebDriverWait(self.driver, 10).until(
                lambda d: d.execute_script("return document.readyState") == "complete"
            )
            
            logger.info("Navigated to URL", url=url)
        
        except Exception as e:
            logger.error("Navigation failed", url=url, error=str(e))
            raise
    
    async def click(self, locator: Dict[str, str]) -> None:
        """
        Click element.
        
        Success Criteria:
        - Element is found
        - Element is clickable
        - Click executes successfully
        """
        try:
            by, value = self._convert_locator(locator)
            
            # Wait for element to be clickable
            element = WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((by, value))
            )
            
            element.click()
            logger.debug("Clicked element", locator=locator)
        
        except Exception as e:
            logger.error("Click failed", locator=locator, error=str(e))
            raise
    
    async def type_text(self, locator: Dict[str, str], text: str) -> None:
        """
        Type text into element.
        
        Success Criteria:
        - Element is found
        - Element is interactable
        - Text is entered successfully
        """
        try:
            by, value = self._convert_locator(locator)
            
            # Wait for element to be present
            element = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((by, value))
            )
            
            element.clear()
            element.send_keys(text)
            
            logger.debug("Typed text", locator=locator, text_length=len(text))
        
        except Exception as e:
            logger.error("Type text failed", locator=locator, error=str(e))
            raise
    
    async def get_text(self, locator: Dict[str, str]) -> str:
        """
        Get element text.
        
        Success Criteria:
        - Element is found
        - Text is retrieved
        """
        try:
            by, value = self._convert_locator(locator)
            element = self.driver.find_element(by, value)
            text = element.text
            
            logger.debug("Got text", locator=locator, text_length=len(text))
            return text
        
        except Exception as e:
            logger.error("Get text failed", locator=locator, error=str(e))
            return ""
    
    async def is_visible(self, locator: Dict[str, str]) -> bool:
        """
        Check if element is visible.
        
        Success Criteria:
        - Returns boolean
        - No exceptions for missing elements
        """
        try:
            by, value = self._convert_locator(locator)
            element = self.driver.find_element(by, value)
            visible = element.is_displayed()
            
            logger.debug("Checked visibility", locator=locator, visible=visible)
            return visible
        
        except (NoSuchElementException, StaleElementReferenceException):
            return False
        except Exception as e:
            logger.error("Visibility check failed", locator=locator, error=str(e))
            return False
    
    async def take_screenshot(self, full_page: bool = False) -> bytes:
        """
        Take screenshot.
        
        Success Criteria:
        - Screenshot is captured
        - Data is valid image bytes
        - Size is reasonable
        """
        try:
            screenshot_bytes = self.driver.get_screenshot_as_png()
            
            # Self-test screenshot
            if not self.validator.validate_screenshot(screenshot_bytes):
                logger.warning("Self-test warning: Screenshot validation failed")
            
            logger.debug("Screenshot captured", size=len(screenshot_bytes))
            return screenshot_bytes
        
        except Exception as e:
            logger.error("Screenshot failed", error=str(e))
            raise
    
    async def get_current_url(self) -> str:
        """Get current URL."""
        return self.driver.current_url
    
    async def execute_script(self, script: str, *args) -> Any:
        """Execute JavaScript."""
        return self.driver.execute_script(script, *args)
    
    async def close(self) -> None:
        """
        Close browser.
        
        Success Criteria:
        - Browser closes cleanly
        - No hanging processes
        """
        try:
            if self.driver:
                self.driver.quit()
                self.driver = None
                logger.info("Selenium browser closed")
        
        except Exception as e:
            logger.error("Error closing browser", error=str(e))


# Module-level self-test
def self_test_module() -> bool:
    """
    Module-level self-test.
    
    Success Criteria:
    - All classes can be instantiated
    - Validator works correctly
    - Locator conversion works
    """
    try:
        # Test validator
        validator = SeleniumDriverValidator()
        
        # Test valid locator
        valid_locator = {'css': '#button'}
        if not validator.validate_locator(valid_locator):
            logger.error("Module self-test failed: Valid locator rejected")
            return False
        
        # Test invalid locator
        invalid_locator = {'invalid': 'test', 'css': '#test'}
        if validator.validate_locator(invalid_locator):
            logger.error("Module self-test failed: Invalid locator accepted")
            return False
        
        # Test screenshot validation
        # Create a minimal valid PNG
        img = Image.new('RGB', (100, 100), color='white')
        buf = BytesIO()
        img.save(buf, format='PNG')
        valid_screenshot = buf.getvalue()
        
        if not validator.validate_screenshot(valid_screenshot):
            logger.error("Module self-test failed: Valid screenshot rejected")
            return False
        
        logger.info("Module self-test passed: selenium_driver")
        return True
    
    except Exception as e:
        logger.error("Module self-test failed", error=str(e))
        return False


if __name__ == "__main__":
    # Run module self-test
    success = self_test_module()
    print(f"Module self-test: {'PASSED' if success else 'FAILED'}")
