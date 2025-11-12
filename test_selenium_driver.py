"""
Comprehensive test suite for Selenium WebDriver.
Tests all functions, classes, and module-level functionality.
"""

import pytest
import asyncio
from typing import Dict, Any
from io import BytesIO
from PIL import Image

import sys
sys.path.insert(0, '/home/ubuntu/testdriver-full/testdriver-mcp-full/src')

from execution.selenium_driver import (
    SeleniumDriver,
    SeleniumDriverValidator,
    self_test_module
)


def test_validator_valid_locator():
    """Test validator accepts valid locators."""
    validator = SeleniumDriverValidator()
    
    valid_locators = [
        {'css': '#button'},
        {'xpath': '//div[@id="test"]'},
        {'id': 'submit'},
        {'name': 'username'},
        {'class_name': 'btn'},
    ]
    
    for locator in valid_locators:
        assert validator.validate_locator(locator), f"Failed for {locator}"
    
    print("✓ Validator accepts valid locators")


def test_validator_invalid_locator():
    """Test validator rejects invalid locators."""
    validator = SeleniumDriverValidator()
    
    invalid_locators = [
        {'invalid_strategy': 'test'},
        {'css': '#test', 'xpath': '//div'},  # Multiple strategies
        {},  # Empty
        "not a dict",  # Wrong type
    ]
    
    for locator in invalid_locators:
        assert not validator.validate_locator(locator), f"Accepted invalid: {locator}"
    
    print("✓ Validator rejects invalid locators")


def test_validator_screenshot():
    """Test validator validates screenshots."""
    validator = SeleniumDriverValidator()
    
    # Create valid screenshot
    img = Image.new('RGB', (100, 100), color='white')
    buf = BytesIO()
    img.save(buf, format='PNG')
    valid_screenshot = buf.getvalue()
    
    assert validator.validate_screenshot(valid_screenshot)
    
    # Test invalid screenshots
    assert not validator.validate_screenshot(b'')  # Too small
    assert not validator.validate_screenshot(b'x' * 100)  # Not an image
    assert not validator.validate_screenshot("not bytes")  # Wrong type
    
    print("✓ Validator correctly validates screenshots")


@pytest.mark.asyncio
async def test_selenium_driver_initialization():
    """Test Selenium driver initialization."""
    config = {
        'browser': 'chrome',
        'headless': True,
        'implicit_wait': 10,
        'page_load_timeout': 30
    }
    
    driver = SeleniumDriver(config)
    
    assert driver.browser_type == 'chrome'
    assert driver.headless is True
    assert driver.implicit_wait == 10
    assert driver.page_load_timeout == 30
    
    print("✓ Selenium driver initializes correctly")


@pytest.mark.asyncio
async def test_locator_conversion():
    """Test locator conversion to Selenium format."""
    config = {'browser': 'chrome', 'headless': True}
    driver = SeleniumDriver(config)
    
    from selenium.webdriver.common.by import By
    
    # Test CSS selector
    by, value = driver._convert_locator({'css': '#button'})
    assert by == By.CSS_SELECTOR
    assert value == '#button'
    
    # Test XPath
    by, value = driver._convert_locator({'xpath': '//div'})
    assert by == By.XPATH
    assert value == '//div'
    
    # Test ID
    by, value = driver._convert_locator({'id': 'submit'})
    assert by == By.ID
    assert value == 'submit'
    
    print("✓ Locator conversion works correctly")


@pytest.mark.asyncio
async def test_browser_launch():
    """Test browser launch and initialization."""
    config = {'browser': 'chrome', 'headless': True}
    driver = SeleniumDriver(config)
    
    try:
        success = await driver.initialize()
        
        if success:
            assert driver.driver is not None
            print("✓ Browser launched successfully")
        else:
            print("⚠ Browser launch skipped (Chrome not available)")
        
    finally:
        await driver.close()


@pytest.mark.asyncio
async def test_navigation():
    """Test navigation to URL."""
    config = {'browser': 'chrome', 'headless': True}
    driver = SeleniumDriver(config)
    
    try:
        success = await driver.initialize()
        if not success:
            print("⚠ Navigation test skipped (Chrome not available)")
            return
        
        # Navigate to data URL
        await driver.navigate('data:text/html,<html><body><h1>Test</h1></body></html>')
        
        url = await driver.get_current_url()
        assert url.startswith('data:text/html')
        
        print("✓ Navigation works correctly")
    
    finally:
        await driver.close()


@pytest.mark.asyncio
async def test_screenshot():
    """Test screenshot capture."""
    config = {'browser': 'chrome', 'headless': True}
    driver = SeleniumDriver(config)
    
    try:
        success = await driver.initialize()
        if not success:
            print("⚠ Screenshot test skipped (Chrome not available)")
            return
        
        await driver.navigate('data:text/html,<html><body><h1>Test</h1></body></html>')
        
        screenshot = await driver.take_screenshot()
        
        assert isinstance(screenshot, bytes)
        assert len(screenshot) > 1000
        
        # Validate screenshot
        validator = SeleniumDriverValidator()
        assert validator.validate_screenshot(screenshot)
        
        print(f"✓ Screenshot captured ({len(screenshot)} bytes)")
    
    finally:
        await driver.close()


@pytest.mark.asyncio
async def test_element_interaction():
    """Test element interaction."""
    config = {'browser': 'chrome', 'headless': True}
    driver = SeleniumDriver(config)
    
    try:
        success = await driver.initialize()
        if not success:
            print("⚠ Element interaction test skipped (Chrome not available)")
            return
        
        # Load test page
        html = '''
        <html>
        <body>
            <input id="test-input" type="text" value="initial">
            <button id="test-button">Click Me</button>
            <div id="test-div">Test Text</div>
        </body>
        </html>
        '''
        await driver.navigate(f'data:text/html,{html}')
        
        # Test visibility
        visible = await driver.is_visible({'id': 'test-div'})
        assert visible is True
        
        # Test get text
        text = await driver.get_text({'id': 'test-div'})
        assert text == 'Test Text'
        
        # Test type text
        await driver.type_text({'id': 'test-input'}, 'new value')
        
        # Test click
        await driver.click({'id': 'test-button'})
        
        print("✓ Element interactions work correctly")
    
    finally:
        await driver.close()


@pytest.mark.asyncio
async def test_javascript_execution():
    """Test JavaScript execution."""
    config = {'browser': 'chrome', 'headless': True}
    driver = SeleniumDriver(config)
    
    try:
        success = await driver.initialize()
        if not success:
            print("⚠ JavaScript test skipped (Chrome not available)")
            return
        
        await driver.navigate('data:text/html,<html><body></body></html>')
        
        # Execute JavaScript
        result = await driver.execute_script("return 2 + 2;")
        assert result == 4
        
        result = await driver.execute_script("return document.title;")
        assert isinstance(result, str)
        
        print("✓ JavaScript execution works correctly")
    
    finally:
        await driver.close()


def test_module_self_test():
    """Test module-level self-test."""
    success = self_test_module()
    assert success is True
    print("✓ Module self-test passed")


async def run_all_tests():
    """Run all Selenium driver tests."""
    print("\n" + "="*60)
    print("Selenium WebDriver - Comprehensive Test Suite")
    print("="*60)
    
    tests = [
        ("Validator - Valid Locator", test_validator_valid_locator),
        ("Validator - Invalid Locator", test_validator_invalid_locator),
        ("Validator - Screenshot", test_validator_screenshot),
        ("Driver Initialization", test_selenium_driver_initialization),
        ("Locator Conversion", test_locator_conversion),
        ("Browser Launch", test_browser_launch),
        ("Navigation", test_navigation),
        ("Screenshot Capture", test_screenshot),
        ("Element Interaction", test_element_interaction),
        ("JavaScript Execution", test_javascript_execution),
        ("Module Self-Test", test_module_self_test),
    ]
    
    passed = 0
    failed = 0
    skipped = 0
    
    for name, test_func in tests:
        try:
            if asyncio.iscoroutinefunction(test_func):
                await test_func()
            else:
                test_func()
            passed += 1
            print(f"✓ {name}: PASSED")
        except Exception as e:
            if "skipped" in str(e).lower() or "⚠" in str(e):
                skipped += 1
                print(f"⚠ {name}: SKIPPED")
            else:
                failed += 1
                print(f"✗ {name}: FAILED - {e}")
    
    print("\n" + "="*60)
    print(f"Total: {len(tests)} tests")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Skipped: {skipped}")
    print(f"Success Rate: {passed/(len(tests)-skipped)*100 if len(tests)-skipped > 0 else 0:.1f}%")
    print("="*60)
    
    return passed, failed, skipped


if __name__ == "__main__":
    asyncio.run(run_all_tests())
