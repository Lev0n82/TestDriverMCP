# TestDriver MCP Framework - Quick Start Script
# This script helps you verify the installation and start using TestDriver

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "TestDriver MCP Framework - Quick Start" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if virtual environment is activated
if ($env:VIRTUAL_ENV) {
    Write-Host "✓ Virtual environment is activated" -ForegroundColor Green
} else {
    Write-Host "○ Activating virtual environment..." -ForegroundColor Yellow
    .\venv\Scripts\Activate.ps1
}

Write-Host ""
Write-Host "Checking installation..." -ForegroundColor Cyan
Write-Host ""

# Check Python version
Write-Host "Python Version:" -ForegroundColor Yellow
python --version

# Check key packages
Write-Host ""
Write-Host "Key Packages Installed:" -ForegroundColor Yellow
$packages = @("playwright", "selenium", "openai", "anthropic", "sqlalchemy", "qdrant-client", "pytest")
foreach ($pkg in $packages) {
    $versionLine = pip show $pkg 2>$null | Select-String "Version:"
    if ($versionLine) {
        $version = $versionLine.ToString().Split(":")[1].Trim()
        Write-Host "  ✓ $pkg ($version)" -ForegroundColor Green
    } else {
        Write-Host "  ✗ $pkg (not installed)" -ForegroundColor Red
    }
}

# Check Playwright browsers
Write-Host ""
Write-Host "Playwright Browsers:" -ForegroundColor Yellow
$playwrightPath = "$env:USERPROFILE\SnapQA\ms-playwright"
if (Test-Path $playwrightPath) {
    $browsers = Get-ChildItem $playwrightPath -Directory | Where-Object { $_.Name -match "chromium|firefox|webkit" }
    foreach ($browser in $browsers) {
        Write-Host "  ✓ $($browser.Name)" -ForegroundColor Green
    }
} else {
    Write-Host "  ○ Browsers not found at $playwrightPath" -ForegroundColor Yellow
}

# Check configuration file
Write-Host ""
Write-Host "Configuration:" -ForegroundColor Yellow
if (Test-Path ".env") {
    Write-Host "  ✓ .env file exists" -ForegroundColor Green
    Write-Host "  ○ Remember to update your API keys in .env" -ForegroundColor Yellow
} else {
    Write-Host "  ✗ .env file not found" -ForegroundColor Red
    Write-Host "  ○ Run: Copy-Item .env.example .env" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Next Steps:" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "1. Edit .env file with your configuration:" -ForegroundColor White
Write-Host "   notepad .env" -ForegroundColor Gray
Write-Host ""
Write-Host "2. Add your OpenAI API key (or configure Ollama):" -ForegroundColor White
Write-Host "   OPENAI_API_KEY=your_key_here" -ForegroundColor Gray
Write-Host ""
Write-Host "3. Run the server:" -ForegroundColor White
Write-Host "   python server.py" -ForegroundColor Gray
Write-Host ""
Write-Host "4. Run tests to verify everything works:" -ForegroundColor White
Write-Host "   pytest -v" -ForegroundColor Gray
Write-Host ""
Write-Host "5. Read the documentation:" -ForegroundColor White
Write-Host "   - README.md (full documentation)" -ForegroundColor Gray
Write-Host "   - SETUP.md (setup guide)" -ForegroundColor Gray
Write-Host ""
Write-Host "For help, see: https://github.com/Lev0n82/TestDriverMCP" -ForegroundColor Cyan
Write-Host ""
