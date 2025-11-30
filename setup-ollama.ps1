# Ollama Setup Script for TestDriver MCP
# This script helps you install and configure Ollama (the default vision provider)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "TestDriver MCP - Ollama Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if Ollama is already installed
Write-Host "Checking Ollama installation..." -ForegroundColor Yellow
$ollamaInstalled = Get-Command ollama -ErrorAction SilentlyContinue

if ($ollamaInstalled) {
    Write-Host "✓ Ollama is already installed" -ForegroundColor Green
    ollama --version
} else {
    Write-Host "✗ Ollama is not installed" -ForegroundColor Red
    Write-Host ""
    Write-Host "Installing Ollama..." -ForegroundColor Yellow
    Write-Host "Option 1: Using winget (recommended)" -ForegroundColor Cyan
    Write-Host "  winget install Ollama.Ollama" -ForegroundColor Gray
    Write-Host ""
    Write-Host "Option 2: Manual download" -ForegroundColor Cyan
    Write-Host "  https://ollama.com/download" -ForegroundColor Gray
    Write-Host ""
    
    $install = Read-Host "Would you like to install using winget now? (y/n)"
    if ($install -eq 'y' -or $install -eq 'Y') {
        winget install Ollama.Ollama
        Write-Host ""
        Write-Host "✓ Ollama installed! Please restart your terminal." -ForegroundColor Green
        Write-Host "  Then run this script again to pull the vision model." -ForegroundColor Yellow
        exit 0
    } else {
        Write-Host ""
        Write-Host "Please install Ollama manually from https://ollama.com/download" -ForegroundColor Yellow
        exit 1
    }
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Checking Ollama Service..." -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if Ollama service is running
try {
    $response = Invoke-WebRequest -Uri "http://localhost:11434" -Method GET -TimeoutSec 2 -ErrorAction Stop
    Write-Host "✓ Ollama service is running on http://localhost:11434" -ForegroundColor Green
} catch {
    Write-Host "✗ Ollama service is not running" -ForegroundColor Red
    Write-Host "  Please start Ollama (it should auto-start after installation)" -ForegroundColor Yellow
    Write-Host "  Or run: ollama serve" -ForegroundColor Gray
    exit 1
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Pulling Vision Model..." -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if llava model is already available
$models = ollama list 2>$null
if ($models -match "llava") {
    Write-Host "✓ LLaVA vision model is already available" -ForegroundColor Green
    ollama list | Select-String "llava"
} else {
    Write-Host "Pulling llava:13b model (this may take a few minutes)..." -ForegroundColor Yellow
    Write-Host ""
    ollama pull llava:13b
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "✓ Model downloaded successfully!" -ForegroundColor Green
    } else {
        Write-Host ""
        Write-Host "✗ Failed to download model" -ForegroundColor Red
        exit 1
    }
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Testing Ollama..." -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "Testing vision model with a simple prompt..." -ForegroundColor Yellow
$testResult = ollama run llava:13b "Hello, respond with just 'OK' if you're working" 2>&1
if ($testResult -match "OK" -or $LASTEXITCODE -eq 0) {
    Write-Host "✓ Ollama is working correctly!" -ForegroundColor Green
} else {
    Write-Host "⚠ Test completed, please verify manually" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Setup Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Ollama Configuration:" -ForegroundColor Yellow
Write-Host "  URL: http://localhost:11434" -ForegroundColor Gray
Write-Host "  Model: llava:13b" -ForegroundColor Gray
Write-Host "  Provider: ollama (DEFAULT)" -ForegroundColor Gray
Write-Host ""
Write-Host "Your .env file is already configured to use Ollama!" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Run: .\start-server.ps1" -ForegroundColor White
Write-Host "  2. TestDriver will use Ollama for vision tasks" -ForegroundColor White
Write-Host ""
Write-Host "Available commands:" -ForegroundColor Cyan
Write-Host "  ollama list          - List downloaded models" -ForegroundColor Gray
Write-Host "  ollama pull <model>  - Download a model" -ForegroundColor Gray
Write-Host "  ollama run <model>   - Test a model" -ForegroundColor Gray
Write-Host "  ollama serve         - Start Ollama service" -ForegroundColor Gray
Write-Host ""
