# TestDriver MCP Server Startup Script
Write-Host 'Starting TestDriver MCP Server...' -ForegroundColor Cyan
Write-Host ''
if (Test-Path '.\venv\Scripts\Activate.ps1') {
    . .\venv\Scripts\Activate.ps1
    Write-Host 'Virtual environment activated' -ForegroundColor Green
} else {
    Write-Host 'Virtual environment not found!' -ForegroundColor Red
    exit 1
}
Write-Host ''
Write-Host 'Starting server on http://localhost:8000...' -ForegroundColor Cyan
python run_server.py
