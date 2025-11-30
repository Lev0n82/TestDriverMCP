# 🚀 TestDriver MCP Framework - Setup Guide

## Quick Start Installation (Windows)

This guide will walk you through installing and configuring the TestDriver MCP Framework on Windows.

---

## Prerequisites

Before installing, ensure you have:

✅ **Python 3.11+** installed
   - Download from: https://www.python.org/downloads/
   - Verify: `python --version`

✅ **Git** installed
   - Download from: https://git-scm.com/download/win
   - Verify: `git --version`

✅ **4GB RAM minimum** (8GB recommended)

✅ **Visual Studio Code** (optional but recommended)

---

## Step 1: Repository Setup

The repository has already been cloned to `c:\TestDriverMCP`. Verify the contents:

```powershell
cd c:\TestDriverMCP
ls
```

You should see files like `main.py`, `server.py`, `README.md`, etc.

---

## Step 2: Create Virtual Environment

It's highly recommended to use a virtual environment:

```powershell
# Create virtual environment
python -m venv venv

# Activate it
.\venv\Scripts\Activate.ps1
```

If you get a script execution policy error, run:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

## Step 3: Install Python Dependencies

Install all required packages:

```powershell
# Upgrade pip first
python -m pip install --upgrade pip

# Install all dependencies
pip install -r requirements.txt
```

This will install:
- Web automation frameworks (Playwright, Selenium)
- AI/ML libraries (OpenAI, Anthropic, Ollama)
- Database drivers (SQLAlchemy, PostgreSQL, Qdrant)
- Testing frameworks (pytest)
- Monitoring tools (Prometheus)
- And many more...

---

## Step 4: Install Playwright Browsers

Playwright requires browser binaries:

```powershell
# Install all browsers
playwright install

# Or install specific browsers
playwright install chromium
playwright install firefox
playwright install webkit
```

---

## Step 5: Configure Environment Variables

Create your `.env` file from the template:

```powershell
# Copy the example (already configured with Ollama defaults)
Copy-Item .env.example .env

# Optional: Edit if you want to customize
notepad .env
# or
code .env
```

**Good news:** The default `.env` is pre-configured to use Ollama, so you can use it as-is!

### Minimal Configuration

For a quick start with local development, the default values work out of the box:

```bash
# Database (SQLite for local dev) - DEFAULT
DATABASE_URL=sqlite+aiosqlite:///./testdriver.db

# Vision API (Ollama - NO API KEY NEEDED) - DEFAULT
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llava:13b
DEFAULT_VISION_PROVIDER=ollama

# Execution Framework
DEFAULT_BROWSER_DRIVER=playwright
HEADLESS_MODE=false

# Vector Store (in-memory for quick start)
QDRANT_URL=:memory:
```

**Note:** The default configuration uses Ollama (local VLM) so you don't need any API keys to get started!

### Setting Up Ollama (Default - Recommended)

Ollama is the default vision provider and requires NO API keys:

**Step 1: Install Ollama**
```powershell
# Download and install from https://ollama.com/download
# Or use winget:
winget install Ollama.Ollama
```

**Step 2: Pull a vision model**
```powershell
# Default model (recommended)
ollama pull llava:13b

# Or lighter/faster alternatives:
ollama pull llava:7b    # Faster, uses less RAM
ollama pull llava       # Latest version
```

**Step 3: Verify Ollama is running**
```powershell
curl http://localhost:11434
```

The `.env` file is already configured for Ollama by default!

### Alternative: Use OpenAI (Requires API Key)

If you prefer OpenAI GPT-4V instead:

1. Get API key from https://platform.openai.com/api-keys
2. Update `.env`:
   ```bash
   # Comment out Ollama
   # OLLAMA_BASE_URL=http://localhost:11434
   # OLLAMA_MODEL=llava:13b
   
   # Enable OpenAI
   OPENAI_API_KEY=your_openai_api_key_here
   OPENAI_MODEL=gpt-4-vision-preview
   DEFAULT_VISION_PROVIDER=openai
   ```

---

## Step 6: Initialize Database

Initialize the database schema:

```powershell
# Run database initialization
python -c "from database import init_database; import asyncio; asyncio.run(init_database())"
```

This creates the necessary tables for:
- Test executions
- Healing events
- Learning data
- Element memories

---

## Step 7: Verify Installation

Run a quick verification:

```powershell
# Check Python version
python --version

# Check installed packages
pip list | Select-String "playwright|selenium|openai|sqlalchemy|qdrant"

# Check Playwright browsers
playwright --version
```

---

## Step 8: Run Tests

Verify everything is working:

```powershell
# Run all tests
pytest -v

# Run specific test file
pytest test_integration.py -v

# Run with coverage
pytest --cov=. --cov-report=html
```

---

## Step 9: Start the MCP Server

Launch the TestDriver MCP server:

```powershell
# Start the server
python server.py
```

The server will start on `http://localhost:8000`

Check health:
```powershell
# In another terminal
curl http://localhost:8000/health
```

Or open in browser: http://localhost:8000/health

---

## Optional: Docker Setup

If you prefer using Docker:

### Prerequisites
- Docker Desktop for Windows
- Docker Compose

### Start All Services

```powershell
# Start all services (database, vector store, metrics, etc.)
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

This starts:
- ✅ TestDriver MCP Server
- ✅ PostgreSQL database
- ✅ Qdrant vector database
- ✅ Prometheus metrics
- ✅ Grafana dashboards

Access points:
- MCP Server: http://localhost:8000
- Grafana: http://localhost:3000 (admin/admin)
- Prometheus: http://localhost:9090
- Qdrant: http://localhost:6333

---

## Troubleshooting

### Issue: Import errors or module not found

**Solution:**
```powershell
# Ensure virtual environment is activated
.\venv\Scripts\Activate.ps1

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### Issue: Playwright browsers not found

**Solution:**
```powershell
# Reinstall browsers
playwright install --force
```

### Issue: Database connection errors

**Solution:**
```powershell
# For SQLite (local dev), ensure path exists
mkdir -p data

# Update .env
DATABASE_URL=sqlite+aiosqlite:///./data/testdriver.db
```

### Issue: OpenAI API errors

**Solutions:**
1. Verify API key is correct in `.env`
2. Check API quota: https://platform.openai.com/account/usage
3. Try alternative provider (Ollama)

### Issue: Port already in use

**Solution:**
```powershell
# Find process using port 8000
netstat -ano | findstr :8000

# Kill the process (replace PID)
taskkill /PID <PID> /F

# Or use a different port in .env
MCP_PORT=8080
```

---

## Next Steps

Now that TestDriver MCP is installed:

1. **Read the README.md** for comprehensive documentation
2. **Explore example tests** in the test files
3. **Try the sample workflows** in the documentation
4. **Configure additional features**:
   - Self-healing strategies
   - Cross-layer validation
   - Performance testing
   - Security scanning

---

## Quick Reference Commands

```powershell
# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Run server
python server.py

# Run tests
pytest -v

# Install new package
pip install <package>

# Update dependencies
pip install -r requirements.txt --upgrade

# Check service health
curl http://localhost:8000/health

# View logs
Get-Content ./logs/testdriver.log -Tail 50 -Wait
```

---

## Getting Help

- 📖 Full documentation: See `README.md`
- 🐛 Issues: https://github.com/Lev0n82/TestDriverMCP/issues
- 💬 Discussions: Check GitHub Discussions

---

## Configuration Summary

✅ Repository cloned
✅ Virtual environment created
✅ Python dependencies installed
✅ Playwright browsers installed
✅ Environment configured (`.env`)
✅ Database initialized
✅ Tests passing
✅ Server running

**You're ready to use TestDriver MCP Framework! 🎉**
