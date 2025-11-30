# 🚀 Quick Start - TestDriver MCP with Ollama

## Get Started in 3 Steps (No API Keys Required!)

TestDriver MCP is now pre-configured to use **Ollama** - a local AI vision model that runs on your machine. No cloud services, no API keys, completely free!

---

## Step 1: Install Dependencies

```powershell
# Navigate to project directory
cd c:\TestDriverMCP

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Verify installation
pip list | Select-String "playwright|selenium|ollama"
```

---

## Step 2: Set Up Ollama (Default Vision Provider)

Run the automated setup script:

```powershell
.\setup-ollama.ps1
```

This will:
- ✅ Check if Ollama is installed
- ✅ Install Ollama if needed (via winget)
- ✅ Pull the LLaVA vision model (llava:13b)
- ✅ Verify everything is working

**Manual installation (alternative):**
```powershell
# Install Ollama
winget install Ollama.Ollama

# Pull vision model
ollama pull llava:13b

# Verify it's running
curl http://localhost:11434
```

---

## Step 3: Start TestDriver Server

```powershell
.\start-server.ps1
```

The server will start on **http://localhost:8000**

---

## ✅ That's It!

Your TestDriver MCP is now running with:
- ✅ **Ollama** for AI vision (no API keys!)
- ✅ **Playwright** & **Selenium** for browser automation
- ✅ **SQLite** for local database
- ✅ **Qdrant** in-memory for vector storage

---

## 🎯 Quick Test

1. **Check server health:**
   ```powershell
   curl http://localhost:8000/health
   ```

2. **Run tests:**
   ```powershell
   pytest -v
   ```

3. **Try a simple test:**
   ```powershell
   python test_integration.py
   ```

---

## 📝 Configuration Overview

The `.env` file is pre-configured with these defaults:

```bash
# Vision Provider (NO API KEY NEEDED!)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llava:13b
DEFAULT_VISION_PROVIDER=ollama

# Database
DATABASE_URL=sqlite+aiosqlite:///./testdriver.db

# Browser
DEFAULT_BROWSER_DRIVER=playwright
HEADLESS_MODE=false

# Vector Store
QDRANT_URL=:memory:
```

---

## 🔄 Want to Use OpenAI Instead?

Edit `.env` and change:

```bash
# Comment out Ollama
# OLLAMA_BASE_URL=http://localhost:11434
# OLLAMA_MODEL=llava:13b

# Enable OpenAI
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4-vision-preview
DEFAULT_VISION_PROVIDER=openai
```

---

## 🆘 Troubleshooting

### Ollama not found
```powershell
winget install Ollama.Ollama
# Restart terminal, then run: ollama pull llava:13b
```

### Model not downloaded
```powershell
ollama pull llava:13b
```

### Ollama service not running
```powershell
ollama serve
```

### Virtual environment not activated
```powershell
.\venv\Scripts\Activate.ps1
```

---

## 📚 Documentation

- **Full Setup Guide**: `SETUP.md`
- **Complete Documentation**: `README.md`
- **Installation Summary**: `INSTALLATION_COMPLETE.md`

---

## 🎉 Key Benefits of Using Ollama

✅ **No API costs** - Runs completely locally  
✅ **No API keys** - No cloud service setup needed  
✅ **Privacy** - Your data never leaves your machine  
✅ **Fast** - Low latency, no network calls  
✅ **Offline** - Works without internet connection  

---

**Happy Testing with TestDriver MCP! 🚀**
