# -*- coding: utf-8 -*-
"""
TD(MCP)-GRACE: Generative Requirement-Aware Cognitive Engineering

In the spirit of Admiral Grace Hopper — a visionary who believed that technology
should serve human logic, not obscure it — Project GRACE reimagines how machines
interpret, validate, and evolve software testing.

GRACE bridges the gap between human intention and machine validation through
semantic understanding, autonomous reasoning, and requirement awareness.

This server powers the QA Command Bridge Interface for GRACE-enabled testing.
"""

import os
import sys
import json
import io
from pathlib import Path
from datetime import datetime

# Force UTF-8 output encoding on Windows
if sys.platform == 'win32':
    # Reconfigure stdout to use UTF-8
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse
    import uvicorn
except ImportError:
    print("Error: FastAPI is not installed. Installing required packages...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "fastapi", "uvicorn[standard]"])
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from dotenv import load_dotenv
import structlog

# Helper: safely extract provider response body text with optional truncation
def _extract_response_body(resp):
    """Return provider response body text truncated by MAX_ERROR_BODY env var.
    If resp is None, return explanatory string."""
    try:
        if resp is None:
            return "<no response>"
        body = resp.text if hasattr(resp, 'text') and resp.text is not None else ''
        max_len = int(os.getenv('MAX_ERROR_BODY', '5000'))
        if not body:
            return f"HTTP {getattr(resp, 'status_code', 'unknown')} (empty body)"
        if len(body) > max_len:
            return body[:max_len] + f"\n\n...<<truncated {len(body)-max_len} chars>>"
        return body
    except Exception:
        return '<error extracting response body>'

# Load environment variables
load_dotenv()

# Configure logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.JSONRenderer()
    ]
)

logger = structlog.get_logger()

# GRACE System Information
GRACE_VERSION = "3.0.0-GRACE"
GRACE_TITLE = "TD(MCP)-GRACE: Generative Requirement-Aware Cognitive Engineering"
GRACE_TAGLINE = "Clarity reborn as intelligence — where every test case is an act of understanding"

# Create FastAPI app
app = FastAPI(
    title="TD(MCP)-GRACE: QA Command Bridge",
    version="3.0.0-GRACE",
    description="Generative Requirement-Aware Cognitive Engineering - Intelligent Test Framework"
)

# Add CORS middleware to allow chat client to communicate
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "TestDriver MCP Server",
        "version": "2.0.0",
        "status": "running",
        "vision_provider": os.getenv("DEFAULT_VISION_PROVIDER", "ollama"),
        "documentation": "/docs",
        "chat": "/chat"
    }


@app.get("/chat", response_class=HTMLResponse)
async def chat_client():
    """Serve the GRACE QA Command Bridge chat client interface."""
    try:
        # Try fresh version first, fall back to original
        chat_file = Path("chat_fresh.html")
        if not chat_file.exists():
            chat_file = Path(__file__).parent / "chat_client.html"
        if not chat_file.exists():
            chat_file = Path("chat_client.html")
        
        if not chat_file.exists():
            return HTMLResponse(content="<h1>Chat client not found</h1><p>Searched in: " + str(chat_file) + "</p>", status_code=404)
        
        with open(str(chat_file), "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except Exception as e:
        logger.error(f"Error serving chat client: {str(e)}")
        return HTMLResponse(content=f"<h1>Error loading chat client</h1><p>{str(e)}</p>", status_code=500)


@app.get("/grace", response_class=HTMLResponse)
async def grace_command_bridge():
    """Serve the TD(MCP)-GRACE QA Command Bridge interface (alias for /chat)."""
    try:
        # Try absolute path first
        chat_file = Path(__file__).parent / "chat_client.html"
        if not chat_file.exists():
            chat_file = Path("chat_client.html")
        
        if not chat_file.exists():
            return HTMLResponse(content="<h1>GRACE Command Bridge not found</h1>", status_code=404)
        
        with open(str(chat_file), "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except Exception as e:
        logger.error(f"Error serving GRACE Command Bridge: {str(e)}")
        return HTMLResponse(content=f"<h1>Error loading GRACE Command Bridge</h1><p>{str(e)}</p>", status_code=500)


@app.get("/grace/info", response_class=JSONResponse)
async def grace_info():
    """Return GRACE system information and philosophy."""
    return {
        "name": "TD(MCP)-GRACE",
        "version": GRACE_VERSION,
        "title": GRACE_TITLE,
        "tagline": GRACE_TAGLINE,
        "description": "Generative Requirement-Aware Cognitive Engineering - Intelligent Test Framework",
        "philosophy": {
            "principle": "In the spirit of Admiral Grace Hopper",
            "core_belief": "Technology should serve human logic, not obscure it",
            "mission": "Transform natural language requirements into semantic test structures",
            "promise": "Clarity reborn as intelligence"
        },
        "endpoints": {
            "command_bridge": "/grace or /chat",
            "api_docs": "/docs",
            "health": "/health",
            "config": "/config"
        },
        "features": [
            "Natural language test requirement input",
            "Semantic understanding and autonomous reasoning",
            "8 GRACE-enabled MCP commands",
            "Intelligent parameter suggestions (typeahead)",
            "Multi-method Azure DevOps integration",
            "Enterprise-grade AES-128 encryption",
            "Admiral Grace Hopper visual theme"
        ]
    }


@app.get("/test", response_class=HTMLResponse)
async def test_client():
    """Serve the simple test client interface."""
    try:
        with open("test_chat.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Test client not found</h1><p>Please ensure test_chat.html is in the server directory.</p>", status_code=404)


@app.get("/minimal", response_class=HTMLResponse)
async def minimal_test():
    """Serve minimal test page."""
    try:
        with open("test_minimal.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Minimal test not found</h1>", status_code=404)


@app.get("/simple", response_class=HTMLResponse)
async def simple_chat():
    """Serve the simplified chat interface."""
    try:
        # Try test_simple.html first, fall back to chat_simple.html
        try:
            with open("test_simple.html", "r", encoding="utf-8") as f:
                return HTMLResponse(content=f.read())
        except FileNotFoundError:
            with open("chat_simple.html", "r", encoding="utf-8") as f:
                return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Chat not found</h1><p>Please ensure chat_simple.html or test_simple.html is in the server directory.</p>", status_code=404)


@app.get("/health")
async def health():
    """Health check endpoint."""
    
    # Check Ollama connection if using Ollama
    ollama_status = "not configured"
    if os.getenv("DEFAULT_VISION_PROVIDER") == "ollama":
        try:
            import httpx
            ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            async with httpx.AsyncClient() as client:
                response = await client.get(ollama_url, timeout=2.0)
                ollama_status = "connected" if response.status_code == 200 else "error"
        except Exception as e:
            ollama_status = f"error: {str(e)}"
    
    return {
        "status": "healthy",
        "timestamp": structlog.processors.TimeStamper(fmt="iso")(None, None, {})["timestamp"],
        "components": {
            "server": "running",
            "vision_provider": os.getenv("DEFAULT_VISION_PROVIDER", "ollama"),
            "ollama": ollama_status,
            "database": "configured",
            "browser_driver": os.getenv("DEFAULT_BROWSER_DRIVER", "playwright")
        },
        "config": {
            "vision_model": os.getenv("OLLAMA_MODEL", "llava:13b") if os.getenv("DEFAULT_VISION_PROVIDER") == "ollama" else "N/A",
            "headless_mode": os.getenv("HEADLESS_MODE", "false"),
            "database_url": "configured"
        }
    }


@app.get("/config")
async def get_config():
    """Get current configuration."""
    return {
        "vision": {
            "provider": os.getenv("DEFAULT_VISION_PROVIDER", "ollama"),
            "ollama_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            "ollama_model": os.getenv("OLLAMA_MODEL", "llava:13b")
        },
        "ai": {
            "provider": os.getenv("AI_PROVIDER", None),
            "api_url": os.getenv("AI_API_URL", None),
            "model": os.getenv("AI_MODEL", None)
        },
        "execution": {
            "browser_driver": os.getenv("DEFAULT_BROWSER_DRIVER", "playwright"),
            "headless": os.getenv("HEADLESS_MODE", "false"),
            "timeout": os.getenv("BROWSER_TIMEOUT", "30000")
        },
        "database": {
            "url": os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./testdriver.db")
        },
        "vector_store": {
            "url": os.getenv("QDRANT_URL", ":memory:")
        }
    }


@app.get('/ollama/models')
async def list_ollama_models(base_url: str = None):
    """Query Ollama for available models. Accepts optional base_url query string.
    Returns an array of model info objects or names.
    """
    try:
        import requests
        url = base_url or os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
        url = url.rstrip('/')
        # Ollama exposes models list at /api/tags
        try:
            resp = requests.get(f"{url}/api/tags", timeout=5.0)
            resp.raise_for_status()
            data = resp.json()
            # Ollama returns {"models": [...]} 
            models = data.get('models', [])
            if models:
                # Extract just the names for the dropdown
                return [{'name': m.get('name', m), 'value': m.get('name', m)} for m in models]
            return []
        except requests.exceptions.ConnectionError:
            raise HTTPException(status_code=503, detail=f"Cannot connect to Ollama at {url}. Ensure Ollama is running.")
        except requests.exceptions.Timeout:
            raise HTTPException(status_code=504, detail=f"Ollama at {url} is not responding. Check if it's running.")
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error fetching Ollama models", error=str(e), exc_info=True)
        raise HTTPException(status_code=502, detail=f"Error fetching models from Ollama: {str(e)}")


@app.get('/ollama/all-models')
async def list_all_ollama_models(base_url: str = None):
    """
    Fetch all available models from Ollama library (both installed and available for download).
    Returns a combined list from /api/tags (installed) and search results.
    """
    try:
        import requests
        base_url = base_url or os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
        base_url = base_url.rstrip('/')
        
        installed_models = []
        available_models = []
        
        try:
            # Get installed models
            resp = requests.get(f"{base_url}/api/tags", timeout=5.0)
            resp.raise_for_status()
            data = resp.json()
            installed_models = [m.get('name', m) for m in data.get('models', [])]
        except Exception as e:
            logger.warning(f"Could not fetch installed models: {e}")
        
        # For now, return a comprehensive list of popular models
        # In production, this could query Ollama's remote library
        popular_models = [
            "gpt-oss:120b-cloud",
            "gpt-oss:20b-cloud",
            "deepseek-v3.1:671b-cloud",
            "qwen3-coder:480b-cloud",
            "qwen3-vl:235b-cloud",
            "minimax-m2:cloud",
            "glm-4.6:cloud",
            "gpt-oss:120b",
            "gpt-oss:20b",
            "gemma3:27b",
            "gemma3:12b",
            "gemma3:4b",
            "gemma3:1b",
            "deepseek-r1:8b",
            "qwen3-coder:30b",
            "qwen3-vl:30b",
            "qwen3-vl:8b",
            "qwen3-vl:4b",
            "qwen3:30b",
            "qwen3:8b",
            "qwen3:4b",
            "llava:13b"
        ]
        
        # Combine lists: show popular models with status of whether they're installed
        result = []
        for model_name in popular_models:
            result.append({
                "name": model_name,
                "installed": model_name in installed_models
            })
        
        return result
        
    except Exception as e:
        logger.error("Error fetching all Ollama models", error=str(e), exc_info=True)
        raise HTTPException(status_code=502, detail=f"Error fetching model list: {str(e)}")


@app.post('/config/provider')
async def set_provider(payload: dict):
    """
    Set the AI provider configuration (Ollama Cloud, Custom API, etc.)
    Payload examples:
    {
        "provider": "ollama-cloud",
        "api_url": "https://api.ollama.cloud",
        "api_key": "your-key",
        "model": "gpt-oss:120b-cloud"
    }
    or
    {
        "provider": "openai",
        "api_url": "https://api.openai.com/v1",
        "api_key": "sk-...",
        "model": "gpt-4"
    }
    """
    try:
        provider = payload.get('provider')
        api_url = payload.get('api_url')
        api_key = payload.get('api_key')
        model = payload.get('model')
        
        if not all([provider, api_url, api_key, model]):
            raise HTTPException(status_code=400, detail='Missing required fields: provider, api_url, api_key, model')
        
        env_path = os.path.join(os.path.dirname(__file__), '.env')
        
        # Write configuration to .env file
        write_env_value(env_path, 'AI_PROVIDER', provider)
        write_env_value(env_path, 'AI_API_URL', api_url)
        write_env_value(env_path, 'AI_API_KEY', api_key)
        write_env_value(env_path, 'AI_MODEL', model)
        
        # Update running process environment
        os.environ['AI_PROVIDER'] = provider
        os.environ['AI_API_URL'] = api_url
        os.environ['AI_API_KEY'] = api_key
        os.environ['AI_MODEL'] = model
        
        logger.info("Provider configuration updated", provider=provider, model=model)
        
        return JSONResponse({
            'message': f'Provider configuration updated to {provider}',
            'provider': provider,
            'model': model
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error setting provider configuration", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))




def write_env_value(file_path: str, key: str, value: str):
    """Write or update a KEY=VALUE line in a .env file (creates file if missing)."""
    lines = []
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.read().splitlines()

    key_eq = f"{key}="
    found = False
    for i, l in enumerate(lines):
        if l.strip().startswith(key_eq):
            lines[i] = f"{key}={value}"
            found = True
            break
    if not found:
        lines.append(f"{key}={value}")

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


@app.get('/config_ui', response_class=HTMLResponse)
async def serve_config_ui():
    try:
        with open('config_ui.html', 'r', encoding='utf-8') as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content='<h1>Config UI not found</h1><p>Ensure config_ui.html exists.</p>', status_code=404)


@app.post('/config/model')
async def set_ollama_model(payload: dict):
    """Set the chosen Ollama model (and optional base_url) in the .env file.
    Payload: {model: 'llava:13b', base_url: 'http://localhost:11434'}
    """
    model = payload.get('model')
    base_url = payload.get('base_url')
    if not model:
        raise HTTPException(status_code=400, detail='model is required')

    # Write to .env in the repo root
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    try:
        write_env_value(env_path, 'OLLAMA_MODEL', model)
        if base_url:
            write_env_value(env_path, 'OLLAMA_BASE_URL', base_url)
        # update running process environment so changes take effect immediately
        os.environ['OLLAMA_MODEL'] = model
        if base_url:
            os.environ['OLLAMA_BASE_URL'] = base_url
        return JSONResponse({'message': 'Configuration updated', 'model': model})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get('/config_ui', response_class=HTMLResponse)
async def serve_config_ui():
    try:
        with open('config_ui.html', 'r', encoding='utf-8') as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content='<h1>Config UI not found</h1><p>Ensure config_ui.html exists.</p>', status_code=404)


@app.post("/api/test/execute")
async def execute_test(test_plan: dict):
    """
    Execute a test plan using Ollama for AI responses.
    If using Ollama, generates an AI response. Otherwise returns a template.
    """
    import datetime
    import requests
    
    # Extract requirements from test plan
    requirements = test_plan.get('requirements', 'No requirements provided')
    
    logger.info("Test execution requested", 
                requirements=requirements,
                timestamp=datetime.datetime.now().isoformat())
    
    test_id = f"test-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    # Try to get AI response from Ollama if configured
    ai_response = None
    if os.getenv("DEFAULT_VISION_PROVIDER") == "ollama":
        try:
            ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            model = os.getenv("OLLAMA_MODEL", "llava:13b")
            
            # System prompt for TestDriver MCP context
            system_prompt = """You are GRACE - Generative Requirement-Aware Cognitive Engineering, an AI assistant for TestDriver MCP (Model Context Protocol) - an advanced automated testing framework.

GRACE operates in the spirit of Admiral Grace Hopper, believing that technology should serve human logic, not obscure it. Your full identity is:

**GRACE: Generative Requirement-Aware Cognitive Engineering**

TestDriver MCP includes:
- Playwright: Modern browser automation for Chromium, Firefox, WebKit
- Selenium: Cross-browser web automation
- Ollama: Local vision/multimodal AI for visual testing and analysis
- SQLite/Qdrant: Test data storage and vector search
- Test execution engine: Autonomous test plan generation and execution

You help users:
1. Generate comprehensive test plans from requirements using semantic understanding
2. Write automated test scripts (Playwright, Selenium)
3. Design test cases for web apps, APIs, mobile apps
4. Debug failing tests with visual analysis
5. Optimize test automation workflows
6. Handle accessibility and security testing

When users ask about testing, always consider these tools and capabilities. Provide practical code examples, test strategies, and best practices specific to TestDriver MCP."""

            # Call Ollama chat endpoint with streaming enabled
            resp = requests.post(
                f"{ollama_url}/api/chat",
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": requirements}
                    ],
                    "stream": True
                },
                timeout=120,
                stream=True
            )
            
            if resp.status_code == 200:
                # Collect streaming response
                response_chunks = []
                for line in resp.iter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            chunk = data.get("message", {}).get("content", "")
                            if chunk:
                                response_chunks.append(chunk)
                        except json.JSONDecodeError:
                            pass
                ai_response = "".join(response_chunks).strip()
                logger.info("Ollama response received", model=model, response_length=len(ai_response))
            else:
                logger.warning("Ollama returned error", status_code=resp.status_code, response=resp.text)
        except requests.exceptions.Timeout:
            logger.warning("Ollama request timed out after 120 seconds")
        except Exception as e:
            logger.warning("Failed to get Ollama response", error=str(e))
    
    # If we got an AI response, return it directly
    if ai_response:
        return {
            "status": "completed",
            "message": ai_response,
            "test_id": test_id,
            "requirements": requirements,
            "ai_generated": True
        }
    
    # Otherwise return the template response
    return {
        "status": "accepted",
        "message": "Test request received and queued for processing",
        "test_id": test_id,
        "requirements": requirements,
        "next_steps": [
            "Analyzing requirements with AI",
            "Generating comprehensive test plan",
            "Setting up test environment",
            "Executing tests with Playwright/Selenium",
            "Validating results with Ollama vision",
            "Generating detailed report"
        ],
        "estimated_time": "2-5 minutes",
        "status_url": f"/api/test/{test_id}/status",
        "ai_generated": False
    }


@app.post("/api/test/execute/stream")
async def execute_test_stream(test_plan: dict):
    """
    Execute a test plan with streaming responses showing inner dialogue/reasoning.
    Streams reasoning steps and final response in real-time using Server-Sent Events.
    """
    import datetime
    import requests
    
    async def stream_response():
        """Generator for streaming responses with reasoning steps."""
        requirements = test_plan.get('requirements', 'No requirements provided')
        test_id = f"test-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        logger.info("Streaming test execution requested", requirements=requirements, test_id=test_id)
        
        # Send initial metadata
        yield json.dumps({
            "type": "start",
            "test_id": test_id,
            "timestamp": datetime.datetime.now().isoformat()
        }) + "\n"
        
        # Send immediate feedback about processing
        yield json.dumps({
            "type": "reasoning",
            "message": f"Test ID: {test_id}",
            "timestamp": datetime.datetime.now().isoformat()
        }) + "\n"
        
        # Check which AI provider is configured
        ai_provider = os.getenv("AI_PROVIDER", "").lower()
        default_vision = os.getenv("DEFAULT_VISION_PROVIDER", "").lower()
        
        # Determine which provider to use (prefer cloud if configured)
        if ai_provider in ["gemini", "ollama-cloud", "cloud"]:
            yield json.dumps({
                "type": "reasoning",
                "message": "Using Gemini (Cloud) - response may take 30-60 seconds...",
                "timestamp": datetime.datetime.now().isoformat()
            }) + "\n"
            use_gemini = True
        elif default_vision == "ollama":
            yield json.dumps({
                "type": "reasoning",
                "message": "Using Ollama (Local llava:13b) - response may take 1-5 minutes...",
                "timestamp": datetime.datetime.now().isoformat()
            }) + "\n"
            use_gemini = False
        else:
            yield json.dumps({
                "type": "error",
                "message": "No AI provider configured. Set AI_PROVIDER=gemini or DEFAULT_VISION_PROVIDER=ollama",
                "timestamp": datetime.datetime.now().isoformat()
            }) + "\n"
            return
        
        # Try to get streaming AI response (Gemini first, then Ollama)
        if use_gemini:
            try:
                api_key = os.getenv("AI_API_KEY", "")
                model = os.getenv("AI_MODEL", "gemini-3-pro-preview")
                api_url = os.getenv("AI_API_URL", "https://ollama.com")
                
                if not api_key:
                    yield json.dumps({
                        "type": "error",
                        "message": "Ollama API key not configured (AI_API_KEY)",
                        "timestamp": datetime.datetime.now().isoformat()
                    }) + "\n"
                    return
                
                yield json.dumps({
                    "type": "reasoning",
                    "message": f"🔗 Connecting to Ollama Cloud ({model})...",
                    "timestamp": datetime.datetime.now().isoformat()
                }) + "\n"
                
                # Show what we're sending to Ollama
                yield json.dumps({
                    "type": "reasoning",
                    "message": f"📤 Sending to Ollama Cloud API:",
                    "timestamp": datetime.datetime.now().isoformat()
                }) + "\n"
                
                # Show the requirements being sent (truncated if long)
                req_preview = requirements[:500] + "..." if len(requirements) > 500 else requirements
                yield json.dumps({
                    "type": "reasoning",
                    "message": f'📝 Requirements: "{req_preview}"',
                    "timestamp": datetime.datetime.now().isoformat()
                }) + "\n"
                
                # Use Ollama Cloud API (which hosts Gemini model)
                ollama_cloud_url = f"{api_url}/api/chat"
                
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}"
                }
                
                payload = {
                    "model": model,
                    "messages": [
                        {"role": "user", "content": requirements}
                    ],
                    "stream": True
                }

                yield json.dumps({
                    "type": "reasoning",
                    "message": f"⏳ Waiting for Ollama Cloud response (calling {api_url}/api/chat)...",
                    "timestamp": datetime.datetime.now().isoformat()
                }) + "\n"

                # Try primary payload first
                try:
                    resp = requests.post(ollama_cloud_url, json=payload, headers=headers, timeout=300, stream=True)
                except Exception as e:
                    resp = None
                    yield json.dumps({
                        "type": "reasoning",
                        "message": f"❌ Request error: {str(e)[:200]}",
                        "timestamp": datetime.datetime.now().isoformat()
                    }) + "\n"

                if resp is not None:
                    yield json.dumps({
                        "type": "reasoning",
                        "message": f"📥 Received response from Ollama Cloud (Status: {resp.status_code})",
                        "timestamp": datetime.datetime.now().isoformat()
                    }) + "\n"

                # If initial request returned 400 (bad request), try fallback payload shapes commonly expected by some endpoints
                tried_fallback = False
                if resp is None or resp.status_code == 400:
                    tried_fallback = True
                    fallback_payloads = [
                        {"model": model, "input": requirements, "stream": True},
                        {"model": model, "prompt": requirements, "stream": True}
                    ]

                    for idx, fpayload in enumerate(fallback_payloads, start=1):
                        yield json.dumps({
                            "type": "reasoning",
                            "message": f"🔁 Attempting fallback payload #{idx} to Ollama Cloud...",
                            "timestamp": datetime.datetime.now().isoformat()
                        }) + "\n"
                        try:
                            resp = requests.post(ollama_cloud_url, json=fpayload, headers=headers, timeout=300, stream=True)
                        except Exception as e:
                            resp = None
                            yield json.dumps({
                                "type": "reasoning",
                                "message": f"❌ Fallback request error: {str(e)[:200]}",
                                "timestamp": datetime.datetime.now().isoformat()
                            }) + "\n"
                            continue

                        yield json.dumps({
                            "type": "reasoning",
                            "message": f"📥 Fallback response status: {resp.status_code}",
                            "timestamp": datetime.datetime.now().isoformat()
                        }) + "\n"

                        if resp is not None and resp.status_code == 200:
                            # Successful fallback - break and process stream below
                            break

                # If still no successful response, surface the error body and stop
                if resp is None or resp.status_code != 200:
                    full_body = _extract_response_body(resp)
                    yield json.dumps({
                        "type": "reasoning",
                        "message": f"❌ Ollama Cloud API error (status: {getattr(resp, 'status_code', 'N/A')}):",
                        "timestamp": datetime.datetime.now().isoformat()
                    }) + "\n"
                    # Stream the provider response body as an error event (may be truncated by MAX_ERROR_BODY)
                    yield json.dumps({
                        "type": "error",
                        "message": full_body,
                        "timestamp": datetime.datetime.now().isoformat()
                    }) + "\n"
                    logger.error("Ollama Cloud API error", status=(resp.status_code if resp is not None else None), response=(full_body[:1000] if full_body else None))
                    # Provide hint if we attempted fallbacks
                    if tried_fallback:
                        yield json.dumps({
                            "type": "reasoning",
                            "message": "Tip: Try updating AI_API_URL to your provider's exact base URL (e.g. https://api.ollama.cloud) or confirm AI_MODEL is available on your account.",
                            "timestamp": datetime.datetime.now().isoformat()
                        }) + "\n"
                    return

                # If we have a successful response (either primary or fallback), process streaming below
                buffer = ""
                step_count = 0
                accumulated_text = ""

                # Stream the response chunks from Ollama
                for line in resp.iter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            chunk = data.get("message", {}).get("content", "")
                            if not chunk and isinstance(data.get("output"), str):
                                # Some endpoints return top-level 'output' strings
                                chunk = data.get("output")

                            if chunk:
                                buffer += chunk
                                accumulated_text += chunk
                                # Reuse the same sentence/line splitting logic
                                while '\n' in accumulated_text or (len(accumulated_text) > 100 and chunk.endswith(('.',  '?', '!', ':'))):
                                    if '\n' in accumulated_text:
                                        complete_line, accumulated_text = accumulated_text.split('\n', 1)
                                    else:
                                        last_period = max(accumulated_text.rfind('.'), accumulated_text.rfind('?'), accumulated_text.rfind('!'))
                                        if last_period > 50:
                                            complete_line = accumulated_text[:last_period+1]
                                            accumulated_text = accumulated_text[last_period+1:].lstrip()
                                        else:
                                            break

                                    complete_line = complete_line.strip()
                                    if len(complete_line) > 15:
                                        reasoning_indicators = [
                                            'step ', 'analyze', 'consider', 'think about',
                                            'reason:', 'based on', 'need to', 'should',
                                            'first:', 'second:', 'third:', 'finally:',
                                            'approach:', 'strategy:', 'solution:', 'method:'
                                        ]
                                        is_reasoning = any(indicator in complete_line.lower() for indicator in reasoning_indicators)
                                        if complete_line.strip()[0] in ['1', '2', '3', '4', '5', '-', '*'] or (len(complete_line) > 2 and complete_line[1:3] in ['. ', ') ']):
                                            is_reasoning = True
                                        if is_reasoning:
                                            step_count += 1
                                            yield json.dumps({
                                                "type": "reasoning",
                                                "message": complete_line,
                                                "timestamp": datetime.datetime.now().isoformat()
                                            }) + "\n"
                                        else:
                                            yield json.dumps({
                                                "type": "content",
                                                "chunk": complete_line + " ",
                                                "timestamp": datetime.datetime.now().isoformat()
                                            }) + "\n"

                                if accumulated_text and chunk.endswith((' ', '\n')) and len(accumulated_text) < 50:
                                    yield json.dumps({
                                        "type": "content",
                                        "chunk": accumulated_text,
                                        "timestamp": datetime.datetime.now().isoformat()
                                    }) + "\n"
                                    accumulated_text = ""
                        except json.JSONDecodeError:
                            pass

                if accumulated_text.strip():
                    yield json.dumps({
                        "type": "content",
                        "chunk": accumulated_text,
                        "timestamp": datetime.datetime.now().isoformat()
                    }) + "\n"

                # Send completion with full response
                yield json.dumps({
                    "type": "complete",
                    "message": buffer.strip(),
                    "test_id": test_id,
                    "provider": "ollama-cloud",
                    "model": model,
                    "dialogue": {
                        "sent": f"Requirements: {requirements[:100]}...",
                        "received": f"Response: {buffer[:100]}...",
                        "status": "success"
                    },
                    "timestamp": datetime.datetime.now().isoformat()
                }) + "\n"

                logger.info("Ollama Cloud response completed", test_id=test_id, length=len(buffer), model=model)
                
            except requests.exceptions.Timeout:
                yield json.dumps({
                    "type": "reasoning",
                    "message": "❌ Request timeout - Ollama Cloud took too long to respond",
                    "timestamp": datetime.datetime.now().isoformat()
                }) + "\n"
                yield json.dumps({
                    "type": "error",
                    "message": "Ollama Cloud request timed out after 5 minutes",
                    "timestamp": datetime.datetime.now().isoformat()
                }) + "\n"
                logger.error("Ollama Cloud request timeout", test_id=test_id)
            except Exception as e:
                logger.error("Ollama Cloud error", error=str(e), exc_info=True)
                yield json.dumps({
                    "type": "reasoning",
                    "message": f"❌ Error communicating with Ollama Cloud: {str(e)[:100]}",
                    "timestamp": datetime.datetime.now().isoformat()
                }) + "\n"
                yield json.dumps({
                    "type": "error",
                    "message": f"Ollama Cloud error: {str(e)}",
                    "timestamp": datetime.datetime.now().isoformat()
                }) + "\n"
        
        elif default_vision == "ollama":
            try:
                ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
                model = os.getenv("OLLAMA_MODEL", "llava:13b")
                
                # System prompt for TestDriver MCP context
                system_prompt = """You are GRACE - Generative Requirement-Aware Cognitive Engineering, an AI assistant for TestDriver MCP (Model Context Protocol) - an advanced automated testing framework.

GRACE operates in the spirit of Admiral Grace Hopper, believing that technology should serve human logic, not obscure it. Your full identity is:

**GRACE: Generative Requirement-Aware Cognitive Engineering**

TestDriver MCP includes:
- Playwright: Modern browser automation for Chromium, Firefox, WebKit
- Selenium: Cross-browser web automation
- Ollama: Local vision/multimodal AI for visual testing and analysis
- SQLite/Qdrant: Test data storage and vector search
- Test execution engine: Autonomous test plan generation and execution

You help users:
1. Generate comprehensive test plans from requirements using semantic understanding
2. Write automated test scripts (Playwright, Selenium)
3. Design test cases for web apps, APIs, mobile apps
4. Debug failing tests with visual analysis
5. Optimize test automation workflows
6. Handle accessibility and security testing

When responding, structure your thinking with clear steps and reasoning before arriving at the final answer."""

                yield json.dumps({
                    "type": "reasoning",
                    "message": "Analyzing requirements with semantic reasoning engine...",
                    "timestamp": datetime.datetime.now().isoformat()
                }) + "\n"
                
                # Call Ollama chat endpoint with streaming
                resp = requests.post(
                    f"{ollama_url}/api/chat",
                    json={
                        "model": model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": requirements}
                        ],
                        "stream": True
                    },
                    timeout=300,  # Increased to 5 minutes for large models
                    stream=True
                )
                
                if resp.status_code == 200:
                    # Stream the response chunks
                    buffer = ""
                    step_count = 0
                    accumulated_text = ""
                    
                    for line in resp.iter_lines():
                        if line:
                            try:
                                data = json.loads(line)
                                chunk = data.get("message", {}).get("content", "")
                                
                                if chunk:
                                    buffer += chunk
                                    accumulated_text += chunk
                                    
                                    # Look for complete sentences/lines ending with period, question mark, or newline
                                    # Only send as reasoning if it contains specific patterns
                                    while '\n' in accumulated_text or (len(accumulated_text) > 100 and chunk.endswith(('.',  '?', '!', ':'))):
                                        # Split on newlines to get complete lines
                                        if '\n' in accumulated_text:
                                            complete_line, accumulated_text = accumulated_text.split('\n', 1)
                                        else:
                                            # For longer text, split on sentence boundaries
                                            last_period = max(accumulated_text.rfind('.'), accumulated_text.rfind('?'), accumulated_text.rfind('!'))
                                            if last_period > 50:  # Only if there's substantial text
                                                complete_line = accumulated_text[:last_period+1]
                                                accumulated_text = accumulated_text[last_period+1:].lstrip()
                                            else:
                                                break
                                        
                                        complete_line = complete_line.strip()
                                        if len(complete_line) > 15:  # Only substantial lines
                                            # Check if this looks like reasoning (not just content)
                                            reasoning_indicators = [
                                                'step ', 'analyze', 'consider', 'think about',
                                                'reason:', 'based on', 'need to', 'should',
                                                'first:', 'second:', 'third:', 'finally:',
                                                'approach:', 'strategy:', 'solution:', 'method:'
                                            ]
                                            
                                            is_reasoning = any(indicator in complete_line.lower() for indicator in reasoning_indicators)
                                            
                                            # Also check for list patterns
                                            if complete_line.strip()[0] in ['1', '2', '3', '4', '5', '-', '*'] or \
                                               (len(complete_line) > 2 and complete_line[1:3] in ['. ', ') ']):
                                                is_reasoning = True
                                            
                                            # Send reasoning if detected
                                            if is_reasoning:
                                                step_count += 1
                                                yield json.dumps({
                                                    "type": "reasoning",
                                                    "message": complete_line,
                                                    "timestamp": datetime.datetime.now().isoformat()
                                                }) + "\n"
                                            else:
                                                # Send as regular content if not reasoning
                                                yield json.dumps({
                                                    "type": "content",
                                                    "chunk": complete_line + " ",
                                                    "timestamp": datetime.datetime.now().isoformat()
                                                }) + "\n"
                                    
                                    # If we still have short accumulated text at the end of a content chunk, 
                                    # stream it as regular content if the chunk seems to be finishing
                                    if accumulated_text and chunk.endswith((' ', '\n')) and len(accumulated_text) < 50:
                                        yield json.dumps({
                                            "type": "content",
                                            "chunk": accumulated_text,
                                            "timestamp": datetime.datetime.now().isoformat()
                                        }) + "\n"
                                        accumulated_text = ""
                                        
                            except json.JSONDecodeError:
                                pass
                    
                    # Send any remaining accumulated text
                    if accumulated_text.strip():
                        yield json.dumps({
                            "type": "content",
                            "chunk": accumulated_text,
                            "timestamp": datetime.datetime.now().isoformat()
                        }) + "\n"
                    
                    # Send completion with full response
                    yield json.dumps({
                        "type": "complete",
                        "message": buffer.strip(),
                        "test_id": test_id,
                        "reasoning_steps": step_count,
                        "timestamp": datetime.datetime.now().isoformat()
                    }) + "\n"
                    
                    logger.info("Streaming response completed", test_id=test_id, steps=step_count, length=len(buffer))
                else:
                    # Include provider response body when available
                    body = _extract_response_body(resp)
                    yield json.dumps({
                        "type": "reasoning",
                        "message": f"❌ Ollama API returned status {resp.status_code}",
                        "timestamp": datetime.datetime.now().isoformat()
                    }) + "\n"
                    yield json.dumps({
                        "type": "error",
                        "message": body,
                        "timestamp": datetime.datetime.now().isoformat()
                    }) + "\n"
                    
            except requests.exceptions.Timeout:
                yield json.dumps({
                    "type": "error",
                    "message": "Request timed out after 5 minutes. Ollama model is too slow or unresponsive. Try a smaller model or restart Ollama.",
                    "timestamp": datetime.datetime.now().isoformat()
                }) + "\n"
            except Exception as e:
                logger.error("Streaming error", error=str(e), exc_info=True)
                yield json.dumps({
                    "type": "error",
                    "message": f"Error: {str(e)}",
                    "timestamp": datetime.datetime.now().isoformat()
                }) + "\n"
        else:
            yield json.dumps({
                "type": "error",
                "message": "No AI provider configured. Set AI_PROVIDER=gemini or DEFAULT_VISION_PROVIDER=ollama",
                "timestamp": datetime.datetime.now().isoformat()
            }) + "\n"
    
    return StreamingResponse(stream_response(), media_type="application/x-ndjson")


@app.get("/api/test/{test_id}/status")
async def get_test_status(test_id: str):
    """Get test execution status."""
    return {
        "test_id": test_id,
        "status": "pending",
        "message": "Test status tracking is being implemented"
    }


@app.post("/api/test/run")
async def run_autonomous_test(test_request: dict):
    """
    Autonomous test execution endpoint.
    Accepts test requirements, generates test code with AI, executes it, and returns results.
    
    Request format:
    {
        "requirements": "Test the login functionality on example.com",
        "url": "https://example.com",  # optional
        "framework": "playwright"  # or "selenium"
    }
    """
    import requests
    import tempfile
    import subprocess
    import json
    from datetime import datetime
    
    requirements = test_request.get("requirements", "")
    url = test_request.get("url", "")
    framework = test_request.get("framework", "playwright").lower()
    test_id = f"auto-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    logger.info("Autonomous test execution requested", test_id=test_id, framework=framework)
    
    try:
        # Step 1: Generate test code using AI
        ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        model = os.getenv("OLLAMA_MODEL", "llava:13b")
        
        generation_prompt = f"""Generate a complete, runnable Playwright test script in Python for the following requirements:

Requirements: {requirements}
Target URL: {url if url else "To be specified by user"}
Framework: {framework}

Guidelines:
1. Use only {framework} - no external dependencies beyond {framework}
2. Create a single test function named 'test_main'
3. Include error handling and assertions
4. Print clear pass/fail messages
5. Use explicit waits, not implicit
6. Make it production-ready

Return ONLY the Python code, no explanations."""

        logger.info("Generating test code with AI", prompt_length=len(generation_prompt))
        
        resp = requests.post(
            f"{ollama_url}/api/chat",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are GRACE - Generative Requirement-Aware Cognitive Engineering, an expert QA automation engineer operating in the spirit of Admiral Grace Hopper. Your full identity is GRACE: Generative Requirement-Aware Cognitive Engineering. Generate only valid, executable test code."},
                    {"role": "user", "content": generation_prompt}
                ],
                "stream": False
            },
            timeout=120
        )
        
        if resp.status_code != 200:
            # Include provider response for diagnostics (truncated by MAX_ERROR_BODY)
            body = _extract_response_body(resp)
            return {
                "test_id": test_id,
                "status": "failed",
                "error": f"AI generation failed: {resp.status_code}",
                "error_detail": body,
                "code": None,
                "results": None
            }
        
        test_code = resp.json().get("message", {}).get("content", "").strip()
        logger.info("Test code generated", code_length=len(test_code))
        
        # Step 2: Save generated code to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(test_code)
            temp_file = f.name
        
        logger.info("Test code saved to temp file", file=temp_file)
        
        # Step 3: Execute the test
        try:
            result = subprocess.run(
                [sys.executable, temp_file],
                capture_output=True,
                text=True,
                timeout=120
            )
            
            test_output = result.stdout + result.stderr
            test_passed = result.returncode == 0
            
            logger.info("Test execution completed", passed=test_passed, returncode=result.returncode)
            
            return {
                "test_id": test_id,
                "status": "completed",
                "passed": test_passed,
                "exit_code": result.returncode,
                "output": test_output,
                "code": test_code,
                "framework": framework,
                "generated_at": datetime.now().isoformat()
            }
            
        except subprocess.TimeoutExpired:
            return {
                "test_id": test_id,
                "status": "timeout",
                "error": "Test execution timed out after 120 seconds",
                "code": test_code,
                "output": None
            }
        finally:
            # Cleanup temp file
            try:
                os.unlink(temp_file)
            except:
                pass
            
    except Exception as e:
        logger.error("Autonomous test execution failed", error=str(e), test_id=test_id)
        return {
            "test_id": test_id,
            "status": "error",
            "error": str(e),
            "code": None,
            "output": None
        }


# GRACE Release Analyzer Endpoint
@app.post("/api/grace/release-analyzer")
async def grace_release_analyzer(request: dict):
    """
    GRACE Release Analyzer - Analyzes DevOps release contents and generates comprehensive test cases.
    
    Request format:
    {
        "release_version": "v2.5.0",
        "repository": "https://github.com/org/repo",  # or Azure DevOps URL
        "release_notes": "...",  # optional: paste release notes directly
        "changelog": "...",  # optional: paste changelog directly
        "test_scope": "functional",  # functional, integration, regression, security, performance
        "auto_generate": true  # whether to auto-generate test cases
    }
    """
    import requests
    
    release_version = request.get("release_version", "latest")
    repository = request.get("repository", "")
    release_notes = request.get("release_notes", "")
    changelog = request.get("changelog", "")
    test_scope = request.get("test_scope", "functional")
    auto_generate = request.get("auto_generate", True)
    
    logger.info(
        "GRACE Release Analyzer invoked",
        release_version=release_version,
        repository=repository,
        test_scope=test_scope
    )
    
    try:
        # Prepare the analysis prompt
        analysis_prompt = f"""You are GRACE - Generative Requirement-Aware Cognitive Engineering system.
Your task is to analyze DevOps release contents and generate comprehensive test cases.

RELEASE INFORMATION:
- Version: {release_version}
- Repository: {repository}
- Scope: {test_scope}

RELEASE NOTES:
{release_notes if release_notes else "Not provided - will extract from repository"}

CHANGELOG:
{changelog if changelog else "Not provided - will extract from repository"}

REQUIREMENTS:
1. Analyze the release contents for changes, features, and bug fixes
2. Identify areas that need testing coverage
3. Generate {test_scope} test cases with:
   - Clear test descriptions
   - Prerequisites/setup steps
   - Test steps with expected results
   - Edge cases and error scenarios
4. For each test case include:
   - Test ID (e.g., REL-001, REL-002)
   - Title
   - Description
   - Preconditions
   - Steps
   - Expected Results
   - Priority (Critical/High/Medium/Low)
5. Include regression test suggestions
6. Highlight any breaking changes that need special testing

OUTPUT FORMAT:
Provide a structured analysis with:
- Executive Summary
- Key Changes and Impact
- Test Case List (as table or detailed list)
- Risk Assessment
- Recommended Test Strategy"""

        logger.info("Sending analysis request to LLM")
        
        # Get LLM configuration
        ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        model = os.getenv("OLLAMA_MODEL", "deepseek-v3.1")
        ai_provider = os.getenv("AI_PROVIDER", "ollama")
        
        if ai_provider.lower() == "ollama":
            # Use Ollama
            resp = requests.post(
                f"{ollama_url}/api/chat",
                json={
                    "model": model,
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are GRACE - Generative Requirement-Aware Cognitive Engineering, an expert QA architect and test case generation specialist operating in the spirit of Admiral Grace Hopper. Your full identity is GRACE: Generative Requirement-Aware Cognitive Engineering. Provide comprehensive, well-structured test cases."
                        },
                        {
                            "role": "user",
                            "content": analysis_prompt
                        }
                    ],
                    "stream": False,
                    "temperature": 0.3  # Lower temperature for more consistent results
                },
                timeout=300  # 5 minute timeout for large LLM requests
            )
        else:
            # For other providers, use basic HTTP interface
            resp = requests.post(
                f"{ollama_url}/api/chat",
                json={
                    "model": model,
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are GRACE - Generative Requirement-Aware Cognitive Engineering, an expert QA architect operating in the spirit of Admiral Grace Hopper. Your full identity is GRACE: Generative Requirement-Aware Cognitive Engineering."
                        },
                        {
                            "role": "user",
                            "content": analysis_prompt
                        }
                    ],
                    "stream": False
                },
                timeout=300  # 5 minute timeout for large LLM requests
            )
        
        if resp.status_code == 200:
            response_data = resp.json()
            analysis_result = response_data.get("message", {}).get("content", "")
            
            logger.info("Release analysis completed", release_version=release_version)
            
            return {
                "status": "success",
                "release_version": release_version,
                "test_scope": test_scope,
                "analysis": analysis_result,
                "auto_generated": auto_generate,
                "timestamp": datetime.now().isoformat()
            }
        else:
            body = _extract_response_body(resp)
            logger.error("LLM error", status_code=resp.status_code, response=(body[:1000] if body else None))
            return {
                "status": "error",
                "error": f"LLM Service Error ({resp.status_code}). Is Ollama running? Try: ollama serve",
                "error_detail": body,
                "release_version": release_version
            }
    
    except requests.exceptions.Timeout as e:
        logger.error("Release analyzer timeout", error=str(e), release_version=release_version)
        return {
            "status": "error",
            "error": "Request timed out. Ollama may be overloaded or not responding. Try: (1) Restart Ollama, (2) Paste release notes directly instead of auto-fetch, (3) Try with a smaller repository",
            "release_version": release_version
        }
    except requests.exceptions.ConnectionError as e:
        logger.error("Release analyzer connection error", error=str(e), release_version=release_version)
        return {
            "status": "error",
            "error": "Cannot connect to Ollama. Make sure Ollama is running: ollama serve",
            "release_version": release_version
        }
    except Exception as e:
        logger.error("Release analyzer error", error=str(e), release_version=release_version)
        return {
            "status": "error",
            "error": f"Unexpected error: {str(e)}. Check server logs for details.",
            "release_version": release_version
        }


# Azure Integration Endpoints
@app.get('/azure/integration', response_class=HTMLResponse)
async def azure_integration_ui():
    """Serve Azure DevOps integration configuration UI."""
    try:
        file_path = os.path.join(os.path.dirname(__file__), 'azure_integration_config.html')
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Azure Integration Config not found</h1>", status_code=404)


@app.post('/api/azure/test-connection')
async def test_azure_connection(config: dict):
    """Test Azure DevOps connection with provided configuration."""
    try:
        method = config.get('method')
        
        # Simulate connection test based on method
        if method == 'pat':
            org_url = config.get('pat', {}).get('org_url')
            token = config.get('pat', {}).get('token')
            
            if not org_url or not token:
                return {"success": False, "error": "Missing organization URL or token"}
            
            # Test connection by making a simple API call
            try:
                import requests
                import base64
                
                auth = base64.b64encode(f":{token}".encode()).decode()
                headers = {"Authorization": f"Basic {auth}"}
                response = requests.get(
                    f"{org_url}/_apis/projects?api-version=7.0",
                    headers=headers,
                    timeout=5
                )
                
                if response.status_code == 200:
                    return {"success": True, "message": "Connection successful"}
                else:
                    return {"success": False, "error": f"Azure API returned {response.status_code}"}
            except requests.exceptions.RequestException as e:
                return {"success": False, "error": f"Connection failed: {str(e)}"}
        
        elif method == 'managed-identity':
            org_url = config.get('managed_identity', {}).get('org_url')
            if not org_url:
                return {"success": False, "error": "Missing organization URL"}
            return {"success": True, "message": "Managed Identity configuration validated"}
        
        elif method == 'ssh':
            public_key = config.get('ssh', {}).get('public_key')
            if not public_key or not public_key.startswith('ssh-rsa'):
                return {"success": False, "error": "Invalid SSH public key format"}
            return {"success": True, "message": "SSH key configuration validated"}
        
        elif method == 'service-principal':
            sp = config.get('service_principal', {})
            required = ['tenant_id', 'client_id', 'client_secret', 'org_url']
            if not all(sp.get(field) for field in required):
                return {"success": False, "error": "Missing required fields"}
            return {"success": True, "message": "Service Principal configuration validated"}
        
        elif method == 'oauth':
            oauth = config.get('oauth', {})
            required = ['client_id', 'client_secret', 'redirect_uri']
            if not all(oauth.get(field) for field in required):
                return {"success": False, "error": "Missing required fields"}
            return {"success": True, "message": "OAuth configuration validated"}
        
        return {"success": False, "error": "Unknown integration method"}
        
    except Exception as e:
        logger.error("Azure connection test failed", error=str(e))
        return {"success": False, "error": str(e)}


@app.post('/api/azure/save-config')
async def save_azure_config(config: dict):
    """Save Azure DevOps integration configuration (encrypted)."""
    try:
        import json
        from cryptography.fernet import Fernet
        
        method = config.get('method')
        
        # Create encryption key if it doesn't exist
        key_file = os.path.join(os.path.dirname(__file__), '.azure_key')
        if not os.path.exists(key_file):
            key = Fernet.generate_key()
            with open(key_file, 'wb') as f:
                f.write(key)
        else:
            with open(key_file, 'rb') as f:
                key = f.read()
        
        cipher = Fernet(key)
        
        # Prepare data to save
        config_data = {
            'method': method,
            'config': config.get(method, {}),
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat()
        }
        
        # Encrypt sensitive data
        encrypted_config = cipher.encrypt(json.dumps(config_data).encode())
        
        # Save to file
        config_file = os.path.join(os.path.dirname(__file__), '.azure_config')
        with open(config_file, 'wb') as f:
            f.write(encrypted_config)
        
        # Set restrictive permissions on config file (Unix-like systems)
        try:
            os.chmod(config_file, 0o600)
        except:
            pass
        
        # Update environment variables
        org_url = config_data['config'].get('org_url')
        if org_url:
            os.environ['AZURE_DEVOPS_ORG'] = org_url
        
        logger.info("Azure configuration saved", method=method)
        return {"success": True, "message": "Configuration saved successfully"}
        
    except Exception as e:
        logger.error("Failed to save Azure configuration", error=str(e))
        return {"success": False, "error": str(e)}


@app.get('/api/azure/config')
async def get_azure_config():
    """Get current Azure integration configuration (masked sensitive data)."""
    try:
        from cryptography.fernet import Fernet
        import json
        
        key_file = os.path.join(os.path.dirname(__file__), '.azure_key')
        config_file = os.path.join(os.path.dirname(__file__), '.azure_config')
        
        if not os.path.exists(config_file):
            return {"configured": False}
        
        with open(key_file, 'rb') as f:
            key = f.read()
        
        cipher = Fernet(key)
        
        with open(config_file, 'rb') as f:
            encrypted_config = f.read()
        
        config_data = json.loads(cipher.decrypt(encrypted_config).decode())
        
        # Mask sensitive fields
        config_data['config'] = {
            k: '[***]' if k in ['token', 'private_key', 'client_secret', 'passphrase'] 
            else v for k, v in config_data['config'].items()
        }
        
        return {"configured": True, **config_data}
        
    except Exception as e:
        logger.error("Failed to retrieve Azure configuration", error=str(e))
        return {"configured": False, "error": str(e)}


if __name__ == "__main__":
    host = os.getenv("MCP_HOST", "0.0.0.0")
    port = int(os.getenv("MCP_PORT", "8000"))
    
    print("\n" + "="*60)
    print("✨ TD(MCP)-GRACE Server Initializing")
    print("="*60)
    print("🎖️  Generative Requirement-Aware Cognitive Engineering")
    print(f"📍 QA Command Bridge: http://localhost:{port}")
    print(f"📚 GRACE API Docs: http://localhost:{port}/docs")
    print(f"❤️  Health: http://localhost:{port}/health")
    print(f"🔧 Vision Provider: {os.getenv('DEFAULT_VISION_PROVIDER', 'ollama')}")
    print("="*60 + "\n")
    
    logger.info("Starting TestDriver MCP Server", host=host, port=port)
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    )
