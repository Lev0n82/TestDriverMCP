# TestDriver MCP Installation & Component Guide

This document centralizes every infrastructure component the TestDriver MCP stack relies on, where its configuration lives in the repository, and how to install or launch it locally. Use it as a checklist when bringing up a new machine.

## Component Map

| Component | Purpose | Key Files/Scripts | Install / Start Command |
|-----------|---------|-------------------|-------------------------|
| Python FastAPI App | Runs the MCP/GRACE API server | `requirements.txt`, `run_server.py` | `python -m venv .venv && .venv\\Scripts\\activate` (Windows) / `source .venv/bin/activate` (Linux/macOS), then `pip install -r requirements.txt` and `python run_server.py` |
| Qdrant via Podman | Vector store for embeddings | `podman-compose.yml`, `scripts/start_qdrant_podman.ps1` | `podman volume create qdrant_data && podman run -d --name qdrant -p 6333:6333 -v qdrant_data:/qdrant/storage qdrant/qdrant:latest` or run the script |
| Redis / Valkey | Key-value cache used for session state & locks | Documented in `system_design_part4_data_models.md`, `TestDriver_System_Design_Specification_Final.md` | Install any Redis-compatible server (e.g., `brew install redis`, `sudo apt install redis`, or Azure Cache). Set `REDIS_URL=redis://host:6379/0`. |
| Ollama + llava:13b | Local vision/LLM provider | `scripts/set_ollama_system_env.ps1`, `local_vlm_adapter.py` | Install Ollama from [ollama.com](https://ollama.com), run `ollama pull llava:13b`, ensure `OLLAMA_HOST`/`OLLAMA_API_KEY` env vars are set (script helps on Windows). |
| Podman CLI | Container runtime for Qdrant & optional app images | Instructions in `system_design_part5_deployment.md`, `SETUP.md` | Linux: `sudo apt-get install -y podman`. Windows/macOS: install Podman Desktop or use Docker Desktop (compose file is compatible). |
| Node.js Tooling (optional) | Needed only if running Playwright tooling | `package.json`, `package-lock.json`, `node_modules/` | `npm install` (already vendored but prefer reinstall). |
| Indexer Utility | Builds local embedding index into Qdrant (uses `sentence-transformers/all-MiniLM-L6-v2`, 384-d vectors) | `scripts/index_codebase.py` | Activate `.venv_indexer` (or create), install `sentence-transformers` + `qdrant-client`, then `python scripts/index_codebase.py --collection code_index_384`. |

## Recommended Installation Order

1. **System Dependencies & Runtimes**
   - Install Python 3.11+, Git, and either Podman or Docker (plus Podman Compose support if using Podman).
2. **Code Checkout**
   - Clone the repository and, if desired, unpack one of the tarball distributions.
3. **Python Virtual Environment**
   - Create `.venv`, install `requirements.txt`, ensure `python run_server.py` starts without optional services (it will log connection failures until Redis/Qdrant are up).
4. **Redis / Valkey Cache**
   - Bring up a local or managed Redis instance and export `REDIS_URL`.
5. **Qdrant via Podman**
   - Install/start Podman, run `scripts/start_qdrant_podman.ps1` (or manual compose) to launch Qdrant on `localhost:6333`.
6. **Ollama & Models**
   - Install Ollama, pull `llava:13b`, configure `OLLAMA_HOST`/`OLLAMA_API_KEY`.
7. **Indexer Environment (Optional but recommended)**
   - Create `.venv_indexer`, install `sentence-transformers` + `qdrant-client`, then run `scripts/index_codebase.py` once Qdrant is healthy.
8. **Node/Playwright Tooling (Optional)**
   - Run `npm install` and `npx playwright install` if browser automation/tests are needed.
9. **Final Verification**
   - Start Redis, Qdrant, Ollama, then run `python run_server.py`, load `chat_fresh.html`, and confirm streaming responses.

## Detailed Instructions

### 1. Python Environment & Server

1. Create and activate a virtual environment:
   - **Windows PowerShell**:
     ```powershell
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1
     ```
   - **Linux/macOS**:
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```
2. Install dependencies:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
3. Start the MCP server:
   ```bash
   python run_server.py
   ```
   The API listens on `http://localhost:8000`, with docs at `/docs` and health at `/health`.

### 2. Podman & Qdrant Vector Store

1. Install Podman (or Docker). Minimal Linux commands:
   ```bash
   sudo apt-get update
   sudo apt-get install -y podman
   ```
2. Verify Podman Compose support: `podman compose version` (or `podman-compose version`).
3. **Recommended (PowerShell script)**:
   ```powershell
   .\scripts\start_qdrant_podman.ps1
   ```
   - Detects whether `podman compose`, `podman-compose`, or plain `podman` is available.
   - Creates the `qdrant_data` volume and enforces health checks on `http://localhost:6333`.
4. **Manual compose alternative**:
   ```powershell
   podman compose -f podman-compose.yml up -d
   ```
5. Confirm the collection endpoint responds:
   ```powershell
   Invoke-WebRequest http://localhost:6333/collections -UseBasicParsing
   ```

### 3. Redis / Valkey Cache

Redis is referenced throughout the system design but not shipped with automation scripts. Choose one of the following deployments:

- **Local service**:
  ```bash
  # macOS (Homebrew)
  brew install redis
  brew services start redis

  # Ubuntu
  sudo apt-get install -y redis-server
  sudo systemctl enable --now redis-server
  ```
- **Container**:
  ```bash
  podman run -d --name redis -p 6379:6379 redis:7
  ```
- **Managed service**: provision Azure Cache for Redis or AWS ElastiCache.

Set the connection string for the app:
```bash
$env:REDIS_URL = "redis://localhost:6379/0"   # PowerShell
export REDIS_URL=redis://localhost:6379/0       # Bash
```

### 4. Ollama & Vision Model

1. Install Ollama from the official installer (supports macOS, Linux, Windows WSL).
2. Start the Ollama service: `ollama serve`.
3. Pull the model the adapters expect:
   ```bash
   ollama pull llava:13b
   ```
4. Configure environment variables. On Windows you can persist them via the helper script:
   ```powershell
   .\scripts\set_ollama_system_env.ps1 -Host http://localhost:11434 -ApiKey "your-optional-key"
   ```
   Otherwise set `OLLAMA_HOST` (default `http://localhost:11434`) and, if required, `OLLAMA_API_KEY`.

### 5. Indexing Workflow (MiniLM Embeddings)

Populating Qdrant with the codebase embeddings is optional but required for semantic search. The script relies on the `sentence-transformers/all-MiniLM-L6-v2` encoder (384-dimensional embeddings). Ollama/llava are **not** involved in the embedding step—they power runtime reasoning only.

1. Create/activate a dedicated virtual environment (the repo uses `.venv_indexer`).
2. Install indexer dependencies:
   ```bash
   pip install -r requirements.txt  # or pip install sentence-transformers qdrant-client tqdm
   ```
3. Run the indexer (examples):
   ```bash
   python scripts/index_codebase.py --collection code_index_384 --max-files 50
   python scripts/index_codebase.py --collection code_index_384 --batch-size 64 --max-files 0
   ```
   - Requires Qdrant to be running on `http://localhost:6333`.
   - Adjust `--max-files` for partial runs (0 means entire repo).
   - Uses `all-MiniLM-L6-v2` internally; no additional model downloads are necessary beyond `sentence-transformers`.

### 6. Optional Node/Playwright Tooling

Some UI tests rely on Playwright and the dependencies listed in `package.json`.

```bash
npm install
npx playwright install
```

### 7. Environment Variables Summary

Set these before running the server (PowerShell syntax shown; Bash uses `export`):

```powershell
$env:OPENAI_API_KEY = "sk-..."             # or any provider key the adapters use
$env:REDIS_URL = "redis://localhost:6379/0"
$env:OLLAMA_HOST = "http://localhost:11434"
$env:OLLAMA_API_KEY = "optional-if-required"
$env:QDRANT_URL = "http://localhost:6333"   # optional override
```

### 8. Verifying the Stack

1. Start Redis (or managed equivalent).
2. Start Qdrant via Podman (`scripts/start_qdrant_podman.ps1`).
3. Ensure Ollama is running.
4. Run `python run_server.py` and visit:
   - Chat UI: open `chat_fresh.html` in a browser (or served via your web stack).
   - API docs: `http://localhost:8000/docs`.
5. Inspect logs for successful connections to Redis, Qdrant, and Ollama.

---
By keeping this guide updated alongside the scripts (`scripts/`), `Dockerfile`, and `podman-compose.yml`, every required component for TestDriver MCP can be discovered and provisioned consistently across machines.