# Quick Start: Model Selection

## Access Points
- **Model Config UI**: http://localhost:8000/config_ui
- **Chat Client**: http://localhost:8000/chat
- **API Docs**: http://localhost:8000/docs
- **Server Health**: http://localhost:8000/health

## Quick Steps
1. Go to `http://localhost:8000/config_ui`
2. Select model from dropdown (22 available)
3. Click "Save model"
4. Use chat or autonomous tests - they will use your selected model

## All 22 Models

| # | Model | Type |
|---|-------|------|
| 1 | gpt-oss:120b-cloud | Cloud |
| 2 | gpt-oss:20b-cloud | Cloud |
| 3 | deepseek-v3.1:671b-cloud | Cloud |
| 4 | qwen3-coder:480b-cloud | Cloud |
| 5 | qwen3-vl:235b-cloud | Cloud |
| 6 | minimax-m2:cloud | Cloud |
| 7 | glm-4.6:cloud | Cloud |
| 8 | gpt-oss:120b | Local |
| 9 | gpt-oss:20b | Local |
| 10 | gemma3:27b | Local |
| 11 | gemma3:12b | Local |
| 12 | gemma3:4b | Local |
| 13 | gemma3:1b | Local |
| 14 | deepseek-r1:8b | Local |
| 15 | qwen3-coder:30b | Local |
| 16 | qwen3-vl:30b | Local |
| 17 | qwen3-vl:8b | Local |
| 18 | qwen3-vl:4b | Local |
| 19 | qwen3:30b | Local |
| 20 | qwen3:8b | Local |
| 21 | qwen3:4b | Local |
| 22 | llava:13b | Local |

## Test It
```bash
# View current model
curl http://localhost:8000/config | jq '.vision.ollama_model'

# Change model
curl -X POST http://localhost:8000/config/model \
  -H "Content-Type: application/json" \
  -d '{"model":"deepseek-r1:8b"}'

# Verify change
curl http://localhost:8000/config | jq '.vision.ollama_model'
```

## Using with Tests
```bash
# Send test request - will use currently selected model
curl -X POST http://localhost:8000/api/test/run \
  -H "Content-Type: application/json" \
  -d '{
    "requirements": "Test that example.com homepage loads",
    "url": "https://example.com",
    "framework": "playwright"
  }'
```

## Configuration File
Location: `c:\TestDriverMCP\.env`
Key setting: `OLLAMA_MODEL=<selected-model>`
