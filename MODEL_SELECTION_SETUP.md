# TestDriver MCP - Model Selection Setup

## Overview
All 22 models have been successfully configured in the TestDriver MCP model selection system. Users can now select any model from the dropdown and all subsequent test executions will use the selected model.

## Available Models

The following models are now selectable from the Model Configuration UI:

### Cloud Models (7)
1. **gpt-oss:120b-cloud** - GPT-OSS 120B Cloud variant
2. **gpt-oss:20b-cloud** - GPT-OSS 20B Cloud variant
3. **deepseek-v3.1:671b-cloud** - DeepSeek v3.1 671B Cloud variant
4. **qwen3-coder:480b-cloud** - Qwen3 Coder 480B Cloud variant
5. **qwen3-vl:235b-cloud** - Qwen3 VL 235B Cloud variant
6. **minimax-m2:cloud** - Minimax M2 Cloud variant
7. **glm-4.6:cloud** - GLM 4.6 Cloud variant

### Local Models (15)
8. **gpt-oss:120b** - GPT-OSS 120B
9. **gpt-oss:20b** - GPT-OSS 20B
10. **gemma3:27b** - Gemma3 27B
11. **gemma3:12b** - Gemma3 12B
12. **gemma3:4b** - Gemma3 4B
13. **gemma3:1b** - Gemma3 1B
14. **deepseek-r1:8b** - DeepSeek R1 8B
15. **qwen3-coder:30b** - Qwen3 Coder 30B
16. **qwen3-vl:30b** - Qwen3 VL 30B
17. **qwen3-vl:8b** - Qwen3 VL 8B
18. **qwen3-vl:4b** - Qwen3 VL 4B
19. **qwen3:30b** - Qwen3 30B
20. **qwen3:8b** - Qwen3 8B
21. **qwen3:4b** - Qwen3 4B
22. **llava:13b** - LLaVA 13B

## How to Use

### Step 1: Open Model Configuration UI
Navigate to: `http://localhost:8000/config_ui`

### Step 2: Select a Model
- Click the "Available Models" dropdown
- All 22 models are now available
- Select the model you want to use

### Step 3: Save Selection
- Click the "Save model" button
- The selection is immediately saved to `.env` and the running process
- You'll see a success message

### Step 4: Use Selected Model
The selected model will automatically be used for:
- **Chat responses** in `/api/test/execute` endpoint
- **Test generation and execution** in `/api/test/run` endpoint

## Technical Details

### Model Selection Flow
```
User selects model in config_ui.html
                ↓
POST /config/model endpoint receives selection
                ↓
write_env_value() updates .env file
                ↓
os.environ['OLLAMA_MODEL'] updated in runtime
                ↓
Subsequent API calls use os.getenv("OLLAMA_MODEL")
```

### Key Files Modified
- **config_ui.html**: Updated with hardcoded list of all 22 models in dropdown (lines 29-51)
- **run_server.py**: No changes needed - already reads OLLAMA_MODEL from environment at:
  - Line 269: Chat execution endpoint
  - Line 385: Autonomous test execution endpoint

### Configuration Files
- **.env**: Contains `OLLAMA_MODEL` setting (automatically updated when model is selected)
  - Current value example: `OLLAMA_MODEL=llava:13b`
  - Persists across server restarts

## Workflow Examples

### Example 1: Run Test with DeepSeek Model
```
1. Go to http://localhost:8000/config_ui
2. Select "DeepSeek v3.1 671B Cloud" from dropdown
3. Click "Save model"
4. Open chat at http://localhost:8000/chat
5. Ask: "Create and run a test for Google homepage"
6. Test will be generated and executed using deepseek-v3.1:671b-cloud
```

### Example 2: Chat with Qwen3 Coder
```
1. Go to http://localhost:8000/config_ui
2. Select "Qwen3 Coder 30B" from dropdown
3. Click "Save model"
4. Go to http://localhost:8000/chat
5. Chat responses will now use qwen3-coder:30b
```

## Verification

Run the following to verify model selection works:

```bash
# Check current model
curl http://localhost:8000/config

# Change to a different model
curl -X POST http://localhost:8000/config/model \
  -H "Content-Type: application/json" \
  -d '{"model":"deepseek-r1:8b","base_url":"http://localhost:11434"}'

# Verify it was changed
curl http://localhost:8000/config
```

## Server Status

The TestDriver MCP server includes endpoints for:
- **GET /health** - Server health check and component status
- **GET /config** - Current configuration including selected model
- **GET /config_ui** - Model selection interface
- **GET /chat** - Chat client UI
- **POST /api/test/execute** - Chat with AI (uses selected model)
- **POST /api/test/run** - Generate and run autonomous tests (uses selected model)
- **POST /config/model** - Save model selection

## Automatic Model Usage

Once a model is selected via the config UI:
1. ✅ All chat responses use the selected model
2. ✅ All autonomous test generation uses the selected model
3. ✅ All test execution uses the selected model
4. ✅ Selection persists across server restarts
5. ✅ No manual configuration needed after initial selection

## Notes

- Models are loaded from the hardcoded list in `config_ui.html` (no external API call needed)
- Model availability depends on whether they're installed in your Ollama instance
- If a selected model is not available in Ollama, the chat/test execution will fail with an appropriate error message
- The 120-second timeout allows for complex test generation even with larger models
