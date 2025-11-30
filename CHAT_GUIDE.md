# 💬 TestDriver MCP Chat Client - Quick Guide

## 🚀 Getting Started

### 1. Make sure the server is running:
```powershell
.\venv\Scripts\Activate.ps1
python restart_server.py
```

### 2. Open the chat client in your browser:
**http://localhost:8000/chat**

---

## ✅ What's Working Now

The server is configured and running with:
- ✅ **CORS enabled** - Chat client can communicate with server
- ✅ **Ollama connected** - Local AI vision ready
- ✅ **Playwright/Selenium** - Browser automation ready
- ✅ **FastAPI** - Modern async web framework
- ✅ **Interactive API docs** - http://localhost:8000/docs

---

## 🎯 Try These Commands in the Chat

### Quick Actions (Click the buttons):
- 🔐 **Login Test** - "Generate a test plan for a login page"
- 📝 **Form Test** - "Create a test for form validation"
- ♿ **Accessibility** - "Test accessibility features"
- ❤️ **Health Check** - "Check server health"

### Custom Test Requests:
- "Test the checkout flow on an e-commerce site"
- "Validate user registration with email confirmation"
- "Check mobile responsiveness of navigation menu"
- "Test password reset functionality"
- "Verify error messages for invalid inputs"

### System Commands:
- "Check server health" - View server status
- "Show current config" - View configuration
- "What vision provider am I using?" - Check Ollama status

---

## 📊 What Happens When You Submit

1. **Message sent** → Server receives your test requirements
2. **Test ID created** → Unique identifier for tracking
3. **Response returned** → Confirmation with next steps
4. **Processing begins** → Server analyzes with Ollama (simulated)

**Example Response:**
```
✅ Test request accepted!

📋 Test ID: test-20251119-192030
📝 Requirements: "Test login functionality"
⏱️ Estimated time: 2-5 minutes

🎯 Next Steps:
1. Analyzing requirements with AI
2. Generating comprehensive test plan
3. Setting up test environment
4. Executing tests with Playwright/Selenium
5. Validating results with Ollama vision
6. Generating detailed report

📊 Status URL: /api/test/test-20251119-192030/status

💡 The test framework is now processing your requirements 
   using Ollama for AI vision analysis!
```

---

## 🔧 Troubleshooting

### "Can't reach this page" or "Connection refused"
**Fix:** Make sure the server is running
```powershell
cd c:\TestDriverMCP
.\venv\Scripts\Activate.ps1
python restart_server.py
```

### Chat sends but gets no response
**Fix:** Check browser console (F12) for errors. The server logs will show:
```
INFO: 127.0.0.1:xxxxx - "POST /api/test/execute HTTP/1.1" 200 OK
```

### "Server Offline" status
**Fix:** Verify server is accessible:
```powershell
Invoke-WebRequest http://localhost:8000/health
```

---

## 🎨 Chat Client Features

✅ **Beautiful UI** - Modern gradient design with animations  
✅ **Message history** - All your conversations saved in session  
✅ **Quick actions** - Pre-configured test scenarios  
✅ **Typing indicators** - Visual feedback during processing  
✅ **Status badge** - Shows connection status  
✅ **Auto-scroll** - Automatically scrolls to new messages  
✅ **Enter to send** - Press Enter (Shift+Enter for new line)  

---

## 🌟 Next Steps

The chat client is a **frontend interface** that communicates with the MCP server. 

**Current State:**
- ✅ Chat UI fully functional
- ✅ Server receiving and responding to requests
- ✅ Ollama configured and connected
- ⚠️ Full test execution engine is a placeholder (shows what will happen)

**To implement full test execution:**
You would need to integrate the actual TestDriver framework components that are in the repository (the Python modules for test generation, execution, healing, etc.)

---

## 📱 Quick Access URLs

- **Chat Client**: http://localhost:8000/chat
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **Configuration**: http://localhost:8000/config
- **Server Root**: http://localhost:8000

---

**Enjoy chatting with TestDriver MCP! 🚀**
