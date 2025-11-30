# Restart Server Script
# Stops any running server and starts a fresh one

import subprocess
import sys
import time
import psutil

print("🔄 Restarting TestDriver MCP Server...")
print()

# Find and kill any running Python processes on port 8000
print("📍 Checking for running server processes...")
killed = False
for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
    try:
        if proc.info['name'] and 'python' in proc.info['name'].lower():
            cmdline = proc.info['cmdline']
            if cmdline and any('run_server.py' in str(arg) for arg in cmdline):
                print(f"   Stopping process {proc.info['pid']}...")
                proc.terminate()
                proc.wait(timeout=3)
                killed = True
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired):
        pass

if killed:
    print("✅ Previous server stopped")
    time.sleep(1)
else:
    print("✅ No running server found")

print()
print("🚀 Starting new server...")
print()

# Start the new server
subprocess.Popen([sys.executable, "run_server.py"])

print("✅ Server starting in background")
print()
print("📍 Access points:")
print("   • Main: http://localhost:8000")
print("   • Chat: http://localhost:8000/chat")
print("   • API Docs: http://localhost:8000/docs")
print("   • Health: http://localhost:8000/health")
print()
