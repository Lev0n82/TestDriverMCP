#!/usr/bin/env python3
import sys
import os

# Change to the script directory
os.chdir(r'c:\TestDriverMCP')

# Add current directory to path
sys.path.insert(0, '.')

print("Testing server imports...")

try:
    print("✓ Importing FastAPI...")
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse, HTMLResponse
    print("✓ FastAPI imported")
    
    print("✓ Importing other dependencies...")
    import uvicorn
    from fastapi.middleware.cors import CORSMiddleware
    print("✓ Dependencies imported")
    
    print("\n✓ All imports successful!")
    print("✓ Server code should work")
    
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"✗ Unexpected error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
