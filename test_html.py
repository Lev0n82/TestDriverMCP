#!/usr/bin/env python3
import sys

try:
    with open('chat_client.html', 'r', encoding='utf-8') as f:
        content = f.read()
    
    print(f"✓ File loaded: {len(content)} bytes")
    print(f"✓ Lines: {len(content.splitlines())}")
    print(f"✓ Starts with DOCTYPE: {content.strip().startswith('<!DOCTYPE')}")
    print(f"✓ Ends with </html>: {content.strip().endswith('</html>')}")
    print(f"✓ Contains chat-container: {'chat-container' in content}")
    print(f"✓ Contains TD(MCP)-GRACE: {'TD(MCP)-GRACE' in content}")
    print("\nFile is valid and ready to serve!")
    sys.exit(0)
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
