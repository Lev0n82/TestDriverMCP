import requests
import json
import sys

API_URL = "http://localhost:8000/api/test/execute/stream"

requirements = """
Platform/Application EDCS
Release Name EDCS 25.12
Release Date 2025-12-04
Category Release
Label 25.12
Code Freeze 2025-11-14

MCU Projects\u200bCGRT Enhancements: 115399 - As a Ministry Admin, I would like to be able to access CSER and CGRT External Audit Feature

PFAAM-C/U/Consents Enhancements: 42483 - PFAAMC - Include Program Delivery Location (Campus and Address)
113944 - PFAAMC - Add new attachment option to new submission dropdown
115400 - PFAAMC - As a Ministry Admin, I would like to generate a monthly report of new and modified submissions within the UI itself.
120500 - PFAAMC - SR - DB script to update Implementation No Later Than Date
120593 - PFAAMC - SR - DB script to adjust tuition fee for LACI03004

EDU Projects\u200bDual Credits Enhancements: 113582 - As a EDCS User, I want the columns values and titles in reports 504, 504B, 504C, and 504I to be updated
113583 - As a EDCS user, I want columns B and C in the 505 report to be removed
"""

payload = {
    "requirements": requirements,
    "timestamp": "manual-test"
}

print(f"Posting to {API_URL} with payload length {len(requirements)}")

try:
    resp = requests.post(API_URL, json=payload, stream=True, timeout=600)
except Exception as e:
    print(f"Request error: {e}")
    sys.exit(2)

print(f"HTTP status: {resp.status_code}")

if resp.status_code != 200:
    try:
        print("Response body:")
        print(resp.text)
    except Exception:
        pass
    sys.exit(1)

print("Streaming events:")
for line in resp.iter_lines():
    if not line:
        continue
    try:
        event = json.loads(line)
        print(json.dumps(event, indent=2, ensure_ascii=False))
    except Exception:
        print("<non-json line>", line)

print("Stream ended")
