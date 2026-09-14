"""Trusted protocol example: one request and action per line."""
import json
import sys

for line in sys.stdin:
    request = json.loads(line)
    print(json.dumps({"farmer": ["WAIT"]}), flush=True)
