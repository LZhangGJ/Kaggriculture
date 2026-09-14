"""One idempotent receipt comment; event strings never enter shell syntax."""
import json
import os
from pathlib import Path
import subprocess

e = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
repo = e["repository"]["full_name"]
number = int(e["issue"]["number"])
endpoint = f"repos/{repo}/issues/{number}/comments"
comments = json.loads(subprocess.check_output(["gh", "api", endpoint, "--paginate", "--slurp"]))
previous = next((c for page in comments for c in page if c.get("user", {}).get("login") == "github-actions[bot]" and c["body"].startswith("<!-- arena-receipt -->")), None)
body = Path("arena-receipt.txt").read_text()
target = f"repos/{repo}/issues/comments/{previous['id']}" if previous else endpoint
subprocess.run(["gh", "api", target, "-X", "PATCH" if previous else "POST", "--input", "-"],
               input=json.dumps({"body": body}), text=True, check=True)
