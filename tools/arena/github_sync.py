"""Read approved private-repository issues/assets. No writes or arbitrary URLs."""
import json
from pathlib import Path
import re
import subprocess
import tempfile

from .check_issue import parse
from .intake import submit
from .store import digest, event, read, write, now


def api(endpoint):
    return json.loads(subprocess.check_output(["gh", "api", endpoint], timeout=30))


def sync(root):
    root = Path(root)
    config = read(root / "config.json").get("github", {})
    repo = config.get("repository", "")
    if not config.get("intake_enabled"):
        return []
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo):
        raise ValueError("Invalid approved repository")
    allowed = set(config.get("allowed_authors", []))
    if not allowed:
        raise ValueError("Configure authorized submission authors")
    imported = []
    for page in range(1, 11):
        issues = api(f"repos/{repo}/issues?state=open&per_page=100&page={page}")
        for issue in issues:
            if not issue["title"].startswith("[Arena]") or issue["user"]["login"] not in allowed or "pull_request" in issue:
                continue
            key = f"issue-{issue['number']}-{digest(issue.get('body'))[:12]}"
            if read(root / "private/github-receipts" / (key + ".json")):
                continue
            try:
                m = parse(issue.get("body") or "")
                asset = m.get("archive_asset")
                if type(asset) is not int or asset <= 0:
                    raise ValueError("Provide a repository release asset ID or use local submit")
                meta = api(f"repos/{repo}/releases/assets/{asset}")
                if meta["size"] > 256*1024*1024:
                    raise ValueError("Archive exceeds 256 MiB")
                if not m.get("sha256"):
                    raise ValueError("Remote intake requires SHA-256")
                with tempfile.TemporaryDirectory() as td:
                    archive = Path(td) / "agent.zip"
                    with open(archive, "wb") as f:
                        subprocess.run(["gh", "api", f"repos/{repo}/releases/assets/{asset}", "-H", "Accept: application/octet-stream"],
                                       stdout=f, check=True, timeout=120)
                    result = submit(root, archive, m, receipt=key)
                write(root / "private/github-receipts" / (key + ".json"), {"submission":result["id"], "at":now()})
                imported.append(result["id"])
            except (ValueError, KeyError, subprocess.SubprocessError) as e:
                event(root, "github_intake_failure", key, {"issue":issue["number"], "reason":str(e)[:300]})
        if len(issues) < 100:
            break
    return imported
