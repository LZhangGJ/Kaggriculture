"""Verify the copied opponent pool against the original match receipt."""

from pathlib import Path
import hashlib
import json


ROOT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    pool = json.loads((ROOT / "POOL.json").read_text(encoding="utf-8"))
    receipt = ROOT / pool["source_receipt"]
    assert digest(receipt) == pool["source_receipt_sha256"]
    original = {row["id"]: row for row in json.loads(receipt.read_text(encoding="utf-8"))}
    assert {row["id"] for row in pool["opponents"]} == set(original)
    checked = 0
    for row in pool["opponents"]:
        assert (ROOT / row["entry"]).is_file(), row["entry"]
        assert {item["path"]: item["sha256"] for item in row["files"]} == original[row["id"]]["hashes"]
        for item in row["files"]:
            path = ROOT / item["path"]
            assert path.stat().st_size == item["bytes"], item["path"]
            assert digest(path) == item["sha256"], item["path"]
            checked += 1
    host = ROOT.parent / "tools/p16_match_host"
    tools = json.loads((host / "MANIFEST.json").read_text(encoding="utf-8"))["files"]
    for item in tools:
        path = host / item["path"]
        assert path.stat().st_size == item["bytes"] and digest(path) == item["sha256"], item["path"]
    print(json.dumps({"status": "PASS", "opponents": len(original),
                      "opponent_files": checked, "frozen_tool_files": len(tools),
                      "scope": "Exact bytes from recorded evaluations; no matches rerun."}, indent=2))


if __name__ == "__main__":
    main()
