"""Reject files not emitted by the private report builder."""
import json
from pathlib import Path
import sys


def check(root):
    root = Path(root)
    if not (root / "index.html").exists() or not (root / "data.json").exists():
        raise ValueError("Missing generated export")
    for p in root.rglob("*"):
        if p.is_symlink():
            raise ValueError("Export link forbidden")
        if p.is_file() and (p.suffix not in (".html", ".json", ".md") or p.parent != root):
            raise ValueError("Unexpected export file")
    data = json.loads((root / "data.json").read_text())
    if set(data) != {"schema", "updated", "agents", "runs", "champion", "champion_evidence", "roster"}:
        raise ValueError("Unexpected export schema")
    def visit(value):
        if isinstance(value, dict):
            if set(value) & {"seed", "seeds", "manifest", "archive_path", "archive_asset", "run_command", "source", "token", "secret"}:
                raise ValueError("Private field in export")
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
    visit(data)


if __name__ == "__main__":
    check(sys.argv[1])
