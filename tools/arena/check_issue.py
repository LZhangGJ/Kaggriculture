"""Trusted workflow reads issue text as JSON, never as commands."""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.arena.intake import manifest_check


def parse(body):
    value = body.split("### Agent manifest", 1)[1].split("\n### ", 1)[0].strip()
    if value.startswith("```"):
        value = value.split("\n", 1)[1].rsplit("```", 1)[0]
    return manifest_check(json.loads(value))


if __name__ == "__main__":
    event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
    try:
        parse(event["issue"]["body"] or "")
        text = "Metadata accepted. Artifact download, hash verification and placement evaluation are pending. This receipt does not mean the agent has been tested."
    except (ValueError, KeyError, IndexError, TypeError):
        text = "Metadata needs correction. Supply the Agent manifest JSON with name, author, version, run argv and valid resources. No code has been executed."
    Path("arena-receipt.txt").write_text("<!-- arena-receipt -->\n" + text)
