"""Runs only inside the unprivileged, network-disabled container."""
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile


def main():
    # Host validates archive paths; recheck before extraction as defense in depth.
    with zipfile.ZipFile("/input.zip") as z:
        for i in z.infolist():
            target = (Path("/work") / i.filename).resolve()
            if not target.is_relative_to(Path("/work")):
                raise ValueError("Unsafe member")
        z.extractall("/work")
    m = json.loads(Path("/manifest.json").read_text())
    if m.get("build"):
        subprocess.run(m["build"], check=True, stdout=sys.stderr, stderr=sys.stderr, timeout=240)
    if sys.argv[1] == "check":
        # Full protocol verification occurs in placement; this receipt proves build only.
        print(json.dumps({"ok": True, "stage": "build"}))
    else:
        print(json.dumps({"arena_ready": True}), flush=True)
        os.environ["ARENA_AGENT_SEED"] = sys.argv[2]
        os.execvp(m["run"][0], m["run"])


if __name__ == "__main__":
    main()
