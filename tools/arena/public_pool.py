"""Daily per-notebook refresh with visible failures and version-safe rollover."""
from datetime import datetime
from pathlib import Path
import subprocess
from zoneinfo import ZoneInfo

from .intake import submit, set_roster
from .store import digest, event, ident, now, read, write


def refresh_daily(root):
    root = Path(root)
    cfg = read(root / "config.json")
    date = datetime.now(ZoneInfo(cfg["timezone"])).date().isoformat()
    result = []
    for source in cfg.get("public_sources", []):
        if source.get('enabled') is False:
            continue
        sid = ident(source["id"])
        path = root / "private/public-refresh" / (sid + ".json")
        old = read(path, {})
        config_hash = digest(source)
        if old.get("date") == date and old.get("config_hash") == config_hash:
            continue
        state = {**old, "id":sid, "notebook":source["notebook"], "date":date,
                 "config_hash":config_hash, "attempted":now(), "status":"failed"}
        try:
            command = source["command"]
            if not isinstance(command, list) or not command or not all(isinstance(v, str) for v in command):
                raise ValueError("Public source command must be trusted argv")
            # Remove no files. A successful adapter must replace its receipt each run.
            output = Path(source["export"])
            before = output.stat().st_mtime_ns if output.exists() else None
            subprocess.run(command, check=True, timeout=120, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if not output.exists() or output.stat().st_mtime_ns == before:
                raise ValueError("Adapter did not refresh its export receipt")
            exported = read(output)
            origin = exported["manifest"].get("origin", {})
            if origin.get("kind") != "public" or origin.get("notebook") != source["notebook"]:
                raise ValueError("Export notebook identity differs from configured source")
            accepted = submit(root, exported["archive_path"], exported["manifest"])
            changed = old.get("agent") != accepted["agent"]
            state.update(status="updated" if changed else "unchanged", agent=accepted["agent"],
                         version=origin["version"], successful=now())
            state.pop("error", None)
            result.append(accepted["id"])
        except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as e:
            state["error"] = type(e).__name__
            event(root, "public_refresh_failed", sid+":"+date, {"source":sid, "reason":str(e)[:300]})
        write(path, state)
    return result


def apply_ready_versions(root):
    """Update next roster only. Frozen run manifests are never edited."""
    root = Path(root)
    roster = read(root / "roster.json", [])
    changed = False
    for path in sorted((root / "private/public-refresh").glob("*.json")):
        status = read(path)
        if status.get("status") not in ("updated", "unchanged"):
            continue
        aid = status.get("agent")
        new = read(root / "agents" / (aid + ".json"))
        if not new or not new.get("build_verified"):
            continue
        for entry in roster:
            previous = read(root / "agents" / (entry["agent"] + ".json"))
            origin = previous.get("manifest", {}).get("origin", {})
            if origin.get("notebook") == status["notebook"] and entry["agent"] != aid:
                if any(e["agent"] == aid for e in roster):
                    continue
                entry.update(agent=aid, reason="Daily public refresh: " + status["notebook"] + " version " + status["version"])
                changed = True
    if changed:
        set_roster(root, roster)
    return changed
