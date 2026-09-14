"""Untrusted packages are data here; all builds happen in the sandbox."""
from __future__ import annotations

import json
import shutil
import stat
import zipfile
import re
from pathlib import Path, PurePosixPath

from .store import digest, event, file_hash, ident, now, read, write


def manifest_check(m):
    origin = m.get("origin", {"kind": "team"})
    if not isinstance(origin, dict) or origin.get("kind") not in ("team", "public"):
        raise ValueError("origin.kind must be team or public")
    if origin["kind"] == "public":
        if not re.fullmatch(r"[A-Za-z0-9_-]+/[A-Za-z0-9_-]+", origin.get("notebook", "")):
            raise ValueError("Public origin requires owner/notebook")
        if not isinstance(origin.get("version"), str) or not origin["version"].strip():
            raise ValueError("Public origin requires an exact notebook version")
    for field in ("name", "author", "version"):
        if not isinstance(m.get(field), str) or not 1 <= len(m[field]) <= 120:
            raise ValueError(f"Missing/invalid {field}")
    for field in ("run", "build"):
        argv = m.get(field, [])
        if not isinstance(argv, list) or not all(isinstance(s, str) and 0 < len(s) <= 4096 for s in argv):
            raise ValueError(f"{field} must be an argv array, not shell text")
    if not m.get("run"):
        raise ValueError("run is required")
    if not isinstance(m.get("resources", {}), dict):
        raise ValueError("resources must be an object")
    for field, maximum in (("cpus", 64), ("memory_mb", 65536), ("scratch_mb", 4096)):
        v = m.get("resources", {}).get(field, 1 if field == "cpus" else 512)
        if type(v) not in (int, float) or not 0 < v <= maximum:
            raise ValueError(f"Invalid resource {field}")
    return m


def inspect_zip(path, maximum=256 * 1024 * 1024):
    total, seen = 0, set()
    with zipfile.ZipFile(path) as z:
        if len(z.infolist()) > 10000:
            raise ValueError("Too many archive entries")
        for info in z.infolist():
            name = info.filename
            p = PurePosixPath(name)
            if (not name or "\\" in name or ":" in name or p.is_absolute()
                    or ".." in p.parts or any(part.endswith((".", " ")) for part in p.parts)):
                raise ValueError("Unsafe archive path")
            key = name.rstrip("/").casefold()
            if key in seen:
                raise ValueError("Duplicate archive path")
            seen.add(key)
            mode = info.external_attr >> 16
            if stat.S_ISLNK(mode) or (stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR)):
                raise ValueError("Archive links/special files forbidden")
            total += info.file_size
            if total > maximum or info.flag_bits & 1:
                raise ValueError("Archive too large or encrypted")
        # Read all members to verify CRC and size, never extract onto the host.
        for info in z.infolist():
            with z.open(info) as f:
                while f.read(1024 * 1024):
                    pass


def submit(root, archive, manifest, receipt=None):
    root, archive = Path(root), Path(archive)
    m = manifest_check(manifest)
    sha = file_hash(archive)
    if m.get("sha256") and m["sha256"].lower() != sha:
        raise ValueError("Archive SHA-256 mismatch")
    inspect_zip(archive)
    # Author/display metadata do not change executable identity.
    identity = {"archive": sha, "run": m["run"], "build": m.get("build", []), "resources": m.get("resources", {})}
    if m.get("origin", {}).get("kind") == "public":
        identity["public_version"] = m["origin"]
    aid = digest(identity)
    sid = ident(receipt) if receipt else digest([aid, m["author"], m["name"], m["version"]])
    existing = read(root / "submissions" / f"{sid}.json")
    if existing:
        if existing["agent"] != aid:
            raise ValueError("Receipt already references another artifact; submit a new revision")
        return existing
    target = root / "artifacts" / f"{sha}.zip"
    if not target.exists():
        tmp = target.with_suffix(".pending")
        shutil.copyfile(archive, tmp)
        if file_hash(tmp) != sha:
            raise ValueError("Archive changed during copy")
        tmp.replace(target)
    if not (root / "agents" / f"{aid}.json").exists():
        write(root / "agents" / f"{aid}.json", {"id": aid, "archive": sha, "manifest": m,
              "status": "registered", "created": now(), "build_verified": False})
    record = {"id": sid, "agent": aid, "author": m["author"], "created": now(), "status": "registered"}
    write(root / "submissions" / f"{sid}.json", record)
    event(root, "submission", sid, {"agent": aid})
    return record


def set_roster(root, entries):
    root = Path(root)
    cfg = read(root / "config.json")
    allowed = {"champion", "established", "public", "counter", "candidate"}
    if not 2 <= len(entries) <= cfg["roster_cap"]:
        raise ValueError("Roster must contain 2..cap agents")
    if len({e["agent"] for e in entries}) != len(entries):
        raise ValueError("Duplicate roster identity")
    for e in entries:
        if e["category"] not in allowed or not e.get("reason"):
            raise ValueError("Category and admission reason required")
        a = read(root / "agents" / (ident(e["agent"]) + ".json"))
        if not a or not a["build_verified"]:
            raise ValueError("Roster agent has not passed sandbox build/protocol checks")
    old = read(root / "roster.json", [])
    write(root / "snapshots" / (digest([now(), old, entries]) + ".json"), {"created": now(), "before": old, "after": entries})
    write(root / "roster.json", entries)
    new_ids = {e["agent"] for e in entries}
    for aid in new_ids | {e["agent"] for e in old}:
        path = root / "agents" / (aid + ".json")
        a = read(path)
        a["status"] = "active" if aid in new_ids else "archived"
        if any(e["agent"] == aid and e["category"] == "public" for e in entries):
            a["public_notebook"] = True
        write(path, a)


def import_issue(root, payload_path, archive):
    """An operator supplies the downloaded archive. Never fetch arbitrary issue URLs."""
    payload = read(payload_path)
    issue = payload.get("issue", payload)
    body = issue["body"]
    marker = "### Agent manifest"
    if marker not in body:
        raise ValueError("Missing Agent manifest section")
    text = body.split(marker, 1)[1].split("\n### ", 1)[0].strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
    m = json.loads(text)
    return submit(root, archive, m, receipt=f"issue-{int(issue['number'])}-{digest(m)[:12]}")
