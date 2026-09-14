"""Deterministic ticks: no model calls, no shell evaluation of manifests."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
import subprocess
import sys
import uuid
from zoneinfo import ZoneInfo

from .intake import submit
from .sandbox import cleanup, preflight, validate_agent
from .schedule import complete, create_placements, plan, save_result
from .store import event, now, read, records, write


def refresh(root):
    """Import explicit local notebook exports; never run notebooks during discovery."""
    root = Path(root)
    imported = []
    for p in sorted((root / "inbox").glob("*/*.json")):
        item = read(p)
        if not item.get("archive_path") or not item.get("manifest"):
            continue
        try:
            result = submit(root, item["archive_path"], item["manifest"])
            imported.append(result["id"])
        except (OSError, ValueError) as e:
            event(root, "intake_failed", str(p), {"reason": str(e)[:500]})
    return imported


def run_one(root, manifest, game, cfg):
    root = Path(root)
    attempts = root / "runs" / manifest["id"] / "attempts" / game["id"]
    attempts.mkdir(parents=True, exist_ok=True)
    # Retry infrastructure failures at most three times; explicit review then required.
    if len(list(attempts.glob("*.json"))) >= 3:
        return
    attempt = uuid.uuid4().hex
    output = attempts / (attempt + ".result")
    record = {"attempt": attempt, "started": now(), "game": game["id"]}
    write(attempts / (attempt + ".json"), record)
    try:
        command = [sys.executable, "-m", "tools.arena.worker", str(root.resolve()), manifest["id"], game["id"], str(output.resolve())]
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       cwd=Path(__file__).resolve().parents[2], timeout=cfg["game_timeout_seconds"] + 30)
        result = read(output)
        save_result(root, manifest, game, result)
        record.update(finished=now(), resolved=result["resolved"], reason=result["reason"])
        if not result["resolved"]:
            event(root, "evaluation_failure", game["id"], {"run":manifest["id"], "reason":result["reason"]})
    except (subprocess.SubprocessError, OSError, ValueError, TypeError) as e:
        record.update(finished=now(), resolved=False, reason=type(e).__name__)
        event(root, "evaluation_failure", game["id"], {"run": manifest["id"], "reason": record["reason"]})
    finally:
        for seat in (0, 1):
            try:
                cleanup("arena-" + game["id"][:24] + "-" + str(seat))
            except (OSError, subprocess.SubprocessError):
                pass
        write(attempts / (attempt + ".json"), record)


def work(root, cfg, remote_only=False):
    root = Path(root)
    pending, placement_by_author, daily = [], {}, []
    for path in sorted((root / "runs").glob("*/manifest.json")):
        m = read(path)
        is_remote = m['kind']=='continuous' or bool(m.get('parent_daily'))
        if is_remote != remote_only:
            continue
        jobs = [(m, g) for g in m["games"] if not read(path.parent / "games" / (g["id"] + ".json"), {}).get("resolved")
                and len(list((path.parent / "attempts" / g["id"]).glob("*.json"))) < 3]
        if m['kind'] == 'daily':
            if not remote_only:
                from .daily_dispatch import assignments
                owners = assignments(root, m)
                jobs = [(mm, g) for mm, g in jobs if owners.get(g['id'], 'coordinator') == 'coordinator']
            daily.extend(jobs)
            continue
        if m["kind"] == "placement":
            author = m["agents"][m["candidate"]]["manifest"]["author"]
            placement_by_author.setdefault(author, []).extend(jobs)
        else:
            pending.append(jobs)
    # Round-robin runs, not draining one submission before all others.
    def interleave(groups):
        result = []
        for i in range(max(map(len, groups), default=0)):
            result.extend(g[i] for g in groups if len(g) > i)
        return result
    authors = sorted(placement_by_author)
    cursor = read(root / "worker_cursor.json", {}).get("author", "")
    authors = [a for a in authors if a > cursor] + [a for a in authors if a <= cursor]
    p, other = interleave([placement_by_author[a] for a in authors]), interleave(pending)
    if remote_only:
        other=[job for group in pending for job in group]  # Drain the first round while its successor stays queued.
    cap = cfg["max_games_per_tick"]
    quota = max(1, round(cap * cfg["placement_fraction"]))
    selected = p[:quota] + other[:cap-quota]
    selected += (p[quota:] + other[cap-quota:])[:cap-len(selected)]
    if daily:
        selected = daily[:cap]  # Finish tournament work before any ordinary queue.
    chosen_placement = [m for m, _ in selected if m["kind"] == "placement"]
    if chosen_placement:
        last = chosen_placement[-1]
        write(root / "worker_cursor.json", {"author": last["agents"][last["candidate"]]["manifest"]["author"]})
    with ThreadPoolExecutor(max_workers=cfg["workers"]) as pool:
        list(pool.map(lambda pair: run_one(root, *pair, cfg), selected))
    for path in sorted((root / "runs").glob("*/manifest.json")):
        m = read(path)
        if complete(root, m):
            event(root, "evaluation_complete", m["id"], {"run": m["id"], "kind": m["kind"]})
            if m["kind"] == "placement":
                ap = root / "agents" / (m["candidate"] + ".json")
                a = read(ap)
                if a["status"] not in ("active", "archived"):
                    a["status"] = "placement-rated"
                write(ap, a)
    return len(selected)


def tick(root):
    root = Path(root)
    from .web_uploads import import_pending
    import_pending(root)
    cfg = read(root / "config.json")
    from .github_sync import sync
    from .public_pool import refresh_daily, apply_ready_versions
    if cfg.get('public_discovery_enabled'):
        try:
            subprocess.run([sys.executable,'-m','tools.arena.discovery',str(root.absolute())],check=True,timeout=600)
        except (OSError,subprocess.SubprocessError) as e:
            event(root,'public_discovery_failed',now()[:10],{'reason':str(e)[:300]})
    imported = sync(root) + refresh(root) + refresh_daily(root)
    today = datetime.now(ZoneInfo(cfg["timezone"])).date().isoformat()
    refresh_marker = root / "private" / ("refresh-" + today + ".json")
    if not cfg.get("public_sources") and cfg.get("public_refresh_commands") and not refresh_marker.exists():
        # Administrator-configured download/conversion tools, never issue-provided code.
        for command in cfg["public_refresh_commands"]:
            if not isinstance(command, list) or not all(isinstance(s, str) for s in command):
                raise ValueError("Refresh command must be trusted argv")
            subprocess.run(command, check=True, timeout=120, stdout=subprocess.DEVNULL)
        imported += refresh(root)
        write(refresh_marker, {"at": now()})
    jobs = 0
    if cfg["execution_enabled"]:
        if not 1 <= cfg["workers"] <= 64 or not 1 <= cfg["max_games_per_tick"] <= 10000:
            raise ValueError("Invalid execution bounds")
        preflight(cfg)
        for a in records(root, "agents"):
            if not a["build_verified"] and not a.get("validation_failure"):
                try:
                    validate_agent(root, a["id"])
                except (ValueError, subprocess.SubprocessError) as e:
                    a["validation_failure"] = str(e)[:500]
                    write(root / "agents" / (a["id"] + ".json"), a)
                    event(root, "validation_failure", a["id"], {"reason": a["validation_failure"]})
        create_placements(root)
        apply_ready_versions(root)
        from .discovery import advance
        advance(root)
        if cfg["daily_enabled"]:
            try:
                plan(root, "daily-" + today)
            except ValueError as e:
                event(root, "daily_pending", today, {"reason": str(e)})
            from .continuous import sync as sync_executors
            sync_executors(root)
        jobs = work(root, cfg)
    from .reporting import build
    build(root, bootstrap=cfg["bootstrap"])
    write(root / "tick.json", {"at": now(), "execution_enabled": cfg["execution_enabled"], "games_attempted": jobs})
    return {"imported": imported, "games_attempted": jobs,
            "attention": [e for e in records(root, "events") if not e["acknowledged"] and e["kind"] not in ("run_planned", "submission")]}
