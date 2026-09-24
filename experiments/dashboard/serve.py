#!/usr/bin/env python3
"""Read-only live dashboard for the Kaggriculture training pipeline."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import signal
import statistics
import subprocess
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "work"
BC_ROOT = ROOT / "data" / "bc" / "midgame-v1"
STUDENT_ROOT = WORK / "student-v1"
REPLAY_ROOT = WORK / "latest_top_routes_20260922_top100_e12"
OLD_REPLAY_ROOT = WORK / "latest_top_routes_20260922_top60_e8" / "replays"
RAW_REPLAY_ROOT = ROOT / "data" / "replays" / "raw"
NATIVE_ROOT = ROOT / "experiments" / "native_opponents"
NEW_OPPONENT_ROOT = WORK / "new_public_opponents" / "eval"
RUNTIME_ROOT = WORK / "dashboard"
CONTINUOUS_RL_ROOTS = (WORK / "continuous-rl-nointraday-expiry", WORK / "continuous-rl-nointraday", WORK / "continuous-rl")
ECONOMIC_RL_ROOT = WORK / "continuous-rl-economic-v1"
CONTINUOUS_RL_STALE_SECONDS = 30.0
CONTINUOUS_RL_LEGACY_STALE_SECONDS = 300.0
ACTION_SCHEMA = "autoregressive-action-event-bc-v3"
ACTION_TRAINING_TASK = "autoregressive_action_event_actor_v3"
ACTION_MANIFEST_SCHEMAS = {
    ACTION_SCHEMA,
    "student_state_dagger_action_event_v3",
}
FORMAL_RL_ALGORITHMS = {
    "on_policy_clipped_ppo_day_bundle_stratified_loo_unscaled_advantage",
    "on_policy_clipped_ppo_day_bundle_paired_seed_loo_unscaled_advantage",
    "on_policy_clipped_ppo_day_bundle_paired_seed_loo_unscaled_advantage_day_state_crossfit_hgb",
}
FORMAL_RL_BACKEND = "native_cpp"
FORMAL_RL_SCOPES = {"native_cpp_mixed_pool", "native_cpp_job_batch"}
RL_ALGORITHM_PREFIX = "on_policy_clipped_ppo"

_CACHE_LOCK = threading.Lock()
_PAYLOAD_CACHE: tuple[float, dict[str, Any]] = (0.0, {})
_RATE_SAMPLES: dict[str, tuple[float, int]] = {}
_JSON_CACHE: dict[Path, tuple[tuple[int, int], Any]] = {}
_HASH_CACHE: dict[Path, tuple[tuple[int, int], str]] = {}
_CPU_SAMPLE: tuple[int, int] | None = None
_NPU_CACHE: tuple[float, dict[str, Any]] = (0.0, {})


def _iso(timestamp: float | None) -> str | None:
    if not timestamp:
        return None
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat(timespec="seconds")


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _json(path: Path) -> Any:
    try:
        stat = path.stat()
        stamp = (stat.st_mtime_ns, stat.st_size)
        cached = _JSON_CACHE.get(path)
        if cached and cached[0] == stamp:
            return cached[1]
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    _JSON_CACHE[path] = (stamp, value)
    return value


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    index = (len(values) - 1) * q
    lo, hi = math.floor(index), math.ceil(index)
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - index) + values[hi] * (index - lo)


def _episode_ids(paths: Iterable[Path]) -> set[int]:
    ids: set[int] = set()
    for path in paths:
        match = re.fullmatch(r"episode-(\d+)-replay\.json", path.name)
        if match:
            ids.add(int(match.group(1)))
    return ids


def _rate(name: str, count: int, now: float) -> float | None:
    previous = _RATE_SAMPLES.get(name)
    _RATE_SAMPLES[name] = (now, count)
    if not previous or now <= previous[0] or count < previous[1]:
        return None
    return (count - previous[1]) / (now - previous[0])


def _system() -> dict[str, Any]:
    global _CPU_SAMPLE
    load = [None, None, None]
    try:
        load = [float(value) for value in Path("/proc/loadavg").read_text().split()[:3]]
    except (OSError, ValueError):
        pass
    cpu_fraction = None
    try:
        values = [int(value) for value in Path("/proc/stat").read_text().splitlines()[0].split()[1:]]
        total_ticks = sum(values)
        idle_ticks = values[3] + (values[4] if len(values) > 4 else 0)
        if _CPU_SAMPLE and total_ticks > _CPU_SAMPLE[0]:
            cpu_fraction = 1.0 - (idle_ticks - _CPU_SAMPLE[1]) / (total_ticks - _CPU_SAMPLE[0])
        _CPU_SAMPLE = (total_ticks, idle_ticks)
    except (OSError, ValueError, IndexError):
        pass
    memory: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, value = line.split(":", 1)
            memory[key] = int(value.split()[0]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    total = memory.get("MemTotal", 0)
    available = memory.get("MemAvailable", 0)
    disk = shutil.disk_usage(ROOT)
    return {
        "cpu_count": os.cpu_count(),
        "load_1m": load[0],
        "load_5m": load[1],
        "load_15m": load[2],
        "cpu_used_fraction": cpu_fraction,
        "memory_total_bytes": total,
        "memory_used_bytes": max(0, total - available),
        "memory_used_fraction": (total - available) / total if total else None,
        "disk_total_bytes": disk.total,
        "disk_used_bytes": disk.used,
        "disk_free_bytes": disk.free,
        "disk_used_fraction": disk.used / disk.total if disk.total else None,
    }


def _npu() -> dict[str, Any]:
    """Sample the installed vendor collector, cached to keep HTTP cheap."""
    global _NPU_CACHE
    now = time.monotonic()
    if now - _NPU_CACHE[0] < 30:
        return _NPU_CACHE[1]
    try:
        output = subprocess.run(
            ["npu-smi", "info"], check=False, capture_output=True, text=True,
            timeout=4,
        ).stdout
    except (OSError, subprocess.TimeoutExpired):
        output = ""
    devices: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for line in output.splitlines():
        columns = [part.strip() for part in line.strip().strip("|").split("|")]
        if len(columns) != 3:
            continue
        header = re.match(r"^(\d+)\s+(\S+)", columns[0])
        if header and not re.match(r"^\d+\s*$", columns[0]):
            values = columns[2].split()
            current = {
                "id": int(header.group(1)), "name": header.group(2),
                "health": columns[1],
                "power_watts": float(values[0]) if values else None,
                "temperature_c": float(values[1]) if len(values) > 1 else None,
            }
            devices.append(current)
            continue
        if current is None or not re.fullmatch(r"[0-9A-Fa-f:.]+", columns[1]):
            continue
        detail = re.match(
            r"^([0-9.]+)\s+(\d+)\s*/\s*(\d+)\s+(\d+)\s*/\s*(\d+)",
            columns[2],
        )
        if detail:
            current.update({
                "aicore_fraction": float(detail.group(1)) / 100.0,
                "memory_used_mb": int(detail.group(2)),
                "memory_total_mb": int(detail.group(3)),
                "hbm_used_mb": int(detail.group(4)),
                "hbm_total_mb": int(detail.group(5)),
            })
    utilizations = [row["aicore_fraction"] for row in devices
                    if isinstance(row.get("aicore_fraction"), (int, float))]
    hbm_used = sum(int(row.get("hbm_used_mb", 0) or 0) for row in devices)
    hbm_total = sum(int(row.get("hbm_total_mb", 0) or 0) for row in devices)
    value = {
        "available": bool(devices), "devices": devices,
        "mean_aicore_fraction": statistics.mean(utilizations) if utilizations else None,
        "max_aicore_fraction": max(utilizations) if utilizations else None,
        "hbm_used_mb": hbm_used, "hbm_total_mb": hbm_total,
        "hbm_used_fraction": hbm_used / hbm_total if hbm_total else None,
        "sampled_at": _iso(time.time()),
    }
    _NPU_CACHE = (now, value)
    return value


def _family_map() -> dict[str, str]:
    result: dict[str, str] = {}
    for name in (
        "manifest_public7_training.json",
        "manifest_public3_families_training.json",
        "manifest_all10_training.json",
    ):
        data = _json(NEW_OPPONENT_ROOT / name)
        for row in data.get("agents", []) if isinstance(data, dict) else []:
            slug = row.get("slug")
            family = row.get("source_family")
            if isinstance(slug, str) and isinstance(family, str):
                result[slug] = family
    return result


def _corpus() -> dict[str, Any]:
    games = errors = shards = 0
    timed_games = 0
    wall_seconds = 0.0
    latest = 0.0
    by_source: dict[str, int] = defaultdict(int)
    opponents: set[str] = set()
    families: set[str] = set()
    family_map = _family_map()
    accepted_files: list[str] = []
    for path in sorted(BC_ROOT.glob("train-*.json")):
        data = _json(path)
        if not isinstance(data, dict) or data.get("trajectory_format") != "kaggriculture-bc-v1":
            continue
        summary = data.get("summary")
        if not isinstance(summary, dict):
            continue
        if isinstance(summary.get("candidate"), dict):
            summary = summary["candidate"]
        shard_games = 0
        for opponent, row in summary.items():
            if not isinstance(row, dict):
                continue
            count = int(row.get("games", 0) or 0)
            shard_games += count
            errors += int(row.get("errors", 0) or 0)
            opponents.add(str(opponent))
            families.add(family_map.get(str(opponent), str(opponent)))
        if not shard_games:
            continue
        source = str(data.get("opponent_source", "unknown"))
        if source == "unknown" and isinstance(data.get("opponents"), dict):
            source = "public7-anchor-legacy"
        by_source[source] += shard_games
        games += shard_games
        shard_wall = float(data.get("wall_seconds", 0.0) or 0.0)
        if shard_wall > 0:
            timed_games += shard_games
            wall_seconds += shard_wall
        shards += 1
        latest = max(latest, _mtime(path))
        accepted_files.append(path.name)
    return {
        "format": "kaggriculture-bc-v1",
        "shards": shards,
        "games": games,
        "frames": games * 719,
        "errors": errors,
        "opponents": len(opponents),
        "families": len(families),
        "by_source": dict(sorted(by_source.items())),
        "latest_games_per_second": (
            timed_games / wall_seconds if wall_seconds > 0 else None
        ),
        "latest_at": _iso(latest),
        "latest_age_seconds": max(0.0, time.time() - latest) if latest else None,
        "accepted_manifest_count": len(accepted_files),
    }


def _replays(now: float) -> dict[str, Any]:
    selection = _json(REPLAY_ROOT / "selection.json")
    target_map = selection.get("episode_targets", {}) if isinstance(selection, dict) else {}
    target_ids = {int(value) for value in target_map if str(value).isdigit()}
    downloaded_paths = list((REPLAY_ROOT / "replays").glob("episode-*-replay.json"))
    downloaded = _episode_ids(downloaded_paths)
    previous = _episode_ids(RAW_REPLAY_ROOT.glob("episode-*-replay.json"))
    previous.update(_episode_ids(OLD_REPLAY_ROOT.glob("episode-*-replay.json")))
    target_downloaded = downloaded & target_ids if target_ids else downloaded
    already_local = target_ids & previous
    newest = max((_mtime(path) for path in downloaded_paths), default=0.0)
    count = len(target_downloaded)
    per_second = _rate("replays", count, now)
    return {
        "target": len(target_ids),
        "downloaded": count,
        "effective_available": len(target_ids & (downloaded | previous)),
        "new_ids": len(target_downloaded - previous),
        "downloaded_duplicates": len(target_downloaded & previous),
        "already_local_targets": len(already_local),
        "remaining": max(0, len(target_ids) - len(target_ids & (downloaded | previous))),
        "progress": count / len(target_ids) if target_ids else None,
        "files_per_second": per_second,
        "latest_at": _iso(newest),
        "latest_age_seconds": max(0.0, now - newest) if newest else None,
        "active": bool(newest and now - newest < 180 and count < len(target_ids)),
    }


def _suffix() -> dict[str, Any]:
    states: dict[str, dict[str, Any]] = {}
    files = 0
    latest = 0.0
    for path in sorted(WORK.glob("r1-real-suffix-*.json")):
        if "smoke" in path.name:
            continue
        data = _json(path)
        if not isinstance(data, dict) or data.get("format") != "kaggriculture-r1-real-suffix-paired-v1":
            continue
        files += 1
        latest = max(latest, _mtime(path))
        for state in data.get("states", []):
            if isinstance(state, dict) and isinstance(state.get("state_id"), str):
                states[state["state_id"]] = state
    candidate_count = replica_count = 0
    noise: list[float] = []
    paired_abs: list[float] = []
    for state in states.values():
        candidates = state.get("candidates", [])
        candidate_count += len(candidates)
        for candidate in candidates:
            replicas = candidate.get("replicas", []) if isinstance(candidate, dict) else []
            replica_count += len(replicas)
            if candidate.get("canonical_anchor"):
                continue
            values = [
                float(row["paired_margin_delta"])
                for row in replicas
                if isinstance(row, dict) and isinstance(row.get("paired_margin_delta"), (int, float))
            ]
            paired_abs.extend(abs(value) for value in values)
            if len(values) >= 2:
                noise.append(statistics.stdev(values))
    return {
        "files": files,
        "states": len(states),
        "candidates": candidate_count,
        "replicas": replica_count,
        "mean_candidates_per_state": candidate_count / len(states) if states else None,
        "median_paired_margin_noise": _percentile(noise, 0.5),
        "p90_paired_margin_noise": _percentile(noise, 0.9),
        "median_absolute_paired_delta": _percentile(paired_abs, 0.5),
        "latest_at": _iso(latest),
        "latest_age_seconds": max(0.0, time.time() - latest) if latest else None,
    }


def _flatten_numbers(value: Any, prefix: str = "") -> dict[str, float]:
    result: dict[str, float] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            result.update(_flatten_numbers(child, child_prefix))
    elif isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
        result[prefix.lower()] = float(value)
    return result


def _metric(numbers: dict[str, float], names: tuple[str, ...]) -> float | None:
    for name in names:
        if name in numbers:
            return numbers[name]
    for key, value in numbers.items():
        if any(key.endswith(f".{name}") for name in names):
            return value
    return None


def _sha256(path: Path) -> str | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    stamp = (stat.st_mtime_ns, stat.st_size)
    cached = _HASH_CACHE.get(path)
    if cached and cached[0] == stamp:
        return cached[1]
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return None
    value = digest.hexdigest()
    _HASH_CACHE[path] = (stamp, value)
    return value


def _do_not_use_entries() -> set[str]:
    """Return only relative artifacts explicitly named by local audit markers."""
    entries: set[str] = set()
    for path in sorted(STUDENT_ROOT.glob("DO_NOT_USE*")):
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for quoted in re.findall(r"`([^`]+)`", text):
            name = quoted.strip().rstrip("/")
            candidate = Path(name)
            if name and not candidate.is_absolute() and ".." not in candidate.parts:
                entries.add(candidate.as_posix())
    return entries


def _is_do_not_use(path: Path, entries: set[str]) -> bool:
    try:
        relative = path.relative_to(STUDENT_ROOT).as_posix().rstrip("/")
    except ValueError:
        return True
    return any(relative == entry or relative.startswith(entry + "/") for entry in entries)


def _accepted_v3_shards(entries: set[str]) -> dict[str, dict[str, str]]:
    """Index accepted action-event ABI-v3 shards by exact manifest digest."""
    accepted: dict[str, dict[str, str]] = {}
    for path in sorted(STUDENT_ROOT.rglob("manifest.json")):
        if _is_do_not_use(path, entries):
            continue
        data = _json(path)
        schema = data.get("schema", {}) if isinstance(data, dict) else {}
        if not (
            isinstance(schema, dict)
            and data.get("validation_status") == "accepted"
            and schema.get("schema_name") in ACTION_MANIFEST_SCHEMAS
        ):
            continue
        manifest_sha256 = _sha256(path)
        schema_sha256 = data.get("schema_sha256")
        if manifest_sha256:
            accepted[manifest_sha256] = {
                "name": path.parent.relative_to(STUDENT_ROOT).as_posix(),
                "schema_name": str(schema.get("schema_name")),
                "schema_sha256": str(schema_sha256 or ""),
            }
    return accepted


def _valid_v3_metric(record: Any, accepted: dict[str, dict[str, str]]) -> bool:
    """Fail closed unless a BC metric proves action-event-v3 shard lineage."""
    if not isinstance(record, dict):
        return False
    status = str(record.get("status", "")).upper()
    manifest_sha256 = record.get("shard_manifest_sha256")
    shard = accepted.get(manifest_sha256) if isinstance(manifest_sha256, str) else None
    return bool(
        record.get("training_task") == ACTION_TRAINING_TASK
        and record.get("schema_name") == ACTION_SCHEMA
        and not status.startswith(("REJECT", "FAIL"))
        and shard
    )


def _training_records() -> tuple[list[dict[str, Any]], list[str], float,
                                 dict[str, dict[str, str]], int]:
    records: list[dict[str, Any]] = []
    sources: list[str] = []
    latest = 0.0
    excluded = 0
    blocked = _do_not_use_entries()
    accepted = _accepted_v3_shards(blocked)
    candidates = list(STUDENT_ROOT.rglob("action-event-v3*.metrics.json"))
    candidates += list(STUDENT_ROOT.rglob("action-event-v3*.metrics.jsonl"))
    candidates.sort(key=_mtime)
    for path in candidates:
        if _is_do_not_use(path, blocked):
            excluded += 1
            continue
        try:
            if path.suffix == ".jsonl":
                values = []
                with path.open(encoding="utf-8") as handle:
                    for line in handle:
                        try:
                            if line.strip():
                                values.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
            else:
                value = _json(path)
                if isinstance(value, list):
                    values = value
                elif isinstance(value, dict) and isinstance(value.get("records"), list):
                    values = value["records"]
                elif isinstance(value, dict) and isinstance(value.get("history"), list):
                    values = value["history"]
                else:
                    values = [value]
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            excluded += 1
            continue
        valid = [value for value in values[-500:] if _valid_v3_metric(value, accepted)]
        if not valid:
            excluded += 1
            continue
        latest = max(latest, _mtime(path))
        sources.append(path.relative_to(STUDENT_ROOT).as_posix())
        records.extend(valid)
    return records[-1000:], sources, latest, accepted, excluded


def _latest_training_progress(
    accepted: dict[str, dict[str, str]], blocked: set[str]
) -> tuple[dict[str, Any], str | None, float]:
    valid: list[tuple[float, Path, dict[str, Any]]] = []
    for path in STUDENT_ROOT.rglob("*.progress.json"):
        if _is_do_not_use(path, blocked):
            continue
        value = _json(path)
        if _valid_v3_metric(value, accepted):
            valid.append((_mtime(path), path, value))
    if not valid:
        return {}, None, 0.0
    latest, path, value = max(valid, key=lambda row: row[0])
    return value, path.relative_to(STUDENT_ROOT).as_posix(), latest


def _training() -> dict[str, Any]:
    raw_records, sources, latest, accepted, excluded = _training_records()
    progress, progress_source, progress_time = _latest_training_progress(
        accepted, _do_not_use_entries())
    series: list[dict[str, Any]] = []
    for index, record in enumerate(raw_records):
        numbers = _flatten_numbers(record)
        loss = _metric(numbers, (
            "train.final.loss", "train.final_loss", "final_loss", "loss", "train_loss", "total_loss",
        ))
        val_loss = _metric(numbers, (
            "heldout.final.loss", "heldout.loss", "heldout.final_loss", "val_loss", "validation_loss",
        ))
        top1 = _metric(numbers, (
            "heldout.final.masked_accuracy", "heldout.final_top1", "heldout.top1",
            "train.final.masked_accuracy", "train.final_top1", "top1", "val_top1",
            "top1_accuracy", "validation_top1",
        ))
        legal = _metric(numbers, (
            "teacher_label_legal_fraction", "label_legal_fraction", "legal_rate",
            "valid_rate", "validation_legal_rate",
        ))
        if legal is None:
            illegal = _metric(numbers, ("illegal_rate", "invalid_rate"))
            legal = 1.0 - illegal if illegal is not None else None
        throughput = _metric(numbers, (
            "samples_per_second", "samples_per_sec", "examples_per_second",
            "timing.event_epochs_per_second", "event_epochs_per_second",
            "timing.state_epochs_per_second", "state_epochs_per_second",
            "timing.slot_epochs_per_second", "slot_epochs_per_second",
            "steps_per_second", "games_per_second", "throughput",
        ))
        epoch = _metric(numbers, ("epoch", "step", "iteration", "global_step"))
        if any(value is not None for value in (loss, val_loss, top1, legal, throughput)):
            series.append({
                "x": epoch if epoch is not None else index,
                "loss": loss,
                "val_loss": val_loss,
                "top1": top1,
                "legal": legal,
                "throughput": throughput,
            })
    latest_row = series[-1] if series else {}
    latest_record = raw_records[-1] if raw_records else {}
    epoch_series = [
        {"x": row.get("epoch"), "loss": row.get("train_loss")}
        for row in progress.get("events", [])
        if isinstance(row, dict) and isinstance(row.get("train_loss"), (int, float))
    ]
    progress_age = max(0.0, time.time() - progress_time) if progress_time else None
    progress_status = str(progress.get("status", "")).lower()
    reported_status = str(latest_record.get("status", "")).upper()
    return {
        "records": len(series),
        "sources": sources,
        "accepted_v3_shards": len(accepted),
        "excluded_metric_files": excluded,
        "latest": latest_row,
        "series": series[-120:],
        "epoch_series": epoch_series[-120:],
        "progress_source": progress_source,
        "progress_age_seconds": progress_age,
        "completed_epochs": progress.get("completed_epochs", 0),
        "epochs": progress.get("epochs"),
        "phase": progress.get("phase"),
        "current_train_loss": epoch_series[-1]["loss"] if epoch_series else None,
        "latest_at": _iso(max(latest, progress_time)),
        "latest_age_seconds": (
            max(0.0, time.time() - max(latest, progress_time))
            if latest or progress_time else None
        ),
        "model": (
            progress.get("metrics_file")
            if progress_status == "complete" and progress.get("metrics_file")
            else progress_source if progress_time > latest
            else (sources[-1] if sources else None)
        ),
        "device": latest_record.get("device", "cpu" if progress else None),
        "parameters": progress.get("parameters", latest_record.get("parameters")),
        "states": progress.get("states", latest_record.get("states")),
        "decisions": progress.get(
            "events", latest_record.get("events", latest_record.get("slots"))),
        "class_recall": latest_record.get("heldout", {}).get("final", {}).get("class_recall", {}),
        "status": (
            "running" if progress_status == "running" and progress_age is not None and progress_age < 600
            else "ready" if reported_status in {"PASS", "COMPLETE", "COMPLETED"}
            else "rejected" if reported_status.startswith("REJECT")
            else "failed" if reported_status.startswith("FAIL")
            else "running" if latest and time.time() - latest < 600
            else "idle" if series
            else "waiting"
        ),
    }


def _slot_pipeline() -> dict[str, Any]:
    paths = [STUDENT_ROOT / "action-event-v3-actor-owned-width3074-full" / "manifest.json"]
    paths += sorted(STUDENT_ROOT.glob(
        "action-event-v3-actor-owned-dagger-r*-seeds*/manifest.json"))
    manifests: list[tuple[Path, dict[str, Any]]] = []
    latest = 0.0
    for path in paths:
        data = _json(path)
        if not isinstance(data, dict) or data.get("validation_status") != "accepted":
            continue
        schema = data.get("schema", {})
        if (not isinstance(schema, dict) or
                schema.get("schema_name") not in ACTION_MANIFEST_SCHEMAS):
            continue
        manifests.append((path, data))
        latest = max(latest, _mtime(path))
    base = next((data for path, data in manifests if "dagger-r" not in path.as_posix()), {})
    dagger = [data for path, data in manifests if "dagger-r" in path.as_posix()]
    dagger_counts = {
        key: sum(int(row.get("counts", {}).get(key, 0) or 0) for row in dagger)
        for key in ("games", "states", "events")
    }
    required_checks = ("execution_contract", "label_fixed_point",
                       "resource_replay_exact", "no_identity_in_model_inputs")
    validation_failures = sum(
        any(data.get("validation", {}).get(key) != "PASS" for key in required_checks)
        for _path, data in manifests
    )
    return {
        "accepted_shards": len(manifests),
        "base_counts": base.get("counts", {}) if isinstance(base, dict) else {},
        "dagger_rounds": len(dagger),
        "dagger_counts": dagger_counts,
        "validation_failures": validation_failures,
        "latest_at": _iso(latest),
        "latest_age_seconds": max(0.0, time.time() - latest) if latest else None,
    }


_UNIT_PREFIXES = ("kag-bc-", "kag-live-slot-", "kag-merge-", "kag-ppo-",
                  "kag-rl-", "kaggriculture-rl-", "kaggriculture-dashboard")


def _services() -> list[dict[str, Any]]:
    try:
        listed = subprocess.run(
            ["systemctl", "list-units", "--all", "--plain", "--no-legend",
             "kag-*.service", "kaggriculture-rl-*.service",
             "kaggriculture-dashboard.service"],
            check=False, capture_output=True, text=True, timeout=2,
        ).stdout.splitlines()
        units = [line.split()[0] for line in listed if line.split() and
                 line.split()[0].startswith(_UNIT_PREFIXES)]
        if not units:
            return []
        output = subprocess.run(
            ["systemctl", "show", *units,
             "--property=Id,ActiveState,SubState,Result,CPUUsageNSec,MemoryCurrent,TasksCurrent"],
            check=False, capture_output=True, text=True, timeout=2,
        ).stdout
    except (OSError, subprocess.TimeoutExpired):
        return []
    rows = []
    for block in output.strip().split("\n\n"):
        fields = dict(line.split("=", 1) for line in block.splitlines() if "=" in line)
        unit = fields.get("Id", "")
        if not unit.startswith(_UNIT_PREFIXES):
            continue
        def number(key: str) -> int | None:
            value = fields.get(key, "")
            return int(value) if value.isdigit() else None
        active = fields.get("ActiveState", "unknown")
        result = fields.get("Result", "")
        state = "running" if active == "active" else "failed" if result == "failed" else "done"
        rows.append({
            "name": unit.removesuffix(".service"), "state": state,
            "substate": fields.get("SubState"), "result": result,
            "cpu_seconds": (number("CPUUsageNSec") or 0) / 1e9,
            "memory_bytes": number("MemoryCurrent"), "tasks": number("TasksCurrent"),
        })
    return sorted(rows, key=lambda row: (row["state"] != "running", row["name"]))


def _normalise_parity(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        return {}
    summary = data.get("summary", {}) if isinstance(data.get("summary"), dict) else {}
    run_rows = data.get("runs", []) if isinstance(data.get("runs"), list) else []
    raw_runs = data.get("total_runs", summary.get("seat_runs", data.get("seat_runs", 0)))
    runs = int(raw_runs or len(run_rows))
    exact = int(data.get("exact_runs", summary.get("exact_seat_runs", data.get("exact", 0))) or 0)
    raw_mismatch = data.get("first_mismatch")
    if not isinstance(raw_mismatch, dict):
        mismatches = [row.get("first_mismatch") for row in run_rows
                      if isinstance(row, dict) and isinstance(row.get("first_mismatch"), dict)]
        raw_mismatch = min(mismatches, key=lambda row: int(row.get("step", 1 << 30)),
                           default=None)
    mismatch = (
        {key: raw_mismatch.get(key) for key in ("seed", "seat", "step", "field") if key in raw_mismatch}
        if isinstance(raw_mismatch, dict)
        else None
    )
    if not mismatch and runs and exact < runs:
        counts = summary.get("first_difference_step_counts", {})
        mismatches = [int(key) for key in counts if str(key).isdigit()]
        mismatch = {"step": min(mismatches)} if mismatches else {"present": True}
    return {
        "runs": runs,
        "exact_runs": exact,
        "exact_fraction": exact / runs if runs else None,
        "steps": data.get("steps"),
        "seed_count": data.get("seed_count"),
        "both_seats": data.get("both_seats"),
        "first_mismatch": mismatch,
    }


def _native_one(name: str, directory: str | None = None,
                legacy_parity: Path | None = None) -> dict[str, Any]:
    root = NATIVE_ROOT / (directory or name)
    assets_path = root / "assets.manifest.json"
    parity_candidates = [
        root / "parity_active_holdout.json", root / "parity_active_report.json",
        root / "parity_report.json", root / "parity_probe.json",
    ]
    if legacy_parity is not None:
        parity_candidates.append(legacy_parity)
    valid_parity = [(path, _json(path)) for path in parity_candidates]
    valid_parity = [(path, value) for path, value in valid_parity if isinstance(value, dict)]
    parity_source, parity = max(
        valid_parity,
        key=lambda row: (_normalise_parity(row[1]).get("runs", 0), _mtime(row[0])),
        default=(root / "parity_probe.json", {}),
    )
    source_files = [path for path in root.rglob("*") if path.is_file() and "__pycache__" not in path.parts]
    newest = max((_mtime(path) for path in source_files + [parity_source]), default=0.0)
    assets = _json(assets_path)
    throughput = _json(root / "throughput_report.json")
    if not isinstance(throughput, dict):
        throughput = _json(root / "benchmark_report.json")
    norm = _normalise_parity(parity)
    built = any(path.suffix in {".so", ".a"} for path in source_files)
    if norm.get("runs") and norm.get("exact_runs") == norm.get("runs"):
        status = "parity" if norm["runs"] >= 8 else "probe-exact"
    elif norm.get("runs"):
        status = "mismatch"
    elif source_files:
        status = "porting"
    else:
        status = "waiting"
    speed = None
    if isinstance(throughput, dict):
        result = throughput.get("result", {})
        if isinstance(result, dict):
            speed = result.get("games_per_second")
        repeats = throughput.get("parallel_repeats", [])
        if speed is None and isinstance(repeats, list) and repeats:
            row = repeats[-1]
            speed = row.get("games_per_second") if isinstance(row, dict) else None
    return {
        "name": name,
        "status": status,
        "source_files": len(source_files),
        "assets_ready": isinstance(assets, dict),
        "built": built,
        "parity": norm,
        "parity_source": parity_source.name,
        "build_sha256": parity.get("native_build_sha256"),
        "asset_sha256": (
            assets.get("asset_sha256") if isinstance(assets, dict) else None),
        "games_per_second": speed,
        "updated_at": _iso(newest),
        "updated_age_seconds": max(0.0, time.time() - newest) if newest else None,
    }


def _native() -> list[dict[str, Any]]:
    return [
        _native_one("thomas_2945_cpp", "thomas_2945_cpp",
                    WORK / "thomas-native-prefix-parity-current-32seed-2615700000.json"),
        _native_one("metav4_2965"),
        _native_one(
            "fieldcraft_2887", legacy_parity=NATIVE_ROOT / "fieldcraft_2887" /
            "parity_live_disjoint_report.json"),
    ]


def _student_artifact(value: Any, declared_sha256: Any,
                      suffixes: tuple[str, ...] = (".pt",)) -> dict[str, Any]:
    """Attest a basename inside the fixed student directory; never trust paths."""
    name = Path(value).name if isinstance(value, str) and value else None
    path = STUDENT_ROOT / name if name and name not in {".", ".."} else None
    present = bool(path and path.is_file() and path.suffix in suffixes)
    actual = _sha256(path) if present and path is not None else None
    declared = declared_sha256 if isinstance(declared_sha256, str) else None
    return {
        "name": name, "present": present,
        "bytes": path.stat().st_size if present and path is not None else None,
        "declared_sha256": declared, "actual_sha256": actual,
        "sha256_match": bool(declared and actual == declared),
    }


def _latest_dagger_parent() -> dict[str, Any]:
    candidates: list[tuple[int, float, Path, dict[str, Any]]] = []
    for path in STUDENT_ROOT.glob(
            "action-event-v3-actor-owned-dagger-r*-*.metrics.json"):
        match = re.search(r"dagger-r(\d+)", path.name)
        data = _json(path)
        if (not match or not isinstance(data, dict) or
                data.get("training_task") != ACTION_TRAINING_TASK or
                data.get("schema_name") != ACTION_SCHEMA or
                str(data.get("status", "")).upper() != "PASS"):
            continue
        candidates.append((int(match.group(1)), _mtime(path), path, data))
    if not candidates:
        return {}
    round_number, updated, path, data = max(candidates, key=lambda row: (row[0], row[1]))
    checkpoint_value = data.get("checkpoint") or path.name.removesuffix(
        ".metrics.json") + ".pt"
    checkpoint = _student_artifact(checkpoint_value, None)
    if checkpoint["present"]:
        checkpoint["declared_sha256"] = checkpoint["actual_sha256"]
        checkpoint["sha256_match"] = True
    final = data.get("heldout", {}).get("final", {})
    return {
        "round": round_number, "source": path.name,
        "status": data.get("status"), "checkpoint": checkpoint,
        "states": data.get("states"), "events": data.get("events"),
        "parameters": data.get("parameters"), "device": data.get("device"),
        "heldout_loss": final.get("loss") if isinstance(final, dict) else None,
        "heldout_accuracy": (
            final.get("masked_accuracy") if isinstance(final, dict) else None),
        "updated_at": _iso(updated),
        "updated_age_seconds": max(0.0, time.time() - updated),
    }


def _rl_family(name: str) -> str:
    return {
        "thomas_2945_cpp": "ahmed-v31-descendant",
        "metav4_2965": "ahmed-v31-descendant",
        "fieldcraft_2887": "fieldcraft-independent",
        "replay_clean": "clean-replay-pool",
    }.get(name, _family_map().get(name, name))


def _rl_known_artifacts(native: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Attest only the three fixed formal-pool artifact locations."""
    known = {row["name"]: row for row in native}
    fast_builds = sorted((ROOT / "fast_kaggriculture/python/fast_kaggriculture").glob(
        "_fast_kaggriculture*.so"))
    replay_asset = ROOT / "agent/route_actions.json.zlib"
    if len(fast_builds) == 1 and replay_asset.is_file():
        known["replay_clean"] = {
            "status": "native-replay",
            "build_sha256": _sha256(fast_builds[0]),
            "asset_sha256": _sha256(replay_asset),
        }
    return known


def _normalise_continuous_rl(data: Any, updated: float, now: float,
                             legacy_alive: bool | None = None) -> dict[str, Any]:
    if not isinstance(data, dict):
        return {}
    heartbeat_value = data.get("heartbeat_at")
    has_heartbeat = isinstance(heartbeat_value, (int, float))
    heartbeat = float(heartbeat_value) if has_heartbeat else updated
    age = max(0.0, now - heartbeat) if heartbeat else None
    raw_status = str(data.get("status", ""))
    if raw_status == "failed":
        status = "failed"
    elif raw_status in {"training", "exporting", "stopping"}:
        if not has_heartbeat and legacy_alive is not None:
            status = "training" if legacy_alive else "failed"
        else:
            stale_after = (
                CONTINUOUS_RL_STALE_SECONDS if has_heartbeat
                else CONTINUOUS_RL_LEGACY_STALE_SECONDS)
            status = (
                "training" if age is not None and age <= stale_after
                else "stale")
    else:
        status = raw_status or "waiting"
    return {
        "status": status,
        "raw_status": raw_status,
        "round": data.get("round"),
        "stage": data.get("stage"),
        "driver_pid": data.get("driver_pid"),
        "child_pid": data.get("child_pid"),
        "driver_lock_held": legacy_alive,
        "exit_code": data.get("exit_code", data.get("child_exit_code")),
        "heartbeat_at": _iso(heartbeat),
        "heartbeat_age_seconds": age,
    }


def _continuous_rl(now: float, roots: tuple[Path, ...] = CONTINUOUS_RL_ROOTS) -> dict[str, Any]:
    root = max(roots,
               key=lambda candidate: _mtime(candidate / "state.json"))
    state = root / "state.json"
    lock_path = root / "driver.lock"
    legacy_alive = None
    try:
        with lock_path.open("r") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                legacy_alive = True
            else:
                legacy_alive = False
                fcntl.flock(lock, fcntl.LOCK_UN)
    except OSError:
        pass
    return _normalise_continuous_rl(
        _json(state), _mtime(state), now,
        legacy_alive)


def _normalise_rl_metric(path: Path, data: dict[str, Any],
                         native: list[dict[str, Any]]) -> dict[str, Any]:
    games = int(data.get("games", 0) or 0)
    wins = int(data.get("wins", 0) or 0)
    draws = int(data.get("draws", 0) or 0)
    events = int(data.get("events", 0) or 0)
    actionable = int(data.get("actionable_events", 0) or 0)
    illegal = int(data.get("illegal", 0) or 0)
    fallbacks = int(data.get("fallbacks", 0) or 0)
    parent = _student_artifact(
        data.get("checkpoint_in"), data.get("checkpoint_in_sha256"))
    child = _student_artifact(
        data.get("checkpoint_out"), data.get("checkpoint_out_sha256"))
    rollout = _student_artifact(
        data.get("rollout"), data.get("rollout_sha256"), (".pt", ".npz"))
    known_native = _rl_known_artifacts(native)
    native_artifacts = []
    raw_artifacts = data.get("native_opponent_artifacts", {})
    if isinstance(raw_artifacts, dict):
        for name, artifact in sorted(raw_artifacts.items()):
            if not isinstance(artifact, dict):
                continue
            known = known_native.get(str(name), {})
            module_sha = artifact.get("module_sha256")
            asset_sha = artifact.get("asset_sha256")
            native_artifacts.append({
                "name": str(name), "family": _rl_family(str(name)),
                "module_sha256": module_sha, "asset_sha256": asset_sha,
                "module_attested": bool(module_sha and module_sha == known.get("build_sha256")),
                "asset_attested": bool(asset_sha and asset_sha == known.get("asset_sha256")),
                "parity_status": known.get("status"),
            })
    by_opponent = []
    raw_opponents = data.get("by_opponent", {})
    if isinstance(raw_opponents, dict):
        for name, row in sorted(raw_opponents.items()):
            if not isinstance(row, dict):
                continue
            count = int(row.get("games", 0) or 0)
            opponent_wins = int(row.get("wins", 0) or 0)
            by_opponent.append({
                "name": str(name), "family": _rl_family(str(name)),
                "games": count,
                "ratio": row.get("ratio", count / games if games else None),
                "wins": opponent_wins,
                "win_rate": opponent_wins / count if count else None,
                "draws": int(row.get("draws", 0) or 0),
                "mean_margin": row.get("mean_margin"),
                "mean_reward": row.get("mean_reward"),
            })
    by_seat = []
    raw_seats = data.get("by_seat", {})
    if isinstance(raw_seats, dict):
        for seat, row in sorted(raw_seats.items()):
            if not isinstance(row, dict):
                continue
            count = int(row.get("games", 0) or 0)
            seat_wins = int(row.get("wins", 0) or 0)
            by_seat.append({
                "seat": str(seat), "games": count, "wins": seat_wins,
                "win_rate": seat_wins / count if count else None,
                "draws": int(row.get("draws", 0) or 0),
                "mean_margin": row.get("mean_margin"),
                "mean_reward": row.get("mean_reward"),
            })
    backend = data.get("opponent_backend")
    scope = data.get("scope")
    pool_names = {row["name"] for row in by_opponent}
    artifact_names = {row["name"] for row in native_artifacts}
    artifact_attested = bool(native_artifacts) and all(
        row["module_attested"] and row["asset_attested"]
        for row in native_artifacts)
    pool_attested = bool(pool_names) and pool_names == artifact_names
    algorithm = data.get("algorithm")
    formal_eligible = bool(
        str(data.get("status", "")).upper() == "PASS"
        and algorithm in FORMAL_RL_ALGORITHMS
        and backend == FORMAL_RL_BACKEND and scope in FORMAL_RL_SCOPES
        and parent["sha256_match"] and child["sha256_match"]
        and rollout["sha256_match"] and artifact_attested and pool_attested
    )
    validation_messages = []
    if algorithm not in FORMAL_RL_ALGORITHMS:
        validation_messages.append("algorithm 不是已验真的 day-bundle PPO 版本")
    if backend != FORMAL_RL_BACKEND:
        validation_messages.append(f"formal RL backend 必须是 {FORMAL_RL_BACKEND}")
    if scope not in FORMAL_RL_SCOPES:
        validation_messages.append("scope 不是已验真的 native C++ rollout")
    if not artifact_attested or not pool_attested:
        validation_messages.append("native opponent pool/hash 未完整验真")
    if not all(row["sha256_match"] for row in (parent, child, rollout)):
        validation_messages.append("checkpoint/rollout hash chain 不闭合")
    ppo = data.get("ppo", {}) if isinstance(data.get("ppo"), dict) else {}
    hyper = data.get("hyperparameters", {}) if isinstance(
        data.get("hyperparameters"), dict) else {}
    timing = data.get("timing", {}) if isinstance(data.get("timing"), dict) else {}
    total_seconds = timing.get(
        "total_wall_seconds", timing.get("rollout_and_update_seconds"))
    update_seconds = timing.get("update_seconds")
    rollout_seconds = timing.get("rollout_seconds")
    if (rollout_seconds is None and isinstance(total_seconds, (int, float)) and
            isinstance(update_seconds, (int, float))):
        rollout_seconds = max(0.0, total_seconds - update_seconds)
    phase_seconds = {
        "rollout_seconds": rollout_seconds,
        "save_seconds": timing.get("rollout_save_seconds"),
        "cpu_replay_seconds": timing.get("cpu_replay_seconds"),
        "device_replay_seconds": timing.get("device_replay_seconds"),
        "update_seconds": update_seconds,
    }
    complete_timing = all(
        isinstance(value, (int, float)) for value in phase_seconds.values())
    accounted_seconds = sum(phase_seconds.values()) if complete_timing else None
    other_seconds = (
        max(0.0, total_seconds - accounted_seconds)
        if isinstance(total_seconds, (int, float)) and accounted_seconds is not None
        else None
    )
    replay_gate = data.get("old_logprob_replay_gate", {})
    replay_gate = replay_gate if isinstance(replay_gate, dict) else {}
    replay_checks = [
        (data.get("old_logprob_replay_max_abs_error"),
         replay_gate.get("max_abs_tolerance")),
        (data.get("old_logprob_replay_mean_abs_error"),
         replay_gate.get("mean_abs_tolerance")),
        (data.get("old_logprob_replay_approx_kl"),
         replay_gate.get("approx_kl_tolerance")),
    ]
    replay_checks = [
        (value, limit) for value, limit in replay_checks
        if isinstance(limit, (int, float))
    ]
    replay_gate_passed = (
        all(isinstance(value, (int, float)) and value <= limit
            for value, limit in replay_checks)
        if replay_checks else None
    )
    optimizer_step_before = data.get("optimizer_step_before")
    optimizer_step_after = data.get("optimizer_step_after")
    optimizer_step_derived = False
    if (not isinstance(optimizer_step_after, (int, float)) and
            isinstance(optimizer_step_before, (int, float)) and
            isinstance(ppo.get("gradient_steps"), (int, float))):
        optimizer_step_after = optimizer_step_before + ppo["gradient_steps"]
        optimizer_step_derived = True
    day_kl = ppo.get("day_chain_approx_kl", ppo.get("approx_kl"))
    day_clip = ppo.get("day_clip_fraction", ppo.get("clip_fraction"))
    return {
        "source": path.name, "raw_status": data.get("status"),
        "algorithm": algorithm,
        "student_intraday": data.get("student_intraday"),
        "deployment_eligible": data.get("deployment_eligible", False),
        "backend": backend, "scope": scope,
        "formal_eligible": formal_eligible,
        "validation_messages": validation_messages,
        "checkpoint_parent": parent, "checkpoint_child": child,
        "rollout": rollout, "binary_sha256": data.get("binary_sha256"),
        "native_artifacts": native_artifacts,
        "games": games, "wins": wins, "draws": draws,
        "win_rate": wins / games if games else None,
        "mean_margin": data.get("mean_margin"),
        "mean_reward": data.get("mean_reward"),
        "reward_min": data.get("reward_min"), "reward_max": data.get("reward_max"),
        "by_opponent": by_opponent, "by_seat": by_seat,
        "unique_environment_seeds": data.get("unique_environment_seeds"),
        "unique_policy_seeds": data.get("unique_policy_seeds"),
        "environment_seed_min": data.get("environment_seed_min"),
        "environment_seed_max": data.get("environment_seed_max"),
        "both_seats": {row["seat"] for row in by_seat} == {"0", "1"},
        "policy_seeds_unique": data.get("unique_policy_seeds") == games if games else None,
        "events": events, "actionable_events": actionable,
        "actionable_fraction": actionable / events if events else None,
        "legal_events": max(0, events - illegal),
        "legal_fraction": (events - illegal) / events if events else None,
        "illegal": illegal, "fallbacks": fallbacks,
        "old_logprob_replay_max_abs_error": data.get(
            "old_logprob_replay_max_abs_error"),
        "logprob_replay": {
            "cpu": {
                "max_abs_error": data.get("old_logprob_replay_max_abs_error"),
                "mean_abs_error": data.get("old_logprob_replay_mean_abs_error"),
                "approx_kl": data.get("old_logprob_replay_approx_kl"),
                "gate": replay_gate,
                "gate_passed": replay_gate_passed,
            },
            "device": {
                "max_abs_error": data.get(
                    "device_logprob_replay_max_abs_error"),
                "approx_kl": data.get("device_logprob_replay_approx_kl"),
                "mean_shift": data.get("device_logprob_replay_mean_shift"),
            },
        },
        "parameter_delta_l2": data.get("parameter_delta_l2"),
        "learning_rate": hyper.get("learning_rate"),
        "target_kl": hyper.get("target_kl"),
        "optimizer": {
            "state_restored": data.get("optimizer_state_restored"),
            "step_before": optimizer_step_before,
            "step_after": optimizer_step_after,
            "step_after_derived": optimizer_step_derived,
        },
        "ppo": {
            key: ppo.get(key) for key in (
                "epochs_requested", "epochs_completed", "gradient_steps",
                "early_stopped", "loss", "policy_loss", "entropy",
                "gradient_norm", "kl_gate", "event_forward_kl",
                "event_approx_kl", "event_clip_fraction", "day_forward_kl",
                "day_chain_approx_kl", "day_bundle_approx_kl",
                "day_clip_fraction")
        } | {"approx_kl": day_kl, "clip_fraction": day_clip},
        "timing": phase_seconds | {
            "total_seconds": total_seconds,
            "accounted_seconds": accounted_seconds,
            "other_seconds": other_seconds,
            "games_per_second": timing.get(
                "rollout_games_per_second",
                games / rollout_seconds if games and rollout_seconds else None),
            "events_per_second": timing.get(
                "rollout_events_per_second",
                events / rollout_seconds if events and rollout_seconds else None),
        },
        "updated_at": _iso(_mtime(path)),
        "updated_age_seconds": max(0.0, time.time() - _mtime(path)),
        "updated_timestamp": _mtime(path),
    }


def _rl(native: list[dict[str, Any]], services: list[dict[str, Any]],
        continuous: dict[str, Any], economic: bool = False) -> dict[str, Any]:
    runs = []
    paths = sorted((path for path in STUDENT_ROOT.glob("*.metrics.json")
                    if ("economic-v1" in path.name) == economic), key=_mtime)
    for path in paths[-1:]:
        data = _json(path)
        algorithm = data.get("algorithm") if isinstance(data, dict) else None
        if isinstance(algorithm, str) and algorithm.startswith(RL_ALGORITHM_PREFIX):
            runs.append(_normalise_rl_metric(path, data, native))
    dagger = _latest_dagger_parent()
    known = {}
    dagger_checkpoint = dagger.get("checkpoint", {}) if dagger else {}
    if dagger_checkpoint.get("actual_sha256"):
        known[dagger_checkpoint["actual_sha256"]] = f"DAgger R{dagger['round']}"
    for run in runs:
        parent_sha = run["checkpoint_parent"].get("declared_sha256")
        run["parent_origin"] = known.get(parent_sha, "unlinked")
        child_sha = run["checkpoint_child"].get("declared_sha256")
        if child_sha:
            known[child_sha] = run["checkpoint_child"].get("name") or run["source"]
        run["parent_is_latest_dagger"] = bool(
            parent_sha and parent_sha == dagger_checkpoint.get("actual_sha256"))
    latest = runs[-1] if runs else {}
    active = any(row.get("state") == "running" and
                 ("ppo" in row.get("name", "").lower() or
                  "rl" in row.get("name", "").lower()) and
                 (("economic" in row.get("name", "").lower()) == economic)
                 for row in services)
    if continuous.get("status") in {"training", "stale", "failed", "stopped"}:
        status = continuous["status"]
    elif active:
        status = "running"
    elif not latest:
        status = "waiting"
    elif str(latest.get("raw_status", "")).upper().startswith("FAIL"):
        status = "failed"
    elif latest.get("formal_eligible"):
        status = "ready"
    else:
        status = "audit-only"
    series = []
    for index, path in enumerate(paths[-120:]):
        data = _json(path)
        if not isinstance(data, dict) or not str(data.get("algorithm", "")).startswith(RL_ALGORITHM_PREFIX):
            continue
        ppo = data.get("ppo") or {}
        timing = data.get("timing") or {}
        games = data.get("games") or 0
        series.append({
            "x": len(paths) - min(len(paths), 120) + index + 1,
            "win_rate": data.get("wins", 0) / games if games else None,
            "mean_reward": data.get("mean_reward"),
            "policy_loss": ppo.get("policy_loss"),
            "approx_kl": ppo.get("day_chain_approx_kl"),
            "clip_fraction": ppo.get("day_clip_fraction"),
            "games_per_second": timing.get("rollout_games_per_second"),
            "events_per_second": timing.get("rollout_events_per_second"),
        })
    return {
        "status": status, "must_algorithms": sorted(FORMAL_RL_ALGORITHMS),
        "must_backend": FORMAL_RL_BACKEND,
        "must_scopes": sorted(FORMAL_RL_SCOPES), "runs": runs[-20:],
        "run_count": len(paths), "latest": latest, "series": series,
        "continuous": continuous,
        "dagger_parent": dagger,
        "latest_at": latest.get("updated_at") if latest else None,
        "latest_age_seconds": latest.get("updated_age_seconds") if latest else None,
    }


def _replay_pool() -> dict[str, Any]:
    path = WORK / "native-replay-pool" / "smoke-final.json"
    data = _json(path)
    if not isinstance(data, dict) or data.get("schema") != "native-replay-pool-smoke-v1":
        return {}
    audited = 2 * int(data.get("audit_games", 0) or 0)
    return {
        key: data.get(key) for key in (
            "pool_routes", "routes_exercised", "games", "terminal_games",
            "games_per_second", "threads", "route_switches_per_player_game",
            "mean_macro_unit_failures", "mean_macro_market_failures",
        )
    } | {
        "failure_free_fraction": (
            int(data.get("zero_macro_failure_player_games", 0) or 0) / audited
            if audited else None
        ),
        "updated_at": _iso(_mtime(path)),
        "updated_age_seconds": max(0.0, time.time() - _mtime(path)),
    }


def _task(name: str, updated: float | None, state: str, detail: str) -> dict[str, Any]:
    return {"name": name, "state": state, "detail": detail, "updated_at": _iso(updated)}


def _tasks(
    now: float,
    corpus: dict[str, Any],
    replay: dict[str, Any],
    suffix: dict[str, Any],
    training: dict[str, Any],
    native: list[dict[str, Any]],
    rl: dict[str, Any],
) -> list[dict[str, Any]]:
    def stamp(section: dict[str, Any], key: str = "latest_at") -> float | None:
        value = section.get(key)
        try:
            return datetime.fromisoformat(value).timestamp() if value else None
        except (TypeError, ValueError):
            return None

    corpus_time = stamp(corpus)
    suffix_time = stamp(suffix)
    train_time = stamp(training)
    rows = [
        _task(
            "BC rollout",
            corpus_time,
            "active" if corpus_time and now - corpus_time < 600 else "idle",
            f"{corpus['games']:,} games / {corpus['frames']:,} frames",
        ),
        _task(
            "Kaggle replay download",
            stamp(replay),
            "active" if replay.get("active") else ("done" if replay.get("remaining") == 0 else "idle"),
            f"{replay['downloaded']:,}/{replay['target']:,}; new {replay['new_ids']:,}",
        ),
        _task(
            "Student training",
            train_time,
            training["status"],
            (f"epoch {training.get('completed_epochs', 0)}/{training.get('epochs') or '?'} · "
             f"{training['records']} final metric records"),
        ),
        _task(
            "RL / PPO",
            stamp(rl.get("continuous", {}), "heartbeat_at") or stamp(rl),
            rl["status"],
            (f"round {rl.get('continuous', {}).get('round', '?')} · "
             f"{rl.get('continuous', {}).get('stage', 'idle')} · "
             f"{rl['run_count']} completed updates"),
        ),
        _task(
            "Real suffix labels",
            suffix_time,
            "active" if suffix_time and now - suffix_time < 600 else ("ready" if suffix["states"] else "waiting"),
            f"{suffix['states']} states / {suffix['replicas']} replicas",
        ),
    ]
    for row in native:
        timestamp = None
        if row.get("updated_at"):
            timestamp = datetime.fromisoformat(row["updated_at"]).timestamp()
        rows.append(_task(f"C++ {row['name']}", timestamp, row["status"], f"{row['parity'].get('exact_runs', 0)}/{row['parity'].get('runs', 0)} exact"))
    rows.append(_task("Dashboard", now, "active", "read-only · fixed allowlist"))
    return rows


def _payload_uncached() -> dict[str, Any]:
    now = time.time()
    corpus = _corpus()
    replay = _replays(now)
    suffix = _suffix()
    native = _native()
    services = _services()
    training = _training()
    rl = _rl(native, services, _continuous_rl(now, (ECONOMIC_RL_ROOT,)), True)
    baseline_rl = _rl(native, services, _continuous_rl(now))
    replay_pool = _replay_pool()
    slot_pipeline = _slot_pipeline()
    return {
        "schema": "kaggriculture-experiment-dashboard-v1",
        "generated_at": _iso(now),
        "system": _system(),
        "npu": _npu(),
        "training": training,
        "rl": rl,
        "rl_tracks": {"economic": rl, "baseline": baseline_rl},
        "slot_pipeline": slot_pipeline,
        "corpus": corpus,
        "suffix": suffix,
        "native_opponents": native,
        "replay_pool": replay_pool,
        "replays": replay,
        "services": services,
        "tasks": _tasks(now, corpus, replay, suffix, training, native, rl),
        "privacy": {
            "raw_private_data_exposed": False,
            "absolute_paths_exposed": False,
            "api_mode": "fixed-allowlist-read-only",
        },
    }


def _payload() -> dict[str, Any]:
    global _PAYLOAD_CACHE
    now = time.monotonic()
    with _CACHE_LOCK:
        if now - _PAYLOAD_CACHE[0] < 2.0:
            return _PAYLOAD_CACHE[1]
        value = _payload_uncached()
        _PAYLOAD_CACHE = (now, value)
        return value


PAGE = r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Kaggriculture 实验监控</title><style>
:root{color-scheme:dark;--bg:#071019;--panel:#0d1b27;--panel2:#102432;--line:#20394b;--text:#e8f2f5;--muted:#8ba7b6;--cyan:#4fd1c5;--green:#74d680;--amber:#f2c66d;--red:#ff7a7a;--blue:#6aa9ff}*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 10% 0,#123047 0,transparent 34%),var(--bg);font:14px/1.45 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;color:var(--text)}main{max-width:1500px;margin:auto;padding:24px}.top{display:flex;gap:18px;align-items:end;justify-content:space-between;margin-bottom:18px}h1{font-size:28px;margin:0}.muted{color:var(--muted)}.live{display:inline-flex;align-items:center;gap:7px}.dot{width:9px;height:9px;border-radius:50%;background:var(--green);box-shadow:0 0 13px var(--green)}.grid{display:grid;grid-template-columns:repeat(12,1fr);gap:14px}.panel{background:linear-gradient(145deg,var(--panel),#091722);border:1px solid var(--line);border-radius:14px;padding:16px;min-width:0}.wide{grid-column:span 12}.half{grid-column:span 6}.third{grid-column:span 4}.title{display:flex;align-items:center;justify-content:space-between;margin-bottom:13px}.title h2{font-size:16px;margin:0}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(145px,1fr));gap:10px}.card{background:var(--panel2);border:1px solid #203d50;border-radius:11px;padding:12px}.label{font-size:12px;color:var(--muted);white-space:nowrap}.value{font-size:22px;font-weight:700;margin-top:3px;overflow-wrap:anywhere}.hint{font-size:11px;color:var(--muted);margin-top:2px}.ok{color:var(--green)}.warn{color:var(--amber)}.bad{color:var(--red)}.guard{border:1px solid var(--line);border-radius:10px;padding:10px 12px;margin-bottom:12px}.guard.good{color:var(--green);border-color:#285b49;background:#0d251e}.guard.bad{color:var(--red);border-color:#6c3434;background:#291416}.charts{display:grid;grid-template-columns:repeat(5,1fr);gap:10px}.chart{background:#091722;border:1px solid var(--line);border-radius:10px;padding:10px}.chart svg{width:100%;height:80px;display:block}.chart .value{font-size:17px}.bar{height:8px;background:#172d3d;border-radius:9px;overflow:hidden;margin-top:8px}.bar>i{display:block;height:100%;background:linear-gradient(90deg,var(--cyan),var(--blue));border-radius:9px}table{width:100%;border-collapse:collapse}th,td{text-align:left;padding:9px 8px;border-bottom:1px solid var(--line);vertical-align:top}th{font-size:11px;text-transform:uppercase;color:var(--muted)}td{font-variant-numeric:tabular-nums}.badge{display:inline-block;border:1px solid var(--line);border-radius:999px;padding:2px 8px;font-size:11px}.badge.active,.badge.running,.badge.training,.badge.parity,.badge.done,.badge.ready{color:var(--green);border-color:#285b49}.badge.mismatch,.badge.rejected,.badge.failed,.badge.audit-only{color:var(--red);border-color:#6c3434}.badge.waiting,.badge.idle,.badge.porting,.badge.stale{color:var(--amber);border-color:#67552c}.sources{display:flex;gap:6px;flex-wrap:wrap;margin-top:10px}.source{background:#102737;color:var(--muted);border-radius:6px;padding:3px 7px;font-size:11px}@media(max-width:950px){.half,.third{grid-column:span 12}.charts{grid-template-columns:repeat(2,1fr)}}@media(max-width:560px){main{padding:14px}.top{align-items:start;flex-direction:column}.charts{grid-template-columns:1fr}.value{font-size:19px}.panel{overflow:auto}}
.badge.probe-exact{color:var(--amber);border-color:#67552c}
</style></head><body><main><div class="top"><div><h1>Kaggriculture 实验监控</h1><div class="muted">中盘 student · teacher suffix · C++ 对手 · 数据管线</div></div><div><span class="live"><i class="dot"></i><span id="status">连接中</span></span><div id="updated" class="muted"></div></div></div><div class="grid">
<section class="panel wide"><div class="title"><h2>RL / PPO · step288 后</h2><div><select id="rl-track" aria-label="训练链" style="background:#102432;color:#e8f2f5;border:1px solid #20394b;border-radius:6px;padding:4px"><option value="economic">经济特征 RL（当前）</option><option value="baseline">旧基线（已停）</option></select> <span id="rl-status" class="badge">—</span></div></div><div id="rl-guard" class="guard bad">等待指标</div><div id="rl-cards" class="cards"></div><div id="rl-charts" class="charts" style="margin-top:12px"></div><div class="title" style="margin-top:16px"><h2>Checkpoint parent → child</h2><span class="muted">哈希闭环；rollout 只读</span></div><div id="rl-chain"></div><div class="title" style="margin-top:16px"><h2>对手池 / 座位 / native artifacts</h2><span class="muted">实际采样比例</span></div><div id="rl-pool"></div></section>
<section class="panel wide"><div class="title"><h2>Action-event ABI v3 BC / DAgger</h2><span id="train-status" class="badge">—</span></div><div id="train-cards" class="cards"></div><div id="train-progress" style="margin-top:12px"></div><div id="charts" class="charts" style="margin-top:12px"></div><div id="metric-sources" class="sources"></div><div id="recall" style="margin-top:12px"></div></section>
<section class="panel half"><div class="title"><h2>Action-event ABI v3 数据</h2><span class="muted">fixed-point + resource replay</span></div><div id="slot-pipeline" class="cards"></div></section>
<section class="panel half"><div class="title"><h2>后台运行任务</h2><span class="muted">systemd 实际状态</span></div><div id="services"></div></section>
<section class="panel half"><div class="title"><h2>Rollout / BC corpus</h2><span class="muted">只统计完整 manifest</span></div><div id="corpus" class="cards"></div><div id="families" class="sources"></div></section>
<section class="panel half"><div class="title"><h2>Kaggle 高手 replay</h2><span id="replay-state" class="badge">—</span></div><div id="replays" class="cards"></div><div class="bar"><i id="replay-bar"></i></div></section>
<section class="panel half"><div class="title"><h2>真实 suffix 标签</h2><span class="muted">checkpoint 内 replica 成组</span></div><div id="suffix" class="cards"></div></section>
<section class="panel half"><div class="title"><h2>C++ 对手移植 / parity</h2><span class="muted">Python ↔ C++ 逐步动作</span></div><div id="native"></div></section>
<section class="panel wide"><div class="title"><h2>系统</h2><span class="muted">CPU / NPU 只读采样</span></div><div id="system" class="cards"></div><div id="npu-devices" class="sources"></div></section>
<section class="panel wide"><div class="title"><h2>任务与最近产物</h2><span class="muted">状态由固定 allowlist 产物新鲜度推断</span></div><table><thead><tr><th>任务</th><th>状态</th><th>进度</th><th>最后更新 UTC</th></tr></thead><tbody id="tasks"></tbody></table></section>
</div></main><script>
const $=id=>document.getElementById(id),esc=s=>String(s??'—').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])),num=(v,d=0)=>v==null?'—':Number(v).toLocaleString(undefined,{maximumFractionDigits:d}),pct=v=>v==null?'—':num(Number(v)*100,1)+'%',bytes=v=>{if(v==null)return'—';let n=Number(v),u=['B','KiB','MiB','GiB','TiB'],i=0;while(n>=1024&&i<u.length-1){n/=1024;i++}return num(n,n<10?1:0)+' '+u[i]},age=v=>v==null?'—':v<60?num(v,0)+' 秒前':v<3600?num(v/60,1)+' 分钟前':num(v/3600,1)+' 小时前',hash=v=>v?String(v).slice(0,12):'—';
function card(label,value,hint='',klass=''){return `<div class="card"><div class="label">${esc(label)}</div><div class="value ${klass}">${esc(value)}</div><div class="hint">${esc(hint)}</div></div>`}
function spark(values,color='#4fd1c5'){const xs=values.filter(v=>v!=null&&Number.isFinite(Number(v))).map(Number);if(xs.length<2)return'<svg viewBox="0 0 200 80"><text x="8" y="43" fill="#6f8b9b" font-size="12">等待指标</text></svg>';let lo=Math.min(...xs),hi=Math.max(...xs);if(hi===lo){hi+=1;lo-=1}const points=xs.map((v,i)=>`${i/(xs.length-1)*196+2},${76-(v-lo)/(hi-lo)*70}`).join(' ');return `<svg viewBox="0 0 200 80" preserveAspectRatio="none"><polyline fill="none" stroke="${color}" stroke-width="2.3" points="${points}"/></svg>`}
function chart(label,key,rows,color,format=num){const values=rows.map(r=>r[key]),latest=[...values].reverse().find(v=>v!=null);return `<div class="chart"><div class="label">${esc(label)}</div><div class="value">${esc(format(latest,4))}</div>${spark(values,color)}</div>`}
function render(p){
  $('status').textContent='在线 · 每 5 秒刷新';$('updated').textContent='数据时间 '+(p.generated_at||'—');
  const q=(p.rl_tracks||{})[$('rl-track').value]||p.rl||{},live=q.continuous||{},u=q.latest||{},pp=u.ppo||{},tm=u.timing||{},lg=u.logprob_replay||{},cpu=lg.cpu||{},dev=lg.device||{},cg=cpu.gate||{},opt=u.optimizer||{},pool=u.by_opponent||[],dp=q.dagger_parent||{},dpc=dp.checkpoint||{};
  $('rl-status').textContent=q.status||'waiting';$('rl-status').className='badge '+(q.status||'waiting');
  const problems=u.validation_messages||[],mustAlgorithms=q.must_algorithms||[],mustScopes=q.must_scopes||[];$('rl-guard').className='guard '+(u.formal_eligible?'good':'bad');$('rl-guard').textContent=u.formal_eligible?'正式 RL 链通过：算法、native_cpp rollout scope、checkpoint/rollout 与对手 artifacts 哈希均闭合':(problems.length?problems.join(' · '):'等待 PPO 指标；正式 RL 必须使用已验真的 native_cpp scope 与 leave-one-out baseline');
  const opponentWins=pool.map(x=>`${x.name}: ${num(x.wins)}/${num(x.games)}`).join(' · '),cpuGate=cpu.gate_passed==null?'—':(cpu.gate_passed?'PASS':'FAIL');
  $('rl-cards').innerHTML=[
    card('Continuous driver','R'+num(live.round)+' · '+(live.stage||'—'),'PID '+num(live.driver_pid)+' / child '+num(live.child_pid)+' · lock '+(live.driver_lock_held?'held':'free')+' · heartbeat '+age(live.heartbeat_age_seconds)+' · exit '+num(live.exit_code),live.status==='training'?'ok':(live.status==='failed'?'bad':'warn')),
    card('运行 / 数据更新',q.status||'waiting',age(q.latest_age_seconds),u.formal_eligible?'ok':'bad'),
    card('当前 checkpoint',u.checkpoint_child?.name||'未声明',hash(u.checkpoint_child?.actual_sha256)+' · '+(u.checkpoint_child?.sha256_match?'hash PASS':'hash FAIL'),u.checkpoint_child?.sha256_match?'ok':'bad'),
    card('日内自主补种',u.student_intraday===0?'关闭':u.student_intraday===1?'开启':'旧链未标记','当前 student rollout 的执行配置',u.student_intraday===0?'ok':'warn'),
    card('Algorithm',u.algorithm||'未声明','正式门槛 '+mustAlgorithms.join(' / '),mustAlgorithms.includes(u.algorithm)?'ok':'warn'),
    card('Backend / scope',(u.backend||'—')+' / '+(u.scope||'—'),'正式门槛 '+q.must_backend+' / '+mustScopes.join(' / '),u.backend===q.must_backend&&mustScopes.includes(u.scope)?'ok':'warn'),
    card('Parent → child',hash(u.checkpoint_parent?.declared_sha256)+' → '+hash(u.checkpoint_child?.declared_sha256),(u.parent_origin||'unlinked')+' · '+(u.checkpoint_parent?.sha256_match&&u.checkpoint_child?.sha256_match?'hash PASS':'hash FAIL')),
    card('Games / wins',num(u.games)+' / '+num(u.wins),pct(u.win_rate)+' · '+num(u.draws)+' draws'),
    card('Wins by opponent',opponentWins||'—','当前 rollout'),
    card('Margin / reward',num(u.mean_margin,1)+' / '+num(u.mean_reward,4),num(u.reward_min,3)+' … '+num(u.reward_max,3)),
    card('Seeds / 双座',num(u.unique_environment_seeds)+' env · '+num(u.unique_policy_seeds)+' policy',u.both_seats?'seat0+1':'双座未证实',u.both_seats?'ok':'warn'),
    card('Events / actionable',num(u.events)+' / '+num(u.actionable_events),pct(u.actionable_fraction)),
    card('Legal / illegal',pct(u.legal_fraction)+' / '+num(u.illegal),num(u.fallbacks)+' fallback',u.illegal===0&&u.fallbacks===0?'ok':'bad'),
    card('CPU logprob gate',cpuGate,'max '+num(cpu.max_abs_error,8)+' ≤ '+num(cg.max_abs_tolerance,8)+' · mean '+num(cpu.mean_abs_error,8)+' ≤ '+num(cg.mean_abs_tolerance,8)+' · KL '+num(cpu.approx_kl,10)+' ≤ '+num(cg.approx_kl_tolerance,10),cpu.gate_passed?'ok':(cpu.gate_passed===false?'bad':'warn')),
    card('Device logprob replay',num(dev.max_abs_error,8)+' / '+num(dev.approx_kl,10),'max |Δ| / approx KL · shift '+num(dev.mean_shift,10)),
    card('Day KL / clip',num(pp.approx_kl,7)+' / '+pct(pp.clip_fraction),'event '+num(pp.event_approx_kl,7)+' / '+pct(pp.event_clip_fraction)+' · target '+num(u.target_kl,4)),
    card('Optimizer step',num(opt.step_before)+' → '+num(opt.step_after),(opt.state_restored?'state restored':'fresh state')+' · '+num(pp.gradient_steps)+' updates'+(opt.step_after_derived?' · after derived':''),opt.state_restored?'ok':'warn'),
    card('PPO epochs / loss',num(pp.epochs_completed)+' / '+num(pp.epochs_requested),num(pp.loss,6)+' / policy '+num(pp.policy_loss,6)),
    card('Entropy / grad norm',num(pp.entropy,6)+' / '+num(pp.gradient_norm,5),'early stop '+String(pp.early_stopped??'—')),
    card('Parameter Δ / LR',num(u.parameter_delta_l2,7)+' / '+num(u.learning_rate,8),'L2 / learning rate'),
    card('Wall rollout / save',num(tm.rollout_seconds,1)+' / '+num(tm.save_seconds,1),'seconds'),
    card('Wall CPU / device replay',num(tm.cpu_replay_seconds,1)+' / '+num(tm.device_replay_seconds,1),'seconds'),
    card('Wall update / total',num(tm.update_seconds,1)+' / '+num(tm.total_seconds,1),'seconds · other '+num(tm.other_seconds,1)),
    card('吞吐',num(tm.games_per_second,3)+' games/s',num(tm.events_per_second,1)+' events/s'),
    card('最新 DAgger parent','R'+num(dp.round),hash(dpc.actual_sha256)+' · '+num(dp.states)+' states')
  ].join('');
  $('rl-charts').innerHTML=chart('Win rate','win_rate',q.series||[],'#74d680',pct)+chart('Mean reward','mean_reward',q.series||[],'#4fd1c5',(v)=>num(v,5))+chart('Policy loss','policy_loss',q.series||[],'#f2c66d',(v)=>num(v,6))+chart('Day-chain KL','approx_kl',q.series||[],'#ff7a7a',(v)=>num(v,7))+chart('Day clip','clip_fraction',q.series||[],'#cf88ff',pct)+chart('Rollout games/s','games_per_second',q.series||[],'#6aa9ff',(v)=>num(v,3))+chart('Rollout events/s','events_per_second',q.series||[],'#4fd1c5',(v)=>num(v,1));
  const chain=q.runs||[];$('rl-chain').innerHTML=chain.length?'<table><thead><tr><th>Checkpoint / hash</th><th>状态</th><th>Wins</th><th>吞吐</th><th>wall s: rollout/save/cpu/device/update/total</th><th>day KL / clip</th><th>optimizer</th></tr></thead><tbody>'+chain.map(x=>{const t=x.timing||{},p=x.ppo||{},o=x.optimizer||{},by=(x.by_opponent||[]).map(y=>`${y.name} ${num(y.wins)}/${num(y.games)}`).join(' · ');return `<tr><td>${esc(x.checkpoint_child?.name||x.source)}<div class="hint">${esc(hash(x.checkpoint_child?.actual_sha256))} · ${esc(x.updated_at)}</div></td><td><span class="badge ${x.formal_eligible?'ready':'audit-only'}">${x.formal_eligible?'formal':'audit-only'}</span><div class="hint">${esc(x.backend||'—')} / ${esc(x.scope||'—')}</div></td><td>${num(x.wins)}/${num(x.games)}<div class="hint">${esc(by)}</div></td><td>${num(t.games_per_second,2)} g/s<div class="hint">${num(t.events_per_second,0)} events/s</div></td><td>${num(t.rollout_seconds,1)} / ${num(t.save_seconds,1)} / ${num(t.cpu_replay_seconds,1)} / ${num(t.device_replay_seconds,1)} / ${num(t.update_seconds,1)} / ${num(t.total_seconds,1)}</td><td>${num(p.approx_kl,7)} / ${pct(p.clip_fraction)}<div class="hint">event ${num(p.event_approx_kl,7)} / ${pct(p.event_clip_fraction)}</div></td><td>${num(o.step_before)} → ${num(o.step_after)}<div class="hint">${o.state_restored?'restored':'fresh'}</div></td></tr>`}).join('')+'</tbody></table>':'<span class="muted">等待 work/student-v1/*.metrics.json 中的 PPO 记录</span>';
  const seats=u.by_seat||[],arts=u.native_artifacts||[];$('rl-pool').innerHTML=(pool.length?'<table><thead><tr><th>类别 / opponent</th><th>实际比例</th><th>Wins / games</th><th>胜率</th><th>margin</th><th>reward</th></tr></thead><tbody>'+pool.map(x=>`<tr><td>${esc(x.family)}<div class="hint">${esc(x.name)}</div></td><td>${pct(x.ratio)}</td><td>${num(x.wins)} / ${num(x.games)}</td><td>${pct(x.win_rate)}</td><td>${num(x.mean_margin,1)}</td><td>${num(x.mean_reward,4)}</td></tr>`).join('')+'</tbody></table>':'<span class="muted">暂无 opponent pool 明细</span>')+(seats.length?'<table><thead><tr><th>Seat</th><th>Games</th><th>胜率</th><th>margin</th><th>reward</th></tr></thead><tbody>'+seats.map(x=>`<tr><td>${esc(x.seat)}</td><td>${num(x.games)}</td><td>${pct(x.win_rate)}</td><td>${num(x.mean_margin,1)}</td><td>${num(x.mean_reward,4)}</td></tr>`).join('')+'</tbody></table>':'')+(arts.length?'<table><thead><tr><th>Native artifact</th><th>module hash</th><th>asset hash</th><th>验真</th></tr></thead><tbody>'+arts.map(x=>`<tr><td>${esc(x.name)}<div class="hint">${esc(x.family)}</div></td><td>${esc(hash(x.module_sha256))}</td><td>${esc(hash(x.asset_sha256))}</td><td class="${x.module_attested&&x.asset_attested?'ok':'bad'}">${x.module_attested&&x.asset_attested?'PASS':'FAIL'}</td></tr>`).join('')+'</tbody></table>':'');
  const t=p.training,l=t.latest||{},epoch=t.epoch_series||[],curve=epoch.length?epoch:(t.series||[]);
  $('train-status').textContent=t.status;$('train-status').className='badge '+t.status;
  $('train-cards').innerHTML=card('当前 epoch',`${num(t.completed_epochs)}/${num(t.epochs)}`,t.phase||'—')+card('当前 train loss',num(t.current_train_loss??l.loss,5),epoch.length?'逐 epoch 实时':'最终记录')+card('Held-out loss',num(l.val_loss,5),'按 seed × family 隔离')+card('Held-out Top-1',pct(l.top1),'action-event')+card('Legal rate',pct(l.legal),'规则 mask 后')+card('吞吐',num(l.throughput,1),'event epochs/s')+card('参数',num(t.parameters),num(t.accepted_v3_shards)+' accepted v3 shards')+card('训练集',num(t.states)+' states',num(t.decisions)+' events')+card('当前产物',t.model||'等待',(t.device||'—')+' · '+age(t.latest_age_seconds));
  const ep=Math.max(0,Number(t.completed_epochs||0)),eps=Math.max(0,Number(t.epochs||0));
  $('train-progress').innerHTML=`<div class="label">训练进度 · ${num(ep)}/${num(eps)}</div><div class="bar"><i style="width:${eps?Math.min(100,ep/eps*100):0}%"></i></div>`;
  $('charts').innerHTML=chart('Epoch train loss','loss',curve,'#4fd1c5',(v)=>num(v,5))+chart('Final held-out loss','val_loss',t.series||[],'#f2c66d',(v)=>num(v,5))+chart('Held-out Top-1','top1',t.series||[],'#74d680',pct)+chart('Legal','legal',t.series||[],'#6aa9ff',pct)+chart('Throughput','throughput',t.series||[],'#cf88ff',(v)=>num(v,1));
  const src=[...(t.sources||[]),...(t.progress_source?[t.progress_source]:[])];$('metric-sources').innerHTML=src.map(x=>`<span class="source">${esc(x)}</span>`).join('')||'<span class="muted">等待经过 accepted manifest 哈希验证的 action-event ABI v3 指标；旧 slot ABI v2 已拒绝</span>';
  const kinds={'-1':'SKIP','0':'WHEAT','1':'CARROT','2':'TOMATO','3':'STRAWBERRY','4':'MELON','9':'GOOSE','10':'COW','11':'SHEEP'},rec=Object.entries(t.class_recall||{});
  $('recall').innerHTML=rec.length?'<div class="label">Held-out 每类召回</div><table><thead><tr><th>类别</th><th>样本</th><th>Recall</th></tr></thead><tbody>'+rec.map(([k,v])=>`<tr><td>${esc(kinds[k]||k)}</td><td>${num(v.support)}</td><td>${pct(v.recall)}</td></tr>`).join('')+'</tbody></table>':'<span class="muted">当前训练完成后显示逐类别 held-out recall</span>';
  const d=p.slot_pipeline||{},bc=d.base_counts||{},dc=d.dagger_counts||{};$('slot-pipeline').innerHTML=card('Accepted manifests',num(d.accepted_shards),'base + DAgger')+card('Base 游戏 / states',num(bc.games)+' / '+num(bc.states),num(bc.events)+' events')+card('DAgger rounds',num(d.dagger_rounds),num(dc.games)+' games')+card('DAgger states / events',num(dc.states)+' / '+num(dc.events),'追加分布')+card('验证失败',num(d.validation_failures),'必须为 0',d.validation_failures===0?'ok':'bad')+card('最近更新',age(d.latest_age_seconds),d.latest_at||'—');
  $('services').innerHTML=(p.services||[]).length?'<table><thead><tr><th>服务</th><th>状态</th><th>CPU time</th><th>内存 / tasks</th></tr></thead><tbody>'+p.services.map(x=>`<tr><td>${esc(x.name)}</td><td><span class="badge ${esc(x.state)}">${esc(x.state)}</span><div class="hint">${esc(x.substate||x.result)}</div></td><td>${num(x.cpu_seconds,1)}s</td><td>${bytes(x.memory_bytes)} / ${num(x.tasks)}</td></tr>`).join('')+'</tbody></table>':'<span class="muted">当前没有匹配的后台服务</span>';
  const c=p.corpus;$('corpus').innerHTML=card('完整游戏',num(c.games),num(c.shards)+' shards')+card('逐步帧',num(c.frames),'719 / 完整局')+card('源码 family',num(c.families),num(c.opponents)+' opponent labels')+card('错误',num(c.errors),'完整 manifests')+card('产出吞吐',num(c.latest_games_per_second,2),'games/s')+card('最近 shard',age(c.latest_age_seconds),c.latest_at||'—');$('families').innerHTML=Object.entries(c.by_source||{}).map(([k,v])=>`<span class="source">${esc(k)} · ${num(v)} games</span>`).join('');
  const r=p.replays;$('replay-state').textContent=r.active?'下载中':(r.remaining===0?'完成':'暂停/等待');$('replay-state').className='badge '+(r.active?'active':(r.remaining===0?'done':'idle'));$('replays').innerHTML=card('已下载',num(r.downloaded),'/ '+num(r.target))+card('新增 ID',num(r.new_ids),'排除本地存货')+card('下载重复',num(r.downloaded_duplicates),'本地已有')+card('有效覆盖',num(r.effective_available),num(r.remaining)+' remaining')+card('瞬时速度',r.files_per_second==null?'采样中':num(r.files_per_second*60,1),'files/min')+card('最近文件',age(r.latest_age_seconds),r.latest_at||'—');$('replay-bar').style.width=Math.max(0,Math.min(100,Number(r.progress||0)*100))+'%';
  const s=p.suffix;$('suffix').innerHTML=card('状态',num(s.states),num(s.files)+' shards')+card('候选',num(s.candidates),num(s.mean_candidates_per_state,2)+' / state')+card('Replicas',num(s.replicas),'同 checkpoint 成组')+card('Margin 噪声 p50',num(s.median_paired_margin_noise,0),'paired Δ 标准差')+card('Margin 噪声 p90',num(s.p90_paired_margin_noise,0),'paired Δ 标准差')+card('|paired Δ| p50',num(s.median_absolute_paired_delta,0),'候选效应尺度');
  $('native').innerHTML='<table><thead><tr><th>对手</th><th>状态</th><th>Parity</th><th>build / asset</th><th>吞吐</th></tr></thead><tbody>'+p.native_opponents.map(n=>{const q=n.parity||{},m=q.first_mismatch||{};return `<tr><td>${esc(n.name)}<div class="hint">${esc(n.parity_source)}</div></td><td><span class="badge ${esc(n.status)}">${esc(n.status)}</span></td><td>${num(q.exact_runs)}/${num(q.runs)} · ${num(q.seed_count)} seeds · ${q.both_seats?'双座':'单座'}<div class="hint">${m.step==null?'无首差':'step '+esc(m.step)+' '+esc(m.field||'')}</div></td><td>${esc(hash(n.build_sha256))} / ${esc(hash(n.asset_sha256))}</td><td>${num(n.games_per_second,2)} games/s</td></tr>`}).join('')+'</tbody></table>';
  const y=p.system,z=p.npu||{};$('system').innerHTML=card('CPU 使用率',pct(y.cpu_used_fraction),num(y.cpu_count)+' cores')+card('Load 1m',num(y.load_1m,1),num(y.load_5m,1)+' / '+num(y.load_15m,1))+card('NPU AICore',pct(z.mean_aicore_fraction),'max '+pct(z.max_aicore_fraction))+card('NPU HBM',pct(z.hbm_used_fraction),num(z.hbm_used_mb)+' / '+num(z.hbm_total_mb)+' MiB')+card('内存',pct(y.memory_used_fraction),bytes(y.memory_used_bytes)+' / '+bytes(y.memory_total_bytes))+card('磁盘',pct(y.disk_used_fraction),bytes(y.disk_free_bytes)+' free')+card('隐私边界',p.privacy.raw_private_data_exposed?'FAIL':'PASS','无 private / 绝对路径 / 任意文件 API',p.privacy.raw_private_data_exposed?'bad':'ok');$('npu-devices').innerHTML=(z.devices||[]).map(x=>`<span class="source">NPU ${num(x.id)} · ${pct(x.aicore_fraction)} · HBM ${num(x.hbm_used_mb)}/${num(x.hbm_total_mb)} MiB</span>`).join('')||'<span class="muted">npu-smi 暂不可用</span>';
  $('tasks').innerHTML=p.tasks.map(x=>`<tr><td>${esc(x.name)}</td><td><span class="badge ${esc(x.state)}">${esc(x.state)}</span></td><td>${esc(x.detail)}</td><td>${esc(x.updated_at||'—')}</td></tr>`).join('')
}
async function update(){try{const r=await fetch('/api/status',{cache:'no-store'});if(!r.ok)throw Error('HTTP '+r.status);render(await r.json())}catch(e){$('status').textContent='连接失败 · '+e.message;$('status').className='bad'}}$('rl-track').onchange=update;update();setInterval(update,5000);
</script></body></html>'''.encode()


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class Handler(BaseHTTPRequestHandler):
    server_version = "KaggricultureDashboard/1"

    def _send(self, status: int, content_type: str, body: bytes, head: bool = False) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'; img-src 'self' data:; frame-ancestors 'none'")
        self.end_headers()
        if not head:
            self.wfile.write(body)

    def _route(self, head: bool = False) -> None:
        path = urlparse(self.path).path
        if path == "/":
            self._send(200, "text/html; charset=utf-8", PAGE, head)
        elif path == "/healthz":
            self._send(200, "application/json", b'{"ok":true,"mode":"read-only"}', head)
        elif path == "/api/status":
            body = json.dumps(_payload(), ensure_ascii=False, separators=(",", ":")).encode()
            self._send(200, "application/json; charset=utf-8", body, head)
        else:
            self._send(404, "application/json", b'{"error":"not found"}', head)

    def do_GET(self) -> None:  # noqa: N802
        self._route()

    def do_HEAD(self) -> None:  # noqa: N802
        self._route(head=True)

    def do_POST(self) -> None:  # noqa: N802
        self._send(405, "application/json", b'{"error":"read-only"}')

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"{self.address_string()} [{self.log_date_time_string()}] {fmt % args}", flush=True)


def _self_check() -> None:
    blocked = _do_not_use_entries()
    accepted = _accepted_v3_shards(blocked)
    assert "autofill-student-smoke.metrics.json" in blocked
    assert "candidate-student-v1.metrics.json" in blocked
    assert accepted, "expected at least one accepted action-event ABI-v3 shard"
    manifest_sha256 = next(iter(accepted))
    valid = {
        "status": "PASS",
        "training_task": ACTION_TRAINING_TASK,
        "schema_name": ACTION_SCHEMA,
        "shard_manifest_sha256": manifest_sha256,
    }
    assert _valid_v3_metric(valid, accepted)
    assert not _valid_v3_metric({**valid, "schema_name": "autoregressive-slot-bc-v2"}, accepted)
    assert not _valid_v3_metric({**valid, "shard_manifest_sha256": "0" * 64}, accepted)
    assert not _valid_v3_metric({"status": "PASS", "heldout": {"top1": 1.0}}, accepted)
    payload = _payload_uncached()
    encoded = json.dumps(payload, ensure_ascii=False)
    assert payload["schema"] == "kaggriculture-experiment-dashboard-v1"
    assert str(ROOT) not in encoded
    assert '"private":' not in encoded.lower()
    assert '"observation":' not in encoded.lower()
    assert '"action":' not in encoded.lower()
    assert set(payload["rl"]["must_algorithms"]) == FORMAL_RL_ALGORITHMS
    assert payload["rl"]["must_backend"] == FORMAL_RL_BACKEND
    assert set(payload["rl"]["must_scopes"]) == FORMAL_RL_SCOPES
    assert payload["rl"]["dagger_parent"].get("round", 0) >= 3
    runs = {row["source"]: row for row in payload["rl"]["runs"]}
    for name in (
        "v3-ppo-native-job-v10-1536g.metrics.json",
        "v3-ppo-native-job-v11-1536g.metrics.json",
        "v3-ppo-native-job-v13-seedloo-1536g.metrics.json",
    ):
        if (STUDENT_ROOT / name).is_file() and name in runs:
            assert runs[name]["checkpoint_child"]["sha256_match"]
            assert runs[name]["by_opponent"]
            assert runs[name]["timing"]["games_per_second"]
            assert runs[name]["timing"]["events_per_second"]
    v11 = runs.get("v3-ppo-native-job-v11-1536g.metrics.json")
    if v11:
        assert all(isinstance(v11["timing"].get(key), (int, float)) for key in (
            "rollout_seconds", "save_seconds", "cpu_replay_seconds",
            "device_replay_seconds", "update_seconds", "total_seconds"))
        assert isinstance(v11["ppo"]["approx_kl"], (int, float))
        assert isinstance(v11["ppo"]["clip_fraction"], (int, float))
        assert v11["logprob_replay"]["cpu"]["gate_passed"] is True
        assert v11["optimizer"]["step_after"] > v11["optimizer"]["step_before"]
    v13 = runs.get("v3-ppo-native-job-v13-seedloo-1536g.metrics.json")
    if v13:
        assert v13["algorithm"] in FORMAL_RL_ALGORITHMS
        assert v13["scope"] == "native_cpp_job_batch"
        assert v13["rollout"]["name"].endswith(".npz")
        assert v13["rollout"]["sha256_match"] is True
        assert v13["formal_eligible"] is True
    assert all("candidate-student" not in source for source in payload["training"]["sources"])
    assert all("slot-v2" not in source for source in payload["training"]["sources"])
    now = time.time()
    assert _normalise_continuous_rl(
        {"status": "training", "heartbeat_at": now}, now, now)["status"] == "training"
    assert _normalise_continuous_rl(
        {"status": "training", "heartbeat_at": now - 60}, now - 60, now)["status"] == "stale"
    assert _normalise_continuous_rl(
        {"status": "failed", "heartbeat_at": now - 60}, now - 60, now)["status"] == "failed"
    assert _normalise_continuous_rl(
        {"status": "training"}, now - 600, now, True)["status"] == "training"
    assert _normalise_continuous_rl(
        {"status": "training"}, now, now, False)["status"] == "failed"
    server = Server(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        from urllib.error import HTTPError
        from urllib.request import urlopen

        base = f"http://127.0.0.1:{server.server_port}"
        assert json.load(urlopen(base + "/healthz", timeout=3))["ok"] is True
        assert json.load(urlopen(base + "/api/status", timeout=10))["privacy"]["raw_private_data_exposed"] is False
        try:
            urlopen(base + "/api/file?path=/etc/passwd", timeout=3)
        except HTTPError as error:
            assert error.code == 404
        else:
            raise AssertionError("arbitrary file endpoint unexpectedly accepted")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
    print("dashboard self-check: PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=39765)
    parser.add_argument("--pid-file", type=Path, default=RUNTIME_ROOT / "dashboard.pid")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        _self_check()
        return
    if not (1024 <= args.port <= 65535):
        parser.error("--port must be between 1024 and 65535")
    server = Server((args.host, args.port), Handler)
    args.pid_file.parent.mkdir(parents=True, exist_ok=True)
    args.pid_file.write_text(f"{os.getpid()}\n", encoding="ascii")

    def stop(_signum: int, _frame: Any) -> None:
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    print(f"dashboard listening on http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        server.server_close()
        try:
            args.pid_file.unlink()
        except OSError:
            pass


if __name__ == "__main__":
    main()
