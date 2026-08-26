#!/usr/bin/env python3
"""Download every completed Public replay for a live leaderboard score band.

The official leaderboard response supplies the active submission id for each
team.  Episode lists are fetched from Kaggle's EpisodeService and replay JSONs
are stored once in a shared content directory, so a match between two selected
teams is not duplicated on disk.  Per-submission manifests retain each team's
seat, result, opponent and rating metadata.
"""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any
import urllib.error
import urllib.request


LIST_URL = "https://www.kaggle.com/api/i/competitions.EpisodeService/ListEpisodes"
REPLAY_URL = "https://www.kaggle.com/competitions/episodes/{episode_id}/replay.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def request_json(url: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"User-Agent": "Mozilla/5.0", "Content-Type": "application/json"},
        method="GET" if body is None else "POST",
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.load(response)


def validate_replay(path: Path, episode_id: int) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size < 1024:
        raise RuntimeError("replay missing or too small")
    payload = json.loads(path.read_text(encoding="utf-8"))
    actual = int(payload.get("info", {}).get("EpisodeId", -1))
    if actual != episode_id:
        raise RuntimeError(f"EpisodeId mismatch {actual} != {episode_id}")
    if len(payload.get("steps", [])) < 2:
        raise RuntimeError("replay has no usable steps")
    return payload


def cached_candidates(reuse_roots: list[Path]) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for root in reuse_roots:
        if not root.exists():
            continue
        for path in root.rglob("*.json"):
            stem = path.stem
            if stem.startswith("episode-") and stem.endswith("-replay"):
                stem = stem[len("episode-") : -len("-replay")]
            if stem.isdigit():
                result.setdefault(int(stem), path.resolve())
    return result


def materialize_cached(source: Path, destination: Path, episode_id: int) -> str:
    validate_replay(source, episode_id)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(".json.part")
    partial.unlink(missing_ok=True)
    try:
        os.link(source, partial)
        method = "hardlinked_cache"
    except OSError:
        shutil.copy2(source, partial)
        method = "copied_cache"
    validate_replay(partial, episode_id)
    os.replace(partial, destination)
    return method


def download_one(
    episode_id: int,
    output_dir: Path,
    retries: int,
    cache: dict[int, Path],
) -> dict[str, Any]:
    path = output_dir / "episodes" / f"{episode_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        replay = validate_replay(path, episode_id)
        status = "existing_valid"
    except Exception:
        source = cache.get(episode_id)
        if source is not None:
            try:
                status = materialize_cached(source, path, episode_id)
                replay = validate_replay(path, episode_id)
            except Exception:
                source = None
        if source is None:
            status = "downloaded"
            partial = path.with_suffix(".json.part")
            error = ""
            for attempt in range(1, retries + 1):
                try:
                    request = urllib.request.Request(
                        REPLAY_URL.format(episode_id=episode_id),
                        headers={
                            "User-Agent": "Mozilla/5.0",
                            "Accept-Encoding": "gzip",
                        },
                    )
                    with urllib.request.urlopen(request, timeout=180) as response, partial.open("wb") as handle:
                        encoding = str(response.headers.get("Content-Encoding", "")).lower()
                        source = gzip.GzipFile(fileobj=response) if encoding == "gzip" else response
                        try:
                            while chunk := source.read(1024 * 1024):
                                handle.write(chunk)
                        finally:
                            if source is not response:
                                source.close()
                    replay = validate_replay(partial, episode_id)
                    os.replace(partial, path)
                    break
                except Exception as exc:  # noqa: BLE001 - receipt records exact network error
                    error = str(exc)
                    partial.unlink(missing_ok=True)
                    if attempt == retries:
                        return {
                            "episode_id": episode_id,
                            "status": "failed",
                            "error": error,
                        }
                    time.sleep(min(20, 2**attempt))
    return {
        "episode_id": episode_id,
        "status": status,
        "error": "",
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "module_version": str(
            replay.get(
                "module_version",
                replay.get("info", {}).get("ModuleVersion", ""),
            )
        ),
        "step_frames": len(replay.get("steps", [])),
    }


def materialize_local_only(
    episode_id: int,
    output_dir: Path,
    cache: dict[int, Path],
    trust_local_cache: bool = False,
) -> dict[str, Any]:
    """Validate/materialize a local replay without ever touching the network."""
    path = output_dir / "episodes" / f"{episode_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if trust_local_cache:
        source = path if path.is_file() and path.stat().st_size >= 1024 else cache.get(episode_id)
        if source is None or not source.is_file() or source.stat().st_size < 1024:
            return {
                "episode_id": episode_id,
                "status": "cache_failed",
                "error": "trusted cache candidate missing or too small",
            }
        if source.resolve() == path.resolve():
            status = "trusted_existing"
        else:
            partial = path.with_suffix(".json.part")
            partial.unlink(missing_ok=True)
            try:
                os.link(source, partial)
                status = "trusted_hardlinked_cache"
            except OSError:
                shutil.copy2(source, partial)
                status = "trusted_copied_cache"
            os.replace(partial, path)
        return {
            "episode_id": episode_id,
            "status": status,
            "error": "",
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": "",
            "module_version": "",
            "step_frames": "",
        }
    try:
        replay = validate_replay(path, episode_id)
        status = "existing_valid"
    except Exception as existing_error:
        source = cache.get(episode_id)
        if source is None:
            return {
                "episode_id": episode_id,
                "status": "cache_failed",
                "error": f"no cache candidate; existing={existing_error}",
            }
        try:
            status = materialize_cached(source, path, episode_id)
            replay = validate_replay(path, episode_id)
        except Exception as cache_error:
            return {
                "episode_id": episode_id,
                "status": "cache_failed",
                "error": str(cache_error),
            }
    return {
        "episode_id": episode_id,
        "status": status,
        "error": "",
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "module_version": str(
            replay.get(
                "module_version",
                replay.get("info", {}).get("ModuleVersion", ""),
            )
        ),
        "step_frames": len(replay.get("steps", [])),
    }


def fetch_submission_episodes(
    submission_id: int,
    retries: int,
) -> tuple[int, dict[str, Any]]:
    error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return submission_id, request_json(
                LIST_URL,
                {
                    "ids": [],
                    "submissionId": submission_id,
                    "successfulOnly": True,
                    "includeInProgress": False,
                },
            )
        except urllib.error.HTTPError as exc:
            error = exc
            if exc.code != 429 or attempt == retries:
                raise
            retry_after = exc.headers.get("Retry-After")
            delay = int(retry_after) if retry_after and retry_after.isdigit() else 5 * attempt
            time.sleep(min(60, max(5, delay)))
        except Exception as exc:  # noqa: BLE001 - retry transient API failures
            error = exc
            if attempt == retries:
                raise
            time.sleep(min(60, 3 * attempt))
    raise RuntimeError(f"episode-list request failed for {submission_id}: {error}")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--leaderboard-json", type=Path, required=True)
    parser.add_argument("--score-min", type=float, default=2700.0)
    parser.add_argument(
        "--rank-min",
        type=int,
        default=None,
        help="Optionally keep only leaderboard entries with rank >= this value.",
    )
    parser.add_argument(
        "--rank-max",
        type=int,
        default=None,
        help="Optionally keep only leaderboard entries with rank <= this value.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=18)
    parser.add_argument("--api-workers", type=int, default=8)
    parser.add_argument(
        "--latest-per-submission",
        type=int,
        default=None,
        help="Keep only the newest N completed Public episodes per selected submission.",
    )
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--reuse-root", type=Path, action="append", default=[])
    parser.add_argument(
        "--trust-local-cache",
        action="store_true",
        help="Skip repeat JSON parsing for previously validated local replay files.",
    )
    args = parser.parse_args()
    if not 1 <= args.workers <= 18:
        raise ValueError("--workers must be between 1 and 18")
    if not 1 <= args.api_workers <= 18:
        raise ValueError("--api-workers must be between 1 and 18")

    root = args.output_root.resolve()
    api_dir = root / "api_raw"
    manifest_dir = root / "submission_manifests"
    api_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    leaderboard = json.loads(args.leaderboard_json.read_text(encoding="utf-8-sig"))
    teams = {int(row["teamId"]): row for row in leaderboard.get("teams", [])}
    selected: list[dict[str, Any]] = []
    for entry in leaderboard.get("publicLeaderboard", []):
        if args.rank_min is not None and int(entry.get("rank", 0) or 0) < args.rank_min:
            continue
        if args.rank_max is not None and int(entry.get("rank", 0) or 0) > args.rank_max:
            continue
        if float(entry.get("displayScore", 0) or 0) < args.score_min:
            continue
        team_id = int(entry["teamId"])
        team = teams[team_id]
        selected.append(
            {
                "rank": int(entry["rank"]),
                "team_id": team_id,
                "submission_id": int(entry["submissionId"]),
                "team_name": str(team.get("teamName", "")),
                "score": float(entry.get("displayScore", 0) or 0),
                "members": "|".join(
                    str(member.get("userName", ""))
                    for member in team.get("teamMembers", [])
                ),
                "last_submission_utc": str(team.get("lastSubmissionDate", "")),
            }
        )
    selected.sort(key=lambda row: row["rank"])
    if not selected:
        raise RuntimeError("score band selected no leaderboard entries")
    write_csv(root / "selected_teams.csv", selected)

    api_payloads: dict[int, dict[str, Any]] = {}
    missing_api_rows: list[dict[str, Any]] = []
    for row in selected:
        submission_id = int(row["submission_id"])
        path = api_dir / f"list_episodes_submission_{submission_id}.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload.get("episodes"), list):
                raise ValueError("missing episodes list")
            api_payloads[submission_id] = payload
            print(
                f"episode-list submission={submission_id} episodes={len(payload['episodes'])} status=existing",
                flush=True,
            )
        except Exception:
            missing_api_rows.append(row)
    with ThreadPoolExecutor(max_workers=args.api_workers) as pool:
        futures = {
            pool.submit(
                fetch_submission_episodes,
                int(row["submission_id"]),
                max(5, args.retries),
            ): row
            for row in missing_api_rows
        }
        for future in as_completed(futures):
            submission_id, payload = future.result()
            api_payloads[submission_id] = payload
            path = api_dir / f"list_episodes_submission_{submission_id}.json"
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"episode-list submission={submission_id} episodes={len(payload.get('episodes', []))}", flush=True)

    selected_by_submission = {int(row["submission_id"]): row for row in selected}
    rows_by_submission: dict[int, list[dict[str, Any]]] = {
        submission_id: [] for submission_id in selected_by_submission
    }
    unique_episode_ids: set[int] = set()
    for submission_id, payload in api_payloads.items():
        for episode in payload.get("episodes", []):
            agents = episode.get("agents", [])
            own_seats = [
                index
                for index, agent in enumerate(agents)
                if int(agent.get("submissionId", 0) or 0) == submission_id
            ]
            episode_type = str(episode.get("type", ""))
            if (
                len(agents) != 2
                or len(own_seats) != 1
                or str(episode.get("state", "")) != "COMPLETED"
                or episode_type not in {"PUBLIC", "EPISODE_TYPE_PUBLIC"}
            ):
                continue
            own_seat = own_seats[0]
            opponent_seat = 1 - own_seat
            own_reward = float(agents[own_seat].get("reward", 0) or 0)
            opponent_reward = float(agents[opponent_seat].get("reward", 0) or 0)
            result = "WIN" if own_reward > opponent_reward else "LOSS" if own_reward < opponent_reward else "TIE"
            episode_id = int(episode["id"])
            unique_episode_ids.add(episode_id)
            rows_by_submission[submission_id].append(
                {
                    "episode_id": episode_id,
                    "create_time_utc": str(episode.get("createTime", "")),
                    "end_time_utc": str(episode.get("endTime", "")),
                    "own_seat": own_seat,
                    "opponent_seat": opponent_seat,
                    "opponent_team_id": int(agents[opponent_seat].get("teamId", 0) or 0),
                    "opponent_submission_id": int(agents[opponent_seat].get("submissionId", 0) or 0),
                    "own_initial_score": float(agents[own_seat].get("initialScore", 0) or 0),
                    "own_updated_score": float(agents[own_seat].get("updatedScore", 0) or 0),
                    "opponent_initial_score": float(agents[opponent_seat].get("initialScore", 0) or 0),
                    "opponent_updated_score": float(agents[opponent_seat].get("updatedScore", 0) or 0),
                    "own_reward": own_reward,
                    "opponent_reward": opponent_reward,
                    "margin": own_reward - opponent_reward,
                    "result": result,
                }
            )

    if args.latest_per_submission is not None:
        if args.latest_per_submission < 1:
            raise ValueError("--latest-per-submission must be positive")
        for submission_id, rows in rows_by_submission.items():
            rows_by_submission[submission_id] = sorted(
                rows,
                key=lambda row: (row["create_time_utc"], row["episode_id"]),
                reverse=True,
            )[: args.latest_per_submission]
        unique_episode_ids = {
            int(row["episode_id"])
            for rows in rows_by_submission.values()
            for row in rows
        }

    cache = cached_candidates([path.resolve() for path in args.reuse_root])
    downloads: dict[int, dict[str, Any]] = {}
    local_ids = [
        episode_id
        for episode_id in sorted(unique_episode_ids)
        if (root / "episodes" / f"{episode_id}.json").is_file()
        or episode_id in cache
    ]
    remote_ids = [
        episode_id
        for episode_id in sorted(unique_episode_ids)
        if episode_id not in set(local_ids)
    ]
    print(
        f"replay-plan local={len(local_ids)} remote={len(remote_ids)} "
        f"local_workers={args.workers} network_workers=1",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                materialize_local_only,
                episode_id,
                root,
                cache,
                args.trust_local_cache,
            ): episode_id
            for episode_id in local_ids
        }
        for index, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            if result["status"] == "cache_failed":
                remote_ids.append(int(result["episode_id"]))
            else:
                downloads[int(result["episode_id"])] = result
            if index % 100 == 0 or result["status"] == "cache_failed" or index == len(futures):
                print(
                    f"local-replays {index}/{len(futures)} "
                    f"episode={result['episode_id']} status={result['status']}",
                    flush=True,
                )

    for index, episode_id in enumerate(sorted(set(remote_ids)), start=1):
        result = download_one(episode_id, root, args.retries, {})
        downloads[int(result["episode_id"])] = result
        if index % 25 == 0 or result["status"] == "failed" or index == len(remote_ids):
            print(
                f"network-replays {index}/{len(set(remote_ids))} "
                f"episode={result['episode_id']} status={result['status']}",
                flush=True,
            )

    all_rows: list[dict[str, Any]] = []
    per_submission_summary: list[dict[str, Any]] = []
    for submission_id, team in selected_by_submission.items():
        episodes = sorted(
            rows_by_submission[submission_id],
            key=lambda row: (row["create_time_utc"], row["episode_id"]),
        )
        enriched = []
        for row in episodes:
            download = downloads[row["episode_id"]]
            merged = {**row, **download}
            enriched.append(merged)
            all_rows.append({**team, **merged})
        counts = {
            result: sum(row["result"] == result for row in episodes)
            for result in ("WIN", "LOSS", "TIE")
        }
        manifest = {
            "schema": "kaggriculture-score-band-submission-replays-v1",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            **team,
            "episode_count": len(enriched),
            "result_counts": counts,
            "episodes": enriched,
        }
        (manifest_dir / f"rank{team['rank']:02d}_{submission_id}.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        per_submission_summary.append({**team, "episode_count": len(enriched), **counts})

    write_csv(root / "submission_summary.csv", per_submission_summary)
    write_csv(root / "episode_rows.csv", all_rows)
    failed = [row for row in downloads.values() if row["status"] == "failed"]
    summary = {
        "schema": "kaggriculture-live-score-band-all-public-replays-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "official LeaderboardService, EpisodeService/ListEpisodes and replay.json",
        "score_min": args.score_min,
        "rank_min": args.rank_min,
        "rank_max": args.rank_max,
        "latest_per_submission": args.latest_per_submission,
        "selected_teams": len(selected),
        "per_submission_episode_rows": len(all_rows),
        "unique_episode_count": len(unique_episode_ids),
        "valid_unique_replays": len(downloads) - len(failed),
        "failed_unique_replays": len(failed),
        "cache_candidates": len(cache),
        "materialization_counts": {
            status: sum(row["status"] == status for row in downloads.values())
            for status in (
                "existing_valid",
                "hardlinked_cache",
                "copied_cache",
                "trusted_existing",
                "trusted_hardlinked_cache",
                "trusted_copied_cache",
                "downloaded",
                "failed",
            )
        },
        "unique_bytes": sum(int(row.get("bytes", 0)) for row in downloads.values()),
        "leaderboard_sha256": sha256(args.leaderboard_json),
        "failed": failed,
    }
    (root / "DOWNLOAD_SUMMARY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 2 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
