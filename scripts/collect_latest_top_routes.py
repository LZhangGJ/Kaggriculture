#!/usr/bin/env python3
"""Collect recent public replay routes from the current Kaggriculture top N.

Unlike the original cold-start collector, metadata calls have timeouts and are
checkpointed before the much larger replay downloads begin. Kaggle access is
serial by contract; CPU-heavy processing happens after download.
Rerunning the same command resumes valid replay files and reuses selection.json.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, TypeVar


T = TypeVar("T")


class KaggleRateLimited(RuntimeError):
    pass


@dataclass(frozen=True)
class Target:
    team_id: int
    team_name: str
    leaderboard_score: float
    submission_id: int


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--competition", default="kaggriculture")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-teams", type=int, default=60)
    parser.add_argument("--episodes-per-team", type=int, default=8)
    parser.add_argument("--metadata-workers", type=int, default=1)
    parser.add_argument("--download-workers", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=45.0)
    args = parser.parse_args()
    if args.metadata_workers != 1 or args.download_workers != 1:
        parser.error("Kaggle metadata and replay access must remain single-threaded")
    return args


def _raise_rate_limit(message: str) -> None:
    if "429" in message or "Too Many Requests" in message:
        raise KaggleRateLimited(message)


def _json_stdout(stdout: str) -> Any:
    lines = stdout.splitlines()
    payload = next(
        ("\n".join(lines[index:]) for index, line in enumerate(lines)
         if line.startswith(("[", "{"))),
        "",
    )
    return json.loads(payload)


def require_replay_cli(cli: str) -> None:
    result = subprocess.run(
        [cli, "competitions", "--help"], check=False, capture_output=True, text=True
    )
    if result.returncode or "replay" not in (result.stdout + result.stderr):
        raise RuntimeError(f"Kaggle CLI does not support competitions replay: {cli}")


def kaggle_json(cli: str, timeout: float, *args: str) -> Any:
    last = ""
    for attempt in range(8):
        try:
            result = subprocess.run(
                [cli, "competitions", *args, "--format", "json", "--quiet"],
                check=False, capture_output=True, text=True, timeout=timeout,
            )
            if result.returncode == 0:
                return _json_stdout(result.stdout)
            last = result.stderr.strip() or result.stdout.strip()
            _raise_rate_limit(last)
        except (subprocess.TimeoutExpired, json.JSONDecodeError) as error:
            last = f"{type(error).__name__}: {error}"
        # Metadata endpoints apply a fairly strict shared-account rate limit.
        # Long capped backoff is preferable to discarding already resolved teams.
        time.sleep(min(60.0, 3.0 * 2.0 ** attempt))
    raise RuntimeError(f"Kaggle CLI failed: {' '.join(args)}: {last}")


def cached_kaggle_json(
    cache_path: Path, cli: str, timeout: float, *args: str
) -> Any:
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    value = kaggle_json(cli, timeout, *args)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return value


def parallel_map(
    values: list[T], function: Callable[[T], Any], workers: int, label: str
) -> list[Any]:
    if workers == 1:
        results = []
        for value in values:
            results.append(function(value))
            print(f"{label} {value}: ok [{len(results)}/{len(values)}]", flush=True)
        return results
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(function, value): value for value in values}
        for future in concurrent.futures.as_completed(futures):
            value = futures[future]
            result = future.result()
            results.append(result)
            print(f"{label} {value}: ok [{len(results)}/{len(values)}]", flush=True)
    return results


def replay_summary(path: Path) -> dict[str, Any] | None:
    try:
        import orjson

        payload = orjson.loads(path.read_bytes())
        if not isinstance(payload.get("steps"), list) or len(payload["steps"]) < 2:
            return None
        return {
            "team_names": [
                str(value) for value in (payload.get("info", {}) or {}).get("TeamNames", [])
            ],
            "rewards": list(payload.get("rewards", []) or []),
            "bytes": path.stat().st_size,
        }
    except (OSError, ValueError, AttributeError):
        return None


def valid_replay(path: Path) -> bool:
    return replay_summary(path) is not None


def download(cli: str, timeout: float, replay_dir: Path, episode_id: int) -> dict[str, Any]:
    path = replay_dir / f"episode-{episode_id}-replay.json"
    summary = replay_summary(path)
    if summary is not None:
        return {"episode_id": episode_id, "error": None, "cached": True, **summary}
    path.unlink(missing_ok=True)
    last = ""
    for attempt in range(4):
        try:
            result = subprocess.run(
                [cli, "competitions", "replay", str(episode_id),
                 "--path", str(replay_dir), "--quiet"],
                check=False, capture_output=True, text=True, timeout=max(120.0, timeout * 4),
            )
            if result.returncode == 0:
                summary = replay_summary(path)
                if summary is not None:
                    return {
                        "episode_id": episode_id, "error": None,
                        "cached": False, **summary,
                    }
            last = result.stderr.strip() or result.stdout.strip() or "invalid replay"
            _raise_rate_limit(last)
        except subprocess.TimeoutExpired as error:
            last = f"TimeoutExpired: {error}"
        path.unlink(missing_ok=True)
        time.sleep(min(8.0, 2.0 ** attempt))
    return {"episode_id": episode_id, "error": last, "cached": False}


def main() -> None:
    args = arguments()
    cli = shutil.which("kaggle")
    if cli is None:
        raise RuntimeError("kaggle CLI is not on PATH")
    require_replay_cli(cli)
    args.output.mkdir(parents=True, exist_ok=True)
    selection_path = args.output / "selection.json"
    metadata_dir = args.output / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    resume_selection = None
    if selection_path.exists():
        resume_selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selection_matches = bool(
        resume_selection
        and int(resume_selection.get("top_teams", -1)) == args.top_teams
        and int(resume_selection.get("episodes_per_team", -1)) == args.episodes_per_team
        and str(resume_selection.get("competition", "")) == args.competition
    )
    if selection_matches:
        selection = resume_selection
        targets = [Target(**row) for row in selection["targets"]]
        episode_targets = {
            int(key): [Target(**row) for row in values]
            for key, values in selection["episode_targets"].items()
        }
        print(f"resuming {len(targets)} targets and {len(episode_targets)} episodes", flush=True)
    else:
        if resume_selection:
            print(
                "selection parameters changed; rebuilding from cached metadata "
                f"({resume_selection.get('episodes_per_team')} -> {args.episodes_per_team} episodes/team)",
                flush=True,
            )
        leaderboard = cached_kaggle_json(
            metadata_dir / f"leaderboard-top{args.top_teams}.json",
            cli, args.timeout, "leaderboard", args.competition, "--show",
            "--page-size", str(args.top_teams),
        )[:args.top_teams]
        if len(leaderboard) != args.top_teams:
            raise RuntimeError(f"leaderboard returned {len(leaderboard)} rows")

        def target_for(row: dict[str, Any]) -> Target:
            submissions = cached_kaggle_json(
                metadata_dir / f"team-submissions-{int(row['teamId'])}.json",
                cli, args.timeout, "team-submissions", str(row["teamId"])
            )
            best = max(
                submissions,
                key=lambda value: (float(value["publicScore"]), value["dateSubmitted"]),
            )
            return Target(
                team_id=int(row["teamId"]), team_name=str(row["teamName"]),
                leaderboard_score=float(row["score"]), submission_id=int(best["id"]),
            )

        targets = parallel_map(
            leaderboard, target_for, args.metadata_workers, "submission"
        )
        targets.sort(key=lambda row: (-row.leaderboard_score, row.team_id))

        def episodes_for(target: Target) -> tuple[Target, list[int]]:
            episodes = cached_kaggle_json(
                metadata_dir / f"episodes-{target.submission_id}.json",
                cli, args.timeout, "episodes", str(target.submission_id)
            )
            public = [
                row for row in episodes
                if str(row.get("state", "")).endswith("COMPLETED")
                and str(row.get("type", "")).endswith("PUBLIC")
            ]
            public.sort(key=lambda row: str(row.get("createTime", "")), reverse=True)
            return target, [int(row["id"]) for row in public[:args.episodes_per_team]]

        episode_targets: dict[int, list[Target]] = {}
        for target, episodes in parallel_map(
            targets, episodes_for, args.metadata_workers, "episodes"
        ):
            for episode_id in episodes:
                episode_targets.setdefault(episode_id, []).append(target)
        selection = {
            "competition": args.competition,
            "selected_at_utc": datetime.now(UTC).isoformat(),
            "top_teams": args.top_teams,
            "episodes_per_team": args.episodes_per_team,
            "targets": [asdict(row) for row in targets],
            "episode_targets": {
                str(key): [asdict(row) for row in values]
                for key, values in sorted(episode_targets.items())
            },
        }
        selection_path.write_text(
            json.dumps(selection, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"checkpointed {selection_path}", flush=True)

    replay_dir = args.output / "replays"
    replay_dir.mkdir(parents=True, exist_ok=True)
    episode_ids = sorted(
        episode_targets,
        key=lambda episode_id: (
            (replay_dir / f"episode-{episode_id}-replay.json").exists(), episode_id
        ),
    )
    results = parallel_map(
        episode_ids,
        lambda episode_id: download(cli, args.timeout, replay_dir, episode_id),
        args.download_workers,
        "replay",
    )
    result_by_id = {int(row["episode_id"]): row for row in results}

    replay_rows = []
    for episode_id, requested in sorted(episode_targets.items()):
        path = replay_dir / f"episode-{episode_id}-replay.json"
        result = result_by_id[episode_id]
        row: dict[str, Any] = {
            "episode_id": episode_id,
            "replay": str(path.relative_to(args.output)),
            "requested_targets": [asdict(value) for value in requested],
            "error": result["error"],
        }
        if result["error"] is None:
            names = list(result["team_names"])
            rewards = list(result["rewards"])
            target_rows = []
            for target in requested:
                for player, name in enumerate(names):
                    if name == target.team_name:
                        target_rows.append({
                            **asdict(target), "player_index": player,
                            "final_reward": rewards[player] if player < len(rewards) else None,
                        })
            row.update({
                "bytes": int(result["bytes"]), "team_names": names,
                "targets": target_rows, "demonstrations_per_target": 719,
                "alignment": "steps[t-1][player_index].observation -> steps[t][player_index].action, t=1..719",
            })
        replay_rows.append(row)
    failures = sum(row["error"] is not None for row in replay_rows)
    manifest = {
        "competition": args.competition,
        "collected_at_utc": datetime.now(UTC).isoformat(),
        "selection": {
            "top_teams": args.top_teams,
            "episodes_per_team": args.episodes_per_team,
            "public_completed_only": True,
        },
        "leaderboard_targets": [asdict(row) for row in targets],
        "replays": replay_rows,
    }
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"wrote {manifest_path}; {len(replay_rows) - failures} downloaded, {failures} failed",
        flush=True,
    )


if __name__ == "__main__":
    main()
