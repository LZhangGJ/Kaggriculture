"""Refresh current top-25 Kaggriculture replay manifests with one API session."""

from __future__ import annotations

import csv
import os
import re
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import requests

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

ROOT = Path(r"D:\Kaggriculture")
TOP_N = 25
RANKED = ROOT / f"top{TOP_N}"
TEAM_MANIFEST = RANKED / f"current_top{TOP_N}_team_manifest.csv"
EPISODE_MANIFEST = RANKED / f"current_top{TOP_N}_episode_manifest.csv"
REPLAY_RE = re.compile(r"episode-(\d+)-replay\.json$")
REPLAY_ROOTS = (
    RANKED,
    ROOT / "top40",
    ROOT / "gold_top10",
    ROOT / "ranks_21_40",
    ROOT / "our_latest_two",
)


def as_dict(value):
    return value.to_dict() if hasattr(value, "to_dict") else dict(value)


def safe_name(value: str) -> str:
    return re.sub(r'[<>:"/\\|?*]', "_", value).rstrip(" .")


def scan_replays() -> dict[int, Path]:
    found = {}
    for replay_root in REPLAY_ROOTS:
        if not replay_root.is_dir():
            continue
        for path in replay_root.rglob("episode-*-replay.json"):
            match = REPLAY_RE.fullmatch(path.name)
            if match and path.is_file() and path.stat().st_size:
                found.setdefault(int(match.group(1)), path)
    return found


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def api_call(func, *args):
    for attempt in range(7):
        try:
            return func(*args)
        except Exception as exc:
            if "429" not in str(exc) or attempt == 6:
                raise
            delay = min(60, 5 * 2**attempt)
            print(f"RATE_LIMIT|retry={attempt + 1}|sleep={delay}s", flush=True)
            time.sleep(delay)


def main() -> int:
    original_send = requests.Session.send

    def send_with_timeout(session, request, **kwargs):
        kwargs.setdefault("timeout", 10.0)
        return original_send(session, request, **kwargs)

    requests.Session.send = send_with_timeout
    import kaggle

    RANKED.mkdir(parents=True, exist_ok=True)
    api = kaggle.api  # The package authenticates this shared client once.
    leaders = [
        as_dict(row)
        for row in api_call(api.competition_leaderboard_view, "kaggriculture", TOP_N)
    ][:TOP_N]

    teams = []
    episode_rows = []
    for rank, leader in enumerate(leaders, 1):
        team_id = int(leader["teamId"])
        submissions = [
            as_dict(row) for row in api_call(api.competition_team_submissions, team_id)
        ]
        scored = [row for row in submissions if str(row.get("publicScore", "")).strip()]
        if not scored:
            print(
                f"TEAM_FAILED|rank={rank}|team={team_id}|no completed scored submission",
                flush=True,
            )
            continue
        selected = max(
            scored,
            key=lambda row: (
                float(row["publicScore"]),
                str(row.get("dateSubmitted", "")),
            ),
        )
        submission_id = int(selected["id"])
        team_name = str(leader["teamName"])
        teams.append(
            {
                "rank": rank,
                "team_id": team_id,
                "team_name": team_name,
                "leaderboard_score": leader["score"],
                "submission_id": submission_id,
                "public_score": selected["publicScore"],
                "date_submitted": selected.get("dateSubmitted", ""),
            }
        )
        episodes = [
            as_dict(row)
            for row in api_call(api.competition_list_episodes, submission_id)
        ]
        completed = [
            row for row in episodes if str(row.get("state", "")).endswith("COMPLETED")
        ]
        target_dir = (
            RANKED
            / f"{team_id}_{safe_name(team_name)}"
            / f"{submission_id}_agent"
            / "replays"
        )
        for episode in completed:
            episode_id = int(episode["id"])
            episode_rows.append(
                {
                    "rank": rank,
                    "team_id": team_id,
                    "team_name": team_name,
                    "submission_id": submission_id,
                    "public_score": selected["publicScore"],
                    "episode_id": episode_id,
                    "episode_type": episode.get("type", ""),
                    "replay_path": str(
                        target_dir / f"episode-{episode_id}-replay.json"
                    ),
                    "was_present": False,
                    "reused_now": False,
                }
            )
        print(
            f"QUERY|rank={rank}|team={team_id}|submission={submission_id}|"
            f"score={selected['publicScore']}|episodes={len(completed)}",
            flush=True,
        )
        time.sleep(1.0)

    for current, previous in (
        (TEAM_MANIFEST, RANKED / f"previous_top{TOP_N}_team_manifest.csv"),
        (EPISODE_MANIFEST, RANKED / f"previous_top{TOP_N}_episode_manifest.csv"),
    ):
        if current.is_file():
            shutil.copy2(current, previous)
    team_fields = [
        "rank",
        "team_id",
        "team_name",
        "leaderboard_score",
        "submission_id",
        "public_score",
        "date_submitted",
    ]
    episode_fields = [
        "rank",
        "team_id",
        "team_name",
        "submission_id",
        "public_score",
        "episode_id",
        "episode_type",
        "replay_path",
        "was_present",
        "reused_now",
    ]
    write_csv(TEAM_MANIFEST, teams, team_fields)
    write_csv(EPISODE_MANIFEST, episode_rows, episode_fields)

    existing = scan_replays()
    grouped = defaultdict(list)
    for row in episode_rows:
        grouped[row["episode_id"]].append(row)

    new_downloads = reused = failed = 0
    failures = []
    consecutive_download_failures = 0
    replay_circuit_open = False
    for index, (episode_id, rows) in enumerate(sorted(grouped.items()), 1):
        requested = False
        targets = [Path(row["replay_path"]) for row in rows]
        source = next(
            (
                path
                for path in targets
                if path.is_file() and path.stat().st_size
            ),
            None,
        )
        if source is not None:
            for row, target in zip(rows, targets):
                row["was_present"] = target.is_file() and target.stat().st_size > 0
        if source is None:
            source = existing.get(episode_id)
        if source is None and replay_circuit_open:
            failed += 1
            failures.append(
                (episode_id, "replay circuit open after 10 consecutive API failures")
            )
        elif source is None:
            requested = True
            target = targets[0]
            target.parent.mkdir(parents=True, exist_ok=True)
            error = ""
            try:
                api.competition_episode_replay(
                    episode_id, path=str(target.parent), quiet=True
                )
                if target.is_file() and target.stat().st_size:
                    source = target
                    existing[episode_id] = target
                    new_downloads += 1
                    error = ""
                else:
                    error = "download returned no non-empty replay"
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            if source is None:
                failed += 1
                failures.append((episode_id, error))
                consecutive_download_failures += 1
                if consecutive_download_failures >= 10:
                    replay_circuit_open = True
                    print(
                        "REPLAY_CIRCUIT_OPEN|consecutive_failures=10", flush=True
                    )
            else:
                consecutive_download_failures = 0
        else:
            consecutive_download_failures = 0
        if source is not None:
            for row, target in zip(rows, targets):
                if target.is_file() and target.stat().st_size:
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.link(source, target)
                    row["reused_now"] = True
                    reused += 1
                except OSError:
                    try:
                        shutil.copy2(source, target)
                        row["reused_now"] = True
                        reused += 1
                    except OSError as exc:
                        failures.append((episode_id, f"reuse {target}: {exc}"))
                        failed += 1
        if index % 50 == 0 or index == len(grouped):
            print(
                f"PROGRESS|episodes={index}/{len(grouped)}|new={new_downloads}|"
                f"reused={reused}|failed={failed}",
                flush=True,
            )
        if requested:
            time.sleep(0.75)

    write_csv(TEAM_MANIFEST, teams, team_fields)
    write_csv(EPISODE_MANIFEST, episode_rows, episode_fields)

    for team in teams:
        rows = [row for row in episode_rows if row["team_id"] == team["team_id"]]
        missing = sum(not Path(row["replay_path"]).is_file() for row in rows)
        print(
            f"TEAM|rank={team['rank']}|team={team['team_id']}:{team['team_name']}|"
            f"submission={team['submission_id']}|score={team['public_score']}|"
            f"episodes={len(rows)}|"
            f"new_or_linked={sum(not row['was_present'] for row in rows) - missing}|"
            f"reused={sum(bool(row['reused_now']) for row in rows)}|missing={missing}",
            flush=True,
        )
    for episode_id, error in failures:
        print(f"FAILED|episode={episode_id}|{error}", flush=True)
    print(
        f"SUMMARY|teams={len(teams)}|episode_rows={len(episode_rows)}|"
        f"distinct_episodes={len(grouped)}|new_downloads={new_downloads}|"
        f"reused={reused}|failed={failed}",
        flush=True,
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
