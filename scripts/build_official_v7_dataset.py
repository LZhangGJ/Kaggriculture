"""Select diverse high-score official replays and convert them to V7 labels."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from kaggriculture_lab.official_replay_v7 import (
    OFFICIAL_V7_MANIFEST_SCHEMA,
    ReplayActorSequence,
    behavior_signature,
    build_v7_label_bundle,
    write_v7_label_bundle,
)


@dataclass
class Candidate:
    episode_id: int
    rank: int
    average_score: float
    raw_path: Path
    teams: tuple[str, str]
    hashes: tuple[str, str]
    signatures: tuple[np.ndarray, np.ndarray]


def _load_candidates(
    replay_dir: Path,
    manifest_path: Path,
    limit: int,
) -> list[Candidate]:
    with manifest_path.open(newline="", encoding="utf-8") as stream:
        rows = sorted(
            csv.DictReader(stream),
            key=lambda row: float(row["avg_score"]),
            reverse=True,
        )[:limit]
    candidates: list[Candidate] = []
    for rank, row in enumerate(rows, start=1):
        episode_id = int(row["episode_id"])
        raw_path = replay_dir / f"{episode_id}.json"
        if not raw_path.is_file():
            raise FileNotFoundError(raw_path)
        with raw_path.open(encoding="utf-8") as stream:
            replay = json.load(stream)
        steps = list(replay.get("steps", []) or [])
        if len(steps) < 2 or any(len(step) < 2 for step in steps):
            raise ValueError(f"invalid two-player replay: {raw_path}")
        info = replay.get("info", {}) or {}
        names = list(info.get("TeamNames", []) or [])
        if len(names) < 2:
            names = [
                str(agent.get("Name", ""))
                for agent in list(info.get("Agents", []) or [])[:2]
            ]
        names.extend(f"unknown-player-{index}" for index in range(len(names), 2))
        rewards = list(replay.get("rewards", []) or [])
        rewards.extend(0.0 for _ in range(len(rewards), 2))
        # Candidate selection needs only immutable action streams.  Avoid
        # canonicalizing/copying ~1,440 full observations per 30 MiB replay.
        sequences = tuple(
            ReplayActorSequence(
                player=player,
                observations=(),
                actions=tuple(
                    dict(step[player].get("action", {}) or {}) for step in steps[1:]
                ),
                final_observation={},
                reward=float(rewards[player] or 0.0),
                teacher=str(names[player]),
            )
            for player in range(2)
        )
        candidates.append(
            Candidate(
                episode_id=episode_id,
                rank=rank,
                average_score=float(row["avg_score"]),
                raw_path=raw_path,
                teams=(sequences[0].teacher, sequences[1].teacher),
                hashes=(sequences[0].action_hash, sequences[1].action_hash),
                signatures=(
                    behavior_signature(sequences[0]),
                    behavior_signature(sequences[1]),
                ),
            )
        )
        print(
            f"candidate={rank}/{len(rows)} episode={episode_id} "
            f"avg_score={float(row['avg_score']):.3f}"
        )
    return candidates


def _select_candidates(
    candidates: list[Candidate],
    *,
    count: int,
    max_per_team: int,
    max_per_trajectory: int,
) -> list[Candidate]:
    selected: list[Candidate] = []
    selected_ids: set[int] = set()
    team_counts: Counter[str] = Counter()
    hash_counts: Counter[str] = Counter()

    def take(candidate: Candidate, team_cap: int, trajectory_cap: int) -> bool:
        if candidate.episode_id in selected_ids:
            return False
        if any(team_counts[name] >= team_cap for name in candidate.teams):
            return False
        if any(hash_counts[value] >= trajectory_cap for value in candidate.hashes):
            return False
        selected.append(candidate)
        selected_ids.add(candidate.episode_id)
        team_counts.update(candidate.teams)
        hash_counts.update(candidate.hashes)
        return True

    # Strict diversity pass, then two bounded relaxation passes so the requested
    # size is met even when the public meta contains many exact clones.
    for team_cap, trajectory_cap in (
        (max_per_team, max_per_trajectory),
        (max_per_team * 2, max_per_trajectory * 2),
        (max(count, max_per_team), max(count, max_per_trajectory)),
    ):
        for candidate in candidates:
            take(candidate, team_cap, trajectory_cap)
            if len(selected) >= count:
                return selected
    return selected


def _cluster_actors(
    selected: list[Candidate], clusters: int, seed: int
) -> dict[tuple[int, int], int]:
    features = np.stack(
        [signature for candidate in selected for signature in candidate.signatures]
    ).astype(np.float64)
    mean = features.mean(axis=0, keepdims=True)
    scale = features.std(axis=0, keepdims=True)
    normalized = (features - mean) / np.where(scale > 1e-8, scale, 1.0)
    cluster_count = min(max(1, clusters), len(normalized))
    rng = np.random.default_rng(seed)
    first = int(rng.integers(len(normalized)))
    centers = [normalized[first]]
    while len(centers) < cluster_count:
        distances = np.min(
            np.stack(
                [np.square(normalized - center).sum(axis=1) for center in centers]
            ),
            axis=0,
        )
        centers.append(normalized[int(np.argmax(distances))])
    center_array = np.stack(centers)
    labels = np.zeros(len(normalized), dtype=np.int64)
    for _ in range(50):
        distances = np.square(
            normalized[:, None, :] - center_array[None, :, :]
        ).sum(axis=2)
        updated = distances.argmin(axis=1)
        if np.array_equal(labels, updated):
            break
        labels = updated
        for cluster in range(cluster_count):
            members = normalized[labels == cluster]
            if len(members):
                center_array[cluster] = members.mean(axis=0)
    return {
        (candidate.episode_id, player): int(labels[2 * index + player])
        for index, candidate in enumerate(selected)
        for player in range(2)
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--replay-dir",
        type=Path,
        default=Path(r"D:\Kaggriculture\data\raw\replays\2026-08-21"),
    )
    parser.add_argument("--source-date", default="2026-08-21")
    parser.add_argument("--candidate-episodes", type=int, default=160)
    parser.add_argument("--max-episodes", type=int, default=64)
    parser.add_argument("--max-per-team", type=int, default=12)
    parser.add_argument("--max-per-trajectory", type=int, default=3)
    parser.add_argument("--clusters", type=int, default=8)
    parser.add_argument("--intent-horizon", type=int, default=24)
    parser.add_argument("--inverse-top-m", type=int, default=3)
    parser.add_argument("--value-gamma", type=float, default=0.997)
    parser.add_argument("--value-reward-scale", type=float, default=10.0)
    parser.add_argument("--value-win-bonus", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--skip-resimulation", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(r"D:\Kaggriculture\data\processed\official_v7_2026-08-21"),
    )
    args = parser.parse_args()
    if min(
        args.candidate_episodes,
        args.max_episodes,
        args.max_per_team,
        args.max_per_trajectory,
        args.clusters,
        args.intent_horizon,
        args.inverse_top_m,
    ) <= 0:
        parser.error("selection, clustering, and inverse-planning sizes must be positive")
    manifest_path = args.replay_dir / "manifest.csv"
    if not manifest_path.is_file():
        parser.error(f"missing official daily manifest: {manifest_path}")
    candidates = _load_candidates(
        args.replay_dir,
        manifest_path,
        min(args.candidate_episodes, 10_000),
    )
    selected = _select_candidates(
        candidates,
        count=min(args.max_episodes, len(candidates)),
        max_per_team=args.max_per_team,
        max_per_trajectory=args.max_per_trajectory,
    )
    if not selected:
        parser.error("selection produced no episodes")
    cluster_by_actor = _cluster_actors(selected, args.clusters, args.seed)
    label_dir = args.output_dir / "labels"
    label_dir.mkdir(parents=True, exist_ok=True)
    index_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    quality_counts: Counter[str] = Counter()
    team_counts: Counter[str] = Counter()
    cluster_counts: Counter[int] = Counter()
    for number, candidate in enumerate(selected, start=1):
        with candidate.raw_path.open(encoding="utf-8") as stream:
            replay = json.load(stream)
        clusters = tuple(
            cluster_by_actor[(candidate.episode_id, player)] for player in range(2)
        )
        bundle = build_v7_label_bundle(
            replay,
            episode_id=candidate.episode_id,
            raw_path=candidate.raw_path,
            source_date=args.source_date,
            daily_score_rank=candidate.rank,
            average_score=candidate.average_score,
            teacher_clusters=clusters,
            horizon=args.intent_horizon,
            top_m=args.inverse_top_m,
            value_gamma=args.value_gamma,
            value_reward_scale=args.value_reward_scale,
            value_win_bonus=args.value_win_bonus,
            validate_resimulation=not args.skip_resimulation,
        )
        label_path = label_dir / f"episode-{candidate.episode_id}-v7.json.gz"
        write_v7_label_bundle(label_path, bundle)
        tier = str(bundle["quality"]["tier"])
        quality_counts[tier] += 1
        episode_rows.append(
            {
                "episode_id": candidate.episode_id,
                "daily_score_rank": candidate.rank,
                "average_score": candidate.average_score,
                "raw_file": candidate.raw_path.name,
                "label_file": str(label_path.relative_to(args.output_dir)),
                "quality_tier": tier,
                "resimulation_matched": bool(
                    bundle["quality"]["resimulation"]["matched"]
                ),
            }
        )
        for actor in bundle["actors"]:
            team_counts[str(actor["teacher"])] += 1
            cluster_counts[int(actor["teacher_cluster"])] += 1
            index_rows.append(
                {
                    "episode_id": candidate.episode_id,
                    "player": int(actor["player"]),
                    "teacher": str(actor["teacher"]),
                    "teacher_cluster": int(actor["teacher_cluster"]),
                    "action_hash": str(actor["action_hash"]),
                    "daily_score_rank": candidate.rank,
                    "average_score": candidate.average_score,
                    "quality_tier": tier,
                    "raw_file": candidate.raw_path.name,
                    "label_file": str(label_path.relative_to(args.output_dir)),
                    "samples": len(actor["traces"]),
                }
            )
        print(
            f"label={number}/{len(selected)} episode={candidate.episode_id} "
            f"tier={tier} resim={bundle['quality']['resimulation']['matched']}"
        )
    with (args.output_dir / "dataset_index.jsonl").open("w", encoding="utf-8") as stream:
        for row in index_rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "schema": OFFICIAL_V7_MANIFEST_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "competition": "kaggriculture",
        "source_date": args.source_date,
        "source_replay_dir": str(args.replay_dir.resolve()),
        "source_manifest": str(manifest_path.resolve()),
        "selection": {
            "candidate_episodes": len(candidates),
            "selected_episodes": len(selected),
            "max_per_team": args.max_per_team,
            "max_per_trajectory": args.max_per_trajectory,
            "method": "score-ordered trajectory-hash and team capped selection",
        },
        "labels": {
            "schema": "kaggriculture.agent-trace.v2",
            "action_alignment": "observation[t] -> replay.steps[t+1][player].action",
            "intent_horizon": args.intent_horizon,
            "inverse_top_m": args.inverse_top_m,
            "high_level_mode_supervision": False,
            "reason": "public replay high-level modes are latent and ambiguous",
        },
        "value_targets": {
            "gamma": args.value_gamma,
            "reward_scale": args.value_reward_scale,
            "win_bonus": args.value_win_bonus,
        },
        "counts": {
            "episodes": len(episode_rows),
            "actors": len(index_rows),
            "samples": sum(int(row["samples"]) for row in index_rows),
            "unique_teams": len(team_counts),
            "unique_action_trajectories": len(
                {str(row["action_hash"]) for row in index_rows}
            ),
            "quality_tiers": dict(sorted(quality_counts.items())),
            "clusters": {
                str(key): value for key, value in sorted(cluster_counts.items())
            },
        },
        "episodes": episode_rows,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest["counts"], indent=2, ensure_ascii=False))
    print(f"output={args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
