#!/usr/bin/env python3
"""Generate generic SWITCH counterfactual datasets across diverse routes.

The orchestration process runs on Windows, while each worker invokes the frozen
WSL C++ extension.  Multiple independent opponent routes are evaluated in
parallel; no opponent-specific score or policy branch is introduced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def to_wsl(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    if not drive:
        return resolved.as_posix()
    tail = resolved.as_posix().split(":", 1)[1].lstrip("/")
    return f"/mnt/{drive}/{tail}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--genomes", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--receipt-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--prefix-seed-base", type=int, default=2_800_001)
    parser.add_argument("--future-seed-base", type=int, default=3_800_001)
    parser.add_argument("--prefix-seed-count", type=int, default=32)
    parser.add_argument("--future-samples", type=int, default=32)
    parser.add_argument("--candidate-ranks", type=int, default=8)
    parser.add_argument("--route-offset", type=int, default=0)
    parser.add_argument("--route-limit", type=int)
    parser.add_argument(
        "--local-edit-hold-days", type=int, default=0,
        help=(
            "Maximum project-commitment window for each counterfactual edit. "
            "The native planner may release it earlier after commissioning "
            "and the official first-production lead time."
        ),
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help=(
            "Forward one generic adaptive-genome override NAME=VALUE to every "
            "worker. May be repeated. This changes the state/candidate "
            "generator only; it must never encode an opponent identity."
        ),
    )
    parser.add_argument("--decision-days", default="7,10,12,15")
    parser.add_argument(
        "--route-day-mode",
        choices=("round_robin", "cross_product"),
        default="round_robin",
        help=(
            "round_robin preserves the original cheap probe; cross_product "
            "evaluates every route at every decision day and avoids "
            "confounding route family with season phase"
        ),
    )
    parser.add_argument(
        "--tag", default="v1",
        help="Artifact suffix used to preserve earlier feature-schema corpora.",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--total-cpu-threads", type=int, default=16)
    parser.add_argument("--wsl-distribution", default="Ubuntu-24.04")
    parser.add_argument("--python", type=Path, default=Path(".venv_wsl_cpp/bin/python"))
    args = parser.parse_args()

    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    all_routes = list(selection["selected_routes"])
    if args.route_offset < 0:
        raise ValueError("route-offset must be non-negative")
    route_end = (
        None if args.route_limit is None
        else args.route_offset + max(0, args.route_limit)
    )
    routes = all_routes[args.route_offset:route_end]
    days = [int(value) for value in args.decision_days.split(",") if value]
    if not routes or not days:
        raise ValueError("selection and decision days must be non-empty")
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    args.receipt_dir.mkdir(parents=True, exist_ok=True)

    project_root = Path.cwd().resolve()
    fast_root = project_root / (
        "research/team_mate/Kaggriculture_main_512631c/agents/"
        "route_clustering_switch_agent/fast_kaggriculture"
    )
    agent_src = project_root / (
        "research/team_mate/Kaggriculture_main_512631c/agents/"
        "route_clustering_switch_agent/src"
    )
    tool = project_root / (
        "experiments/ecobot_adaptive_planner_v1/tools/"
        "calibrate_portfolio_expected_value.py"
    )
    python = project_root / args.python
    omp_threads = max(1, args.total_cpu_threads // max(1, args.workers))

    def execute(
        index: int, route: dict[str, object], day: int
    ) -> dict[str, object]:
        family = str(route["family"])
        prefix_start = args.prefix_seed_base + index * 10_000
        future_start = args.future_seed_base + index * 1_000_000
        stem = (
            f"w6_diverse_value_{family}_d{day}_prefix{args.prefix_seed_count}_"
            f"future{args.future_samples}_rank{args.candidate_ranks}_{args.tag}"
        )
        receipt = args.receipt_dir / f"{stem}.json"
        dataset = args.artifact_dir / f"{stem}.npz"
        command = [
            to_wsl(python), to_wsl(tool),
            "--source", to_wsl(args.source),
            "--actions", to_wsl(args.actions),
            "--metadata", to_wsl(args.metadata),
            "--genomes", to_wsl(args.genomes),
            "--genome-index", "0",
            "--opponent", family,
            "--prefix-seed-start", str(prefix_start),
            "--prefix-seed-count", str(args.prefix_seed_count),
            "--future-seed-start", str(future_start),
            "--future-samples", str(args.future_samples),
            "--candidate-ranks", str(args.candidate_ranks),
            "--minimum-decision-day", str(day),
            "--override", "portfolio_supply_impact_weight=0",
            "--override", (
                "portfolio_local_edit_hold_days="
                f"{args.local_edit_hold_days}"
            ),
        ]
        for override in args.override:
            command.extend(["--override", override])
        command.extend([
            "--output", to_wsl(receipt),
            "--dataset-output", to_wsl(dataset),
        ])
        env_prefix = (
            f"export OMP_NUM_THREADS={omp_threads}; "
            f"export OPENBLAS_NUM_THREADS={omp_threads}; "
            f"export PYTHONPATH={shlex.quote(to_wsl(fast_root / 'python'))}:"
            f"{shlex.quote(to_wsl(agent_src))}; "
        )
        shell_command = env_prefix + " ".join(shlex.quote(part) for part in command)
        started = time.perf_counter()
        completed = subprocess.run(
            [
                "wsl.exe", "-d", args.wsl_distribution,
                "--", "bash", "-lc", shell_command,
            ],
            cwd=project_root,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"{family} failed with {completed.returncode}:\n"
                f"{completed.stdout}\n{completed.stderr}"
            )
        return {
            "family": family,
            "day": day,
            "prefix_seed_start": prefix_start,
            "future_seed_start": future_start,
            "elapsed_seconds": time.perf_counter() - started,
            "dataset": str(dataset),
            "dataset_sha256": sha256(dataset),
            "receipt": str(receipt),
            "receipt_sha256": sha256(receipt),
        }

    if args.route_day_mode == "cross_product":
        jobs = [
            (route, day)
            for route in routes
            for day in days
        ]
    else:
        jobs = [
            (route, days[index % len(days)])
            for index, route in enumerate(routes)
        ]

    started = time.perf_counter()
    results: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(execute, index, route, day): (
                str(route["family"]), day
            )
            for index, (route, day) in enumerate(jobs)
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(
                f"completed {result['family']} day={result['day']} "
                f"elapsed={result['elapsed_seconds']:.1f}s",
                flush=True,
            )
    result_order = {
        (str(route["family"]), day): index
        for index, (route, day) in enumerate(jobs)
    }
    results.sort(key=lambda item: result_order[
        (str(item["family"]), int(item["day"]))
    ])
    payload = {
        "schema": "kaggriculture.diverse-switch-training-corpus.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection": {"path": str(args.selection), "sha256": sha256(args.selection)},
        "generation_contract": {
            "engine": "fast_kaggriculture C++ 1.32.7 via WSL",
            "workers": args.workers,
            "total_cpu_thread_budget": args.total_cpu_threads,
            "omp_threads_per_worker": omp_threads,
            "uses_opponent_identity_for_runtime_policy": False,
            "opponent_routes_used_only_as_state_generators": True,
            "prefix_seed_count_per_route": args.prefix_seed_count,
            "future_samples_per_candidate": args.future_samples,
            "candidate_ranks": args.candidate_ranks,
            "route_offset": args.route_offset,
            "route_limit": args.route_limit,
            "local_edit_hold_days": args.local_edit_hold_days,
            "generic_overrides": list(args.override),
            "decision_days": days,
            "route_day_mode": args.route_day_mode,
            "jobs": len(jobs),
            "artifact_tag": args.tag,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "datasets": results,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "routes": len(routes),
        "jobs": len(results),
        "elapsed_seconds": payload["elapsed_seconds"],
        "manifest": str(args.manifest),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
