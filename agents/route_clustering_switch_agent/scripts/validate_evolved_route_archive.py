#!/usr/bin/env python3
"""Validate evolved route variants and select a complementary holdout portfolio."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from meta_agent.src.native_teammate_executor import NativeTeammateBundle
from meta_agent.src.teammate_expanded_routes import load_action_tapes
from search_native_tape_mutations import _apply_genes, _ints


def _csv(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _representatives(
    families: tuple[str, ...], required: tuple[str, ...], limit: int
) -> tuple[str, ...]:
    selected = list(dict.fromkeys(value for value in required if value in families))
    if limit <= 0 or limit >= len(families):
        return families
    for index in np.linspace(0, len(families) - 1, limit, dtype=int):
        value = families[int(index)]
        if value not in selected:
            selected.append(value)
        if len(selected) >= limit:
            break
    selected.extend(value for value in families if value not in selected)
    return tuple(selected[:limit])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--openings", type=_csv)
    parser.add_argument("--base-targets", type=_csv)
    parser.add_argument("--checkpoints", type=_ints)
    parser.add_argument("--seeds", type=_ints, required=True)
    parser.add_argument("--opponents", type=_csv)
    parser.add_argument("--opponent-limit", type=int, default=0)
    parser.add_argument("--top-global", type=int, default=32)
    parser.add_argument("--per-parent", type=int, default=4)
    parser.add_argument("--portfolio-size", type=int, default=16)
    parser.add_argument("--minimum-score-gain", type=float, default=0.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--matrix-output", type=Path, required=True)
    args = parser.parse_args()

    started = time.perf_counter()
    archive = json.loads(args.archive.read_text(encoding="utf-8"))
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    entries = list(metadata["opponent_routes"])
    base_families = tuple(str(value["family"]) for value in entries)
    route_id = {str(value["family"]): str(value["route_id"]) for value in entries}
    tapes = load_action_tapes(args.actions)
    openings = args.openings or tuple(archive["openings"])
    base_targets = args.base_targets or tuple(archive["base_targets"])
    checkpoints = args.checkpoints or tuple(int(value) for value in archive["checkpoints"])
    opponents = args.opponents or _representatives(
        base_families,
        tuple(dict.fromkeys((*archive["parents"], *base_targets))),
        args.opponent_limit,
    )

    source_rows = list(archive["ranking"])
    chosen_rows = list(source_rows[:args.top_global])
    by_parent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in source_rows:
        by_parent[str(row["parent"])].append(row)
    for parent in archive["parents"]:
        for row in by_parent[str(parent)][:args.per_parent]:
            if row not in chosen_rows:
                chosen_rows.append(row)

    routes: dict[str, list[dict[str, Any]]] = {}
    rows: list[dict[str, Any]] = []
    tape_hashes: set[str] = set()
    for row in chosen_rows:
        parent = str(row["parent"])
        tape = _apply_genes(tapes[route_id[parent]], list(row.get("genes") or []))
        digest = hashlib.sha256(
            json.dumps(tape, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if digest in tape_hashes or not (row.get("genes") or []):
            continue
        tape_hashes.add(digest)
        name = f"HV{len(rows):03d}"
        routes[name] = tape
        rows.append({**row, "validation_family": name, "tape_sha256": digest})
    if not rows:
        raise ValueError("validation shortlist is empty after tape de-duplication")

    bundle = NativeTeammateBundle(
        args.source, args.actions, args.metadata, additional_routes=routes
    )
    target_names = (*base_targets, *(row["validation_family"] for row in rows))
    result = bundle.executor.switch_search(
        [bundle.index(value) for value in openings],
        [bundle.index(value) for value in target_names],
        checkpoints,
        args.seeds,
        [bundle.index(value) for value in opponents],
    )
    scores = np.asarray(result["outcome"], dtype=np.float32) * .5
    margins = np.asarray(result["margin"], dtype=np.float32)
    base_count = len(base_targets)
    base_scores = np.max(scores[:, :, :base_count], axis=2)
    base_margins = np.max(
        np.where(scores[:, :, :base_count] == base_scores[:, :, None],
                 margins[:, :, :base_count], -np.inf),
        axis=2,
    )
    candidate_scores = scores[:, :, base_count:]
    candidate_margins = margins[:, :, base_count:]
    for index, row in enumerate(rows):
        values = candidate_scores[:, :, index]
        value_margins = candidate_margins[:, :, index]
        next_scores = np.maximum(base_scores, values)
        next_margins = np.where(
            values > base_scores,
            value_margins,
            np.where(values < base_scores, base_margins, np.maximum(base_margins, value_margins)),
        )
        row["holdout_candidate_score"] = float(np.mean(values))
        row["holdout_candidate_margin"] = float(np.mean(value_margins))
        row["holdout_incremental_oracle_score"] = float(np.mean(next_scores - base_scores))
        row["holdout_incremental_oracle_margin"] = float(np.mean(next_margins - base_margins))

    selected: list[int] = []
    available = set(range(len(rows)))
    covered_scores = base_scores.copy()
    covered_margins = base_margins.copy()
    trace = []
    for position in range(min(args.portfolio_size, len(rows))):
        def contribution(index: int) -> tuple[float, float, float, float]:
            values = candidate_scores[:, :, index]
            value_margins = candidate_margins[:, :, index]
            next_scores = np.maximum(covered_scores, values)
            next_margins = np.where(
                values > covered_scores,
                value_margins,
                np.where(values < covered_scores, covered_margins,
                         np.maximum(covered_margins, value_margins)),
            )
            return (
                float(np.mean(next_scores)),
                float(np.mean(next_margins)),
                float(rows[index]["holdout_candidate_score"]),
                float(rows[index]["holdout_candidate_margin"]),
            )

        chosen = max(available, key=contribution)
        previous_score = float(np.mean(covered_scores))
        values = candidate_scores[:, :, chosen]
        value_margins = candidate_margins[:, :, chosen]
        next_scores = np.maximum(covered_scores, values)
        next_margins = np.where(
            values > covered_scores,
            value_margins,
            np.where(values < covered_scores, covered_margins,
                     np.maximum(covered_margins, value_margins)),
        )
        gain = float(np.mean(next_scores)) - previous_score
        if position > 0 and gain <= args.minimum_score_gain:
            break
        selected.append(chosen)
        available.remove(chosen)
        covered_scores, covered_margins = next_scores, next_margins
        trace.append({
            "position": position + 1,
            "validation_family": rows[chosen]["validation_family"],
            "parent": rows[chosen]["parent"],
            "source_rank": rows[chosen]["rank"],
            "incremental_score": gain,
            "portfolio_oracle_score": float(np.mean(covered_scores)),
            "portfolio_oracle_margin": float(np.mean(covered_margins)),
        })

    portfolio = []
    for rank, index in enumerate(selected, 1):
        row = dict(rows[index])
        row["portfolio_rank"] = rank
        portfolio.append(row)
    args.matrix_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.matrix_output,
        scores=candidate_scores,
        margins=candidate_margins,
        base_scores=base_scores,
        base_margins=base_margins,
        families=np.asarray([row["validation_family"] for row in rows]),
        openings=np.asarray(openings),
        checkpoints=np.asarray(checkpoints, dtype=np.int16),
        opponents=np.asarray(opponents),
        seeds=np.asarray(args.seeds, dtype=np.int64),
    )
    payload = {
        "schema": "native-evolved-route-holdout-validation-v1",
        "archive": str(args.archive.resolve()),
        "openings": list(openings),
        "base_targets": list(base_targets),
        "checkpoints": list(checkpoints),
        "opponents": list(opponents),
        "seeds": list(args.seeds),
        "candidate_count": len(rows),
        "base_oracle_score": float(np.mean(base_scores)),
        "base_oracle_margin": float(np.mean(base_margins)),
        "portfolio_oracle_score": float(np.mean(covered_scores)),
        "portfolio_oracle_margin": float(np.mean(covered_margins)),
        "selection_trace": trace,
        "portfolio": portfolio,
        "ranking": sorted(
            rows,
            key=lambda row: (
                -row["holdout_incremental_oracle_score"],
                -row["holdout_incremental_oracle_margin"],
                -row["holdout_candidate_score"],
            ),
        ),
        "matrix_output": str(args.matrix_output.resolve()),
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output.resolve()),
        "matrix_output": str(args.matrix_output.resolve()),
        "candidates": len(rows),
        "base_oracle_score": payload["base_oracle_score"],
        "portfolio_oracle_score": payload["portfolio_oracle_score"],
        "selection_trace": trace,
        "top10": [{
            key: row[key] for key in (
                "validation_family", "parent", "rank",
                "holdout_incremental_oracle_score",
                "holdout_incremental_oracle_margin",
                "holdout_candidate_score", "holdout_candidate_margin",
            )
        } for row in payload["ranking"][:10]],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
