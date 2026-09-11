#!/usr/bin/env python3
"""Build an auditable FC2B/Rank14 paired-outcome manifest against PRT."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
RECEIPTS = ROOT / "experiments/fusion_champion_v1/receipts"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def key(row: dict) -> tuple[int, int]:
    return int(row["seed"]), int(row["candidate_seat"])


def collect_fc2b(paths: list[Path]) -> dict[tuple[int, int], dict]:
    rows: dict[tuple[int, int], dict] = {}
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        baseline = [row for row in payload["rows"] if int(row["switch_step"]) == 719]
        if len(baseline) != 1:
            raise AssertionError(f"{path}: expected exactly one switch_step=719 row")
        for row in baseline[0]["per_game"]:
            item_key = key(row)
            if item_key in rows:
                raise AssertionError(f"duplicate FC2B game {item_key}")
            rows[item_key] = row
    return rows


def collect_rank14(paths: list[Path]) -> dict[tuple[int, int], dict]:
    rows: dict[tuple[int, int], dict] = {}
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        matches = [row for row in payload["rows"] if row["opponent"] == "local_prt_v6"]
        if len(matches) != 1:
            raise AssertionError(f"{path}: expected exactly one PRT row")
        for row in matches[0]["per_game"]:
            item_key = key(row)
            if item_key in rows:
                raise AssertionError(f"duplicate Rank14 game {item_key}")
            rows[item_key] = row
    return rows


def outcome(margin: int) -> str:
    return "win" if margin > 0 else ("tie" if margin == 0 else "loss")


def main() -> int:
    fc2b_paths = sorted(RECEIPTS.glob("fc10m_kobe_prt_calibration_seed580001_n512_shard*_v1.json"))
    rank14_paths = sorted(RECEIPTS.glob("fc10n_rank14_vs_prt_seed580001_n512_shard*_v1.json"))
    if len(fc2b_paths) != 4 or len(rank14_paths) != 4:
        raise AssertionError("expected four FC2B and four Rank14 source shards")

    fc2b = collect_fc2b(fc2b_paths)
    rank14 = collect_rank14(rank14_paths)
    if set(fc2b) != set(rank14):
        raise AssertionError("FC2B and Rank14 game keys differ")
    if len(fc2b) != 1024:
        raise AssertionError(f"expected 1024 paired games, got {len(fc2b)}")

    categories: dict[str, list[dict]] = {
        "fc2b_loss_rank14_win": [],
        "fc2b_win_rank14_loss": [],
        "both_win": [],
        "both_nonwin": [],
    }
    for seed, seat in sorted(fc2b):
        base_margin = int(fc2b[(seed, seat)]["margin"])
        alt_margin = int(rank14[(seed, seat)]["margin"])
        row = {
            "seed": seed,
            "candidate_seat": seat,
            "fc2b_margin": base_margin,
            "rank14_margin": alt_margin,
            "fc2b_result": outcome(base_margin),
            "rank14_result": outcome(alt_margin),
            "rank14_margin_delta": alt_margin - base_margin,
        }
        if base_margin <= 0 and alt_margin > 0:
            category = "fc2b_loss_rank14_win"
        elif base_margin > 0 and alt_margin <= 0:
            category = "fc2b_win_rank14_loss"
        elif base_margin > 0 and alt_margin > 0:
            category = "both_win"
        else:
            category = "both_nonwin"
        categories[category].append(row)

    rescue_rows = categories["fc2b_loss_rank14_win"]
    payload = {
        "schema": "kaggriculture.fusion_champion.fc10_prt_rescue_manifest.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "official_package_version": "1.32.7",
        "truth_boundary": "offline paired outcome audit only; not an online selector",
        "seat_protocol": "same event seed, both candidate seats",
        "sources": [
            {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
            for path in (*fc2b_paths, *rank14_paths)
        ],
        "games": len(fc2b),
        "category_counts": {name: len(rows) for name, rows in categories.items()},
        "rescue_unique_seed_count": len({row["seed"] for row in rescue_rows}),
        "rescue_unique_seeds": sorted({row["seed"] for row in rescue_rows}),
        "categories": categories,
    }
    output = RECEIPTS / "fc10q_fc2b_rank14_prt_rescue_manifest_seed580001_n512x2_v1.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "PASS",
                "output": str(output.resolve()),
                "games": len(fc2b),
                "category_counts": payload["category_counts"],
                "rescue_unique_seed_count": payload["rescue_unique_seed_count"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
