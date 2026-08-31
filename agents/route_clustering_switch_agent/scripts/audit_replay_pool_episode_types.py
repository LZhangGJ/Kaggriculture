"""Cross-check a RouteGenome JSONL pool against replay manifest episode types."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence


EPISODE = re.compile(rb'"episode_id":(\d+)')


def manifest_episode_types(path: Path) -> dict[int, set[str]]:
    result: dict[int, set[str]] = defaultdict(set)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("episode_id"):
                result[int(row["episode_id"])].add(str(row.get("episode_type", "")))
    return dict(result)


def pool_episode_ids(path: Path) -> tuple[set[int], int, int]:
    result = set()
    records = 0
    missing = 0
    with path.open("rb") as handle:
        for line in handle:
            records += 1
            match = EPISODE.search(line)
            if match is None:
                missing += 1
            else:
                result.add(int(match.group(1)))
    return result, records, missing


def audit(manifest: Path, pool: Path) -> dict[str, object]:
    kinds = manifest_episode_types(manifest)
    episode_ids, records, missing = pool_episode_ids(pool)
    counts: Counter[str] = Counter()
    for episode in episode_ids:
        counts.update(kinds.get(episode, {"NOT_IN_CURRENT_MANIFEST"}))
    validation = sorted(
        episode for episode in episode_ids
        if "EPISODE_TYPE_VALIDATION" in kinds.get(episode, set())
    )
    public = sorted(
        episode for episode in episode_ids
        if "EPISODE_TYPE_PUBLIC" in kinds.get(episode, set())
    )
    return {
        "pool_records_scanned": records,
        "pool_records_missing_episode_id": missing,
        "pool_unique_episodes": len(episode_ids),
        "current_manifest_unique_episodes": len(kinds),
        "pool_episode_types": dict(sorted(counts.items())),
        "validation_episode_ids_in_pool": len(validation),
        "public_episode_ids_in_pool": len(public),
        "validation_episode_id_sample": validation[:20],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pool", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(audit(args.manifest, args.pool), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
