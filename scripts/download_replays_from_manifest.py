"""Download Kaggle episode replays listed in a CSV manifest with one API session."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--delay", type=float, default=0.75)
    parser.add_argument("--retries", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    from kaggle.api.kaggle_api_extended import KaggleApi

    with args.manifest.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))

    api = KaggleApi()
    api.authenticate()

    new = skipped = failed = 0
    failures: list[tuple[str, str]] = []
    total = len(rows)
    for index, row in enumerate(rows, 1):
        episode_id = int(row["episode_id"])
        replay_path = Path(row["replay_path"])
        if replay_path.is_file() and replay_path.stat().st_size:
            skipped += 1
        else:
            replay_path.parent.mkdir(parents=True, exist_ok=True)
            error = ""
            for attempt in range(args.retries + 1):
                try:
                    api.competition_episode_replay(
                        episode_id, path=str(replay_path.parent), quiet=True
                    )
                    if replay_path.is_file() and replay_path.stat().st_size:
                        new += 1
                        error = ""
                        break
                    error = "download returned without a non-empty replay file"
                except Exception as exc:  # Kaggle surfaces HTTP errors through several classes.
                    error = f"{type(exc).__name__}: {exc}"
                if attempt < args.retries:
                    time.sleep(min(30.0, 2.0 ** attempt))
            if error:
                failed += 1
                failures.append((str(episode_id), error))
            time.sleep(max(0.0, args.delay))

        if index % 50 == 0 or index == total:
            print(
                f"PROGRESS {index}/{total} new={new} skipped={skipped} failed={failed}",
                flush=True,
            )

    for episode_id, error in failures:
        print(f"FAILED\t{episode_id}\t{error}", flush=True)
    print(
        f"SUMMARY total={total} new={new} skipped={skipped} failed={failed} "
        f"manifest={args.manifest}",
        flush=True,
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
