"""Second independent 200-game seed block for JointAFS R1 vs merged T3R1."""
from __future__ import annotations

import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / "p16_jointafs_r1_vs_t3r1_merged_200_20260910"
sys.path.insert(0, str(PREVIOUS))
import run as match  # noqa: E402


match.HERE = HERE
match.SEEDS = list(range(2610120000, 2610120100))


if __name__ == "__main__":
    match.main()
