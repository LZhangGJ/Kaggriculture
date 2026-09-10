"""Run one AFS binary in an isolated process for release parity checking."""
from pathlib import Path
import importlib.util
import json
import sys


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SRC = (
    ROOT
    / "gpt_review/gpt_code/Kaggriculture_P16_JointAFS_R1_20260910"
    / "Kaggriculture_P16_JointAFS_R1_20260910"
)


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    binary = Path(sys.argv[1]).resolve()
    output = Path(sys.argv[2]).resolve()
    seed = int(sys.argv[3])
    arena = load(SRC / "arena.py", "jointafs_isolated_arena")
    opponents = [x["id"] for x in json.loads((SRC / "POOL.json").read_text(encoding="utf-8"))]
    rows = [
        arena.game((opponent, seed, seat, str(binary), "none", {}, 719, ""))
        for opponent in opponents
        for seat in (0, 1)
    ]
    output.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
