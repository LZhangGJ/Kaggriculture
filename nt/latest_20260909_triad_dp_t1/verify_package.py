"""Verify the unchanged supplier payload and recount supplied games; no simulation."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PAYLOAD = ROOT / "package"
MANIFEST_SHA256 = "f5613b345748bbb7a87beea61ddec4a677f23720e3d6ba0aa0e52aeb412f6780"
OPPONENTS = ["g001", "g003", "boatlee_v29", "kaito_v58", "lynn_v5",
             "yhay81_six_day", "yhay81_three_day"]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def recount(path):
    rows = json.loads(path.read_text(encoding="utf-8"))
    if len(rows) != 1400:
        raise ValueError(f"Unexpected panel size: {path}")
    keys = {(r["seed"], r["opponent"], r["seat"]) for r in rows}
    expected_keys = {(s, o, p) for s in range(260909000, 260909100)
                     for o in range(7) for p in range(2)}
    if keys != expected_keys:
        raise ValueError(f"Wrong seed/opponent/seat coverage: {path}")
    if any(r["steps"] != 719 or r.get("error") for r in rows):
        raise ValueError(f"Incomplete or errored game: {path}")
    if any(bool(r["win"]) != (r["cash"] > r["opponent_cash"])
           or r["margin"] != r["cash"] - r["opponent_cash"] for r in rows):
        raise ValueError(f"Outcome/cash inconsistency: {path}")
    wins = Counter(r["opponent"] for r in rows if r["cash"] > r["opponent_cash"])
    return {
        "games": len(rows), "wins": sum(wins.values()),
        "win_rate": sum(wins.values()) / len(rows),
        "mean_cash": sum(r["cash"] for r in rows) / len(rows),
        "mean_margin": sum(r["margin"] for r in rows) / len(rows),
        "per_opponent": {name: {"games": 200, "wins": wins[i],
                                  "win_rate": wins[i] / 200}
                         for i, name in enumerate(OPPONENTS)},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, help="Optional NEW JSON receipt path")
    args = parser.parse_args()
    manifest_path = PAYLOAD / "MANIFEST_T1.json"
    if sha256(manifest_path) != MANIFEST_SHA256:
        raise ValueError("Supplier manifest fingerprint changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if len(manifest) != 1885:
        raise ValueError("Supplier manifest count changed")
    total = manifest_path.stat().st_size
    for name, meta in manifest.items():
        path = (PAYLOAD / name).resolve()
        if not path.is_relative_to(PAYLOAD.resolve()):
            raise ValueError(f"Unsafe path: {name}")
        if not path.is_file() or path.stat().st_size != meta["bytes"]:
            raise ValueError(f"Missing/size mismatch: {name}")
        if sha256(path) != meta["sha256"]:
            raise ValueError(f"SHA256 mismatch: {name}")
        total += path.stat().st_size
    receipt = {
        "status": "PASS", "verification": "file hashes and supplied result recount only",
        "new_simulation_performed": False,
        "payload_files": len(manifest) + 1, "payload_bytes": total,
        "supplier_manifest_sha256": MANIFEST_SHA256,
        "supplier_policy_binary_sha256": sha256(PAYLOAD / "policy/agent.so"),
        "t1": recount(PAYLOAD / "runs/release_holdout100/rows.json"),
        "baseline_c3_j7": recount(PAYLOAD / "runs/baseline_j7_holdout100/rows.json"),
    }
    if receipt["t1"]["wins"] != 1132 or receipt["baseline_c3_j7"]["wins"] != 1090:
        raise ValueError("Unexpected frozen result")
    output = json.dumps(receipt, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        with args.out.open("x", encoding="utf-8") as handle:
            handle.write(output)
    print(output, end="")


if __name__ == "__main__":
    main()
