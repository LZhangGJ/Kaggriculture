"""Freeze the pre-registered S3T 2x2 configurations before outcome runs."""
from pathlib import Path
import hashlib, json, shutil

EXP = Path(__file__).resolve().parents[1]

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main() -> None:
    out = EXP / "profiles" / "s3t"
    out.mkdir(parents=True, exist_ok=False)
    base = json.loads((EXP / "profiles" / "s3s" / "configs.json").read_text())["old"]
    configs = {}
    for label, compile_enabled, hire_enabled in (
        ("old", False, False),
        ("compile_only", True, False),
        ("hire_only", False, True),
        ("compile_hire", True, True),
    ):
        cfg = dict(base)
        cfg.update(
            regret_schedule=False,
            regret_compile=compile_enabled,
            regret_hire_estimate=hire_enabled,
            split_service_jobs=False,
        )
        configs[label] = cfg
    cfg_path = out / "configs.json"
    cfg_path.write_text(json.dumps(configs, indent=2), encoding="utf8")
    src = out / "source"
    src.mkdir()
    source_hashes = {}
    for name in ("policy.hpp", "module.cpp", "test_policy.cpp"):
        original = EXP / "native" / name
        frozen = src / name
        shutil.copy2(original, frozen)
        source_hashes[str(original.relative_to(EXP))] = sha(original)
    freeze = {
        "configurations": configs,
        "config_sha256": sha(cfg_path),
        "source_hashes": source_hashes,
        "A": [20261401, 20261450],
        "B": [20261501, 20261550],
        "conditional_new_J": [20262301, 20262350],
        "final_holdout_used": False,
        "fitted_parameters": 0,
    }
    (out / "freeze.json").write_text(json.dumps(freeze, indent=2), encoding="utf8")
    print(json.dumps({"status": "FROZEN", "config_sha256": freeze["config_sha256"], "source_hashes": source_hashes}))

if __name__ == "__main__":
    main()
