"""Freeze S4A-3 procurement/production overlap controls before strength tests."""
from pathlib import Path
import argparse, hashlib, json, shutil

EXP = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="s4a3")
    args = parser.parse_args()
    if not args.profile.replace("_", "").isalnum():
        raise ValueError("profile name")
    out = EXP / "profiles" / args.profile
    out.mkdir(exist_ok=False)
    base = json.loads((EXP / "profiles/s3w/configs.json").read_text())["old"]

    def config(shared=False, step=False, prep=False):
        return dict(
            base,
            shared_task_atoms_v2=shared,
            stepwise_recoordination=step,
            preparation_pipeline_v2=prep,
            schedule_value_compare=False,
        )

    configs = {
        # Independent controls.
        "old": config(),
        "prep2_only": config(prep=True),
        # Causally linked combinations; the all-on branch is the only possible
        # promotion candidate in this round.
        "step_prep2": config(step=True, prep=True),
        "shared_prep2": config(shared=True, prep=True),
        "shared_step_prep2": config(shared=True, step=True, prep=True),
    }
    cfg = out / "configs.json"
    cfg.write_text(json.dumps(configs, indent=2), encoding="utf8")
    source = out / "source"
    source.mkdir()
    hashes = {}
    for name in ("policy.hpp", "module.cpp", "test_policy.cpp"):
        path = EXP / "native" / name
        shutil.copy2(path, source / name)
        hashes[path.relative_to(EXP).as_posix()] = sha(path)
    build = json.loads((EXP / "native/build/build_receipt.json").read_text())
    freeze = dict(
        status="FROZEN_BEFORE_STRENGTH",
        config_sha256=sha(cfg),
        source_hashes=hashes,
        binary_sha256=build["binary_sha256"],
        development_N=[20262701, 20262750],
        confirmation_O=[20262801, 20262850],
        unseen_P=[20262901, 20262950],
        final_holdout_used=False,
        oracle_used=False,
        note=(
            "S4A-3 overlaps only resource-free service/movement with live market "
            "procurement; it cannot spend projected fills or change market order."
        ),
    )
    (out / "freeze.json").write_text(json.dumps(freeze, indent=2), encoding="utf8")
    print(json.dumps(dict(status="FROZEN", configs=list(configs), config_sha256=sha(cfg), binary_sha256=build["binary_sha256"])))


if __name__ == "__main__":
    main()
