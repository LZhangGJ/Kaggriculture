"""Freeze midroute-delivery independent and causal-combination tests."""
from pathlib import Path
import json
import shutil
import hashlib

EXP = Path(__file__).resolve().parents[1]


def main():
    out = EXP / 'profiles/s4c1'
    out.mkdir(exist_ok=False)
    prior = json.loads((EXP / 'profiles/s4b1/configs.json').read_text())
    base = dict(prior['old'], midroute_delivery=False)
    cfg = {
        'old': base,
        'midroute': dict(base, midroute_delivery=True),
        'shared_step_midroute': dict(base, shared_task_atoms_v2=True,
                                    stepwise_recoordination=True, midroute_delivery=True),
        'all_guard': dict(prior['all_guard'], midroute_delivery=False),
        'all_midroute': dict(prior['all_guard'], midroute_delivery=True),
        'all_midroute_net': dict(prior['all_guard_net'], midroute_delivery=True),
    }
    path = out / 'configs.json'
    path.write_text(json.dumps(cfg, indent=2))
    (out / 'source').mkdir()
    hashes = {}
    for name in ['policy.hpp', 'module.cpp', 'test_policy.cpp']:
        src = EXP / 'native' / name
        shutil.copy2(src, out / 'source' / name)
        hashes['native/' + name] = hashlib.sha256(src.read_bytes()).hexdigest()
    build = json.loads((EXP / 'native/build/build_receipt.json').read_text())
    (out / 'freeze.json').write_text(json.dumps(dict(
        status='FROZEN_BEFORE_STRENGTH', source_hashes=hashes,
        config_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        binary_sha256=build['binary_sha256'], development_N=[20262701, 20262750],
        confirmation_O=[20262801, 20262850], unseen_P=[20262901, 20262950],
        oracle_used=False, final_holdout_used=False), indent=2))
    print(json.dumps(dict(status='FROZEN', configs=list(cfg))))


if __name__ == '__main__':
    main()
