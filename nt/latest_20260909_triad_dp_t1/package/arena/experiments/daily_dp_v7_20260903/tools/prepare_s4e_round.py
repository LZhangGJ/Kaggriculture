"""Freeze live intraday admission and execution-capacity interactions."""
from pathlib import Path
import hashlib
import json
import shutil

EXP = Path(__file__).resolve().parents[1]


def main():
    out = EXP / 'profiles/s4e1'
    out.mkdir(exist_ok=False)
    prior = json.loads((EXP / 'profiles/s4d2/configs.json').read_text())
    def off(cfg):
        return dict(cfg, intraday_admission=False, intraday_procurement=False)
    def on(cfg):
        return dict(cfg, intraday_admission=True, intraday_procurement=True)
    old, guard, combo = (off(prior[x]) for x in ('old', 'all_guard', 'all_midroute_net'))
    cfg = {'old': old,
           'intraday_stock': dict(old, intraday_admission=True),
           'intraday_funded': on(old),
           'intraday_funded_exchange': dict(on(old), resource_aware_exchange=True),
           'all_guard': guard, 'guard_intraday': on(guard),
           'all_midroute_net': combo, 'all_intraday': on(combo),
           'all_intraday_exchange': dict(on(combo), resource_aware_exchange=True)}
    path = out / 'configs.json';path.write_text(json.dumps(cfg, indent=2))
    (out / 'source').mkdir();hashes = {}
    for name in ('policy.hpp', 'module.cpp', 'test_policy.cpp', 'resource_exchange.hpp', 'intraday_admission.hpp'):
        src = EXP / 'native' / name;shutil.copy2(src, out / 'source' / name)
        hashes['native/' + name] = hashlib.sha256(src.read_bytes()).hexdigest()
    build = json.loads((EXP / 'native/build/build_receipt.json').read_text())
    (out / 'freeze.json').write_text(json.dumps(dict(
        status='FROZEN_BEFORE_STRENGTH', source_hashes=hashes,
        config_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), binary_sha256=build['binary_sha256'],
        development_N=[20262701, 20262750], confirmation_O=[20262801, 20262850], unseen_P=[20262901, 20262950],
        oracle_used=False, final_holdout_used=False), indent=2))
    print(json.dumps(dict(status='FROZEN', configs=list(cfg))))


if __name__ == '__main__':
    main()
