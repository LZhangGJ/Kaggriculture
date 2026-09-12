"""Replace the mis-timed day1 experiment while leaving reference tests running."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root = Path(__file__).resolve().parent
os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
prefix = ['wsl', '-d', 'Ubuntu', '--cd', '/mnt/c/Users/Owner/Documents/Codex/2026-09-05/cont/lzhang-kaggriculture', '--',
          'env', 'CUDA_VISIBLE_DEVICES=', 'OMP_NUM_THREADS=1', 'OPENBLAS_NUM_THREADS=1', 'MKL_NUM_THREADS=1',
          'nice', '-n', '10', 'python3']
references = [j for j in json.loads((root / 'REFERENCE_DAY1_STATUS.json').read_text()).get('jobs', [])
              if j['run'].startswith('reference-')]


def status(stage, jobs=()):
    state = dict(time=time.time(), stage=stage, jobs=list(jobs))
    (root / 'DAY3_CONNECTED_STATUS.json').write_text(json.dumps(state, indent=2))
    (root / 'ACTIVE_JOBS.json').write_text(json.dumps(dict(time=time.time(), jobs=list(jobs), resource_cap=16,
                                                          gpu_allowed=False, no_push=True, confirmation_used=False), indent=2))


def execute(name, args, workers):
    with (root / (name + '.log')).open('x') as log:
        process = subprocess.Popen(prefix + args, stdout=log, stderr=subprocess.STDOUT,
                                   creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        status(name, references + [dict(run=name, pid=process.pid, workers=workers)])
        if process.wait():
            raise RuntimeError(name)


def local(args):
    subprocess.run([sys.executable, *map(str, args)], check=True)


try:
    if not (root / 'runs/day1-joint-headroom/STOPPED_FOR_REJECTION.json').exists():
        raise ValueError('Old day1 collector must be stopped first')
    execute('day3-connected-keep-check', ['research/robust90/check_native_package.py',
            '--run', 'research/robust90/runs/early-matched-day9-leader',
            '--package', 'research/robust90/packages/day3-connected-keep-fixture',
            '--out', 'research/robust90/runs/day3-connected-keep-check', '--workers', '8'], 8)
    common = ['research/robust90/future_sweep.py', '--day', '3', '--restore-base-day', '8',
              '--suffix-selectors', 'research/robust90/models/expected-selector-v1/selector.json',
              '--binary', 'research/robust90/candidates/clean-selector-core/build/forecast_features.so',
              '--profile-file', 'research/robust90/DAY3_CONNECTED_PROFILES.json', '--seed-start', '2612022000',
              '--common-randomness', '--no-plan-features']
    execute('day3-connected-parity', common + ['--out', 'research/robust90/runs/day3-connected-parity',
            '--seeds', '1', '--samples', '1', '--workers', '8', '--opponents', 'soil_v219g', 'native:r2', '--parity'], 8)
    rows = [json.loads(line) for line in (root / 'runs/day3-connected-parity/rows.jsonl').open()]
    if len(rows) != 64 or any(sum(v for k, v in row['decision_features'].items() if k.startswith('shop_')) != 1 for row in rows):
        raise ValueError('First-shop clock or parity coverage mismatch')
    plan = json.loads((root / 'DAY3_CONNECTED_PLAN.json').read_text())
    execute('day3-connected-headroom', common + ['--out', 'research/robust90/runs/day3-connected-headroom',
            '--seeds', '8', '--samples', '4', '--workers', '10', '--opponents', *plan['opponents']], 10)
    local([root / 'analyze_future_headroom.py', root / 'runs/day3-connected-headroom',
           '--plan', root / 'DAY3_CONNECTED_PLAN.json', '--out', root / 'DAY3_CONNECTED_HEADROOM.json'])
    names = ['reference-day9-native', 'reference-matched-day9', 'reference-matched-r2']
    while not all((root / 'runs' / name / 'RESULTS.json').exists() for name in names):
        status('waiting_for_reference_results', references)
        time.sleep(15)
    references = []
    for parent, label in [('reference-matched-day9', 'VERSUS_LEADER.json'), ('reference-matched-r2', 'VERSUS_R2.json')]:
        destination = root / 'runs/reference-day9-native' / label
        if not destination.exists():
            local([root / 'assess.py', root / 'runs/reference-day9-native/rows.jsonl',
                   '--parent', root / 'runs' / parent / 'rows.jsonl', '--pool', root / 'POOL.json', '--out', destination])
    execute('reference-final-package-check', ['research/robust90/check_native_package.py',
            '--run', 'research/robust90/runs/reference-day9-native', '--package', 'research/robust90/packages/reference-day9-v1',
            '--out', 'research/robust90/runs/reference-final-package-check', '--workers', '16'], 16)
    status('complete_review_results')
except BaseException as exc:
    (root / 'DAY3_CONNECTED_ERROR.json').write_text(json.dumps(dict(time=time.time(), error=repr(exc)), indent=2))
    raise
