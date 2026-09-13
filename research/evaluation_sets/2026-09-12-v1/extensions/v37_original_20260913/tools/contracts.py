"""Frozen inputs and deterministic job plans for versioned evaluations."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import time
import uuid

HERE = Path(__file__).resolve().parent
DEFAULT_BUNDLE = HERE.parents[1] / 'evaluation_sets/2026-09-12-v1'
SCHEMA = 'kaggriculture-published-benchmark-v37-original-v1'
STRESS_PROFILES = ('crop_producer', 'livestock_producer', 'cash_maximizer',
                   'delayed_seller', 'aggressive_expander')
SHARED_ACTIONS = {'flexon_v5', 'market_smart_v8', 'seven_turn', 'shop0909', 'aurax_shop_v2'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_bytes())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(4 * 1024**2):
            h.update(block)
    return h.hexdigest()


def write(path, value, *, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n'
    if exclusive:
        with path.open('x', encoding='utf8', newline='\n') as stream:
            stream.write(payload)
    else:
        temporary = path.with_suffix(path.suffix + '.tmp')
        temporary.write_text(payload, encoding='utf8', newline='\n')
        temporary.replace(path)


def inside(root, name):
    path = (Path(root) / name).resolve()
    require(path.is_relative_to(Path(root).resolve()), f'Path leaves frozen root: {name}')
    return path


def seeds(path):
    values = read(path)['seeds']
    require(bool(values) and len(values) == len(set(values)), f'Empty or duplicate seeds: {path}')
    require(all(type(s) is int and 0 <= s < 2**31 for s in values), f'Invalid seed: {path}')
    return values


def opponents(bundle):
    result = []
    original = read(bundle / 'opponents.json')['opponents']
    require(len(original) == 16 and len({o['id'] for o in original}) == 16, 'Expected 16 pinned opponents')
    for opponent in original:
        name = opponent['id']
        if name in SHARED_ACTIONS:
            family, basis = 'shared-action-library', 'Identical actions.json hash in the pinned source manifest'
        elif name in {'native:r1', 'native:r2'}:
            family, basis = 'native-afs', 'R1/R2 versions with the shared native policy wrapper'
        else:
            family, basis = 'unclassified:' + name, 'Separate report group; independent ancestry is unverified'
        result.append(dict(opponent, group='primary', family=family, family_basis=basis))
    runtime = bundle / 'evaluation/runtime'
    dependencies = [
        'research/robust90/stress.py',
        'nt/latest_20260911_afs_workflow_repair_r1_r2/r2/policy/agent.so',
        'nt/latest_20260911_afs_workflow_repair_r1_r2/r2/policy/config.json',
        'nt/latest_20260911_p16_jointafs_r1/agent/policy/agent.py',
    ]
    for profile in STRESS_PROFILES:
        files = {name: sha(runtime / name) for name in dependencies}
        result.append(dict(id='stress:' + profile, group='diagnostic', family='native-afs',
                           family_basis='AFS R2 parameter variant; not an independent policy family',
                           runtime_files=files, runtime_tree_sha256=digest(files)))
    return result


def preflight_seeds(bundle):
    representative = seeds(bundle / 'manifests/representative.json')
    stress = seeds(bundle / 'manifests/stress.json')
    require(len(representative) >= 2 and len(stress) >= 2, 'Need two seeds from each development panel')
    rows = [json.loads(line) for line in (bundle / 'features/stress.jsonl').read_text().splitlines()]
    by_seed = {row['seed']: row for row in rows}
    require(set(by_seed) == set(stress), 'Stress features do not match the manifest')
    ordered = sorted(stress, key=lambda s: (by_seed[s]['features']['ref_milk_minus_wool'], s))
    return {'representative': representative[:2], 'stress': [ordered[0], ordered[-1]]}


def jobs(plan, roster, opponent_rows, phase, holdout_seeds=None):
    panels = plan['preflight_seeds'] if phase == 'preflight' else plan['benchmark_seeds'] if phase == 'benchmark' else plan['development_seeds']
    if phase == 'benchmark':
        panels = {name: panels[name] for name in ('representative', 'stress', 'holdout')}
    if phase == 'holdout':
        require(holdout_seeds is not None, 'Holdout needs a fresh release')
        panels = {'holdout': holdout_seeds}
    require(phase in {'preflight', 'development', 'holdout', 'benchmark'}, 'Unknown phase')
    return [(c['id'], panel, seed, o['id'], seat)
            for panel, values in panels.items() for seed in values
            for o in opponent_rows for seat in (0, 1) for c in roster['candidates']]


def create_freeze(bundle, output, roster_path=None, candidate_ids=None):
    bundle, output = Path(bundle).resolve(), Path(output).resolve()
    require(not output.exists(), 'Freeze output already exists; use a new experiment directory')
    roster_path = Path(roster_path) if roster_path else bundle / 'evaluation/roster.json'
    roster = read(roster_path)
    if candidate_ids:
        wanted = set(candidate_ids)
        require(len(wanted) == len(candidate_ids), 'Duplicate requested candidate')
        require(wanted <= {c['id'] for c in roster['candidates']}, 'Unknown candidate')
        roster['candidates'] = [c for c in roster['candidates'] if c['id'] in wanted]
    ids = [c['id'] for c in roster['candidates']]
    require(ids and len(ids) == len(set(ids)), 'Empty or duplicate candidate roster')
    runtime = bundle / 'evaluation/runtime'
    for name, expected in roster['runtime_files'].items():
        require(sha(inside(runtime, name)) == expected, f'Runtime mismatch: {name}')
    for c in roster['candidates']:
        require(c['binary'] in roster['runtime_files'], 'Candidate binary missing from runtime freeze')
        require(sha(inside(runtime, c['binary'])) == c['binary_sha256'], 'Candidate binary hash mismatch')
    op = [row for row in opponents(bundle) if row['group'] == 'primary']
    for o in op:
        for name, expected in o['runtime_files'].items():
            require(sha(inside(runtime, name)) == expected, f'Opponent mismatch: {name}')
    audit = read(bundle / 'campaign_audit/report.json')
    require(audit['status'] == 'PASS', 'Campaign audit did not pass')
    development = {n: seeds(bundle / f'manifests/{n}.json') for n in ('representative', 'stress')}
    require(not set(development['representative']) & set(development['stress']), 'Development panels overlap')
    baselines = [c for c in ('ppo-495ebc48210f14ae', 'ppo-c90da6b7b3ea061d') if c in ids]
    if not baselines:
        baselines = ids[:1]
    plan = dict(schema=SCHEMA, experiment_id=str(uuid.uuid4()), candidate_ids=ids,
                baseline_ids=baselines, development_seeds=development,
                preflight_seeds=preflight_seeds(bundle),
                preflight_rule='First two representative seeds; minimum and maximum stress ref_milk_minus_wool, seed tie-break',
                primary_opponents=[o['id'] for o in op if o['group'] == 'primary'],
                diagnostic_opponents=[o['id'] for o in op if o['group'] == 'diagnostic'],
                timeout_seconds_per_game=180, bootstrap_resamples=4000,
                bootstrap_key='kaggriculture-seed-contract-v2-bootstrap',
                inference='Descriptive paired seed bootstrap; no automatic winner or promotion claim',
                fresh_confirmation_requires_new_holdout=True)
    plan['benchmark_seeds'] = dict(development, holdout=seeds(bundle / 'retired_holdout/holdout.json'))
    require(read(bundle / 'retired_holdout/RETIREMENT.json')['status'] == 'retired_after_one_frozen_comparison', 'Original comparison must be complete')
    plan['holdout_use'] = 'Previously published benchmark; no fresh confirmation claim'
    plan['bootstrap_key'] = 'kaggriculture-seed-contract-v1-bootstrap'
    plan['worker_recycle_games'] = 64
    plan['shards'] = {'local': {'modulo': 3, 'seed_block_indices': [0]},
                      'wrx90': {'modulo': 3, 'seed_block_indices': [1, 2]}}
    files = {'bundle': {}, 'tools': {}, 'experiment': {}}
    required = ['opponents.json', 'evaluation/roster.json', 'evaluation/panel_runner.py', 'evaluation/ANALYSIS_PLAN.md',
                'evaluation/condition_thresholds.json', 'campaign_audit/report.json',
                'campaign_audit/exclusions.json', 'holdout_receipt.json', 'evaluation/HOLDOUT_RELEASE.json']
    required += [f'manifests/{n}.json' for n in ('representative', 'stress', 'stress_pool')]
    required += [f'features/{n}.jsonl' for n in ('representative', 'stress')]
    required += ['tools/economy_features.py', 'retired_holdout/holdout.json', 'retired_holdout/RETIREMENT.json', 'results/RESULTS.json', 'results/raw/development-v1.jsonl.gz', 'results/raw/holdout-v1.jsonl.gz']
    required += [p.relative_to(bundle).as_posix() for p in (bundle / 'previous_nine').iterdir() if p.is_file()]
    required += [p.relative_to(bundle).as_posix() for p in runtime.rglob('*')
                 if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc']
    for name in sorted(set(required)):
        files['bundle'][name] = sha(inside(bundle, name))
    for path in sorted(HERE.glob('*.py')) + [HERE / 'ANALYSIS_PLAN.md']:
        files['tools'][path.name] = sha(path)
    output.mkdir(parents=True)
    write(output / 'roster.json', roster, exclusive=True)
    write(output / 'opponents.json', {'opponents': op}, exclusive=True)
    write(output / 'PLAN.json', plan, exclusive=True)
    for name in ('roster.json', 'opponents.json', 'PLAN.json'):
        files['experiment'][name] = sha(output / name)
    record = dict(schema=SCHEMA, created_unix=time.time(), files=files,
                  source_roster_sha256=sha(roster_path), experiment_id=plan['experiment_id'])
    write(output / 'FREEZE.json', record, exclusive=True)
    return record


def verify_freeze(freeze_path, bundle):
    freeze_path, bundle = Path(freeze_path).resolve(), Path(bundle).resolve()
    record = read(freeze_path)
    require(record.get('schema') == SCHEMA, 'Unsupported freeze schema')
    roots = {'bundle': bundle, 'tools': HERE, 'experiment': freeze_path.parent}
    require(set(record['files']) == set(roots), 'Missing frozen input group')
    for group, mapping in record['files'].items():
        require(bool(mapping), f'Empty frozen input group: {group}')
        for name, expected in mapping.items():
            path = inside(roots[group], name)
            require(path.is_file() and sha(path) == expected, f'Frozen input changed: {group}/{name}')
    current_runtime = {p.relative_to(bundle).as_posix() for p in (bundle / 'evaluation/runtime').rglob('*')
                       if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}
    pinned_runtime = {name for name in record['files']['bundle'] if name.startswith('evaluation/runtime/')}
    require(current_runtime == pinned_runtime, 'Runtime file inventory changed after freeze')
    require({p.name for p in HERE.glob('*.py')} <= set(record['files']['tools']),
            'New tool code was added after freeze')
    plan = read(freeze_path.parent / 'PLAN.json')
    require(record['experiment_id'] == plan['experiment_id'], 'Experiment identity changed')
    roster = read(freeze_path.parent / 'roster.json')
    op = read(freeze_path.parent / 'opponents.json')['opponents']
    return record, plan, roster, op


def resume_manifest(path, current):
    path = Path(path)
    if path.exists():
        require(read(path) == current, 'Resume inputs, environment, job list or verification receipts changed')
    else:
        write(path, current, exclusive=True)


def require_complete(status, expected):
    require(status.get('complete') is True and status.get('invalid') == 0
            and status.get('missing') == 0 and status.get('scheduled') == expected
            and status.get('completed') == expected, 'Verification is incomplete or invalid')


def result_key(row):
    return (row['candidate_id'], row['panel'], row['seed'], row['opponent'], row['opponent_seat'])


def journal_rows(path, expected, *, complete=False):
    expected = set(expected)
    rows = {}
    if Path(path).exists():
        with Path(path).open(encoding='utf8') as stream:
            for line in stream:
                row = json.loads(line)
                key = result_key(row)
                require(key in expected, 'Unexpected result cell')
                require(key not in rows, 'Duplicate result cell')
                rows[key] = row
    if complete:
        require(set(rows) == expected, 'Missing result cells')
        require(all(r.get('valid') is True for r in rows.values()), 'Invalid result cells')
    return rows


def runtime_environment():
    import platform
    require(sys.flags.optimize == 0, 'Run without -O; the frozen parity adapter uses assertions')
    return dict(python=sys.version, platform=platform.platform(), start_method='fork',
                cpu_only=True, threads_per_worker=1, timeout_seconds_per_game=180, worker_recycle_games=64)
