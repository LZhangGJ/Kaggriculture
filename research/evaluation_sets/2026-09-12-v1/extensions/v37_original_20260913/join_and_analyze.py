"""Validate both shards, then join unchanged baseline rows for matched analysis."""
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'tools'))
from contracts import digest, jobs, read, require_complete, result_key, sha, verify_freeze, write
from panel_runner import build_manifest, preflight_receipt

freeze = ROOT / 'inputs/FREEZE.json'
bundle = ROOT / 'bundle'
_, plan, new_roster, opponents = verify_freeze(freeze, bundle)
cross = read(ROOT / 'CROSS_HOST_PREFLIGHT.json')
assert cross['status'] == 'PASS' and cross['freeze_sha256'] == sha(freeze)
all_new_keys = set(jobs(plan, new_roster, opponents, 'benchmark'))
assert len(all_new_keys) == 20480
sources = {}
shard_keys = {}
for host, count in (('local', 6848), ('wrx90', 13632)):
    run = ROOT / 'runs' / ('benchmark-' + host)
    recorded = read(run / 'MANIFEST.json')
    current, expected, _ = build_manifest(freeze, bundle, 'benchmark', 16,
        ROOT / 'runs' / ('preflight-' + host), environment=recorded['environment'], shard=host)
    assert current == recorded
    require_complete(read(run / 'STATUS.json'), count)
    assert len(expected) == count
    shard_keys[host] = set(expected)
    sources[host] = {name: sha(run / name) for name in ('MANIFEST.json', 'STATUS.json', 'rows.jsonl')}
assert not shard_keys['local'] & shard_keys['wrx90']
assert shard_keys['local'] | shard_keys['wrx90'] == all_new_keys

original_roster = read(bundle / 'evaluation/roster.json')
combined = json.loads(json.dumps(original_roster))
combined['candidates'].extend(new_roster['candidates'])
combined['runtime_files'].update(new_roster['runtime_files'])
combined['total_primary_games'] = 204800
assert combined == read(ROOT / 'combined_roster.json')
roster_hash = sha(ROOT / 'combined_roster.json')
old_ids = {row['id'] for row in original_roster['candidates']}
new_ids = {row['id'] for row in new_roster['candidates']}
assert len(old_ids) == 9 and len(new_ids) == 1 and not old_ids & new_ids
destination = ROOT / 'joined'
assert not destination.exists(), 'Preserve an existing join and use a new version'
destination.mkdir()
panels = ROOT / 'analysis_panels'
panels.mkdir(exist_ok=True)
for name in ('manifests', 'features', 'tools', 'source_snapshot'):
    shutil.copytree(bundle / name, panels / name, dirs_exist_ok=True)
shutil.copyfile(bundle / 'opponents.json', panels / 'opponents.json')
(panels / 'sealed').mkdir(exist_ok=True)
shutil.copyfile(bundle / 'retired_holdout/holdout.json', panels / 'sealed/holdout.json')
assert sha(panels / 'sealed/holdout.json') == sha(bundle / 'retired_holdout/holdout.json')
panel_seeds = {name: read(panels / ('sealed' if name == 'holdout' else 'manifests') / (name + '.json'))['seeds']
               for name in ('representative', 'stress', 'holdout')}
opponent_ids = {row['id'] for row in opponents}
expected_all = {(cid, panel, seed, opponent, seat) for cid in old_ids | new_ids
                for panel, seeds in panel_seeds.items() for seed in seeds for opponent in opponent_ids for seat in (0, 1)}
assert len(expected_all) == 204800
seen = set()
counts = {'development': 0, 'holdout': 0}
handles = {}
for phase in counts:
    directory = destination / phase
    directory.mkdir()
    handles[phase] = (directory / 'rows.jsonl').open('wb')

def append(stream, allowed_ids, expected_subset=None):
    local_seen = set()
    for line in stream:
        row = json.loads(line)
        key = result_key(row)
        assert row['candidate_id'] in allowed_ids and key in expected_all and key not in seen
        assert row['valid'] and row['steps'] == 719 and not row.get('runtime_error')
        if expected_subset is not None: assert key in expected_subset
        local_seen.add(key)
        seen.add(key)
        phase = 'holdout' if row['panel'] == 'holdout' else 'development'
        handles[phase].write(line if line.endswith(b'\n') else line + b'\n')
        counts[phase] += 1
    if expected_subset is not None: assert local_seen == expected_subset

try:
    for phase in ('development', 'holdout'):
        with gzip.open(bundle / 'previous_nine' / (phase + '.jsonl.gz'), 'rb') as stream:
            append(stream, old_ids)
    for host in ('local', 'wrx90'):
        with (ROOT / 'runs' / ('benchmark-' + host) / 'rows.jsonl').open('rb') as stream:
            append(stream, new_ids, shard_keys[host])
finally:
    for stream in handles.values(): stream.close()
assert seen == expected_all
assert counts == {'development': 122880, 'holdout': 81920}
for phase, count in counts.items():
    directory = destination / phase
    write(directory / 'STATUS.json', dict(complete=True, completed=count, scheduled=count, invalid=0,
        missing=0, duplicate_cells=0, roster_sha256=roster_hash, phase=phase, scope='Validated join; original baseline games were not rerun'))
    write(directory / 'MANIFEST.json', dict(roster_sha256=roster_hash, phase=phase,
        baseline_commit='981ac6db456155de1b7fbbbd9f54fbfda8c19cd5', source_shards=sources,
        new_freeze_sha256=sha(freeze), cross_host_preflight_sha256=sha(ROOT / 'CROSS_HOST_PREFLIGHT.json'),
        holdout_use='Original six had one-use confirmation; later cohorts reuse the published benchmark'))
subprocess.run([sys.executable, '-B', str(ROOT / 'tools/analyze_master.py'), '--panels', str(panels),
    '--roster', str(ROOT / 'combined_roster.json'), '--development', str(destination / 'development'),
    '--holdout', str(destination / 'holdout'), '--out', str(ROOT / 'results')], check=True)
result = read(ROOT / 'results/RESULTS.json')
baseline = read(bundle / 'previous_nine/RESULTS.json')
for panel in ('representative', 'stress', 'holdout'):
    current, previous = result['panels'][panel], baseline['panels'][panel]
    groups = [(current['overall'], previous['overall'])]
    for field in ('by_opponent', 'by_seat'):
        assert current[field].keys() == previous[field].keys()
        groups.extend((current[field][key], previous[field][key]) for key in current[field])
    for opponent, seats in previous['by_opponent_and_seat'].items():
        groups.extend((current['by_opponent_and_seat'][opponent][seat], value) for seat, value in seats.items())
    for a, b in groups:
        for cid in old_ids: assert a['candidates'][cid] == b['candidates'][cid]
    for cid in old_ids: assert current['diagnostics'][cid] == previous['diagnostics'][cid]
    assert current['bootstrap'] == previous['bootstrap']
for filename in ('REFERENCE_CONDITIONS.json', 'REALIZED_MARKETS.json'):
    current, previous = read(ROOT / 'results' / filename), read(bundle / 'previous_nine' / filename)
    for panel in panel_seeds:
        if filename == 'REFERENCE_CONDITIONS.json':
            for name, group in previous[panel].items():
                for cid in group['candidates']:
                    assert current[panel][name]['candidates'][cid] == group['candidates'][cid]
        else:
            for cid in old_ids: assert current[panel][cid] == previous[panel][cid]
receipt = dict(status='PASS', checked_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    new_games=20480, preserved_baseline_games=184320, total_unique_games=204800, invalid=0, missing=0, duplicate_cells=0,
    source_shards=sources, freeze_sha256=sha(freeze), results_sha256=sha(ROOT / 'results/RESULTS.json'),
    baseline_reconciliation='Every original count, mean, interval, paired difference, diagnostic, condition and market result matches exactly',
    scope='Ten-candidate published benchmark; Pro8 and unchanged v37 reuse the retired holdout')
write(ROOT / 'JOIN_VALIDATION.json', receipt)
print(json.dumps(receipt))
