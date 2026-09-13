"""Require identical deterministic outcomes on both hosts before sharded games."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
freeze_hash = sha(ROOT / 'inputs/FREEZE.json')
fields = ('steps', 'own_cash', 'opponent_cash', 'margin', 'win', 'tie', 'action_hash', 'terminal_shops', 'economy')
rows = {}
sources = {}
timings = {}
for host in ('local', 'wrx90'):
    root = ROOT / 'runs' / ('preflight-' + host)
    status = json.loads((root / 'STATUS.json').read_bytes())
    assert status['complete'] and status['completed'] == status['scheduled'] == 128
    assert status['invalid'] == status['missing'] == 0 and status['freeze_sha256'] == freeze_hash
    sources[host] = {name: sha(root / name) for name in ('STATUS.json', 'MANIFEST.json', 'rows.jsonl', 'VERIFIED.json')}
    mapping = {}
    seconds = defaultdict(list)
    for line in (root / 'rows.jsonl').read_text(encoding='utf8').splitlines():
        row = json.loads(line)
        key = tuple(row[k] for k in ('candidate_id', 'panel', 'seed', 'opponent', 'opponent_seat'))
        assert key not in mapping and row['valid'] and not any(row['comparisons'].values())
        assert row['parity']['observations_checked'] == 1440
        for field in fields:
            assert row['parity'][field] == row['direct'][field]
        mapping[key] = row
        seconds[row['candidate_id']].append(row['direct']['seconds'])
    assert len(mapping) == 128
    rows[host] = mapping
    timings[host] = {cid: sum(values) / len(values) for cid, values in seconds.items()}
assert rows['local'].keys() == rows['wrx90'].keys()
mismatches = []
for key, local in rows['local'].items():
    remote = rows['wrx90'][key]
    for mode in ('parity', 'direct'):
        for field in fields:
            if local[mode][field] != remote[mode][field]:
                mismatches.append(dict(key=key, mode=mode, field=field))
receipt = dict(status='PASS' if not mismatches else 'FAIL', created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    freeze_sha256=freeze_hash, matched_cases=128, full_games_per_host=256, full_games_total=512,
    official_observation_checks_total=128 * 1440 * 2, mismatch_count=len(mismatches), mismatches=mismatches,
    fields_compared=fields, source_files=sources, mean_direct_game_seconds_by_host_and_candidate=timings,
    qualification='Timing comes from the full verification workload under concurrent CPU execution; primary-run throughput may differ.')
(ROOT / 'CROSS_HOST_PREFLIGHT.json').write_bytes((json.dumps(receipt, indent=2, sort_keys=True) + '\n').encode())
print(json.dumps({k: receipt[k] for k in ('status', 'matched_cases', 'full_games_total', 'official_observation_checks_total', 'mismatch_count', 'mean_direct_game_seconds_by_host_and_candidate')}))
assert not mismatches
