"""Read-only same-state material-compatible exchanges, not a terminal Oracle."""
from pathlib import Path
import argparse
import collections
import gzip
import hashlib
import json
import statistics
import sys
import time
import zlib

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / 'native/build'))
import _dp7_native as native


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--panel', required=True)
    p.add_argument('--labels', default='old,all_guard')
    p.add_argument('--opponents', default='g001,g003,boatlee_v29,kaito_v58,lynn_v5,yhay81_six_day,yhay81_three_day,ecobot_v7')
    p.add_argument('--count', type=int, default=50)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    panel_path = Path(a.panel)
    panel = json.loads(panel_path.read_text())
    assert panel['status'] == 'COMPLETE_PANEL_NOT_GOAL_ACCEPTANCE'
    build = json.loads((EXP / 'native/build/build_receipt.json').read_text())
    for rel, h in build['source_hashes'].items():
        assert sha(EXP / rel) == h, rel
    assert sha(next((EXP / 'native/build').glob('_dp7_native*.so'))) == build['binary_sha256']
    for rel, h in panel['build']['source_hashes'].items():
        if rel != 'native/module.cpp':
            assert build['source_hashes'][rel] == h, ('runtime changed', rel)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    kinds = {'native_pass': 0, 'searched_route_native': 1, 'boatlee_v29_native': 2,
             'kaito_v58_native': 3, 'lynn_v5_native': 4, 'fieldbook_native': 5,
             'three_day_native': 6, 'ecobot_v7_native': 7}
    constructors = {1: native.G001, 2: native.BoatleeV29, 3: native.KaitoV58, 4: native.LynnV5}
    agents = {}
    for key in a.opponents.split(','):
        entry = panel['identities'][key]
        kind = kinds[entry['runtime']]
        if key != 'pass':
            assert sha(EXP / entry['source']) == entry['source_sha256']
        agent = None
        if kind in constructors:
            asset = EXP / entry['asset']
            assert sha(asset) == entry['asset_sha256']
            payload = json.loads(zlib.decompress(asset.read_bytes()))
            if kind == 4:
                for rel, h in payload['source_hashes'].items():
                    assert sha((EXP / entry['source']).parent / rel) == h
            agent = constructors[kind](payload)
        agents[key] = kind, agent
    start = time.perf_counter()
    receipt = dict(status='RUNNING_LOCAL_RESOURCE_AUDIT', args=vars(a), build=build,
                   panel_sha256=sha(panel_path), policy_unchanged=True,
                   suffix_rollouts=0, true_future_used=False, threads=16,
                   final_goal_acceptance=False, summary=[])
    (out / 'plan.json').write_text(json.dumps(receipt, indent=2))
    compared = 0
    for label in a.labels.split(','):
        config = panel['configurations'][label]
        for key, (kind, agent) in agents.items():
            refs = {(r['seed'], r['seat']): r for r in panel['rows']
                    if r['variant'] == label and r['opponent'] == key}
            seeds = sorted({s for s, _ in refs})[:a.count]
            keys = [(s, seat) for s in seeds for seat in (0, 1)]
            t0 = time.perf_counter()
            rows = native.resource_handoff_audit_batch([k[0] for k in keys], [k[1] for k in keys], config, kind, agent, 16)
            assert len(rows) == len(keys)
            for row, k in zip(rows, keys):
                assert (row['seed'], row['seat']) == k
                assert row['money'][k[1]] == refs[k]['cash']
                assert row['money'][1 - k[1]] == refs[k]['opponent_cash']
                compared += 1
            samples = [s for row in rows for s in row['samples']]
            proposed = [s for s in samples if s['total_saved'] > 0]
            valid = [s for s in proposed if s['same_effect'] and s['base_noop'] == 0 and s['trial_noop'] == 0]
            valid_per_game = [sum(s['total_saved'] > 0 and s['same_effect'] and s['base_noop'] == 0 and s['trial_noop'] == 0 for s in row['samples']) for row in rows]
            entry = dict(label=label, opponent=key, games=len(rows), sampled_states=len(samples),
                         ready_resources=sum(s['base_resources'] for s in samples),
                         ready_deadline=sum(s['base_deadline'] for s in samples),
                         compatible_pairs=sum(s['compatible'] for s in samples),
                         shorter_pairs=sum(s['shorter'] for s in samples),
                         feasible_pairs=sum(s['feasible'] for s in samples),
                         proposed_states=len(proposed), same_effect_zero_noop_states=len(valid),
                         failed_projection_states=len(proposed) - len(valid),
                         games_with_opportunity=sum(n > 0 for n in valid_per_game),
                         seeds_with_opportunity=len({r['seed'] for r, n in zip(rows, valid_per_game) if n}),
                         mean_opportunity_states_per_game=statistics.fmean(valid_per_game),
                         mean_saved_per_opportunity=statistics.fmean(s['total_saved'] for s in valid) if valid else 0,
                         mean_peak_saved_per_opportunity=statistics.fmean(s['peak_saved'] for s in valid) if valid else 0,
                         mean_max_local_saved_per_game=statistics.fmean(max([0] + [s['total_saved'] for s in r['samples'] if s['same_effect'] and s['base_noop'] == 0 and s['trial_noop'] == 0]) for r in rows),
                         wins=sum(refs[k]['win'] for k in keys), seconds=time.perf_counter() - t0)
            receipt['summary'].append(entry)
            with gzip.open(out / f'{label}_{key}.json.gz', 'wt', compresslevel=1, encoding='utf-8') as f:
                json.dump(dict(summary=entry, rows=rows), f, separators=(',', ':'))
            (out / 'summary.json').write_text(json.dumps(receipt, indent=2))
            print(json.dumps(entry), flush=True)
    receipt.update(status='PASS_READ_ONLY_LOCAL_AUDIT_NOT_STRENGTH', unchanged_result_games=compared,
                   seconds=time.perf_counter() - start,
                   caveat='Conditional within-day unit-stage projection. No markets, clock, RNG or opponent are advanced. Local overlapping savings are not additive game cash.')
    (out / 'summary.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps({k: receipt[k] for k in ('status', 'unchanged_result_games', 'seconds')}))


if __name__ == '__main__':
    main()
