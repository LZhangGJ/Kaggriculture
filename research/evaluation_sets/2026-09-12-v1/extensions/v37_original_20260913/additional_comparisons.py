"""Report weakest opponent rates and the matched feed-reserve comparison."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
def read(path): return json.loads(path.read_bytes())
def write(path, value): path.write_bytes((json.dumps(value, indent=2, sort_keys=True)+'\n').encode())
data = read(ROOT / 'results/RESULTS.json')
assert read(ROOT / 'JOIN_VALIDATION.json')['status'] == 'PASS'
balance = {'definition':'Minimum strict win rate across 16 opponents, combining both seats, separately for each panel.',
    'scope':'Fixed opponent roster; not a round robin among leaderboard candidates. Observed rates do not guarantee future wins.',
    'candidates':{}}
for cid, label in data['candidate_labels'].items():
    item = {'label':label, 'panels':{}}
    for panel, values in data['panels'].items():
        rows = {op:g['candidates'][cid] for op,g in values['by_opponent'].items()}
        minimum = min(r['win_rate'] for r in rows.values())
        item['panels'][panel] = dict(minimum_win_rate=minimum,
            worst_opponents={op:row for op,row in rows.items() if row['win_rate'] == minimum},
            all_16_over_50=all(r['win_rate'] > .5 for r in rows.values()),
            all_individual_ci_lower_bounds_over_50=all(r['win_rate_ci95'][0] > .5 for r in rows.values()))
    item['all_panels_all_opponents_over_50'] = all(p['all_16_over_50'] for p in item['panels'].values())
    item['minimum_across_panels'] = min(p['minimum_win_rate'] for p in item['panels'].values())
    balance['candidates'][cid] = item
write(ROOT / 'results/MATCHUP_BALANCE.json', balance)

parent, guard = 'public-ahmed-v37-original', 'team-v37-feed-reserve-v1'
plan = read(ROOT / 'inputs/PLAN.json')
values = {panel:{cid:{seed:[] for seed in seeds} for cid in (parent,guard)}
          for panel,seeds in plan['benchmark_seeds'].items()}
matched = {panel:{cid:{} for cid in (parent,guard)} for panel in values}
for phase in ('development','holdout'):
    with (ROOT / 'joined' / phase / 'rows.jsonl').open(encoding='utf8') as f:
        for line in f:
            row=json.loads(line)
            if row['candidate_id'] in (parent,guard):
                values[row['panel']][row['candidate_id']][row['seed']].append(int(row['win']))
                key=(row['seed'],row['opponent'],row['opponent_seat'])
                matched[row['panel']][row['candidate_id']][key]={k:row[k] for k in ('win','tie','own_cash','opponent_cash','action_hash')}
comparison = {'definition':'Feed reserve minus unchanged v37 strict win rate on the same seed, opponent and seat cells.',
    'interval':'95 percent percentile interval from the existing 4000 common whole-seed bootstrap draws.',
    'source_results_sha256':hashlib.sha256((ROOT/'results/RESULTS.json').read_bytes()).hexdigest(), 'panels':{}}
for panel, seeds in plan['benchmark_seeds'].items():
    key = 'kaggriculture-seed-contract-v1-bootstrap:'+panel
    rng=np.random.default_rng(int.from_bytes(hashlib.sha256(key.encode()).digest(),'big'))
    draws=rng.integers(0,len(seeds),size=(4000,len(seeds)))
    # Keep the published manifest order, as the frozen analyzer does.
    assert hashlib.sha256(draws.astype('<i8').tobytes()).hexdigest() == data['panels'][panel]['bootstrap']['index_matrix_sha256']
    assert all(len(values[panel][cid][s])==32 for cid in (parent,guard) for s in seeds)
    delta=np.array([sum(values[panel][guard][s])-sum(values[panel][parent][s]) for s in seeds],dtype=float)/32
    distribution=delta[draws].mean(axis=1)
    reference=data['panels'][panel]['overall']['candidates']
    point=reference[guard]['win_rate']-reference[parent]['win_rate']
    assert abs(delta.mean()-point)<1e-12
    comparison['panels'][panel]=dict(unchanged_win_rate=reference[parent]['win_rate'],
        feed_reserve_win_rate=reference[guard]['win_rate'], difference=point,
        ci95=np.quantile(distribution,[.025,.975]).tolist(), seeds=len(seeds),games_per_candidate=len(seeds)*32)
    assert matched[panel][parent].keys()==matched[panel][guard].keys()
    comparison['panels'][panel]['matched_game_differences']={field:sum(matched[panel][parent][key][field]!=matched[panel][guard][key][field] for key in matched[panel][parent]) for field in ('win','tie','own_cash','opponent_cash','action_hash')}
write(ROOT/'results/FEED_RESERVE_COMPARISON.json',comparison)
print(json.dumps({'balance_candidates':len(balance['candidates']), 'feed_comparison':comparison['panels']}))
