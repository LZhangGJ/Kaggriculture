"""Independent stdlib check; intentionally imports no tournament/statistics helpers."""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def fresh_seed_check():
    fresh = {int.from_bytes(hashlib.sha256(f'route-repair-20260911-holdout-{i}'.encode()).digest()[:4], 'big') for i in range(100)}
    assert len(fresh) == 100
    prior = {}
    for path in ROOT.parent.glob('r2*/runs/*/PROTOCOL.json'):
        if ROOT in path.parents:
            continue
        data = read(path)
        seeds = {j[2] for j in data.get('jobs', []) if isinstance(j, list) and len(j) == 4 and type(j[2]) is int}
        prior[str(path.relative_to(ROOT.parent))] = sorted(seeds)
    regression = read(ROOT / 'runs/regression_final/PROTOCOL.json')
    old = set().union(*(set(s) for s in prior.values()))
    development = {j[2] for j in regression['jobs']}
    assert not (fresh & old), sorted(fresh & old)
    assert not (fresh & development)
    return dict(fresh_count=len(fresh), prior_seed_count=len(old), prior_protocols=prior,
                development_seed_count=len(development), overlap=[])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('tags', nargs='*')
    args = parser.parse_args()
    result = dict(status='PASS', seed_check=fresh_seed_check(), panels={})
    for tag in args.tags:
        folder = ROOT / 'runs' / tag
        protocol, summary = read(folder / 'PROTOCOL.json'), read(folder / 'SUMMARY.json')
        rows = read(folder / 'rows.json')
        actual = {(r['arm'], r['opponent'], r['seed'], r['seat']) for r in rows}
        expected = {tuple(j) for j in protocol['jobs']}
        assert actual == expected and len(rows) == len(actual) == len(protocol['jobs'])
        groups = defaultdict(list)
        pairs = defaultdict(dict)
        for row in rows:
            raw = gzip.decompress((folder / 'matches' / (row['name'] + '.json.gz')).read_bytes())
            # The recorded host schema ends with these three top-level fields.
            # Full frame parsing/transition checks are already performed by QA.
            terminal = raw.rfind(b',"rewards":')
            assert terminal >= 0
            replay = json.loads(b'{' + raw[terminal+1:])
            assert set(replay) == {'rewards', 'statuses', 'schema_version'} and replay['schema_version'] == 1
            assert row['frames'] == 720 and replay['statuses'] == ['DONE', 'DONE']
            cash, other = replay['rewards'][row['seat']], replay['rewards'][1-row['seat']]
            assert cash == row['cash'] and other == row['opponent_cash']
            assert row['margin'] == cash - other and row['win'] == (cash > other)
            entry = dict(cash=cash, other=other, win=int(cash > other), draw=int(cash == other))
            groups[row['arm']].append(entry)
            groups[row['arm'], row['opponent']].append(entry)
            pairs[row['opponent'], row['seed'], row['seat']][row['arm']] = entry
        checked = {}
        for key, entries in groups.items():
            source = summary['arms'][key] if isinstance(key, str) else summary['opponents'][key[1]][key[0]]
            n, wins, draws = len(entries), sum(e['win'] for e in entries), sum(e['draw'] for e in entries)
            assert (source['games'], source['wins'], source['draws']) == (n, wins, draws)
            assert abs(source['win_rate'] - 100 * wins / n) < 1e-10
            assert abs(source['mean_margin'] - sum(e['cash']-e['other'] for e in entries)/n) < 1e-8
            checked[str(key)] = dict(games=n, wins=wins, draws=draws, losses=n-wins-draws, win_rate=100*wins/n)
        for comparison in summary['comparisons']:
            old = comparison['baseline']
            delta = [p['fixed']['win']-p[old]['win'] for p in pairs.values()]
            cash = [p['fixed']['cash']-p[old]['cash'] for p in pairs.values()]
            assert comparison['pairs'] == len(delta)
            assert abs(comparison['deltas']['win']['mean']-100*sum(delta)/len(delta)) < 1e-10
            assert abs(comparison['deltas']['cash']['mean']-sum(cash)/len(cash)) < 1e-8
            assert comparison['gained_wins'] == sum(d > 0 for d in delta)
            assert comparison['lost_wins'] == sum(d < 0 for d in delta)
        result['panels'][tag] = dict(replay_terminal_rewards_checked=len(rows), aggregates=checked)
        print(json.dumps(dict(tag=tag, checked=len(rows))), flush=True)
    path = ROOT / ('INDEPENDENT_VERIFY.json' if args.tags else 'SEED_CHECK.json')
    path.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(status='PASS', panels=list(result['panels']), prior_seeds=result['seed_check']['prior_seed_count'])))


if __name__ == '__main__':
    main()
