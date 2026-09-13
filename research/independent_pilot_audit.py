"""Additional evidence audit; does not change the frozen pilot's decisions.

Run after analyze.py. This separate audit was added after training began.
Its paired family contrasts and replay descriptions are supplementary analyses.
"""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import random
import statistics


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mean(values):
    return statistics.fmean(values)


def quantile(values, q):
    values = sorted(values)
    x = (len(values) - 1) * q
    lo = int(x)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (x - lo)


def audit(repo, root):
    freeze = read(root / 'inputs/FREEZE.json')
    for rel, digest in freeze['files'].items():
        assert sha(repo / rel) == digest, ('frozen file', rel)
    for rel, digest in freeze['experiment_files'].items():
        assert sha(root / 'inputs' / rel) == digest, ('experiment file', rel)
    source = repo / 'research/independent_pilot'
    inventory = {p.relative_to(source).as_posix() for p in source.rglob('*')
                 if p.suffix in ('.py', '.cpp', '.hpp', '.so')}
    assert inventory == set(freeze['source_paths'])
    seeds = read(root / 'inputs/seeds.json')
    for name, size in [('train', 1024), ('selection', 8), ('pilot_test', 8)]:
        assert len(seeds[name]) == len(set(seeds[name])) == size
    assert len(set(seeds['train'] + seeds['selection'] + seeds['pilot_test'])) == 1040
    selected = read(root / 'SELECTED.json')
    result = read(root / 'RESULTS.json')
    assert result['complete'] and not result['promotion'] and result['failures'] == 0
    assert result['freeze_sha256'] == sha(root / 'inputs/FREEZE.json')
    assert result['selection_sha256'] == sha(root / 'SELECTED.json')
    assert selected['selection_rows_sha256'] == sha(root / 'selection/rows.jsonl')
    assert result['test_rows_sha256'] == sha(root / 'pilot_test/rows.jsonl')
    phases = {}
    all_rows = {}
    replay_summary = defaultdict(lambda: {'replays': 0, 'worker_commands': Counter(),
                                          'market_commands': Counter(), 'daily_cash': defaultdict(list)})
    for phase, count in [('selection', 1024), ('pilot_test', 640)]:
        status = read(root / phase / 'STATUS.json')
        assert status == {'complete': True, 'scheduled': count, 'completed': count, 'invalid': 0}
        manifest = read(root / phase / 'MANIFEST.json')
        assert manifest['seeds'] == seeds[phase]
        candidates = {c['id']: c for c in manifest['candidates']}
        for c in candidates.values():
            if 'path' in c:
                assert sha(repo / c['path']) == c['sha256']
        expected = {(c, seed, opponent, seat) for c in candidates for seed in manifest['seeds']
                    for opponent in manifest['opponents'] for seat in (0, 1)}
        rows = [json.loads(line) for line in (root / phase / 'rows.jsonl').read_text().splitlines()]
        keys = {(r['candidate'], r['seed'], r['opponent'], r['seat']) for r in rows}
        assert len(rows) == len(keys) == count and keys == expected
        trace_count = 0
        for row in rows:
            assert row['valid'] and row['steps'] == 719
            margin = row['own_cash'] - row['opponent_cash']
            assert row['margin'] == margin and row['win'] == (margin > 0) and row['tie'] == (margin == 0)
            assert row['match_score'] == float(margin > 0) + .5 * float(margin == 0)
            if row['seed'] != manifest['seeds'][0]:
                assert row['official_comparisons'] == 0 and 'trace' not in row
                continue
            trace_count += 1
            assert row['official_comparisons'] == 1440
            path = repo / row['trace']
            assert sha(path) == row['trace_sha256']
            trace = json.loads(gzip.decompress(path.read_bytes()))
            assert len(trace['actions']) == 719 and trace['terminal']['step'] == 719
            assert [d['step'] for d in trace['days']] == list(range(0, 719, 24))
            h = hashlib.sha256()
            for pair in trace['actions']:
                h.update(json.dumps(pair, separators=(',', ':')).encode() + b'\n')
            assert h.hexdigest() == row['action_hash']
            for key, value in trace['result'].items():
                assert row[key] == value
            seat = row['seat']
            assert trace['terminal']['farms'][seat]['money'] == row['own_cash']
            assert trace['terminal']['farms'][1-seat]['money'] == row['opponent_cash']
            if phase == 'pilot_test':
                family = candidates[row['candidate']]['family']
                summary = replay_summary[family]
                summary['replays'] += 1
                for pair in trace['actions']:
                    action = pair[seat]
                    for command in [action['farmer'], *action['hands']]:
                        summary['worker_commands'][command[0] if command else 'PASS'] += 1
                    for command in action['market']:
                        summary['market_commands'][command[0]] += 1
                for day in trace['days']:
                    summary['daily_cash'][str(day['step']//24)].append(day['observations'][seat]['farms'][seat]['money'])
        assert trace_count == len(candidates) * len(manifest['opponents']) * 2
        phases[phase] = {'games': len(rows), 'full_reference_replays': trace_count,
                         'observation_comparisons': sum(r['official_comparisons'] for r in rows),
                         'total_game_worker_wall_seconds': sum(r['seconds'] for r in rows),
                         'rows_sha256': sha(root / phase / 'rows.jsonl')}
        all_rows[phase] = (rows, candidates)
    selection_rows, selection_roster = all_rows['selection']
    scores = {}
    for family in ('search', 'ppo'):
        configs = sorted({c['config'] for c in selection_roster.values() if c['family'] == family})
        scores[family] = {}
        for config in configs:
            ids = {c['id'] for c in selection_roster.values() if c['family'] == family and c['config'] == config}
            rr = [r for r in selection_rows if r['candidate'] in ids]
            scores[family][config] = [mean(r['win'] for r in rr), mean(r['match_score'] for r in rr), mean(r['margin'] for r in rr)]
        assert selected['configs'][family] == max(configs, key=lambda c: scores[family][c])
    test_rows, test_roster = all_rows['pilot_test']
    wanted = {c['id'] for c in selection_roster.values() if c['family'] in ('control', 'untrained')
              or c['config'] == selected['configs'].get(c['family'])}
    assert set(test_roster) == wanted == {c['id'] for c in selected['candidates']}
    for cid, candidate in test_roster.items():
        assert candidate == selection_roster[cid]
    family_ids = {family: {c['id'] for c in test_roster.values() if c['family'] == family}
                  for family in ('search', 'ppo', 'control', 'untrained')}
    assert result['games'] == len(test_rows) == 640 and result['unique_worlds'] == 8
    assert result['official_comparisons'] == phases['pilot_test']['observation_comparisons']
    for family, ids in family_ids.items():
        rr = [r for r in test_rows if r['candidate'] in ids]
        recorded = result['families'][family]
        assert recorded['games'] == len(rr)
        assert recorded['wins'] == sum(r['win'] for r in rr)
        assert recorded['draws'] == sum(r['tie'] for r in rr)
        for report_key, row_key in [('win_rate', 'win'), ('match_score', 'match_score'),
                                    ('mean_cash', 'own_cash'), ('mean_margin', 'margin')]:
            assert abs(recorded[report_key] - mean(r[row_key] for r in rr)) < 1e-8
    family_means = {family: [mean(r['win'] for r in test_rows if r['seed'] == s and r['candidate'] in ids)
                            for s in seeds['pilot_test']] for family, ids in family_ids.items()}
    rng = random.Random(2026091201)
    samples = [[rng.randrange(8) for _ in range(8)] for _ in range(4000)]
    contrasts = {}
    for left, right in [('search', 'ppo'), ('ppo', 'untrained'), ('search', 'control')]:
        delta = [a-b for a, b in zip(family_means[left], family_means[right])]
        bootstrap = [mean(delta[i] for i in indices) for indices in samples]
        contrasts[f'{left}_minus_{right}'] = {'delta_win_rate': mean(delta),
             'paired_ci95': [quantile(bootstrap, .025), quantile(bootstrap, .975)]}
    trials = {}
    for path in sorted((root / 'trials').glob('*/STATUS.json')):
        d = read(path)
        assert d['complete']
        trials[path.parent.name] = d
    assert len(trials) == 12
    training = {family: {'trials': sum(('method' in d) == (family == 'search') for d in trials.values()),
                         'games': sum(d['games'] for d in trials.values() if ('method' in d) == (family == 'search')),
                         'timed_process_cpu_seconds': sum(d['cpu_seconds'] for d in trials.values() if ('method' in d) == (family == 'search'))}
                for family in ('search', 'ppo')}
    training['ppo']['eligible_player_turns'] = sum(d.get('player_turns', 0) for d in trials.values())
    learning = Counter()
    for path in sorted((root / 'trials').glob('ppo-*/updates.jsonl')):
        journal = [json.loads(line) for line in path.read_text().splitlines()]
        assert len(journal) == trials[path.parent.name]['updates']
        assert journal[-1]['games'] == trials[path.parent.name]['games']
        for update in journal:
            assert len(update['cash']) == 4
            for env, cash in enumerate(update['cash']):
                if env < 2:
                    seat = 1 - env % 2
                    learning['control_games'] += 1
                    learning['control_wins'] += cash[seat] > cash[1-seat]
                    learning['control_ties'] += cash[seat] == cash[1-seat]
                    learning['learner_seats'] += 1
                    learning['learner_zero_cash'] += cash[seat] == 0
                else:
                    learning['selfplay_games'] += 1
                    learning['selfplay_draws'] += cash[0] == cash[1]
                    learning['selfplay_both_zero'] += cash == [0, 0]
                    learning['learner_seats'] += 2
                    learning['learner_zero_cash'] += sum(c == 0 for c in cash)
    assert learning['control_games'] + learning['selfplay_games'] == training['ppo']['games']
    assert learning['learner_seats'] * 719 == training['ppo']['eligible_player_turns']
    for family, summary in replay_summary.items():
        summary['daily_cash'] = {day: {'mean': mean(values), 'median': statistics.median(values)}
                                 for day, values in summary['daily_cash'].items()}
    return {'passed': True, 'audit_scope': 'additional check written during selection; leaves frozen decisions unchanged',
            'frozen_input_files': len(freeze['files']), 'source_inventory_files': len(inventory),
            'freeze_sha256': sha(root / 'inputs/FREEZE.json'), 'phases': phases,
            'configuration_selection_scores': scores, 'selected_configs': selected['configs'],
            'supplementary_paired_contrasts': contrasts,
            'supplementary_analysis_limits': 'Eight-world clusters; conditional on three training seeds and these four opponents. No multiplicity adjustment. If all observed win rates are zero, empirical bootstrap intervals collapse to [0,0]; this does not establish population equality. Replay descriptions use one test world only and do not establish causes.',
            'training': training, 'ppo_training_outcomes': dict(learning),
            'training_pipeline_wall_seconds': read(root / 'TRAINING_STATUS.json')['wall_seconds'],
            'compute_limit': 'Trial CPU clocks exclude initialization before the timed loop; pipeline wall time includes startup. Evaluation worker wall time is summed over games and is not elapsed time or CPU time.',
            'one_world_replay_descriptions': dict(replay_summary)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    repo = Path(__file__).resolve().parents[1]
    result = audit(repo, args.root.resolve())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes((json.dumps(result, indent=2, sort_keys=True) + '\n').encode())
    print(json.dumps({k: result[k] for k in ('passed', 'phases', 'selected_configs', 'training')}))


if __name__ == '__main__':
    main()
