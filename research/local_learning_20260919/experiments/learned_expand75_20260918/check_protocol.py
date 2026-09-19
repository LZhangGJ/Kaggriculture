"""Runnable preflight for strict threshold, balanced pool and globally unique seed splits."""
import ast
from collections import Counter
import json
from pathlib import Path
from evaluate import summary

ROOT = Path(__file__).resolve().parent
plan = json.loads((ROOT / 'PLAN.json').read_text(encoding='utf8'))
audit = json.loads((ROOT / 'SEED_AUDIT.json').read_text(encoding='utf8'))
panels = [plan['training_jobs'], *plan['development_panels'], plan['final_jobs']]
seeds = [job[0] for panel in panels for job in panel]
assert len(seeds) == len(set(seeds)) == 1260
assert not set(seeds) & set(audit['excluded_positive_integers'])
for panel, repeats in zip(panels, (2, 8, 32)):
    counts = Counter((opponent, seat) for _, opponent, seat in panel)
    assert set(counts) == {(o, s) for o in plan['opponents'] for s in (0, 1)}
    assert set(counts.values()) == {repeats}
assert plan['threshold'] == .75 and plan['required_wins'] == 721
rows = [dict(win=i < 720, tie=False, error=None, opponent='synthetic_boundary_check') for i in range(960)]
assert not summary(rows, 960, 'check')['meets_target']
rows[720]['win'] = True
assert summary(rows, 960, 'check')['meets_target']
rows[720]['win'], rows[720]['tie'] = False, True
assert not summary(rows, 960, 'check')['meets_target']
rows[720]['win'], rows[720]['tie'], rows[720]['error'] = True, False, 'error'
assert not summary(rows, 960, 'check')['meets_target']
assert not summary(rows[:800], 960, 'check')['meets_target']
for path in ROOT.glob('*.py'):
    ast.parse(path.read_text(encoding='utf8'), filename=str(path))
report = dict(status='PASS', unique_seeds=len(seeds), seed_overlap=0,
              balanced_all_15_opponents_and_both_seats=True, strictly_more_than_75_boundary=True,
              draws_never_wins=True, runtime_errors_reject=True, incomplete_results_never_accept=True,
              python_syntax=True)
(ROOT / 'PROTOCOL_CHECK.json').write_text(json.dumps(report, indent=2), encoding='utf8')
print(json.dumps(report), flush=True)
