"""Runnable preflight for strict threshold, balanced pool and globally unique seed splits."""
import ast
from collections import Counter
import json
from pathlib import Path
from evaluate import summary
from pipeline import should_train
from train import pattern_masks
import numpy as np

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
new, rare = pattern_masks(np.array([10, 20, 30]), np.array([10, 20]), np.array([10]))
assert new.tolist() == [False, False, True]
assert rare.tolist() == [False, True, True], 'Previously introduced class 20 must stay in rare training'
assert len(plan['training']['group_probabilities']) == len(plan['supplemental_folders'])+2
assert abs(sum(plan['training']['group_probabilities'])-1) < 1e-12
assert plan['training']['rare_pattern_probability'] == .05
assert not should_train('TARGET_MET')
for status in ('DEVELOPMENT_BELOW_TARGET', 'FINAL_BELOW_TARGET', 'VALIDATION_REGRESSION'):
    assert should_train(status)
for status in ('DEVELOPMENT', 'FINAL_EVALUATION', 'FAILED'):
    try:
        should_train(status)
    except RuntimeError:
        pass
    else:
        raise AssertionError('A running or failed process must not trigger new training')
for path in ROOT.glob('*.py'):
    ast.parse(path.read_text(encoding='utf8'), filename=str(path))
report = dict(status='PASS', unique_seeds=len(seeds), seed_overlap=0,
              balanced_all_15_opponents_and_both_seats=True, strictly_more_than_75_boundary=True,
              draws_never_wins=True, runtime_errors_reject=True, incomplete_results_never_accept=True,
              previous_pipeline_gate=True,
              retained_and_new_rare_classes_sampled=True,
              python_syntax=True)
(ROOT / 'PROTOCOL_CHECK.json').write_text(json.dumps(report, indent=2), encoding='utf8')
print(json.dumps(report), flush=True)
