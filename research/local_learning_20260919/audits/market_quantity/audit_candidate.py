"""Check candidate provenance and evaluation isolation; never recompute wins from replays."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path('F:/Kaggriculture/experiments/market_quantity75_20260918')
SOURCE = Path('F:/Kaggriculture/experiments/local_teacher_bc_20260917')
OUT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--require-final', action='store_true')
parser.add_argument('--experiment', type=Path, default=ROOT)
args = parser.parse_args()
ROOT = args.experiment.resolve()
OUT = OUT if ROOT.name == OUT.name else OUT.parent / ROOT.name
OUT.mkdir(exist_ok=True)
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
sys.path.insert(0, str(ROOT))
from cpu_budget import CpuBudget
guard = CpuBudget(OUT / 'candidate_audit_cpu')
guard.wait()
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


plan = read(ROOT / 'PLAN.json')
done = read(ROOT / 'run/DONE.json')
assert done['status'] == 'COMPLETE' and done['passes_regression_guard']
for name, expected in read(ROOT / 'CODE_HASHES.json').items():
    assert digest(ROOT / name) == expected, name
for name, expected in plan['frozen_code'].items():
    assert digest(ROOT / 'frozen_code' / name) == expected, name
assert digest(ROOT / 'SEED_AUDIT.json') == plan['seed_audit_sha256']
assert digest(ROOT / 'initial.pt') == plan['initial_sha256']
assert digest(SOURCE / 'frozen/teacher/main.py') == plan['teacher_sha256']
assert digest(SOURCE / 'frozen/pool/POOL.json') == plan['pool_sha256']
pool = read(SOURCE / 'frozen/pool/POOL.json')
assert [x['id'] for x in pool] == plan['opponents'] and len(pool) == 15
for row in pool:
    assert digest(SOURCE / f"frozen/pool/opponents/{row['id']}/main.py") == row['main_sha256']

snapshot = (ROOT / 'run/best.pt').read_bytes()
candidate_hash = hashlib.sha256(snapshot).hexdigest()
checkpoint = torch.load(io.BytesIO(snapshot), map_location='cpu', weights_only=False)
initial = torch.load(ROOT / 'initial.pt', map_location='cpu', weights_only=False)
assert checkpoint['training_protocol']['plan'] == plan
assert checkpoint['expansion_step'] == done['best_step'] > 0
assert checkpoint['patterns'] == read(ROOT / 'run/PATTERNS.json')
assert len(checkpoint['patterns']) == done['patterns'] > len(initial['patterns'])
assert set(map(tuple, initial['patterns'])) <= set(map(tuple, checkpoint['patterns']))
assert checkpoint['quantities'] == initial['quantities'] == plan['quantities']
changed = []
for key, value in checkpoint['model'].items():
    if key not in initial['model'] or not torch.equal(value, initial['model'][key]):
        changed.append(key)
        assert key in {'patterns', 'pattern_codes'} or key.startswith(('pattern_head.', 'pattern_embedding.', 'market_quantity_head.')), key
assert any(key.startswith('pattern_head.') for key in changed)
assert any(key.startswith('pattern_embedding.') for key in changed)
quantity_keys = {key for key in checkpoint['model'] if key.startswith('market_quantity_head.')}
assert quantity_keys == {'market_'+key for key in initial['model'] if key.startswith('quantity_head.')}
quantity_changed = [key for key in sorted(quantity_keys)
                    if not torch.equal(checkpoint['model'][key], initial['model'].get(key, initial['model'][key.removeprefix('market_')]))]
assert quantity_changed, 'The independent quantity branch must actually be trained'
checks = read(ROOT / 'run/CHECKS.json')
assert checks['status'] == 'PASS' and checks['market_quantity_gradient'] > 0
if 'rare_pattern_reference' in plan:
    assert digest(plan['rare_pattern_reference']) == plan['rare_pattern_reference_sha256']
    training = checkpoint['training_protocol']
    assert training['retained_rare_examples'] > 0
    assert training['rare_training_examples'] == training['retained_rare_examples']+training['new_class_training_examples']
    assert plan['training']['rare_pattern_probability'] == .05

training_seeds = {x['seed'] for x in read(SOURCE / 'data/packed/train/episodes.json')}
validation_seeds = {x['seed'] for x in read(SOURCE / 'data/packed/validation/episodes.json')}
for folder in [Path(x) for x in plan['supplemental_folders']] + [ROOT / 'data/fresh_packed']:
    ready = read(folder / 'READY.json')
    assert ready['status'] == 'PASS' and digest(folder / 'episodes.json') == ready['episodes_sha256']
    seeds = {x['seed'] for x in read(folder / 'episodes.json')}
    assert not seeds & (training_seeds | validation_seeds)
    training_seeds.update(seeds)
fresh = read(ROOT / 'data/fresh_packed/episodes.json')
assert {(x['seed'], x['opponent'], x['seat']) for x in fresh} == set(map(tuple, plan['training_jobs']))
all_new = [job[0] for panel in [plan['training_jobs'], *plan['development_panels'], plan['final_jobs']] for job in panel]
assert len(all_new) == len(set(all_new)) == 1260
assert not set(all_new) & set(read(ROOT / 'SEED_AUDIT.json')['excluded_positive_integers'])
evaluation_seeds = {job[0] for panel in [*plan['development_panels'], plan['final_jobs']] for job in panel}
assert not evaluation_seeds & (training_seeds | validation_seeds)

protocol = read(ROOT / 'evaluation/development/PROTOCOL.json')
assert protocol['checkpoint_sha256'] == candidate_hash
assert protocol['jobs'] == plan['development_panels'][0]
assert protocol['script_sha256'] == digest(ROOT / 'evaluate.py')
assert protocol['threshold'] == .75 and protocol['comparison'] == 'strictly greater'
assert not protocol['teacher_in_student'] and not plan['runtime']['teacher_fallback']
assert plan['required_wins'] == 721 and plan['final_games'] == 960
summary = read(ROOT / 'evaluation/development/SUMMARY.json')
assert summary['checkpoint_sha256'] == candidate_hash
report = dict(status='PASS_PRECONDITIONS_RESULTS_PENDING', goal_complete=False,
    observed_at_unix=time.time(), candidate_sha256=candidate_hash, trained_step=checkpoint['expansion_step'],
    old_patterns=len(initial['patterns']), patterns=len(checkpoint['patterns']), changed_tensors=changed,
    trained_market_quantity_tensors=quantity_changed,
    code_and_pool_hashes_match=True, frozen_unit_and_backbone_unchanged=True,
    training_seeds=len(training_seeds), unique_new_seeds=len(all_new), evaluation_seed_overlap=0,
    teacher_runtime_fallback=False, result_source='Runner SUMMARY.json only; no replay win-rate recheck',
    development_summary_at_audit={k: v for k, v in summary.items() if k != 'opponents'},
    final_acceptance='Pending: complete 960 games, at least 721 pure wins, zero runtime errors')
if summary['status'] == 'COMPLETE' and not summary['meets_target']:
    report.update(status='PASS_PROVENANCE_DEVELOPMENT_REJECTED',
                  final_acceptance='Not run: candidate failed the fixed development acceptance gate')
if args.require_final:
    assert summary['status'] == 'COMPLETE' and summary['completed'] == summary['games'] == 240
    assert summary['wins'] >= 181 and summary['errors'] == 0 and summary['meets_target']
    assert digest(ROOT / 'selected.pt') == candidate_hash
    final_protocol = read(ROOT / 'evaluation/fresh_final/PROTOCOL.json')
    assert final_protocol['jobs'] == plan['final_jobs']
    assert final_protocol['checkpoint_sha256'] == candidate_hash
    assert final_protocol['script_sha256'] == digest(ROOT / 'evaluate.py')
    assert final_protocol['threshold'] == .75 and final_protocol['comparison'] == 'strictly greater'
    assert not final_protocol['teacher_in_student']
    final = read(ROOT / 'evaluation/fresh_final/SUMMARY.json')
    assert final['checkpoint_sha256'] == candidate_hash
    assert final['status'] == 'COMPLETE' and final['completed'] == final['games'] == 960
    assert final['wins'] >= 721 and final['errors'] == 0 and final['meets_target']
    assert final['strict_win_rate'] == final['wins']/960
    assert set(final['opponents']) == set(plan['opponents'])
    assert all(row['games'] == 64 for row in final['opponents'].values())
    pipeline = read(ROOT / 'PIPELINE.json')
    assert pipeline['status'] == 'TARGET_MET' and pipeline['selected_sha256'] == candidate_hash
    report.update(status='PASS_FULL_ACCEPTANCE', goal_complete=True,
                  final_acceptance={k: v for k, v in final.items() if k != 'opponents'})
temp = OUT / 'CANDIDATE_AUDIT.tmp'
temp.write_text(json.dumps(report, indent=2), encoding='utf8')
temp.replace(OUT / 'CANDIDATE_AUDIT.json')
print(json.dumps(report), flush=True)
