"""Compare market learning on training-only rare patterns; no match or replay evaluation."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path('F:/Kaggriculture/experiments/market_quantity75_20260918')
OUT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--experiment', type=Path, default=ROOT)
args = parser.parse_args()
ROOT = args.experiment.resolve()
OUT = OUT if ROOT.name == OUT.name else OUT.parent / ROOT.name
OUT.mkdir(exist_ok=True)
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
sys.path.insert(0, str(ROOT))
from cpu_budget import CpuBudget
guard = CpuBudget(OUT / 'pattern_learning_audit_cpu')
guard.wait()
import numpy as np
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
from common import batch, load_checkpoint, save

snapshot = (ROOT / 'run/best.pt').read_bytes()
model, _, checkpoint = load_checkpoint(io.BytesIO(snapshot), 'cpu')
initial_model, _, initial = load_checkpoint(ROOT / 'initial.pt', 'cpu')
plan = checkpoint['training_protocol']['plan']
assert plan == json.loads((ROOT / 'PLAN.json').read_text(encoding='utf8'))
reference_path = Path(plan.get('rare_pattern_reference', str(Path(plan['previous_experiment']) / 'initial.pt')))
if 'rare_pattern_reference_sha256' in plan:
    assert hashlib.sha256(reference_path.read_bytes()).hexdigest() == plan['rare_pattern_reference_sha256']
reference = torch.load(reference_path, map_location='cpu', weights_only=False)
reference_codes = reference['model']['pattern_codes'].numpy()
old_codes = initial['model']['pattern_codes'].numpy()
powers = model.pattern_powers.numpy()
assert set(reference_codes) <= set(old_codes)
report = dict(status='PASS', checkpoint_sha256=hashlib.sha256(snapshot).hexdigest(),
              initial_sha256=hashlib.sha256((ROOT / 'initial.pt').read_bytes()).hexdigest(),
              reference_patterns=len(reference_codes), initial_patterns=len(old_codes),
              patterns=len(checkpoint['patterns']), step=checkpoint['expansion_step'],
              scope='Training-only rare patterns. Before/after use identical teacher observations. Conditional quantity accuracy uses target types; this is not held-out performance or win rate.',
              new_match_seeds_used=0, groups=[], observed_at_unix=time.time())
folders = [Path(x) for x in plan['supplemental_folders']] + [ROOT / 'data/fresh_packed']
for folder, source in zip(folders, checkpoint['training_protocol']['sources'][1:], strict=True):
    assert hashlib.sha256((folder / 'READY.json').read_bytes()).hexdigest() == source['ready_sha256']
    arrays = {p.stem: np.load(p, mmap_mode='r') for p in folder.glob('*.npy')}
    codes = (arrays['market_y'][:, :, 0].astype(np.int64)*powers).sum(1)
    for name, selected in [('new_this_round', ~np.isin(codes, old_codes)),
                           ('previously_expanded', np.isin(codes, old_codes) & ~np.isin(codes, reference_codes))]:
        indices = np.flatnonzero(selected)
        row = dict(folder=str(folder), cohort=name, examples=len(indices), before={}, after={})
        for label, net in [('before', initial_model), ('after', model)]:
            counts = dict(exact_pattern=0, exact_market_action=0, conditional_quantities_correct=0, quantity_labels=0)
            for start in range(0, len(indices), 8):
                guard.wait()
                b = batch(arrays, indices[start:start+8], 'cpu')
                with torch.inference_mode():
                    _, predicted = net.predict(b)
                    quantity_logits = net(b)[3]
                target = b['market_y'].long()
                exact_type = predicted[:, :, 0] == target[:, :, 0]
                needed = net.mq[target[:, :, 0]] & (target[:, :, 0] != 0)
                exact_quantity = ~needed | (predicted[:, :, 1] == target[:, :, 1])
                counts['exact_pattern'] += int(exact_type.all(1).sum())
                counts['exact_market_action'] += int((exact_type & exact_quantity).all(1).sum())
                counts['conditional_quantities_correct'] += int(((quantity_logits.argmax(-1) == target[:, :, 1]) & needed).sum())
                counts['quantity_labels'] += int(needed.sum())
            row[label] = counts
        report['groups'].append(row)
report['totals'] = {}
for cohort in ('new_this_round', 'previously_expanded'):
    rows = [row for row in report['groups'] if row['cohort'] == cohort]
    total = dict(examples=sum(row['examples'] for row in rows))
    for label in ('before', 'after'):
        total[label] = {key: sum(row[label][key] for row in rows) for key in rows[0][label]}
    report['totals'][cohort] = total
assert report['totals']['new_this_round']['examples'] == checkpoint['training_protocol']['new_class_training_examples']
save(OUT / 'LEARNED_MARKET_COMPARISON.json', report)
print(json.dumps({k: v for k, v in report.items() if k != 'groups'}), flush=True)
