"""Verify exact warm-start behavior and learnable market quantities without changing units."""
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent
PREVIOUS = Path('F:/Kaggriculture/experiments/learned_expand75_20260918')
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
from cpu_budget import CpuBudget
guard = CpuBudget(ROOT / 'check_cpu')
guard.wait()
import numpy as np
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.manual_seed(2026091809)
from common import arrays, batch, save
from model_joint_numeric import load_checkpoint, Policy

snapshot = (PREVIOUS / 'run/best.pt').read_bytes()
sha = hashlib.sha256(snapshot).hexdigest()
model, _, checkpoint = load_checkpoint(io.BytesIO(snapshot), 'cpu')
assert checkpoint['expansion_step'] == 6000
assert sha == '2a31a9aa7aca82f22363f89c955c9a524875adfde5258ce2195efd4ac0823cb2'
# Use the unchanged source forward implementation as the reference.
spec = importlib.util.spec_from_file_location('original_joint_reference', PREVIOUS / 'frozen_code/model_joint.py')
reference_code = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference_code)


class Reference(Policy):
    forward = reference_code.Policy.forward


reference = Reference(checkpoint['patterns'], **checkpoint['config'])
reference.load_state_dict(model.state_dict())
reference.eval()
original = arrays('train')
probe = batch(original, np.array([0, 1, 22, 23, 24, 80, 200, 718]), 'cpu')
with torch.inference_mode():
    before = model.predict(probe)
    expected = reference.predict(probe)
    assert all(torch.equal(a, b) for a, b in zip(before, expected))
    delta = max((a-b).abs().max().item() for a, b in zip(model(probe), reference(probe)))
    assert delta == 0
del reference
for name, parameter in model.named_parameters():
    parameter.requires_grad_(name.startswith('market_quantity_head.'))
unchanged = {k: v.clone() for k, v in model.state_dict().items() if not k.startswith('market_quantity_head.')}
old = torch.load(PREVIOUS / 'initial.pt', map_location='cpu', weights_only=False)
old_codes = old['model']['pattern_codes'].numpy()
fresh = {p.stem: np.load(p, mmap_mode='r') for p in (PREVIOUS / 'data/fresh_packed').glob('*.npy')}
codes = (fresh['market_y'][:, :, 0].astype(np.int64)*model.pattern_powers.numpy()).sum(1)
indices = np.flatnonzero(~np.isin(codes, old_codes))[:16]
assert len(indices) == 16
b = batch(fresh, indices, 'cpu')
target = b['market_y'].long()
mask = model.mq[target[:, :, 0]] & (target[:, :, 0] != 0)
assert mask.any()


def loss():
    quantity_logits = model(b)[3]
    return torch.nn.functional.cross_entropy(quantity_logits[mask], target[:, :, 1][mask])


initial_loss = loss().item()
trainable = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(trainable, lr=1e-3)
gradients = None
for _ in range(30):
    guard.wait()
    optimizer.zero_grad(set_to_none=True)
    value = loss()
    assert torch.isfinite(value)
    value.backward()
    if gradients is None:
        gradients = {k: float(p.grad.norm()) for k, p in model.named_parameters() if p.requires_grad}
        assert all(np.isfinite(x) and x > 0 for x in gradients.values())
    torch.nn.utils.clip_grad_norm_(trainable, 1.)
    optimizer.step()
final_loss = loss().item()
assert final_loss < initial_loss*.8, (initial_loss, final_loss)
assert all(torch.equal(value, model.state_dict()[key]) for key, value in unchanged.items())
with torch.inference_mode():
    after = model.predict(probe)
    assert torch.equal(before[0], after[0])
    assert torch.equal(before[1][:, :, 0], after[1][:, :, 0])
report = dict(status='PASS', checkpoint_sha256=sha, original_forward_logit_max_error=delta,
    initial_action_parity=8, probe_training_samples=len(indices), market_quantity_labels=int(mask.sum()),
    learning_check_steps=30, quantity_loss_before=initial_loss, quantity_loss_after=final_loss,
    positive_gradients=gradients, unit_actions_unchanged=True, market_types_unchanged=True,
    all_frozen_tensors_unchanged=True, parameters=sum(p.numel() for p in model.parameters()),
    trainable_parameters=sum(p.numel() for p in trainable), model_family='Separate market quantity MLP copied from shared quantity head',
    new_match_seeds_used=0, inference_uses_teacher=False,
    scope='Architecture and small training-sample learnability check only; no candidate acceptance or win-rate claim',
    checked_at_unix=time.time())
save(ROOT / 'QUANTITY_BRANCH_CHECK.json', report)
print(json.dumps(report), flush=True)
