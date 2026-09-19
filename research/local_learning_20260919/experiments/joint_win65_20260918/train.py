"""Joint U/market fine tuning with predicted previous-action exposure, no oracle U input."""
import json
import math
import os
from pathlib import Path
import time
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
ROOT = Path(__file__).resolve().parent
from cpu_budget import CpuBudget
guard = CpuBudget(ROOT / 'training_cpu')
guard.wait()
import numpy as np
import torch
from common import SOURCE, save, arrays, batch, load_checkpoint, loss_and_counts, canonical_unit_history
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.cuda.set_per_process_memory_fraction(.30)
torch.manual_seed(2026091801)
torch.set_float32_matmul_precision('highest')
rng = np.random.default_rng(2026091801)
run = ROOT / 'run'
run.mkdir(exist_ok=True)
plan = json.loads((ROOT / 'PLAN.json').read_text(encoding='utf8'))
train = arrays('train')
validation = arrays('validation')
model, codec, initial = load_checkpoint(ROOT / 'initial.pt', 'cuda')
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=.001, fused=True)
steps = plan['training']['steps']
bs = plan['training']['batch']
val_indices = np.load(SOURCE / 'runs/bc_single_teacher_001/validation_indices.npy')
assert len(val_indices) > 1000
start_step = 0
if (run / 'latest.pt').exists():
    ck = torch.load(run / 'latest.pt', map_location='cuda', weights_only=False)
    model.load_state_dict(ck['model'])
    optimizer.load_state_dict(ck['optimizer'])
    start_step = ck['joint_step']
    rng.bit_generator.state = ck['numpy_rng']
    torch.set_rng_state(ck['torch_rng'].cpu())
    torch.cuda.set_rng_state_all([a.cpu() for a in ck['cuda_rng']])


def evaluate():
    model.eval()
    total = 0.
    counts = np.zeros(6, np.int64)
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        for start in range(0, len(val_indices), bs):
            guard.wait()
            b = batch(validation, val_indices[start:start+bs], 'cuda')
            value, c = loss_and_counts(model, b, model(b))
            total += float(value) * len(b['step'])
            counts += c.cpu().numpy()
    return dict(loss=total/len(val_indices), full_action_exact=float(counts[4]/counts[5]),
                unit_exact=float(counts[0]/counts[1]), market_exact=float(counts[2]/counts[3]))


def checkpoint(step):
    ck = dict(initial)
    ck.update(model=model.state_dict(), optimizer=optimizer.state_dict(), joint_step=step,
        global_step=int(initial['global_step'])+step, numpy_rng=rng.bit_generator.state,
        torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(), joint_plan=plan)
    temp = run / 'latest.tmp'
    torch.save(ck, temp)
    os.replace(temp, run / 'latest.pt')
    if step in plan['training']['candidates']:
        temp = run / f'step_{step}.tmp'
        torch.save(ck, temp)
        os.replace(temp, run / f'step_{step}.pt')


if start_step == 0:
    guard.wait()
    b = batch(train, np.array([0, 1, 22, 23, 24, 80, 200, 718]), 'cuda')
    model.eval()
    with torch.no_grad():
        expected = model.predict(b)
        changed = {k: v.clone() for k, v in b.items()}
        changed['unit_y'][:] = 0
        changed['market_y'][:] = 0
        actual = model.predict(changed)
        assert all(torch.equal(a, c) for a, c in zip(expected, actual)), 'Current target leaked into prediction'
        from common import UNIT_TYPES, GLOBAL_NAMES, encode_action, CROPS
        raw_u, raw_m = [x.clone() for x in expected]
        raw_u[:, 0, 0] = UNIT_TYPES.index(('PLANT', 'WHEAT'))
        raw_u[:, 0, 1] = 0
        changed['global_exact'][:, GLOBAL_NAMES.index('seeds_WHEAT')] = 0
        canonical = canonical_unit_history(raw_u, changed).cpu().numpy()
        for i in range(len(raw_u)):
            n = int(b['unit_count'][i])
            seeds = {c: int(changed['global_exact'][i, GLOBAL_NAMES.index('seeds_'+c)]) for c in CROPS}
            action = codec.decode(raw_u[i].cpu().numpy(), raw_m[i].cpu().numpy(), n)
            tokens, _, _ = encode_action(action, n, seeds)
            expected_u, _ = codec.encode(tokens, n)
            assert np.array_equal(canonical[i], expected_u), 'History codec disagreement'
    optimizer.zero_grad(set_to_none=True)
    value, _ = loss_and_counts(model, b, model(b))
    value.backward()
    gradients = {name: float(dict(model.named_parameters())[name].grad.norm()) for name in
        ('encoder.layers.0.linear1.weight', 'unit_head.weight', 'pattern_head.2.weight', 'quantity_head.2.weight')}
    assert all(x > 0 and math.isfinite(x) for x in gradients.values())
    optimizer.zero_grad(set_to_none=True)
    save(run / 'CHECKS.json', dict(status='PASS', label_leak=False, history_codec_parity=8, jointly_trainable_gradients=gradients))
    metric = evaluate()
    save(run / 'INITIAL_VALIDATION.json', metric)
    print(json.dumps(dict(phase='INITIAL_VALIDATION', **metric)), flush=True)

started = time.monotonic()
running_loss = 0.
for step in range(start_step+1, steps+1):
    guard.wait()
    ids = rng.integers(len(train['step']), size=bs)
    b = batch(train, ids, 'cuda')
    # Denoising exposure only: observations and labels remain the original teacher data.
    # This is not relabeled on-policy DAgger data and must not be described as such.
    chosen = np.flatnonzero((train['step'][ids] > 0) & (rng.random(bs) < .20))
    if len(chosen):
        previous = batch(train, ids[chosen]-1, 'cuda')
        assert (previous['step']+1 == b['step'][chosen]).all(), 'Crossed episode boundary'
        model.eval()
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            u, m = model.predict(previous)
            u = canonical_unit_history(u, previous)
        b['previous_u'][chosen] = u.to(b['previous_u'].dtype)
        b['previous_m'][chosen] = m.to(b['previous_m'].dtype)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    schedule = min(1., step/100) * (.2 + .8*.5*(1+math.cos(math.pi*step/steps)))
    for group in optimizer.param_groups:
        group['lr'] = 1e-5*schedule
    with torch.autocast('cuda', dtype=torch.bfloat16):
        value, _ = loss_and_counts(model, b, model(b))
    assert torch.isfinite(value)
    value.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    assert torch.isfinite(norm)
    optimizer.step()
    running_loss += float(value)
    if step % 25 == 0:
        status = dict(status='TRAINING', pid=os.getpid(), step=step, total=steps,
            loss=running_loss/(step-start_step), seconds=time.monotonic()-started,
            heartbeat_unix=time.time(), cpu_wait_seconds=guard.wait_seconds)
        save(run / 'STATUS.json', status)
        print(json.dumps(status), flush=True)
    if step % 250 == 0:
        checkpoint(step)
    if step in plan['training']['candidates']:
        metric = evaluate()
        with (run / 'metrics.jsonl').open('a', encoding='utf8') as stream:
            stream.write(json.dumps(dict(step=step, **metric))+'\n')
        print(json.dumps(dict(phase='VALIDATION', step=step, **metric)), flush=True)
save(run / 'DONE.json', dict(status='COMPLETE', steps=steps, candidates=plan['training']['candidates']))
save(run / 'STATUS.json', dict(status='COMPLETE', steps=steps, heartbeat_unix=time.time()))
