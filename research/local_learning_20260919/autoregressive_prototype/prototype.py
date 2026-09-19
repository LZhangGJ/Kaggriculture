"""Training-only learnability check for an autoregressive market decoder.

Run with the existing Torch Python, -B. Does not change a candidate or run games.
"""
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time

for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'

import torch
from torch import nn
from torch.nn import functional as F


class MarketDecoder(nn.Module):
    """Current encoded observation plus a within-turn action/quantity prefix."""
    def __init__(self, base):
        super().__init__()
        width = base.market_type.embedding_dim
        self.context = nn.Linear(11 * width, width)
        self.prefix = nn.GRUCell(width, width)
        self.residual = nn.Linear(width, width)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)
        self.command_embedding = copy.deepcopy(base.market_type)
        self.quantity_embedding = copy.deepcopy(base.previous_q)
        self.command_head = copy.deepcopy(base.market_head)
        self.quantity_head = copy.deepcopy(base.market_quantity_head)
        self.register_buffer('needs_quantity', base.mq.clone())

    def forward(self, encoded, targets=None):
        state = self.context(encoded.flatten(1)).tanh()
        previous = torch.zeros(len(encoded), 2, dtype=torch.long, device=encoded.device)
        active = torch.ones(len(encoded), dtype=torch.bool, device=encoded.device)
        commands, quantities, output = [], [], []
        for depth in range(11):
            prefix = self.command_embedding(previous[:, 0]) + self.quantity_embedding(previous[:, 1])
            state = self.prefix(encoded[:, depth] + prefix, state)
            feature = encoded[:, depth] + self.residual(state)
            logits = self.command_head(feature).clone()
            logits[:, 0] = -1e4
            if depth == 10:  # Protocol: at most ten requests followed by EOS.
                logits = logits * 0 - 1e4
                logits[:, 1] = 0
            command = logits.argmax(-1) if targets is None else targets[:, depth, 0]
            qlogits = self.quantity_head(feature + self.command_embedding(command)).clone()
            qlogits[:, 0] = -1e4  # Integer required; keep the existing 0..1000 domain.
            quantity = qlogits.argmax(-1) if targets is None else targets[:, depth, 1]
            quantity = torch.where(self.needs_quantity[command], quantity, 0)
            chosen = torch.stack((command, quantity), -1)
            chosen = torch.where(active[:, None], chosen, 0)
            output.append(chosen)
            commands.append(logits)
            quantities.append(qlogits)
            active = active & (command != 1)
            previous = chosen
        return torch.stack(commands, 1), torch.stack(quantities, 1), torch.stack(output, 1)

    def loss(self, encoded, targets):
        commands, quantities, _ = self(encoded, targets)
        types = targets[:, :, 0]
        ended_before = ((types == 1).cumsum(1) - (types == 1).long()) > 0
        valid = (types != 0) & ~ended_before
        needed = valid & self.needs_quantity[types]
        type_loss = F.cross_entropy(commands.transpose(1, 2), types, reduction='none')
        quantity_loss = F.cross_entropy(quantities.transpose(1, 2), targets[:, :, 1], reduction='none')
        return ((type_loss * valid).sum(1) / valid.sum(1).clamp_min(1)
                + (quantity_loss * needed).sum(1) / needed.sum(1).clamp_min(1)).mean()

    @torch.no_grad()
    def predict(self, encoded):
        return self(encoded)[2]


def main():
    out = Path(__file__).resolve().parent
    root = Path('F:/Kaggriculture/experiments/market_quantity75_20260918')
    sys.path.insert(0, str(root))
    from cpu_budget import CpuBudget
    guard = CpuBudget(out / 'cpu')
    guard.wait()
    import numpy as np
    from common import batch, load_checkpoint, save
    from features import encode_action

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20260919)  # Parameter initialization only; not a game seed.
    snapshot = (root / 'run/best.pt').read_bytes()
    base, codec, checkpoint = load_checkpoint(io.BytesIO(snapshot), 'cpu')
    base.requires_grad_(False)
    initial = torch.load(root / 'initial.pt', map_location='cpu', weights_only=False)
    data = {p.stem: np.load(p, mmap_mode='r') for p in (root / 'data/fresh_packed').glob('*.npy')}
    codes = (data['market_y'][:, :, 0].astype(np.int64) * base.pattern_powers.numpy()).sum(1)
    rare = np.flatnonzero(~np.isin(codes, initial['model']['pattern_codes'].numpy()))
    assert len(rare) >= 16
    indices = list(rare[np.linspace(0, len(rare)-1, 16, dtype=int)])
    lengths = (data['market_y'][:, :, 0] == 1).argmax(1)
    for length in sorted(set(lengths.tolist())):
        options = np.flatnonzero(lengths == length)
        index = int(options[len(options)//2])
        if index not in indices:
            indices.append(index)
    indices = np.asarray(indices, dtype=np.int64)
    b = batch(data, indices, 'cpu')
    target = b['market_y'].long()
    features = []
    with torch.no_grad():
        for start in range(0, len(indices), 4):
            guard.wait()
            features.append(base.encode({k: v[start:start+4] for k, v in b.items()})[1])
        encoded = torch.cat(features).detach()
        unit_before = base.predict({k: v[:4] for k, v in b.items()})[0].clone()
    frozen = {k: v.clone() for k, v in base.state_dict().items()}
    head = MarketDecoder(base)

    def counts():
        head.eval()
        with torch.no_grad():
            predicted = head.predict(encoded)
            return dict(loss=float(head.loss(encoded, target)),
                        exact_types=int((predicted[:, :, 0] == target[:, :, 0]).all(1).sum()),
                        exact_market_actions=int((predicted == target).all(2).all(1).sum()))

    before = counts()
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=0)
    history = []
    positive_gradients = {}
    started = time.monotonic()
    for step in range(1, 801):
        guard.wait()
        head.train()
        optimizer.zero_grad(set_to_none=True)
        loss = head.loss(encoded, target)
        assert torch.isfinite(loss)
        loss.backward()
        for name, value in head.named_parameters():
            if value.grad is not None:
                assert torch.isfinite(value.grad).all()
                positive_gradients[name] = max(positive_gradients.get(name, 0), float(value.grad.norm()))
        nn.utils.clip_grad_norm_(head.parameters(), 1)
        optimizer.step()
        if step % 25 == 0:
            row = dict(step=step, **counts())
            history.append(row)
            save(out / 'PROGRESS.json', dict(status='RUNNING', samples=len(indices), **row))
            print(json.dumps(row), flush=True)
            if row['exact_market_actions'] == len(indices) and row['loss'] < .01:
                break

    after = counts()
    head.eval()
    with torch.no_grad():
        # Labels at later positions cannot change any earlier logits.
        changed = target.clone()
        changed[:, 4:, 0] = 3
        changed[:, 4:, 1] = 0
        original_logits = head(encoded, target)
        changed_logits = head(encoded, changed)
        assert torch.equal(original_logits[0][:, :4], changed_logits[0][:, :4])
        assert torch.equal(original_logits[1][:, :4], changed_logits[1][:, :4])
        # Corrupt ignored labels after EOS: loss must remain the same.
        ended = ((target[:, :, 0] == 1).cumsum(1) - (target[:, :, 0] == 1).long()) > 0
        ignored = target.clone()
        ignored[:, :, 0][ended] = 3
        ignored[:, :, 1][ended] = 0
        assert torch.equal(head.loss(encoded, target), head.loss(encoded, ignored))
        predicted = head.predict(encoded)
        assert not any('pattern' in k for k in head.state_dict())
        for row in predicted:
            stop = torch.where(row[:, 0] == 1)[0]
            assert len(stop) == 1 and int(stop[0]) <= 10
            assert (row[int(stop[0])+1:] == 0).all()
            assert (row[:int(stop[0]), 0] >= 2).all()
        unit_after = base.predict({k: v[:4] for k, v in b.items()})[0]
        assert torch.equal(unit_before, unit_after)
        assert all(torch.equal(frozen[k], v) for k, v in base.state_dict().items())
        roundtrips = 0
        for row in [*predicted, *target]:
            action = codec.decode(b['unit_y'][0].numpy(), row.numpy(), int(b['unit_count'][0]))
            tokens, _, _ = encode_action(action, int(b['unit_count'][0]))
            assert np.array_equal(codec.encode(tokens, int(b['unit_count'][0]))[1], row.numpy())
            roundtrips += 1
        # Both protocol extremes are enforced, even with pathological logits.
        saved = copy.deepcopy(head.state_dict())
        head.command_head.weight.zero_()
        for forced_type in (1, 3):
            head.command_head.bias.fill_(-1e4)
            head.command_head.bias[forced_type] = 1e4
            result = head.predict(encoded[:2])
            if forced_type == 1:
                assert (result[:, 0, 0] == 1).all() and (result[:, 1:] == 0).all()
            else:
                assert (result[:, :10, 0] == 3).all() and (result[:, 10, 0] == 1).all()
        head.load_state_dict(saved)

        # Warm, CPU-only batch-one timing; this is not official sandbox timing.
        def generated_action(one):
            uh, mh = base.encode(one)
            ul = base.unit_head(uh).clone()
            ul[:, :, 0] = -1e4
            ut = ul.argmax(-1)
            uq = base.quantity_head(uh + base.unit_type(ut)).argmax(-1)
            u = torch.stack((ut, torch.where(base.uq[ut], uq, 0)), -1)
            u.masked_fill_((torch.arange(32)[None] >= one['unit_count'][:, None])[:, :, None], 0)
            return u, head.predict(mh)

        timings = {'template_policy': [], 'autoregressive_policy': []}
        for repeat in range(len(indices) * 2 + 5):
            index = repeat % len(indices)
            one = {k: v[index:index+1] for k, v in b.items()}
            order = [('template_policy', base.predict), ('autoregressive_policy', generated_action)]
            for name, function in (order if repeat % 2 else reversed(order)):
                guard.wait()
                tick = time.perf_counter()
                result = function(one)
                elapsed_ms = (time.perf_counter() - tick) * 1000
                assert torch.equal(result[0], base.predict(one)[0])
                if repeat >= 5:
                    timings[name].append(elapsed_ms)
        timings = {name: dict(samples=len(values), median_ms=float(np.median(values)),
                              p95_ms=float(np.percentile(values, 95)), max_ms=max(values))
                   for name, values in timings.items()}

    passed = (after['exact_market_actions'] == len(indices) and after['loss'] < before['loss'] * .1
              and all(v > 0 for v in positive_gradients.values()))
    report = dict(status='PASS' if passed else 'LEARNABILITY_NOT_ESTABLISHED',
                  scope='Small training-sample fit and decoder checks only; no held-out generalization or win-rate evidence.',
                  checkpoint_sha256=hashlib.sha256(snapshot).hexdigest(),
                  source_ready_sha256=hashlib.sha256((root / 'data/fresh_packed/READY.json').read_bytes()).hexdigest(),
                  code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  samples=len(indices), training_indices=indices.tolist(), rare_samples=16,
                  order_lengths=sorted(set(lengths[indices].tolist())), steps=step,
                  seconds=time.monotonic()-started, before=before, after=after, history=history,
                  decoder_parameters=sum(v.numel() for v in head.parameters()),
                  positive_gradients=positive_gradients, causal_prefix_check=True,
                  loss_ignores_after_eos=True, codec_roundtrips=roundtrips,
                  empty_and_max_length_checks=True, base_tensors_and_unit_actions_unchanged=True,
                  full_template_vocabulary_used=False, teacher_needed_at_inference=False,
                  quantities=checkpoint['quantities'], new_match_seeds_used=0,
                  trained_probe_weights_discarded=True, observed_at_unix=time.time())
    report['warm_batch_one_cpu_timing'] = timings
    report['timing_scope'] = 'One CPU thread, in-process inference on training observations; excludes input feature construction, codec, cold start, official sandbox, and OS deadline enforcement.'
    save(out / 'PROTOTYPE_CHECK.json', report)
    print(json.dumps({k: v for k, v in report.items() if k not in ('history', 'quantities', 'positive_gradients', 'training_indices')}), flush=True)
    assert passed, 'See PROTOTYPE_CHECK.json; no candidate was modified or accepted.'


if __name__ == '__main__':
    main()
