"""feature-ext-v1 submission path: ExactAgent (official observations, CPU) with the attached input extensions.

Identical to exact_decoder.ExactAgent.act except:
  * LongMarketHistory runs beside PublicHistory; x['market_long'] enters the encoder (long-market-v1);
  * quantity features come from ppo.fast_features.QuantityFeatures (the GPU class, parity-tested against the numpy
    builders) with worker_stats / market_stats, which appends the 9 residual + price-impact columns (quantity-ext-v1,
    price-impact-v1);
  * own orders are fed to LongMarketHistory after acting.
"""
import numpy as np
import torch
from bc_runtime import MARKET
from exact_decoder import (ExactAgent, encode_exact, batch_arrays, resolve_worker_phase, worker_features, expand_worker,
                           score_event, needs_quantity, worker_wire, worker_delta, market_state, ledger_vector,
                           market_candidates, remember_requests)
from ppo.fast_features import QuantityFeatures, worker_stats, market_stats
from long_market import LongMarketHistory


class ExtAgent(ExactAgent):
    def reset(self):
        super().reset()
        self.long = LongMarketHistory()
        self.qgen = QuantityFeatures(self.wq, self.mq, self.device)

    @classmethod
    def from_checkpoint(cls, path, device='cpu', greedy=True):
        from ppo.train import load_model
        model, ck = load_model(path, device)
        return cls(model, ck['worker_quantities'], ck['market_quantities'], device, greedy)

    @torch.no_grad()
    def act(self, obs):
        if obs['step'] == 0: self.reset()
        self.history.observe(obs); row = encode_exact(obs, self.history); m = self.model
        row['market_long'] = self.long.observe(obs).numpy()
        actor, critic, _, ctx = m.encode(batch_arrays([row], self.device), *self.state)
        self.state = (actor, critic); prefix = torch.zeros_like(actor); slots = []; density = 0.
        def tensor(a): return torch.as_tensor(a, device=self.device).unsqueeze(0)
        n = 1 + len(obs['farms'][obs['player']]['hands'])
        for depth in range(n):
            provisional, _, _ = resolve_worker_phase(obs, slots)
            data = worker_features(obs, provisional, slots, depth)
            c = expand_worker({k: tensor(v) for k, v in data.items()}); led = tensor(data['ledger'])
            extra = ctx['workers'][:, depth]
            lp, gates, commands, emb = score_event(m, 0, actor, prefix, ctx, c, led, extra)
            index = self.choose_action(lp, gates, commands)
            density += float(lp[0, index]); q = 0; has_q = needs_quantity(provisional, depth, index)
            if has_q:
                stats = torch.as_tensor(worker_stats(provisional, depth), device=self.device)[index][None]
                qf, qm = self.qgen(0, torch.tensor([index], device=self.device), stats)
                qlp = m.worker_head.quantities(actor, prefix, emb[:, index], led, qf, qm)
                qi = self.choose(qlp); q = self.wq[qi]; density += float(qlp[0, qi])
            slots.append(worker_wire(index, q if has_q else None))
            prefix = m.worker_head.advance(prefix, emb[:, index], tensor(np.float32(q)).reshape(1), tensor(worker_delta(index, q)))
        resolved, _, _ = resolve_worker_phase(obs, slots); ledger = market_state(resolved)
        extra = m.summarize_post_worker(tensor(ledger_vector(ledger)), tensor(encode_exact(resolved, self.history)['farms'][0]))
        prefix = torch.zeros_like(actor)
        for depth in range(10):
            c = {k: tensor(v) for k, v in market_candidates(ledger).items()}; before = ledger_vector(ledger); led = tensor(before)
            lp, gates, commands, emb = score_event(m, 1, actor, prefix, ctx, c, led, extra)
            index = self.choose_action(lp, gates, commands)
            density += float(lp[0, index])
            if index == 0: break
            q = 0
            if MARKET[index][1] is not None:
                stats = torch.as_tensor(market_stats(ledger), device=self.device)[index][None]
                qf, qm = self.qgen(1, torch.tensor([index], device=self.device), stats)
                qlp = m.market_head.quantities(actor, prefix, emb[:, index], led, qf, qm)
                qi = self.choose(qlp); q = self.mq[qi]; density += float(qlp[0, qi])
            ledger.add_order(index, q)
            prefix = m.market_head.advance(prefix, emb[:, index], tensor(np.float32(q)).reshape(1), tensor(ledger_vector(ledger) - before))
        remember_requests(self.history, ledger.orders)
        self.long.requests(ledger.orders)
        return dict(farmer=slots[0], hands=slots[1:], market=ledger.orders), density


EXT_PREFIXES = ('worker_head.q_extra.', 'worker_head.q_index', 'market_head.q_extra.', 'market_head.q_index', 'market_long.')


def has_extension(checkpoint):
    return any(k.startswith(EXT_PREFIXES) for k in checkpoint['model'])


def load_agent(path, device='cpu', greedy=True):
    """Evaluation/submission loader: feature-ext checkpoints -> ExtAgent with the modules (strict load of every tensor);
    all other checkpoints -> the unchanged exact_decoder.ExactAgent.from_checkpoint (bit-identical to before)."""
    from exact_decoder import ExactAgent
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if not has_extension(checkpoint):
        return ExactAgent.from_checkpoint(path, device=device, greedy=greedy)
    from exact_identity import validate_identity
    from exact_model import ExactWorkerMarketPolicyV1
    from feature_ext import attach_all
    ci = checkpoint['cache_identity']; validate_identity(ci)
    if checkpoint['worker_quantities'] != ci['worker_quantities'] or checkpoint['market_quantities'] != ci['market_quantities']:
        raise ValueError('Checkpoint quantity vocabulary mismatch')
    if checkpoint.get('architecture') != ExactWorkerMarketPolicyV1.architecture:
        raise ValueError('An exact worker/market checkpoint is required')
    model = attach_all(ExactWorkerMarketPolicyV1(), ci['worker_quantities'], ci['market_quantities'])
    weights = {k: v for k, v in checkpoint['model'].items() if not k.startswith('value_shaped.')}
    model.load_state_dict(weights, strict=True)
    return ExtAgent(model, ci['worker_quantities'], ci['market_quantities'], device, greedy)
