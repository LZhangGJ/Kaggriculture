"""Differentiable complete-turn densities, visited-factor KL, and recurrent replay."""
import torch
import os
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from ppo.fast_features import expand_worker_cached as expand_worker  # replay-sync-v1 (source-v50): device-cached tables, bit-identical; exact_actions.expand_worker built them with torch.tensor(list, device) = one host sync per call
from ppo.fast_heads import score_event
from ppo.compact_market import materialize_market
from ppo.compact_workers import materialize_workers
from ppo.fast_features import QuantityFeatures


def replay_head(model, phase, actor, prefix, context, event, post):
    # Gather inside the checkpoint: otherwise each action retains a copy of
    # the full spatial context until backward.
    ids = event['ids']
    rows = torch.arange(len(ids), device=ids.device)
    ctx = {k: v[ids] for k, v in context.items()}
    candidates = expand_worker(event['compact']) if phase == 0 else event['candidates']
    extra = ctx['workers'][rows, event['compact']['worker']] if phase == 0 else post[ids]
    lp, _, _, emb = score_event(model, phase, actor[ids], prefix[ids], ctx,
                                candidates, event['ledger'], extra)
    return lp, emb


LEAN = os.environ.get('PPO_LEAN_GATHER', '1') == '1'


def lean_enabled():
    from ppo import fast_heads
    return LEAN and fast_heads.ENABLED and fast_heads.WORKER_ENABLED


def lean_context(context, events):
    """v40 lean-gather: gather, once per forward, only the context rows the fast heads read.

    Worker events read 5 cells (current, N, S, E, W), their own worker token and the 12 market tokens; market events
    read the 12 market tokens. The old per-event ctx = {k: v[ids]} copied every row's 200 cells, workers and programs
    (and its backward allocated a full-size gradient per event). One gather per tensor + split gives a single
    full-size backward buffer per step. Values are identical; only gradient summation order changes."""
    cells, workers, market = context['cells'], context['workers'], context['market']
    nc, nw = cells.shape[1], workers.shape[1]
    cell_idx, worker_idx, market_ids, wsizes, msizes = [], [], [], [], []
    for e in events:
        ids = e['ids']
        if e['phase'] == 0:
            cell_idx.append((ids[:, None] * nc + e['compact']['cells'][:, :5]).flatten())
            worker_idx.append(ids * nw + e['compact']['worker'])
            wsizes.append(len(ids))
        market_ids.append(ids); msizes.append(len(ids))
    out = [dict() for _ in events]
    if wsizes:
        c = cells.reshape(-1, cells.shape[-1])[torch.cat(cell_idx)].split([5 * n for n in wsizes])
        w = workers.reshape(-1, workers.shape[-1])[torch.cat(worker_idx)].split(wsizes)
        j = 0
        for i, e in enumerate(events):
            if e['phase'] == 0:
                out[i]['cells5'] = c[j].reshape(wsizes[j], 5, -1); out[i]['worker'] = w[j]; j += 1
    if msizes:
        mk = market[torch.cat(market_ids)].split(msizes)
        for i in range(len(events)): out[i]['market'] = mk[i]
    return out


def replay_head_lean(model, phase, actor, prefix, lean, event, post):
    from ppo.fast_heads import worker_score_lean, market_score
    ids = event['ids']
    if phase == 0:
        candidates = expand_worker(event['compact'])
        lp, _, _, emb = worker_score_lean(model, actor[ids], prefix[ids], lean['cells5'], lean['market'],
                                          candidates, event['ledger'], lean['worker'])
    else:
        lp, _, _, emb = market_score(model, actor[ids], prefix[ids], dict(market=lean['market']),
                                     event['candidates'], event['ledger'], post[ids])
    return lp, emb


def _pinned_copy(t):
    out = torch.empty(t.shape, dtype=t.dtype, pin_memory=True)
    return out.copy_(t, non_blocking=True)


def store_reference(cache, ref):
    """v40 reference-cache: keep the frozen reference's per-event factor densities (pinned host memory) so a later PPO
    pass over the identical minibatch reuses them instead of re-running the reference encoder, GRU and heads. The
    reference is frozen and its forward is deterministic, so the reused tensors equal a recomputation bitwise."""
    cache['factors'] = [(_pinned_copy(lp), None if q is None else _pinned_copy(q)) for lp, q in ref['factors']]
    cache['logp'] = _pinned_copy(ref['logp'])
    cache['bytes'] = sum(t.numel() * t.element_size() for pair in cache['factors'] for t in pair if t is not None)


def load_reference(cache, device):
    return dict(logp=cache['logp'].to(device, non_blocking=True),
                factors=[(lp.to(device, non_blocking=True), None if q is None else q.to(device, non_blocking=True))
                         for lp, q in cache['factors']])


def entropy(logp):
    safe = torch.where(torch.isfinite(logp), logp, 0.)
    return -(logp.exp() * safe).sum(-1)


def categorical_kl(reference, current):
    support = torch.isfinite(reference)
    # Same grammar is required. Mask before subtraction to avoid inf-inf gradients.
    delta = torch.where(support, reference, 0.) - torch.where(support, current, 0.)
    return (reference.exp() * delta).sum(-1)


def squash(x, shape='clip'):
    if shape == 'linear': return x  # shaped-reward-v7: no ceiling
    return torch.tanh(x) if shape == 'tanh' else x.clamp(-1., 1.)


def cash_potential(cash, weight, center, sigma, shape='clip'):
    cash = torch.as_tensor(cash, dtype=torch.float32)
    return weight*squash((cash-center)/sigma, shape)


def margin_terms(margin, beta, sigma, shape='clip'):
    margin = torch.as_tensor(margin, dtype=torch.float32)
    return margin.sign() + beta*squash(margin/sigma, shape)


def dense_rewards(money, final_cash, margin, beta, sigma, cash_weight, cash_center, shape='tanh'):
    """shaped-reward-v6. money: [T, N] own cash before each turn; final_cash: [N]. Returns [T, N] per-turn rewards whose
    sum is margin_terms(margin) + Phi(final_cash) - Phi(money[0]) (potential-based shaping, same optimum)."""
    money = torch.as_tensor(money, dtype=torch.float32); final_cash = torch.as_tensor(final_cash, dtype=torch.float32)
    phi = cash_potential(torch.cat([money, final_cash[None]], 0), cash_weight, cash_center, sigma, shape)
    rewards = phi[1:] - phi[:-1]
    rewards[-1] = rewards[-1] + margin_terms(margin, beta, sigma, shape)
    return rewards


def reward_to_go(rewards):
    return rewards.flip(0).cumsum(0).flip(0)


def shaped_reward(margin, beta, sigma, cash=None, cash_weight=0., cash_center=100000., shape='clip'):
    """shaped-reward-v1 terminal return: sign(margin) + beta*clip(margin/sigma, -1, 1).

    shaped-reward-v5 adds cash_weight*clip((cash-cash_center)/sigma, -1, 1) when cash is given (own terminal cash)."""
    margin = torch.as_tensor(margin, dtype=torch.float32)
    reward = margin_terms(margin, beta, sigma, shape)
    if cash is not None and cash_weight:
        reward = reward + cash_potential(cash, cash_weight, cash_center, sigma, shape)
    return reward


def utility(logits):
    # Canonical replay outcomes: loss=0, draw=1, win=2.
    p = logits.float().softmax(-1)
    return p[:, 2] - p[:, 0]


class ExactPPOChunk(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.quantities = None
        self.sequence_memory=os.environ.get('PPO_SEQUENCE_GRU')=='1'
        if self.sequence_memory:
            from ppo.sequence_memory import sequence_view
            self.actor_sequence=sequence_view(model.actor_memory)
            self.critic_sequence=sequence_view(model.critic_memory)

    def forward(self, batch, state, burn=0, reference=None, reference_state=None, reference_only=False):
        m = self.model
        if 'market_seed' in batch and not batch.get('market_materialized',False):
            if self.quantities is None:
                self.quantities=QuantityFeatures(*batch['quantity_vocab'],batch['post_ledger'].device)
            if 'worker_seed' in batch and not batch.get('workers_materialized',False):
                materialize_workers(batch,self.quantities)
            materialize_market(batch,self.quantities)
        b, steps = batch['batch'], batch['steps']
        if burn and not reference_only and os.environ.get('PPO_SPLIT_BURN_ENCODER')=='1':
            split=burn*b
            # Burn-in contributes recurrent state, but its loss is masked and
            # the state is detached at the boundary. Avoid its encoder backward.
            with torch.no_grad():
                prefix_token,prefix_context=m.encode_features({k:v[:split] for k,v in batch['x'].items()})
            suffix_token,suffix_context=m.encode_features({k:v[split:] for k,v in batch['x'].items()})
            token=torch.cat((prefix_token,suffix_token),0)
            context={k:torch.cat((prefix_context[k],suffix_context[k]),0) for k in suffix_context}
        else:
            token, context = m.encode_features(batch['x'])
        token = token.reshape(steps, b, -1)
        ah, ch = state
        actors, critics = [], []
        if self.sequence_memory:
            from ppo.sequence_memory import run_sequence
            actors,ah=run_sequence(self.actor_sequence,token,ah,burn)
            if not reference_only:critics,ch=run_sequence(self.critic_sequence,token,ch,burn)
            actor=actors.flatten(0,1)
            critic=None if reference_only else critics.flatten(0,1)
        else:
            for t in range(steps):
                if t == burn:
                    ah, ch = ah.detach(), ch.detach()
                ah = m.actor_memory(token[t], ah)
                if not reference_only:ch = m.critic_memory(token[t], ch)
                actors.append(ah)
                critics.append(ch)
            actor, critic = torch.stack(actors).flatten(0, 1), torch.stack(critics).flatten(0, 1)
        post = m.summarize_post_worker(batch['post_ledger'], batch['post_farm'])
        prefix = torch.zeros_like(actor)
        n = len(actor)
        logp = actor.new_zeros(n, dtype=torch.float32)
        ent = logp.clone()
        kl = logp.clone()
        count = logp.clone()
        market_terms = None if reference_only else batch.get('market_entropy_terms')  # market-entropy-v1: (command, quantity) flags or None
        ment = logp.clone() if market_terms and market_terms[0] else None
        mqent = logp.clone() if market_terms and market_terms[1] else None
        traces = []
        ref = None
        if reference is not None:
            cache = batch.get('reference_cache')
            if cache:
                ref = load_reference(cache, logp.device)
            else:
                with torch.no_grad():
                    ref = reference(batch, reference_state, burn=burn, reference_only=True)
                if cache is not None and batch.get('reference_cache_store'):
                    store_reference(cache, ref)
        market = False
        lean = lean_context(context, batch['events']) if lean_enabled() else None
        for event_i, e in enumerate(batch['events']):
            phase = e['phase']
            if phase == 1 and not market:
                prefix = torch.zeros_like(prefix)
                market = True
            ids, idx = e['ids'], e['index']
            rows = torch.arange(len(ids), device=ids.device)
            head = m.worker_head if phase == 0 else m.market_head
            if lean is not None:
                if torch.is_grad_enabled() and not reference_only:
                    lp, emb = checkpoint(replay_head_lean, m, phase, actor, prefix, lean[event_i], e, post,
                                         use_reentrant=False)
                else:
                    lp, emb = replay_head_lean(m, phase, actor, prefix, lean[event_i], e, post)
            elif torch.is_grad_enabled() and not reference_only:
                lp, emb = checkpoint(replay_head, m, phase, actor, prefix, context, e, post,
                                     use_reentrant=False)
            else:
                lp, emb = replay_head(m, phase, actor, prefix, context, e, post)
            chosen = emb[rows, idx]
            logp = logp.index_add(0, ids, lp[rows, idx])
            if not reference_only:
                if ment is not None and phase == 1:
                    h = entropy(lp)
                    ent = ent.index_add(0, ids, h); ment = ment.index_add(0, ids, h)
                else:
                    ent = ent.index_add(0, ids, entropy(lp))
                count = count.index_add(0, ids, torch.ones_like(ids, dtype=torch.float32))
            if ref is not None:
                kl = kl.index_add(0, ids, categorical_kl(ref['factors'][event_i][0], lp))
            qr = e['qr']
            qlp = None
            if len(qr):
                qlp = head.quantities(actor[ids[qr]], prefix[ids[qr]], chosen[qr],
                                      e['ledger'][qr], e['qfeatures'], e['qmask'])
                qrows = torch.arange(len(qr), device=ids.device)
                logp = logp.index_add(0, ids[qr], qlp[qrows, e['qindex']])
                if not reference_only:
                    if mqent is not None and phase == 1:
                        hq = entropy(qlp)
                        ent = ent.index_add(0, ids[qr], hq); mqent = mqent.index_add(0, ids[qr], hq)
                    else:
                        ent = ent.index_add(0, ids[qr], entropy(qlp))
                    count = count.index_add(0, ids[qr], torch.ones_like(qr, dtype=torch.float32))
                if ref is not None:
                    kl = kl.index_add(0, ids[qr], categorical_kl(ref['factors'][event_i][1], qlp))
            traces.append((lp, qlp))
            advance = e['advance_indices']
            if len(advance):
                nxt = head.advance(prefix[ids[advance]], chosen[advance],
                                   e['quantity'][advance], e['delta'][advance])
                prefix = prefix.index_copy(0, ids[advance], nxt.to(prefix.dtype))
        if reference_only:return dict(logp=logp,factors=traces)
        teacher=None
        if batch.get('teacher') is not None:  # distill-v1: teacher turns scored with the same actor states and heads
            from ppo.distill_replay import teacher_logp
            teacher=teacher_logp(self,actor,context,batch['teacher'])
        logits = m.value(critic).float()
        # shaped-reward-v2: calibrated outcome utility carries the sign term; the linear head learns
        # only the residual. Detached inputs keep Huber gradient out of the classifier and the trunk.
        if getattr(self, 'value_trunk_grad', False):  # critic-fix-v1: shaped loss also trains critic_memory (ppo/value_trunk.py)
            from ppo.value_trunk import critic_for_value
            value_shaped = utility(logits).detach() + m.value_shaped(critic_for_value(self, token, state[1], burn)).float().squeeze(-1)
        else:
            value_shaped = utility(logits).detach() + m.value_shaped(critic.detach()).float().squeeze(-1)
        out = dict(logp=logp, logits=logits, value=value_shaped, outcome_value=utility(logits), entropy=ent,
                    kl=kl, factor_count=count, factors=traces, state=(ah, ch),
                    reference_logp=None if ref is None else ref['logp'],teacher=teacher)
        if ment is not None: out['market_entropy'] = ment
        if mqent is not None: out['market_quantity_entropy'] = mqent
        return out


def gae(rewards, values, terminated, bootstrap=0., lam=.95, gamma=1.):
    """A collector cutoff bootstraps; an official terminal never does."""
    rewards, values = torch.as_tensor(rewards), torch.as_tensor(values)
    out = torch.zeros_like(values, dtype=torch.float32)
    carry = 0.
    for t in range(len(values)-1, -1, -1):
        nxt = values[t+1] if t+1 < len(values) else bootstrap
        alive = 1. - float(terminated[t])
        delta = rewards[t] + gamma * alive * nxt - values[t]
        carry = delta + gamma * lam * alive * carry
        out[t] = carry
    return out


def ppo_terms(result, old_logp, advantages, outcome, clip=.1,
              vf=.5, ent=.001, anchor=.01, returns=None, vf_shaped=.5, policy_weight=1., value_loss='huber', market_ent=0., market_qent=0.):
    log_ratio = result['logp'] - old_logp
    ratio = log_ratio.clamp(-20, 20).exp()
    policy = -torch.minimum(ratio * advantages, ratio.clamp(1-clip, 1+clip) * advantages)
    value = F.cross_entropy(result['logits'], outcome, reduction='none')
    if returns is None:
        raise ValueError('shaped-reward-v1 requires per-turn shaped returns')
    if value_loss == 'huber':
        shaped = F.huber_loss(result['value'], returns.to(result['value'].dtype), reduction='none', delta=1.)
    else:  # critic-fix-v1
        from ppo.value_trunk import shaped_value_loss
        shaped = shaped_value_loss(result['value'], returns.to(result['value'].dtype), value_loss)
    factors = result['factor_count'].clamp_min(1)
    loss = policy_weight * policy + vf * value + vf_shaped * shaped - ent * result['entropy']/factors + anchor * result['kl']/factors
    # market-entropy-v1: extra bonus on the market command factors (and optionally the market quantity factors), normalised
    # exactly like the all-head entropy term above (per-turn sum over those factors / all visited factors of the turn).
    if market_ent: loss = loss - market_ent * result['market_entropy']/factors
    if market_qent: loss = loss - market_qent * result['market_quantity_entropy']/factors
    stats = dict(approx_kl=(ratio-1)-log_ratio,
                      clipped=(ratio.sub(1).abs()>clip).float(), policy=policy, value=value,
                      value_shaped=shaped, shaped_prediction=result['value'].detach(),
                      factor_kl=result['kl']/factors, entropy=result['entropy']/factors)
    if 'market_entropy' in result: stats['market_entropy'] = result['market_entropy'].detach()/factors
    if 'market_quantity_entropy' in result: stats['market_quantity_entropy'] = result['market_quantity_entropy'].detach()/factors
    return loss, stats
