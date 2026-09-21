"""Differentiable complete-turn densities, visited-factor KL, and recurrent replay."""
import torch
import os
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from exact_actions import expand_worker
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


def entropy(logp):
    safe = torch.where(torch.isfinite(logp), logp, 0.)
    return -(logp.exp() * safe).sum(-1)


def categorical_kl(reference, current):
    support = torch.isfinite(reference)
    # Same grammar is required. Mask before subtraction to avoid inf-inf gradients.
    delta = torch.where(support, reference, 0.) - torch.where(support, current, 0.)
    return (reference.exp() * delta).sum(-1)


def shaped_reward(margin, beta, sigma):
    """shaped-reward-v1 terminal return: sign(margin) + beta*clip(margin/sigma, -1, 1)."""
    margin = torch.as_tensor(margin, dtype=torch.float32)
    return margin.sign() + beta*(margin/sigma).clamp(-1., 1.)


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
        traces = []
        ref = None
        if reference is not None:
            with torch.no_grad():
                ref = reference(batch, reference_state, burn=burn, reference_only=True)
        market = False
        for event_i, e in enumerate(batch['events']):
            phase = e['phase']
            if phase == 1 and not market:
                prefix = torch.zeros_like(prefix)
                market = True
            ids, idx = e['ids'], e['index']
            rows = torch.arange(len(ids), device=ids.device)
            head = m.worker_head if phase == 0 else m.market_head
            if torch.is_grad_enabled() and not reference_only:
                lp, emb = checkpoint(replay_head, m, phase, actor, prefix, context, e, post,
                                     use_reentrant=False)
            else:
                lp, emb = replay_head(m, phase, actor, prefix, context, e, post)
            chosen = emb[rows, idx]
            logp = logp.index_add(0, ids, lp[rows, idx])
            if not reference_only:
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
        logits = m.value(critic).float()
        # shaped-reward-v2: calibrated outcome utility carries the sign term; the linear head learns
        # only the residual. Detached inputs keep Huber gradient out of the classifier and the trunk.
        value_shaped = utility(logits).detach() + m.value_shaped(critic.detach()).float().squeeze(-1)
        return dict(logp=logp, logits=logits, value=value_shaped, outcome_value=utility(logits), entropy=ent,
                    kl=kl, factor_count=count, factors=traces, state=(ah, ch),
                    reference_logp=None if ref is None else ref['logp'])


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
              vf=.5, ent=.001, anchor=.01, returns=None, vf_shaped=.5, policy_weight=1.):
    log_ratio = result['logp'] - old_logp
    ratio = log_ratio.clamp(-20, 20).exp()
    policy = -torch.minimum(ratio * advantages, ratio.clamp(1-clip, 1+clip) * advantages)
    value = F.cross_entropy(result['logits'], outcome, reduction='none')
    if returns is None:
        raise ValueError('shaped-reward-v1 requires per-turn shaped returns')
    shaped = F.huber_loss(result['value'], returns.to(result['value'].dtype), reduction='none', delta=1.)
    factors = result['factor_count'].clamp_min(1)
    loss = policy_weight * policy + vf * value + vf_shaped * shaped - ent * result['entropy']/factors + anchor * result['kl']/factors
    return loss, dict(approx_kl=(ratio-1)-log_ratio,
                      clipped=(ratio.sub(1).abs()>clip).float(), policy=policy, value=value,
                      value_shaped=shaped, shaped_prediction=result['value'].detach(),
                      factor_kl=result['kl']/factors, entropy=result['entropy']/factors)
