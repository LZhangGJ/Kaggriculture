"""Exact market head without projections of absent spatial/worker references."""
import os
import torch
from model_v2 import gather
from exact_model import factor_log_probs
from exact_decoder import score_event as canonical_score_event

ENABLED=os.environ.get('PPO_FAST_MARKET_HEAD')=='1'
WORKER_ENABLED=os.environ.get('PPO_FAST_WORKER_HEAD')=='1'


def worker_score(model,actor,prefix,context,candidate,ledger,extra):
    # Canonical action order: PASS, four moves, then 39 actions at the
    # current cell. Source and acting worker are shared by all 44 actions.
    head=model.worker_head;raw=candidate['raw']
    source=head.source_cell(gather(context['cells'],candidate['source'][:,:1]))
    targets=head.target_cell(gather(context['cells'],candidate['target'][:,:5]))
    targets=torch.cat((targets,targets[:,:1].expand(-1,39,-1)),1)
    items=gather(head.item(context['market']),candidate['items'])
    workers=head.worker(extra)[:,None]
    embedding=head.norm(head.raw(raw)+source+targets+items+workers)
    query=torch.cat((actor,prefix,head.ledger(ledger),extra),-1)
    logits=(embedding*head.query(query)[:,None]).sum(-1)/16+head.bias(raw).squeeze(-1)
    joint,gates,commands=factor_log_probs(logits,head.gate(query),candidate['legal'])
    return joint,gates,commands,embedding


def market_score(model,actor,prefix,context,candidate,ledger,extra):
    # The exact market schema uses -1 for every source, target and worker
    # reference. gather returns zero; all three corresponding Linear layers
    # have bias=False. Their forward terms and useful gradients are zero.
    head=model.market_head
    raw=candidate['raw']
    embedding=head.norm(head.raw(raw)+head.item(gather(context['market'],candidate['items'])))
    query=torch.cat((actor,prefix,head.ledger(ledger),extra),-1)
    logits=(embedding*head.query(query)[:,None]).sum(-1)/16+head.bias(raw).squeeze(-1)
    joint,gates,commands=factor_log_probs(logits,head.gate(query),candidate['legal'])
    return joint,gates,commands,embedding


def worker_score_lean(model,actor,prefix,cells5,market,candidate,ledger,extra):
    """v40 lean-gather: worker_score with the five referenced cells pre-gathered.

    cells5[:,k] = context['cells'][row, compact['cells'][:,k]] for k=0..4 (current cell, N, S, E, W); in the canonical
    order source = candidate['source'][:,:1] = cells[:,0] and target[:,:5] = cells[:,0:5], never -1. Same values and the
    same Linear input shapes as worker_score, so the forward is bitwise identical."""
    head=model.worker_head;raw=candidate['raw']
    source=head.source_cell(cells5[:,:1])
    targets=head.target_cell(cells5[:,:5])
    targets=torch.cat((targets,targets[:,:1].expand(-1,39,-1)),1)
    items=gather(head.item(market),candidate['items'])
    workers=head.worker(extra)[:,None]
    embedding=head.norm(head.raw(raw)+source+targets+items+workers)
    query=torch.cat((actor,prefix,head.ledger(ledger),extra),-1)
    logits=(embedding*head.query(query)[:,None]).sum(-1)/16+head.bias(raw).squeeze(-1)
    joint,gates,commands=factor_log_probs(logits,head.gate(query),candidate['legal'])
    return joint,gates,commands,embedding


def score_event(model,phase,actor,prefix,context,candidate,ledger,extra):
    if WORKER_ENABLED and phase==0:
        return worker_score(model,actor,prefix,context,candidate,ledger,extra)
    if ENABLED and phase==1:
        return market_score(model,actor,prefix,context,candidate,ledger,extra)
    return canonical_score_event(model,phase,actor,prefix,context,candidate,ledger,extra)
