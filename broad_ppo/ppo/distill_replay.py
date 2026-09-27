"""distill-v1 (source-v36): exact log-probability of teacher turns under the policy, inside PPO minibatches.

build_teacher_batch: for the rows of a GPU minibatch window, pick one teacher per labelled row (rotating with the row's
seat and the window start, so every teacher is used equally in expectation), take its stored wire rows and rebuild the
decoder inputs exactly as replay does for the learner's own turn (gpu_replay.batch_window with those wires: worker
resolution, post-worker market seed, events). Unlabelled rows get no events.
teacher_logp: reuses the minibatch's actor states, spatial context and quantity features; scores every teacher event with
the same heads as ExactPPOChunk (checkpointed replay_head) and returns the joint log-prob per row, a validity mask
(label ok, need-flag agreement with the GPU feature table, finite density) and per-factor argmax agreement counts.
"""
import torch
from torch.utils.checkpoint import checkpoint


def build_teacher_batch(data, begin, end, seat_ids, slots, wq, mq, burn, teachers_per_row=1):
    from ppo.gpu_replay import batch_window
    device = data['logp'].device
    K = data['teacher/workers'].shape[1]
    steps, b = end - begin, len(seat_ids)
    slot = torch.as_tensor(slots, device=device, dtype=torch.long)
    labeled = slot >= 0
    if not bool(labeled.any()):
        return None
    k = (torch.as_tensor(seat_ids, device=device) + begin) % K
    s = slot.clamp_min(0)
    w = data['teacher/workers'][begin:end][:, k, s].long()          # [steps, b, 33, 5]
    m = data['teacher/market'][begin:end][:, k, s].long()           # [steps, b, 10, 5]
    status = data['teacher/status'][begin:end][:, k, s]             # [steps, b]
    ok = (status == 0) & labeled[None]
    w = w * ok[..., None, None]
    m = m * ok[..., None, None]
    wq_t = torch.as_tensor(wq, device=device, dtype=torch.long)
    mq_t = torch.as_tensor(mq, device=device, dtype=torch.long)
    w[..., 1] = torch.where(w[..., 3] > 0, wq_t[w[..., 2]], 0)   # quantities are stored as vocabulary indices
    m[..., 1] = torch.where(m[..., 3] > 0, mq_t[m[..., 2]], 0)
    wires = (w.flatten(0, 1), m.flatten(0, 1))
    dummy = torch.zeros(data['logp'].shape[1], device=device)
    tb, _ = batch_window(data, begin, end, [], wq, mq, dummy.long(), burn=burn, seat_ids=list(seat_ids), wires=wires, features=False)
    return dict(events=tb['events'], worker_seed=tb['worker_seed'], market_seed=tb['market_seed'], post_ledger=tb['post_ledger'],
                post_farm=tb['post_farm'], quantity_vocab=(wq, mq), valid=ok.flatten(), teacher=k.repeat(steps))


@torch.no_grad()
def _materialize_workers(batch, quantities):
    """compact_workers.materialize_workers plus a check that each label's need flag equals the GPU need table."""
    from ppo.worker_state import WorkerState
    from ppo.worker_features_device import WorkerFeatures
    state = WorkerState(batch['worker_seed'])
    features = WorkerFeatures(state)
    indices, amounts = [], []
    n, device = state.n, state.device
    bad = torch.zeros(n, device=device, dtype=torch.bool)
    counts = torch.zeros(n, 5, device=device, dtype=torch.long)
    for event in batch['events']:
        if event['phase'] != 0:
            continue
        ids, index, qr, depth = event['ids'], event['index'], event['qr'], event['depth']
        state.resolve(indices, amounts)
        compact, need, stats = features(state.workers[depth].expand(n), counts)
        flag = torch.zeros(len(ids), device=device, dtype=torch.bool)
        flag[qr] = True
        bad[ids] |= need[ids, index] != flag
        event['compact'] = {k: v[ids] for k, v in compact.items()}
        event['ledger'] = event['compact']['ledger']
        delta = torch.zeros(len(ids), 128, device=device)
        delta.scatter_(1, index[:, None], 1)
        delta[:, 44] = quantities.scaled(event['quantity'].double())
        event['delta'] = delta
        if len(qr):
            event['qfeatures'], event['qmask'] = quantities(0, index[qr], stats[ids[qr], index[qr]])
        all_index = torch.zeros(n, device=device, dtype=torch.long)
        all_quantity = torch.zeros_like(all_index)
        all_index[ids] = index
        all_quantity[ids] = event['quantity']
        indices.append(all_index)
        amounts.append(all_quantity)
        counts.scatter_add_(1, (all_index - 15).clamp(0, 4)[:, None], ((all_index >= 15) & (all_index < 20)).long()[:, None])
    return bad


def teacher_logp(chunk, actor, context, tb):
    """-> dict(logp [n], valid [n] bool, agree/total per factor (tensors)). Gradients flow into actor/context/heads."""
    from ppo.compact_market import materialize_market
    from ppo.replay import replay_head
    from ppo.fast_features import QuantityFeatures
    m = chunk.model
    if chunk.quantities is None:
        chunk.quantities = QuantityFeatures(*tb['quantity_vocab'], tb['post_ledger'].device)
    need_mismatch = _materialize_workers(tb, chunk.quantities)
    materialize_market(tb, chunk.quantities)
    post = m.summarize_post_worker(tb['post_ledger'], tb['post_farm'])
    n = len(actor)
    prefix = torch.zeros_like(actor)
    logp = actor.new_zeros(n, dtype=torch.float32)
    stats = {name: actor.new_zeros((), dtype=torch.float32) for name in
             ('agree_worker', 'events_worker', 'agree_market', 'events_market', 'agree_quantity', 'events_quantity')}
    market = False
    for e in tb['events']:
        phase = e['phase']
        if phase == 1 and not market:
            prefix = torch.zeros_like(prefix)
            market = True
        ids, idx = e['ids'], e['index']
        rows = torch.arange(len(ids), device=ids.device)
        head = m.worker_head if phase == 0 else m.market_head
        if torch.is_grad_enabled():
            lp, emb = checkpoint(replay_head, m, phase, actor, prefix, context, e, post, use_reentrant=False)
        else:
            lp, emb = replay_head(m, phase, actor, prefix, context, e, post)
        chosen = emb[rows, idx]
        logp = logp.index_add(0, ids, lp[rows, idx])
        tag = 'worker' if phase == 0 else 'market'
        stats['agree_' + tag] = stats['agree_' + tag] + (lp.detach().argmax(-1) == idx).float().sum()
        stats['events_' + tag] = stats['events_' + tag] + len(ids)
        qr = e['qr']
        if len(qr):
            qlp = head.quantities(actor[ids[qr]], prefix[ids[qr]], chosen[qr], e['ledger'][qr], e['qfeatures'], e['qmask'])
            qrows = torch.arange(len(qr), device=ids.device)
            logp = logp.index_add(0, ids[qr], qlp[qrows, e['qindex']])
            stats['agree_quantity'] = stats['agree_quantity'] + (qlp.detach().argmax(-1) == e['qindex']).float().sum()
            stats['events_quantity'] = stats['events_quantity'] + len(qr)
        advance = e['advance_indices']
        if len(advance):
            nxt = head.advance(prefix[ids[advance]], chosen[advance], e['quantity'][advance], e['delta'][advance])
            prefix = prefix.index_copy(0, ids[advance], nxt.to(prefix.dtype))
    valid = tb['valid'] & ~need_mismatch & torch.isfinite(logp.detach())
    stats['need_mismatch'] = (tb['valid'] & need_mismatch).float().sum()
    stats['nonfinite'] = (tb['valid'] & ~torch.isfinite(logp.detach())).float().sum()
    return dict(logp=logp, valid=valid, stats=stats)
