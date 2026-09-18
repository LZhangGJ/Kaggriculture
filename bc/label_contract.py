"""Weak macro targets projected onto the executor contract; exact market targets stay separate."""
from collections import Counter
from features_v2 import RequestLedger, JOB_INDEX, admissibility, clock
from bc_runtime import job_key, MARKET

LABEL_CONTRACT = 'macro-bc-v3.1-admissible-events-separated-market'


def rejection_reason(job, obs, pending):
    if len(pending) >= 16: return 'queue_full'
    if any(job_key(j) == job_key(job) for j in pending): return 'duplicate_job_key'
    tile = obs['farms'][obs['player']]['tiles'][job['y']][job['x']]
    if tile == 'LOCKED': return 'locked_tile'
    if any((j['x'], j['y']) == (job['x'], job['y']) for j in pending): return 'same_tile_dependency'
    if job['op'] in ('PLANT', 'BUILD_COOP', 'BUILD_PASTURE') and tile is not None:
        return 'occupied_crop' if isinstance(tile, dict) and tile.get('crop') else 'occupied_other'
    if job['op'] == 'PLACE': return 'missing_structure_or_occupied'
    if job['op'] == 'DIG': return 'empty_or_animal'
    return 'other'


def project_goals(target, obs, executor):
    """Only admitted event IDs are consumed. Deferred events can enter on a later turn.

    Event identity prevents an overlapping horizon from adding the same achievement
    twice. Different events at the same tile remain distinct, even if only one can
    fit the current queue. Chronological order is a convention, not teacher intent.
    """
    stats = Counter(); pending = list(executor.pending)
    seen = getattr(executor, 'seen_event_ids', set())
    visited = set(); remaining = []
    if not hasattr(executor, 'event_payloads'): executor.event_payloads = {}
    for job in target['goals']:
        stats['raw_goal_events'] += 1
        event_id = job.get('evidence_id')
        if event_id is None: raise ValueError('Missing production event identity')
        payload=tuple(job.get(k) for k in ('event_frame','unit','op','item','x','y'))
        if event_id in executor.event_payloads and executor.event_payloads[event_id]!=payload:
            raise ValueError(f'Event ID payload collision: {event_id}: {payload}')
        executor.event_payloads[event_id]=payload
        if event_id in seen or event_id in visited:
            stats['dedup_same_event_id'] += 1
            continue
        visited.add(event_id)
        if job_key(job) not in JOB_INDEX: raise ValueError('Unknown production candidate')
        remaining.append(job)
    ledger = RequestLedger(obs, pending); selected = []
    while remaining and len(ledger.pending) < 16:
        deferred = []; advanced = False
        for job in remaining:
            if admissibility(job, obs, ledger.pending)[1]:
                ledger.add_job(job, inferred=True); selected.append(job); advanced = True
            else: deferred.append(job)
        remaining = deferred
        if not advanced: break
    for job in remaining:
        reason = rejection_reason(job, obs, ledger.pending)
        offset = int(job.get('hours_ahead', job.get('event_frame', clock(obs)) - clock(obs)))
        stats['deferred_goal_events'] += 1
        stats['deferred_' + reason] += 1
        stats['deferred_offset_' + str(offset)] += 1
        stats['deferred_op_' + job['op']] += 1
        if offset == 0:
            raise ValueError(f'Unconsumed current-frame event: frame={clock(obs)} '
                             f'event={job} reason={reason} pending={ledger.pending}')
    stats['projected_add_targets'] += len(selected)
    # Absence after filtering is not evidence for END. Capacity is not END either.
    end_allowed = not remaining and len(ledger.pending) < 16
    return selected, end_allowed, stats


def remember_event(executor, job):
    if not hasattr(executor, 'seen_event_ids'): executor.seen_event_ids = set()
    executor.seen_event_ids.add(job['evidence_id'])


def teacher_workers(raw, obs):
    """Normalize missing worker slots as PASS, without reading future observations."""
    if not isinstance(raw, dict): raise ValueError('BC requires recorded worker actions')
    hands = raw.get('hands') or []
    if not isinstance(hands, list): raise ValueError('Invalid recorded hands')
    def command(value):
        if value is None or value == []: return ['PASS']
        if not isinstance(value, list) or not value or not isinstance(value[0], str):
            raise ValueError('Unsupported recorded worker command')
        return value
    n = len(obs['farms'][obs['player']]['hands'])
    return {'farmer': command(raw.get('farmer')),
            'hands': [command(hands[i] if i < len(hands) else None) for i in range(n)]}


def request_history(target):
    """Keep recorded request identities separate from safe-to-emit wire orders."""
    out=[]
    for slot in target['market']:
        if 'raw' in slot:
            raw=slot['raw']
        else:
            op,item=MARKET[slot['index']]
            raw=[op,item,slot['quantity']] if item else [op]
        # Unsupported non-list requests remain in the raw trajectory; history's
        # numeric representation can only encode recognized list-shaped requests.
        out.append(list(raw) if isinstance(raw,list) else [])
    return out
