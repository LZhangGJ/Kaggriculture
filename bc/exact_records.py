"""Read preserved actions without reusing obsolete macro labels or masks.

This is the shared input boundary for the exact decoder and future cache builder.
It does not encode candidates, train a model, or rewrite canonical trajectories.
"""
from copy import deepcopy
import gzip
import json


def raw_turn(record):
    if record.get('schema') != 'macro-bc-v2':
        raise ValueError('Expected canonical macro-bc-v2 replay record')
    frame = record['input_frame']
    observation = record['observation']
    if record['action_frame'] != frame + 1 or observation['step'] != frame:
        raise ValueError('Observation/action alignment mismatch')
    if observation['player'] != record['seat']:
        raise ValueError('Replay seat mismatch')
    if 'raw_worker_action' not in record:
        raise ValueError('Canonical record is missing raw worker actions')
    slots = record['targets']['market']
    if len(slots) > 10:
        raise ValueError('Unexpected market slot count for the pinned engine')
    market = []
    for position, slot in enumerate(slots):
        if slot.get('position') != position or 'raw' not in slot:
            raise ValueError('Raw market request or ordered position is missing')
        market.append(deepcopy(slot['raw']))
    return dict(observation=deepcopy(observation),
                worker_action=deepcopy(record['raw_worker_action']),
                market_requests=market)


def load_seat(path):
    with gzip.open(path, 'rt') as source:
        records = [json.loads(line) for line in source]
    if len(records) != 719:
        raise ValueError('Expected 719 current-engine player turns')
    identity = (records[0]['episode'], records[0]['seat'])
    turns = []
    for frame, record in enumerate(records):
        if record['input_frame'] != frame or (record['episode'], record['seat']) != identity:
            raise ValueError('Trajectory order or identity mismatch')
        turns.append(raw_turn(record))
    return turns
