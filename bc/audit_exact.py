"""Read-only full-corpus configuration and raw-action vocabulary audit."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
import hashlib
import json
from pathlib import Path
import time
import zipfile
from kgrl.contracts import DEFAULT_CONFIG
from cache_identity import engine_identity, file_sha


def audit_game(row):
    counts = Counter(); worker_q = {1}; market_q = {1}; errors = []
    archive, member = row['source'].rsplit(':', 1)
    with zipfile.ZipFile(archive) as source:
        raw = source.read(member)
    if hashlib.sha256(raw).hexdigest() != row['raw_sha256']:
        raise ValueError('Raw digest mismatch: ' + str(row['episode']))
    replay = json.loads(raw)
    if replay.get('module_version') != '1.32.7' or replay.get('statuses') != ['DONE', 'DONE']:
        raise ValueError('Wrong engine or incomplete game')
    config = replay.get('configuration', {})
    for key, value in DEFAULT_CONFIG.items():
        if config.get(key, value) != value:
            errors.append(f'configuration:{key}')
    unknown = set(config) - set(DEFAULT_CONFIG) - {'seed', 'runTimeout'}
    if unknown: errors.append('unknown_configuration:' + ','.join(sorted(unknown)))
    for seat, path in enumerate(row['files']):
        with gzip.open(path, 'rt') as source:
            length = 0
            for frame, line in enumerate(source):
                r = json.loads(line); length += 1; counts['seat_turns'] += 1
                if (r['schema'] != 'macro-bc-v2' or r['input_frame'] != frame or
                    r['action_frame'] != frame + 1 or r['seat'] != seat or
                    r['episode'] != row['episode'] or r['observation']['step'] != frame):
                    raise ValueError('Canonical continuity mismatch')
                actual = replay['steps'][frame+1][seat].get('action') or {}
                if not isinstance(actual, dict):
                    counts['non_dict_actions'] += 1; actual = {}
                own = dict(replay['steps'][frame][seat]['observation'])
                own.pop('remainingOverageTime', None)
                own['step'] = own.get('step', replay['steps'][frame][0]['observation'].get('step'))
                if own != r['observation']: raise ValueError('Observation changed in canonical data')
                worker = {'farmer': actual.get('farmer'), 'hands': actual.get('hands', [])}
                if worker != r['raw_worker_action']: raise ValueError('Worker action mismatch')
                hands = worker['hands'] if isinstance(worker['hands'], list) else []
                counts['extra_hand_slots'] += max(0, len(hands)-len(own['farms'][seat]['hands']))
                for command in [worker['farmer'], *hands]:
                    if not isinstance(command, list) or not command:
                        counts['worker_missing_or_nonlist'] += 1; continue
                    counts['worker_op:' + str(command[0])] += 1
                    if command[0] in ('PICKUP', 'PLACE'):
                        amount = command[2] if len(command) > 2 else 1
                        counts['worker_quantity_type:' + type(amount).__name__] += 1
                        try:
                            q = int(amount); worker_q.add(q)
                            counts['worker_nonpositive_quantity'] += q <= 0
                        except (ValueError, TypeError, OverflowError):
                            counts['worker_quantity_conversion_failure'] += 1
                orders = actual.get('market', [])
                if not isinstance(orders, list):
                    counts['nonlist_market'] += 1; orders = []
                counts['ignored_market_tail'] += max(0, len(orders)-10)
                if orders[:10] != [s['raw'] for s in r['targets']['market']]:
                    raise ValueError('Market raw request mismatch')
                for order in orders[:10]:
                    if not isinstance(order, list) or not order:
                        counts['market_empty_or_nonlist'] += 1; continue
                    counts['market_op:' + str(order[0])] += 1
                    if len(order) > 2:
                        try:
                            q = int(order[2]); market_q.add(q)
                            counts['market_nonpositive_quantity'] += q <= 0
                        except (ValueError, TypeError, OverflowError):
                            counts['market_quantity_conversion_failure'] += 1
            if length != 719: raise ValueError('Wrong trajectory length')
    return dict(episode=row['episode'], counts=dict(counts), worker=sorted(worker_q),
                market=sorted(market_q), configuration_errors=errors)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); p.add_argument('--workers', type=int, default=16)
    p.add_argument('--limit', type=int); a = p.parse_args()
    identity = engine_identity(); rows = [json.loads(s) for s in (a.root/'manifest.jsonl').read_text().splitlines()]
    if a.limit: rows = rows[:a.limit]
    a.output.mkdir(parents=True, exist_ok=True)
    counts = Counter(); wq = {1}; mq = {1}; issues = []; done = 0; start = time.monotonic()
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures = [pool.submit(audit_game, row) for row in rows]
        for future in as_completed(futures):
            result = future.result(); done += 1; counts.update(result['counts'])
            wq.update(result['worker']); mq.update(result['market'])
            if result['configuration_errors']: issues.append(result)
            if done % 20 == 0 or done == len(rows):
                status = dict(complete=False, games=done, total=len(rows), seconds=time.monotonic()-start,
                              counts=dict(counts), configuration_issues=len(issues))
                tmp=a.output/'status.tmp'; tmp.write_text(json.dumps(status)); tmp.replace(a.output/'status.json')
                print(json.dumps({k:v for k,v in status.items() if k!='counts'}), flush=True)
    result = dict(status, complete=not issues, engine=identity, configuration_details=issues,
        source_manifest_sha=file_sha(a.root/'manifest.jsonl'), all_replays=a.limit is None,
        worker_quantity_count=len(wq), market_quantity_count=len(mq),
        vocabulary_note='Conservative parsed transfer amount domain; ignored amounts produce no quantity loss during exact labeling')
    (a.output/'worker-quantities.json').write_text(json.dumps(sorted(wq)))
    (a.output/'market-quantities.json').write_text(json.dumps(sorted(mq)))
    (a.output/'status.json').write_text(json.dumps(result, indent=2))
    if issues: raise ValueError('Unsupported configurations require resolution; no games were excluded')


if __name__ == '__main__': main()
