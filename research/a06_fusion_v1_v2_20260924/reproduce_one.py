"""Replay one recorded game from the portable release with its frozen referee."""
from pathlib import Path
import argparse
import contextlib
import copy
import gzip
import hashlib
import io
import json
import sys
import time

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pool', type=Path, required=True, help='Path to research/a06_dynamic_variants_public10_20260924 in the previously shared repository')
    parser.add_argument('--opponent', default='internal/r14_cashflow')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--seat', type=int, choices=(0, 1), default=0)
    parser.add_argument('--candidate-version',choices=('v1','v2'),default='v1')
    args = parser.parse_args()
    manifest=json.loads((HERE/'MANIFEST.json').read_text())
    for name,digest in manifest.items():
        if name.startswith(('v1/','v2/','referee/')):
            path=(HERE/name).resolve()
            assert path.is_relative_to(HERE) and hashlib.sha256(path.read_bytes()).hexdigest()==digest,name
    sys.path.insert(0, str(HERE / 'referee'))
    from policy_host import LocalGame, Policy, load_engine
    old=args.candidate_version=='v1'
    protocol = json.loads((HERE / ('evidence/versions/v1/PROTOCOL.json' if old else 'evidence/PROTOCOL.json')).read_text())
    seed = args.seed if args.seed is not None else protocol['seeds'][0]
    group, name = args.opponent.split('/', 1)
    assert group in ('internal', 'external')
    original = next(o for o in protocol['opponents'] if o['id'] == args.opponent)
    opponent = args.pool.resolve() / ('agents' if group == 'internal' else 'opponents') / name / 'main.py'
    assert opponent.is_file(), opponent
    for name, digest in original['files'].items():
        assert hashlib.sha256((opponent.parent / name).read_bytes()).hexdigest() == digest, name
    expected = None
    receipts=['holdout_v1_games.jsonl.gz','paired_external_v1_on_v2_games.jsonl.gz'] if old else ['holdout_games.jsonl.gz']
    for receipt in receipts:
        if not (HERE/'evidence'/receipt).is_file():continue
        with gzip.open(HERE/'evidence'/receipt, 'rt', encoding='utf-8') as f:
            for line in f:
                row = json.loads(line)
                if (row['public'], row['seed'], row['local_seat']) == (args.opponent, seed, args.seat):
                    expected = row
                    break
        if expected:break
    max_call = 0
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        own = Policy(str(HERE / ('v1/main.py' if old else 'v2/main.py')), 'fusion_release')
        rival = Policy(str(opponent), 'release_rival')
        env = LocalGame(seed, load_engine())
        while not env.done:
            actions = [None, None]
            for seat, policy in ((args.seat, own), (1-args.seat, rival)):
                before = time.monotonic()
                actions[seat] = policy(env.observation(seat), copy.deepcopy(env.configuration))
                if seat == args.seat:
                    max_call = max(max_call, time.monotonic()-before)
            env.advance(actions)
        cash = [f['money'] for f in env.state[0].observation.farms]
    assert env.t == 719
    result = dict(opponent=args.opponent, seed=seed, seat=args.seat, steps=env.t,
                  own_cash=cash[args.seat], opponent_cash=cash[1-args.seat], max_action_s=max_call)
    if expected:
        result['recorded_result_matches'] = (cash[args.seat], cash[1-args.seat]) == (expected['local_cash'], expected['public_cash'])
        assert result['recorded_result_matches'], result
    print(json.dumps(result))


if __name__ == '__main__':
    main()
