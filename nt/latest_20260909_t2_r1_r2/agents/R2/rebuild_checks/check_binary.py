"""Fail-closed release-conformance check, no third-party Python dependency.
Each fixture uses a fresh native Agent. Only current official observations reach
it; frozen actions are used by this checker, never by the policy. This is trace
conformance, not a new opponent benchmark or a proof over untested states.
"""
from __future__ import annotations
import argparse, gzip, hashlib, importlib.util, json, sys, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
ROOT = Path(__file__).resolve().parents[1]
CHECKS = Path(__file__).resolve().parent

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def one(args: tuple[str, str, dict]) -> dict:
    binary, filename, entry = args
    sys.path.insert(0, str(CHECKS / 'referee'))
    from cpu_runtime import LocalGame
    spec = importlib.util.spec_from_file_location('t2_r2_checked_policy', ROOT / 'policy/agent.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    trace = json.loads(gzip.decompress((CHECKS / 'fixtures' / filename).read_bytes()))
    seat, seed = trace['seat'], trace['seed']
    row = dict(fixture=filename, seed=seed, seat=seat, opponent=trace['opponent'],
               status='FAIL', checked_actions=0)
    start = time.perf_counter()
    agent = None
    try:
        game = LocalGame(seed)
        config = json.loads((ROOT / 'policy/config.json').read_text())
        agent = module.Agent(config=config, binary_path=binary)
        for step, actions in enumerate(trace['actions']):
            got = agent(game.observation(seat), game.configuration)
            if got != actions[seat]:
                row.update(first_divergence_step=step, actual=got, expected=actions[seat],
                           debug=agent.debug(), reason='ACTION_MISMATCH')
                return row
            row['checked_actions'] += 1
            game.advance(actions)
        farms = game.observation(seat)['farms']
        row.update(cash=farms[seat]['money'], opponent_cash=farms[1-seat]['money'], done=game.done)
        if not game.done or row['cash'] != entry['cash'] or row['opponent_cash'] != entry['opponent_cash']:
            row.update(reason='FINAL_STATE_MISMATCH', expected_cash=entry['cash'],
                       expected_opponent_cash=entry['opponent_cash'])
        else:
            row['status'] = 'PASS'
    except Exception as exc:
        row.update(reason='CHECK_ERROR', error=f'{type(exc).__name__}: {exc}')
    finally:
        if agent is not None:
            agent.close()
        row['seconds'] = time.perf_counter() - start
    return row

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--binary', required=True, type=Path)
    ap.add_argument('--report', required=True, type=Path)
    ap.add_argument('--workers', type=int, default=2)
    args = ap.parse_args()
    if args.report.exists():
        raise FileExistsError('Refusing to overwrite a previous validation report')
    manifest = json.loads((CHECKS / 'manifest.json').read_text())
    binary = args.binary.resolve(strict=True)
    start = time.perf_counter()
    mismatches = []
    for rel, digest in manifest['file_hashes'].items():
        p = ROOT / rel
        if not p.is_file() or sha(p) != digest:
            mismatches.append(rel)
    if mismatches:
        result = dict(status='FAIL', reason='FIXTURE_OR_WRAPPER_CHANGED', files=mismatches)
    else:
        jobs = [(str(binary), name, entry) for name, entry in manifest['fixtures'].items()]
        if args.workers == 1:
            rows = [one(job) for job in jobs]
        else:
            with ProcessPoolExecutor(max_workers=max(1, min(args.workers, 4))) as executor:
                rows = list(executor.map(one, jobs))
        result = dict(status='PASS' if all(r['status'] == 'PASS' for r in rows) else 'FAIL',
                      scope='8 frozen T2-R2 focus-case trajectories; official 1.32.7 state progression; build regression, not new live strength validation',
                      fixtures=len(rows), checked_actions=sum(r['checked_actions'] for r in rows),
                      failures=sum(r['status'] != 'PASS' for r in rows), rows=rows)
    result.update(binary=str(binary), binary_sha256=sha(binary), seconds=time.perf_counter()-start,
                  manifest_sha256=sha(CHECKS/'manifest.json'))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open('x', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows',)}, ensure_ascii=False), flush=True)
    return 0 if result['status'] == 'PASS' else 1

if __name__ == '__main__':
    raise SystemExit(main())
