"""Original public policies versus frozen R2, on the official live referee.

No notebook execution, no opponent tape, no hidden future input. One fresh
opponent module graph and R2 context per game. Run using .venv_wsl_cpp Python.
"""
from pathlib import Path
import argparse
import concurrent.futures as futures
import copy
import gzip
import hashlib
import importlib.util
import json
import multiprocessing
import os
import statistics
import sys
import time
import traceback

os.environ['OMP_NUM_THREADS'] = '1'
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = ROOT / 'experiments/shop_router_0909_vs_r2_200_20260909'
sys.path.insert(0, str(BASE))
import run_match as old

ENGINE = R2_MODULE = None


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def init_worker():
    global ENGINE, R2_MODULE
    ENGINE = old.load_engine()
    R2_MODULE = old.module(old.R2 / 'agent.py', 'frozen_r2_public11')
    assert sha(old.R2 / 'agent.so') == old.EXPECTED[old.R2 / 'agent.so']


class FreshPublic:
    """Reset all agent-owned module state, including modules named policy/main."""
    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self.previous_cwd = Path.cwd()
        self.previous_path = sys.path[:]
        self.names = {p.stem for p in self.directory.glob('*.py')}
        self.displaced = {n: sys.modules.pop(n) for n in self.names if n in sys.modules}
        sys.path.insert(0, str(self.directory))
        os.chdir(self.directory)
        # Some public modules use dataclasses or import their own module by name.
        spec = importlib.util.spec_from_file_location('main', self.directory / 'main.py')
        self.module = importlib.util.module_from_spec(spec)
        sys.modules['main'] = self.module
        try:
            spec.loader.exec_module(self.module)
            self.call = self.module.agent
            assert callable(self.call)
        except Exception:
            self.close()
            raise

    def close(self):
        for name, mod in list(sys.modules.items()):
            fn = getattr(mod, '__file__', None)
            if name in self.names or (fn and Path(fn).resolve().is_relative_to(self.directory)):
                sys.modules.pop(name, None)
        sys.modules.update(self.displaced)
        sys.path[:] = self.previous_path
        os.chdir(self.previous_cwd)


def game(job):
    opponent, directory, seed, opponent_seat, output, config, binary = job
    start = time.perf_counter()
    public = r2 = None
    row = dict(opponent=opponent, seed=seed, opponent_seat=opponent_seat)
    try:
        public = FreshPublic(directory)
        r2 = R2_MODULE.Agent(config=config, binary_path=binary)
        env = old.LocalGame(seed, ENGINE)
        actions, days = [], []
        latency = {'opponent': [], 'r2': []}
        while not env.done:
            step = env.t
            observations = [env.observation(0), env.observation(1)]
            if step % 24 == 0:
                days.append({'step': step, 'observations': copy.deepcopy(observations), 'r2_debug': r2.debug()})
            out = [None, None]
            tick = time.perf_counter()
            # The official interpreter has already stripped the private seed.
            assert env.configuration.seed is None
            out[opponent_seat] = public.call(observations[opponent_seat], copy.deepcopy(env.configuration))
            latency['opponent'].append(time.perf_counter() - tick)
            tick = time.perf_counter()
            out[1-opponent_seat] = r2(observations[1-opponent_seat])
            latency['r2'].append(time.perf_counter() - tick)
            assert all(isinstance(a, dict) and len(a.get('market', [])) <= 10 for a in out)
            actions.append(copy.deepcopy(out))
            env.advance(out)
            assert env.t <= 719
        assert env.t == 719 and all(s.status == 'DONE' for s in env.state)
        final = [env.observation(0), env.observation(1)]
        days.append({'step': env.t, 'observations': final, 'r2_debug': r2.debug()})
        money = [final[0]['farms'][s]['money'] for s in (0, 1)]
        margin = money[1-opponent_seat] - money[opponent_seat]
        row.update(steps=env.t, r2_cash=money[1-opponent_seat], opponent_cash=money[opponent_seat],
                   r2_margin=margin, r2_win=margin > 0, opponent_win=margin < 0, tie=margin == 0,
                   runtime_error=None, shops=final[0]['town']['unlocked_shops'],
                   joint_action_sha256=hashlib.sha256(json.dumps(actions, separators=(',', ':')).encode()).hexdigest(),
                   seconds=time.perf_counter()-start,
                   latency={k: dict(total=sum(v), maximum=max(v), over_1s=sum(x > 1 for x in v))
                            for k, v in latency.items()})
        target = Path(output) / 'traces' / opponent / f'{seed}_seat{opponent_seat}.json.gz'
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(gzip.compress(json.dumps(dict(result=row, actions=actions, days=days),
                                                   separators=(',', ':')).encode(), compresslevel=1))
        row['trace'] = str(target.relative_to(output))
    except Exception:
        row.update(runtime_error=traceback.format_exc(), seconds=time.perf_counter()-start)
    finally:
        if r2 is not None:
            r2.close()
        if public is not None:
            public.close()
    return row


def summarize(rows):
    good = [r for r in rows if not r['runtime_error']]
    result = {'games': len(rows), 'valid_games': len(good), 'errors': len(rows)-len(good)}
    if good:
        result.update(r2_wins=sum(r['r2_win'] for r in good), losses=sum(r['opponent_win'] for r in good),
                      ties=sum(r['tie'] for r in good), r2_win_rate=sum(r['r2_win'] for r in good)/len(rows),
                      r2_mean_cash=statistics.mean(r['r2_cash'] for r in good),
                      opponent_mean_cash=statistics.mean(r['opponent_cash'] for r in good),
                      mean_margin=statistics.mean(r['r2_margin'] for r in good),
                      median_margin=statistics.median(r['r2_margin'] for r in good))
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pool', type=Path, required=True, help='JSON list: id, working (relative to ROOT)')
    p.add_argument('--names', default='')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed-start', type=int, default=2609110000)
    p.add_argument('--seeds', type=int, default=100)
    p.add_argument('--workers', type=int, default=16)
    p.add_argument('--config', type=Path)
    p.add_argument('--binary', type=Path)
    p.add_argument('--pilot', action='store_true')
    args = p.parse_args()
    assert 1 <= args.workers <= 16
    pool = json.loads(args.pool.read_text(encoding='utf-8'))
    if args.names:
        wanted = set(args.names.split(','))
        pool = [item for item in pool if item['id'] in wanted]
        assert {item['id'] for item in pool} == wanted
    assert pool and len({x['id'] for x in pool}) == len(pool)
    args.output = args.output.resolve()
    config = None
    if args.config:
        config = json.loads((old.R2/'config.json').read_text())
        config.update(json.loads(args.config.read_text()))
    binary = str(args.binary.resolve()) if args.binary else None
    seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    source_hashes = {str(p.relative_to(ROOT)): sha(p) for x in pool
                     for p in (ROOT / x['working']).rglob('*')
                     if p.is_file() and p.suffix in {'.py', '.json', '.so'} and '__pycache__' not in str(p)}
    protocol = dict(pool=pool, source_hashes=source_hashes, seeds=seeds, seats=[0, 1],
                    config=config, r2_config_sha256=sha(old.R2/'config.json'),
                    r2_binary_sha256=sha(old.R2/'agent.so'), r2_wrapper_sha256=sha(old.R2/'agent.py'),
                    engine_sha256=sha(old.REF/'official/kaggriculture.py'), host_sha256=sha(old.REF/'cpu_runtime.py'),
                    official_version='1.32.7', real_time_opponents=True,
                    limitation='Official local interpreter, not Kaggle sandbox/CPU/schema qualification')
    if binary:
        protocol['candidate_binary'] = binary
        protocol['candidate_binary_sha256'] = sha(binary)
    protocol_path = args.output/'PROTOCOL.json'
    if protocol_path.exists():
        assert json.loads(protocol_path.read_text()) == protocol, 'Frozen protocol changed'
    else:
        save(protocol_path, protocol)
    init_worker()
    jobs = [(x['id'], str(ROOT/x['working']), seed, seat, str(args.output), config, binary)
            for seed in seeds for x in pool for seat in (0, 1)]
    if args.pilot:
        rows, checks = [], []
        for x in pool:
            pair = [j for j in jobs if j[0] == x['id'] and j[2] == seeds[0]]
            got = [game(j) for j in pair]
            repeat = game(pair[0])
            fields = ['joint_action_sha256', 'r2_cash', 'opponent_cash', 'steps']
            passed = not any(r['runtime_error'] for r in got + [repeat]) and all(repeat[k] == got[0][k] for k in fields)
            checks.append(dict(opponent=x['id'], reset_pass=passed, repeat=repeat))
            rows.extend(got)
            print(json.dumps(dict(opponent=x['id'], rows=[{k:v for k,v in r.items() if k not in {'shops','latency'}} for r in got], reset_pass=passed)), flush=True)
            save(args.output/'PILOT.json', dict(rows=rows, checks=checks))
            assert passed, 'Pilot or reset failed; do not count failure as R2 win'
        return
    assert not (args.output/'RESULTS.json').exists(), 'Completed panel exists'
    pilot = json.loads((args.output/'PILOT.json').read_text())
    assert len(pilot['rows']) == 2*len(pool) and all(c['reset_pass'] for c in pilot['checks'])
    partial = args.output/'rows.partial.json'
    rows = json.loads(partial.read_text()) if partial.exists() else pilot['rows']
    assert not any(r['runtime_error'] for r in rows)
    done = {(r['opponent'],r['seed'],r['opponent_seat']) for r in rows}
    pending = [j for j in jobs if (j[0],j[2],j[3]) not in done]
    started = time.perf_counter()
    with futures.ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn'),
                                    initializer=init_worker) as executor:
        tasks = [executor.submit(game, j) for j in pending]
        for future in futures.as_completed(tasks):
            row = future.result()
            if row['runtime_error']:
                save(args.output/'FAILED.json', row)
                for task in tasks:
                    task.cancel()
                raise RuntimeError(row['runtime_error'])
            rows.append(row)
            if len(rows) % 40 == 0 or len(rows) == len(jobs):
                save(partial, rows)
                progress = dict(done=len(rows), total=len(jobs), seconds_this_resume=time.perf_counter()-started,
                                by_opponent={x['id']: summarize([r for r in rows if r['opponent']==x['id']]) for x in pool})
                save(args.output/'PROGRESS.json', progress)
                print(json.dumps(progress), flush=True)
    assert len(rows) == len(jobs) and len({(r['opponent'],r['seed'],r['opponent_seat']) for r in rows}) == len(jobs)
    assert all(sha(ROOT/p) == h for p,h in source_hashes.items())
    rows.sort(key=lambda r: (r['opponent'], r['seed'], r['opponent_seat']))
    save(args.output/'rows.json', rows)
    by = {x['id']: summarize([r for r in rows if r['opponent']==x['id']]) for x in pool}
    result = dict(status='BASELINE_VALIDATED', by_opponent=by, overall=summarize(rows),
                  macro_win_rate=statistics.mean(v['r2_win_rate'] for v in by.values()),
                  seconds_this_resume=time.perf_counter()-started, total_steps=sum(r['steps'] for r in rows))
    save(args.output/'RESULTS.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
