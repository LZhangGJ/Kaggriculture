"""All frozen opponents, balanced seats, live official transitions, pure-win gate."""
import argparse
import contextlib
import copy
import hashlib
import io
import json
import multiprocessing as mp
from multiprocessing.connection import wait
import os
from pathlib import Path
import sys
import time
import traceback
ROOT = Path(__file__).resolve().parent
SOURCE = Path('F:/Kaggriculture/experiments/local_teacher_bc_20260917')
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'


def observation_message(obs):
    # Transport plain JSON values, not the referee's process-local AttrDict class.
    return dict(kind='infer', obs=json.loads(json.dumps(obs)))


def worker(connection):
    sys.path.insert(0, str(SOURCE / 'frozen/pool'))
    from run_matches import Policy
    from cpu_runtime import LocalGame, load_engine
    while True:
        job = connection.recv()
        if job is None:
            break
        seed, opponent, seat = job
        row = dict(seed=seed, opponent=opponent, seat=seat, error=None, win=False, tie=False)
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                rival = Policy(SOURCE / f'frozen/pool/opponents/{opponent}/main.py', f'opp_{seed}_{seat}_{opponent}')
                game = LocalGame(seed, load_engine())
                while not game.done:
                    connection.send(observation_message(game.observation(seat)))
                    action = connection.recv()
                    opposing = rival(game.observation(1-seat), copy.deepcopy(game.configuration))
                    pair = [None, None]
                    pair[seat], pair[1-seat] = action, opposing
                    game.advance(pair)
                cash = [f['money'] for f in game.state[0].observation.farms]
                row.update(steps=game.t, student_cash=cash[seat], opponent_cash=cash[1-seat],
                    win=cash[seat] > cash[1-seat], tie=cash[seat] == cash[1-seat], margin=cash[seat]-cash[1-seat])
                assert game.done and game.t == 719
        except Exception:
            row['error'] = traceback.format_exc()
        connection.send(dict(kind='result', row=row))
    connection.close()


def summary(rows, total, checkpoint_hash):
    complete = len(rows) == total
    wins = sum(bool(r['win']) and not r['error'] for r in rows)
    ties = sum(bool(r['tie']) and not r['error'] for r in rows)
    return dict(status='COMPLETE' if complete else 'RUNNING', completed=len(rows), games=total,
        wins=wins, ties=ties, errors=sum(bool(r['error']) for r in rows),
        strict_win_rate=wins/total if complete else None, score_rate=(wins+.5*ties)/total if complete else None,
        meets_target=complete and wins/total > .75 and not any(r['error'] for r in rows),
        checkpoint_sha256=checkpoint_hash,
        opponents={o: dict(games=sum(r['opponent'] == o for r in rows),
            wins=sum(r['opponent'] == o and r['win'] and not r['error'] for r in rows),
            ties=sum(r['opponent'] == o and r['tie'] and not r['error'] for r in rows))
            for o in sorted({r['opponent'] for r in rows})})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--name', required=True)
    ap.add_argument('--panel', choices=('development', 'final'), required=True)
    ap.add_argument('--panel-index', type=int, default=0)
    args = ap.parse_args()
    out = ROOT / 'evaluation' / args.name
    out.mkdir(parents=True, exist_ok=True)
    from cpu_budget import CpuBudget
    guard = CpuBudget(out / 'cpu')
    guard.wait()
    import numpy as np
    import torch
    from common import Agent, load_checkpoint, save
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision('highest')
    torch.cuda.set_per_process_memory_fraction(.20)
    plan = json.loads((ROOT / 'PLAN.json').read_text(encoding='utf8'))
    assert hashlib.sha256((SOURCE / 'frozen/pool/POOL.json').read_bytes()).hexdigest() == plan['pool_sha256']
    for row in json.loads((SOURCE / 'frozen/pool/POOL.json').read_text(encoding='utf8')):
        assert hashlib.sha256((SOURCE / f"frozen/pool/opponents/{row['id']}/main.py").read_bytes()).hexdigest() == row['main_sha256']
    for name, expected in plan['frozen_code'].items():
        assert hashlib.sha256((ROOT / 'frozen_code' / name).read_bytes()).hexdigest() == expected
    jobs = [tuple(job) for job in (plan['development_panels'][args.panel_index] if args.panel == 'development' else plan['final_jobs'])]
    repair_path = ROOT / 'EVALUATION_REPAIR.json'
    repair = json.loads(repair_path.read_text(encoding='utf8')) if repair_path.exists() else None
    if repair and args.panel == 'development' and args.panel_index == 0:
        jobs = [(repair['replacement_seeds'].get(str(seed), seed), opponent, seat) for seed, opponent, seat in jobs]
    seeds = [job[0] for job in jobs]
    assert len(set(seeds)) == len(jobs), 'Every match must have a distinct environment seed'
    checkpoint_hash = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    spec = dict(panel=args.panel, panel_index=args.panel_index, jobs=jobs, seeds=seeds, opponents=plan['opponents'], seats=[0, 1],
        checkpoint_sha256=checkpoint_hash, backend='CUDA FP32, serial batch-one model calls',
        workers=2, threshold=.75, comparison='strictly greater', metric='wins / all scheduled games; draws are not wins; zero runtime errors required',
        runtime='Official interpreter, local host without competition sandbox or timeout enforcement',
        history='Only actual own submitted actions after the audited canonical codec',
        teacher_in_student=False, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        seed_repair_sha256=hashlib.sha256(repair_path.read_bytes()).hexdigest() if repair else None)
    if (out / 'PROTOCOL.json').exists():
        assert json.loads((out / 'PROTOCOL.json').read_text(encoding='utf8')) == json.loads(json.dumps(spec)), 'Resume changed the frozen evaluation'
    else:
        save(out / 'PROTOCOL.json', spec)
    if args.panel == 'final':
        reservation = ROOT / 'FINAL_RESERVATION.json'
        if reservation.exists():
            assert json.loads(reservation.read_text(encoding='utf8')) == json.loads(json.dumps(spec))
        else:
            with reservation.open('x', encoding='utf8') as stream:
                json.dump(spec, stream, indent=2)
    rows = [json.loads(line) for line in (out / 'games.jsonl').read_text(encoding='utf8').splitlines()] if (out / 'games.jsonl').exists() else []
    completed = {(r['seed'], r['opponent'], r['seat']) for r in rows}
    assert len(completed) == len(rows) and completed <= set(jobs)
    started_path = out / 'STARTED.jsonl'
    started_jobs = {tuple(json.loads(line)['job']) for line in started_path.read_text(encoding='utf8').splitlines()} if started_path.exists() else set()
    assert completed <= started_jobs
    assert started_jobs == completed, 'An interrupted seed must be retired before any restart; never replay it'
    pending = [job for job in jobs if job not in completed]
    model, codec, _ = load_checkpoint(args.checkpoint, 'cuda')
    context = mp.get_context('spawn')
    connections, agents, durations = {}, {}, {}
    remaining = iter(pending)
    def dispatch(connection, job):
        if job is not None:
            assert job not in started_jobs
            with started_path.open('a', encoding='utf8') as stream:
                stream.write(json.dumps(dict(job=job, time_unix=time.time()))+'\n')
                stream.flush()
                os.fsync(stream.fileno())
            started_jobs.add(job)
        connection.send(job)

    for _ in range(min(2, len(pending))):
        parent, child = context.Pipe()
        process = context.Process(target=worker, args=(child,), daemon=True)
        process.start()
        child.close()
        connections[parent] = process
        dispatch(parent, next(remaining))
    try:
        with (out / 'games.jsonl').open('a', encoding='utf8') as log:
            while connections:
                ready = wait(list(connections), timeout=10)
                if not ready:
                    assert all(p.is_alive() for p in connections.values()), 'Match worker exited'
                    continue
                for connection in ready:
                    message = connection.recv()
                    if message['kind'] == 'infer':
                        guard.wait()
                        obs = message['obs']
                        if obs['step'] == 0:
                            agents[connection] = Agent(model, codec, 'cuda')
                            durations[connection] = []
                        before = time.perf_counter()
                        action = agents[connection](obs)
                        durations[connection].append(time.perf_counter()-before)
                        connection.send(action)
                    else:
                        row = message['row']
                        measured = durations.get(connection, [])
                        row.update(atomic_plant_cancellations=agents[connection].cancellations if connection in agents else None,
                            inference_p99_ms=float(np.percentile(measured, 99)*1000) if measured else None,
                            inference_max_ms=max(measured)*1000 if measured else None)
                        rows.append(row)
                        log.write(json.dumps(row)+'\n')
                        log.flush()
                        report = summary(rows, len(jobs), checkpoint_hash)
                        save(out / 'SUMMARY.json', report)
                        print(json.dumps({k: v for k, v in report.items() if k != 'opponents'}), flush=True)
                        job = next(remaining, None)
                        agents.pop(connection, None)
                        durations.pop(connection, None)
                        dispatch(connection, job)
                        if job is None:
                            process = connections.pop(connection)
                            process.join(timeout=10)
                            connection.close()
        report = summary(rows, len(jobs), checkpoint_hash)
        save(out / 'SUMMARY.json', report)
        assert report['completed'] == len(jobs) and report['errors'] == 0
    finally:
        for connection, process in connections.items():
            if process.is_alive():
                process.terminate()
            process.join(timeout=10)
            connection.close()


if __name__ == '__main__':
    main()
