"""Replay the slowest causal policy trajectories sequentially, checking every action."""
import argparse
import gzip
import hashlib
import json
import time
from pathlib import Path
from run_panel import PACKAGE, EVIDENCE, module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', default='fresh5_r5')
    parser.add_argument('--count', type=int, default=2)
    args = parser.parse_args()
    root = EVIDENCE / 'runs' / args.tag
    rows = json.loads((root / 'rows.json').read_text())
    selected, seen = [], set()
    for row in sorted(rows, key=lambda r: r['max_seconds'], reverse=True):
        key = row['arm'], row['opponent'], row['seed']
        if row['arm'] != 'economic' or key in seen:
            continue
        selected.append(row); seen.add(key)
        if len(selected) == args.count:
            break
    runtime = module(PACKAGE / 'main.py', 'sequential_timing_entry')
    binary = PACKAGE / 'build/revision5/route3.so'
    result = dict(binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                  protocol='Fresh policy instance; its own full causal observations; no concurrent games in this process.',
                  games=[])
    for row in selected:
        path = root / 'smoke' / (row['name'] + '.json.gz')
        frames = json.loads(gzip.decompress(path.read_bytes()))['steps']
        agent = runtime.create_agent(binary)
        wall, cpu = [], []
        try:
            for step in range(719):
                observation = frames[step][row['seat']]['observation']
                start = time.perf_counter(); cpu_start = time.process_time()
                action = agent(observation)
                cpu.append(time.process_time() - cpu_start)
                wall.append(time.perf_counter() - start)
                assert action == frames[step+1][row['seat']]['action'], (row['name'], step)
        finally:
            agent.close()
        checked = dict(name=row['name'], action_parity='PASS', observations=719,
                       replay_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                       max_wall_seconds=max(wall), max_cpu_seconds=max(cpu),
                       peak_step=wall.index(max(wall)),
                       over_one_second=sum(t>1 for t in wall),
                       local_overage_seconds=sum(max(0,t-1) for t in wall),
                       wall_seconds=wall, cpu_seconds=cpu)
        result['games'].append(checked)
        print(json.dumps({k:v for k,v in checked.items() if k not in {'wall_seconds','cpu_seconds'}}), flush=True)
    output = EVIDENCE / ('sequential_timing_' + args.tag + '.json')
    assert not output.exists(), output
    output.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
