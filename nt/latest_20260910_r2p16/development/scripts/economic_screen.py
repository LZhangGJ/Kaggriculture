"""Paired, real-opponent development ablations; never future-oracle routing.

Every option is a single public-state rule parameterization held for a whole
game. The original release binary is used. All 11 source-frozen opponents,
equal seeds and seats are evaluated for every option. No holdout seeds here.
"""
from pathlib import Path
import argparse
import concurrent.futures as futures
import json
import multiprocessing
import statistics
import time

import run_panel as panel

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OPTIONS = {
    'cash_time03': {'discount': .03},
    'cash_time08': {'discount': .08},
    'capital08': {'capital_power': .8},
    'work2': {'work_price': 2},
    'hours14': {'labor_hours': 14},
    'crop135': {'crop_bias': 1.35},
    'cash_labor': {'discount': .03, 'work_price': 2, 'labor_hours': 14},
    'externality1': {'competition': 1},
    'externality3': {'competition': 3},
    'externality4': {'competition': 4},
    'supply125': {'supply': 1.25},
    'rival_replant': {'replant': .7},
    'horizon2': {'scenario': 2},
    'work6': {'work_price': 6},
    'hours8': {'labor_hours': 8},
    'margin_labor': {'competition': 3, 'work_price': 6, 'labor_hours': 8},
    'delivery_pressure': {'batch_delivery': 2, 'feed_finance': 0},
}


def run(names, seed_count, output, spread=False, binary=None):
    pool = json.loads((HERE/'pool_all11.json').read_text())
    baseline = {(r['opponent'], r['seed'], r['opponent_seat']): r
                for r in json.loads((HERE/'baseline_rows_all11.json').read_text())}
    settings = json.loads((panel.old.R2/'config.json').read_text())
    source_hashes = {str(p.relative_to(ROOT)): panel.sha(p) for x in pool
                     for p in (ROOT/x['working']).rglob('*')
                     if p.is_file() and p.suffix in {'.py', '.json', '.so'} and '__pycache__' not in str(p)}
    output = output.resolve()
    offsets = [round(i*99/max(1,seed_count-1)) for i in range(seed_count)] if spread else list(range(seed_count))
    assert len(set(offsets))==seed_count
    protocol = dict(options={n: OPTIONS[n] for n in names}, seeds=[2609110000+i for i in offsets],
                    seats=[0, 1], pool=pool, source_hashes=source_hashes, base_config=settings,
                    binary_sha256=panel.sha(panel.old.R2/'agent.so'), engine_sha256=panel.sha(panel.old.REF/'official/kaggriculture.py'),
                    boundary='Development only; no opponent ID/future/seed to agent; original official live interpreter')
    if binary:
        binary=str(binary.resolve())
        protocol.update(candidate_binary=binary,candidate_binary_sha256=panel.sha(binary))
    if (output/'PROTOCOL.json').exists():
        assert json.loads((output/'PROTOCOL.json').read_text()) == protocol
    else:
        panel.save(output/'PROTOCOL.json', protocol)
    panel.init_worker()
    for name in names:
        folder = output/name
        config = dict(settings, **OPTIONS[name])
        if (folder/'RESULTS.json').exists():
            print(json.dumps(dict(name=name, reused=True)), flush=True)
            continue
        panel.save(folder/'CONFIG.json', config)
        jobs = [(x['id'], str(ROOT/x['working']), seed, seat, str(folder), config, binary)
                for seed in protocol['seeds'] for x in pool for seat in (0,1)]
        # Original binary and opponent module reset verified previously; also
        # require this settings variant to be deterministic over a fresh reset.
        if not (folder/'RESET.json').exists():
            a, b = panel.game(jobs[0]), panel.game(jobs[0])
            assert not a['runtime_error'] and not b['runtime_error'], (a,b)
            assert a['joint_action_sha256'] == b['joint_action_sha256']
            panel.save(folder/'RESET.json', dict(first=a, repeat=b, passed=True))
        rows = json.loads((folder/'partial.json').read_text()) if (folder/'partial.json').exists() else []
        done = {(r['opponent'],r['seed'],r['opponent_seat']) for r in rows}
        started = time.perf_counter()
        with futures.ProcessPoolExecutor(max_workers=16, mp_context=multiprocessing.get_context('spawn'),
                                        initializer=panel.init_worker) as executor:
            tasks = [executor.submit(panel.game,j) for j in jobs if (j[0],j[2],j[3]) not in done]
            for f in futures.as_completed(tasks):
                row = f.result()
                assert not row['runtime_error'], row
                rows.append(row)
                if len(rows)%40==0 or len(rows)==len(jobs):
                    panel.save(folder/'partial.json',rows)
                    print(json.dumps(dict(name=name,done=len(rows),total=len(jobs),seconds=time.perf_counter()-started)),flush=True)
        assert len(rows)==len(jobs)
        assert all(panel.sha(ROOT/p)==h for p,h in source_hashes.items())
        rows.sort(key=lambda r:(r['opponent'],r['seed'],r['opponent_seat']))
        paired=[]
        for r in rows:
            b=baseline[(r['opponent'],r['seed'],r['opponent_seat'])]
            paired.append(dict(opponent=r['opponent'],seed=r['seed'],opponent_seat=r['opponent_seat'],old_win=b['r2_win'],new_win=r['r2_win'],
                               cash_delta=r['r2_cash']-b['r2_cash'],margin_delta=r['r2_margin']-b['r2_margin'],
                               same_actions=r['joint_action_sha256']==b['joint_action_sha256']))
        stats=dict(overall=panel.summarize(rows),by_opponent={x['id']:panel.summarize([r for r in rows if r['opponent']==x['id']]) for x in pool},
                   base_wins=sum(r['old_win'] for r in paired),rescued=sum(not r['old_win'] and r['new_win'] for r in paired),
                   lost_wins=sum(r['old_win'] and not r['new_win'] for r in paired),same_actions=sum(r['same_actions'] for r in paired),
                   mean_margin_delta=statistics.mean(r['margin_delta'] for r in paired),
                   mean_cash_delta=statistics.mean(r['cash_delta'] for r in paired),seconds=time.perf_counter()-started)
        panel.save(folder/'rows.json',rows)
        panel.save(folder/'PAIRED.json',paired)
        panel.save(folder/'RESULTS.json',stats)
        print(json.dumps(dict(name=name,result=stats)),flush=True)
    report={n:json.loads((output/n/'RESULTS.json').read_text()) for n in names}
    panel.save(output/'SUMMARY.json',report)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--names',default=','.join(OPTIONS))
    parser.add_argument('--seeds',type=int,default=8)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--spread',action='store_true')
    parser.add_argument('--binary',type=Path)
    args=parser.parse_args()
    assert 1<=args.seeds<=100
    run(args.names.split(','),args.seeds,args.output,args.spread,args.binary)
