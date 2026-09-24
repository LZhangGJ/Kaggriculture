"""Frozen live-policy evaluation; independent reproducible opponent sampling."""
from pathlib import Path
import argparse
import concurrent.futures as cf
import contextlib
import copy
import hashlib
import json
import multiprocessing as mp
import os
import random
import sys
import time
import traceback

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FUSION = ROOT / 'experiments/a06_dual_panel_fusion_20260924'
VERIFY = ROOT / 'dp/AFS_R2_DP_Fusion_R3_Experimental_Delivery_ver b/Kaggriculture_Fusion_R2_20260916/verification'
sys.path.insert(0, str(VERIFY))
from policy_host import Policy, LocalGame, load_engine

def read(path): return json.loads(path.read_text(encoding='utf-8'))
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path, obj): path.write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
def hashes(folder):
    return {p.relative_to(folder).as_posix():sha(p) for p in sorted(folder.rglob('*'))
        if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc' and 'build' not in p.relative_to(folder).parts}
def key(row): return row['candidate'],row['opponent'],row['seed'],row['seat']

def evaluate(job):
    candidate, own_entry, opponent, rival_entry, seed, seat, actor_seed = job
    row = dict(candidate=candidate, opponent=opponent, seed=seed, seat=seat, actor_seed=actor_seed, error=None)
    started = time.monotonic()
    try:
        import torch
        import numpy as np
        torch.set_num_threads(1)
        torch.manual_seed(actor_seed)
        np.random.seed(actor_seed)
        random.seed(actor_seed)
        with open(os.devnull, 'w') as sink, contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            own = Policy(ROOT / own_entry, 'candidate_entry')
            rival = Policy(ROOT / rival_entry, 'teammate_entry')
            env = LocalGame(seed, load_engine())
            maxima = [0., 0.]
            noninitial = [0., 0.]
            action_digest = [hashlib.sha256(), hashlib.sha256()]
            while not env.done:
                assert env.configuration.seed is None
                actions = [None, None]
                for player, policy in ((seat, own), (1-seat, rival)):
                    tick = time.monotonic()
                    actions[player] = policy(env.observation(player), copy.deepcopy(env.configuration))
                    duration = time.monotonic()-tick
                    maxima[player] = max(maxima[player], duration)
                    if env.t > 0: noninitial[player] = max(noninitial[player], duration)
                    action_digest[player].update(json.dumps(actions[player], sort_keys=True, separators=(',',':')).encode())
                env.advance(actions)
            module = rival.fn.__globals__
            instance = module['_instances'][1-seat]
            student = getattr(instance, 'dynamic', instance)
            audit = student.student_summary()
            row['opponent_audit'] = {k:v for k,v in audit.items() if k != 'event_trace'}
            row['opponent_sample'] = student.sample
            assert student.sample is True, 'Submission sample behavior changed'
            assert audit['days']==17 and audit['fallbacks']==0 and audit['illegal']==0, 'Student silently fell back'
            assert audit['successful_steps']==list(range(288,673,24))
            assert env.t == 719
            cash = [farm['money'] for farm in env.state[0].observation.farms]
            margin = cash[seat]-cash[1-seat]
            row.update(steps=env.t, own_cash=cash[seat], opponent_cash=cash[1-seat], margin=margin,
                win=int(margin>0),tie=int(margin==0),max_own_action_s=maxima[seat],
                max_opponent_action_s=maxima[1-seat],max_own_noninitial_s=noninitial[seat],
                max_opponent_noninitial_s=noninitial[1-seat],
                own_actions_sha256=action_digest[seat].hexdigest(),opponent_actions_sha256=action_digest[1-seat].hexdigest())
    except Exception:
        row['error'] = traceback.format_exc()
    row['seconds'] = time.monotonic()-started
    return row

def freeze(opponent):
    seed_path = HERE/'SEEDS.json'
    if not seed_path.exists():
        seed_rng = random.Random(2026092441)
        seeds = seed_rng.sample(range(100_000_000,900_000_000),100)
        old=set()
        for path in (FUSION/'SEEDS.json', FUSION/'SEEDS_V2.json'):
            for value in read(path).values():
                if isinstance(value,list):old.update(x for x in value if isinstance(x,int))
        assert not old.intersection(seeds)
        actor_rng = random.Random(2026092442)
        save(seed_path,dict(seeds=seeds,actor_seeds={f'{seed}:{seat}':actor_rng.randrange(1,2**31)
             for seed in seeds for seat in (0,1)},disjoint_from_fusion_prior_seed_files=True,
             policy_rng='Independent Torch/NumPy/Python RNG seeds, shared across V1/V2; environment seed never sent to policies.'))
    candidates={}
    for name, rel in [('V1','versions/v1/FINAL_FREEZE.json'),('V2','FINAL_FREEZE_V2.json')]:
        frozen=read(FUSION/rel); folder=FUSION/'candidates'/frozen['candidate']
        assert hashes(folder)==frozen['files']
        candidates[name]=dict(id=frozen['candidate'],entry=(folder/'main.py').relative_to(ROOT).as_posix(),files=frozen['files'])
    rival=HERE/'opponents'/opponent
    assert (rival/'main.py').is_file()
    if opponent in ('student_v306','student_v463'):
        receipt_name={'student_v306':'V306_DOWNLOAD.json','student_v463':'V463_DOWNLOAD.json'}[opponent]
        assert hashes(rival)==read(HERE/'source_evidence'/receipt_name)['files']
    protocol=dict(opponent=opponent,entry=(rival/'main.py').relative_to(ROOT).as_posix(),files=hashes(rival),
        candidates=candidates,randomization=read(seed_path),games_per_candidate=200,total_games=400,
        referee={p.relative_to(VERIFY).as_posix():sha(p) for p in (VERIFY/'policy_host.py', VERIFY/'referee/cpu_runtime.py',
          VERIFY/'referee/official/kaggriculture.py',VERIFY/'referee/official/seed_utils.py')},
        method='Official 1.32.7 Python interpreter, live unchanged submission, 100 common environment seeds, both seats; no Kaggle timeout sandbox.')
    path=HERE/f'PROTOCOL_{opponent}.json'
    if path.exists(): assert read(path)==protocol
    else: save(path,protocol)
    return protocol

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=('smoke','run','serial'))
    parser.add_argument('--opponent',required=True)
    parser.add_argument('--workers',type=int,default=16)
    parser.add_argument('--seed',type=int)
    parser.add_argument('--seat',type=int,choices=(0,1))
    parser.add_argument('--candidate',choices=('V1','V2'))
    parser.add_argument('--out')
    args=parser.parse_args()
    assert 1<=args.workers<=16
    protocol=freeze(args.opponent)
    seeds=protocol['randomization']['seeds']
    if args.mode=='smoke': seeds=seeds[:1]
    if args.seed is not None: assert args.mode=='serial' and args.seed in seeds; seeds=[args.seed]
    jobs=[(name,candidate['entry'],args.opponent,protocol['entry'],seed,seat,
            protocol['randomization']['actor_seeds'][f'{seed}:{seat}'])
          for seed in seeds for seat in (0,1) for name,candidate in protocol['candidates'].items()
          if (args.candidate is None or name==args.candidate) and (args.seat is None or seat==args.seat)]
    out=HERE/'runs'/(args.out or f'{args.mode}_{args.opponent}')
    out.mkdir(parents=True,exist_ok=True)
    log=out/'games.jsonl'
    rows=[json.loads(line) for line in log.read_text().splitlines() if line] if log.exists() else []
    done={key(r) for r in rows}; assert len(done)==len(rows)
    jobs_pending=[j for j in jobs if (j[0],j[2],j[4],j[5]) not in done]
    print(json.dumps(dict(total=len(jobs),done=len(rows),remaining=len(jobs_pending))),flush=True)
    started=time.monotonic()
    with log.open('a',encoding='utf-8') as stream:
        for offset in range(0,len(jobs_pending),128):
            with cf.ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('spawn'),max_tasks_per_child=1) as executor:
                for result in cf.as_completed([executor.submit(evaluate,j) for j in jobs_pending[offset:offset+128]]):
                    row=result.result(); rows.append(row); stream.write(json.dumps(row,ensure_ascii=False)+'\n');stream.flush()
                    if len(rows)%16==0 or row['error'] or len(rows)==len(jobs):
                        print(json.dumps(dict(done=len(rows),total=len(jobs),errors=sum(bool(r['error'])for r in rows),
                            seconds=round(time.monotonic()-started,1))),flush=True)
                    if row['error']: print(row['error'],flush=True)
    summary=dict(games=len(rows),expected=len(jobs),complete=sum(r.get('steps')==719 for r in rows),
        errors=sum(bool(r['error'])for r in rows),by_candidate={})
    for name in protocol['candidates']:
        group=[r for r in rows if r['candidate']==name and not r['error']]
        if group:summary['by_candidate'][name]=dict(games=len(group),wins=sum(r['win'] for r in group),
            ties=sum(r['tie']for r in group),mean_cash=sum(r['own_cash']for r in group)/len(group),
            mean_margin=sum(r['margin']for r in group)/len(group))
    save(out/'SUMMARY.json',summary); print(json.dumps(summary),flush=True)
    assert summary['games']==summary['complete']==summary['expected'] and not summary['errors']

if __name__=='__main__':main()
