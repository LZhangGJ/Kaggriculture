"""Resumable official-engine seed pairs, outside the training GPUs."""
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
import uuid
import numpy as np
from ppo.league_runtime import atomic,digest
from ppo.league import sha

AGENTS={}

def validate_panel(panel):
    names=[]
    for opp in panel['opponents']:
        names.append(opp['name'])
        seeds=opp.get('seeds',panel.get('seeds',[]))
        if not seeds or len(set(seeds))!=len(seeds):raise ValueError('Empty or duplicate evaluation seeds')
        if opp.get('kind')=='pass':continue
        if opp.get('kind')=='checkpoint':
            if not opp.get('path') or not opp.get('sha256'):raise ValueError('Missing neural opponent identity')
        elif opp.get('archive'):
            if not opp.get('file') or not opp.get('manifest'):raise ValueError('Missing archive opponent fields')
        else:raise ValueError('Unsupported opponent descriptor: '+opp['name'])
    if len(names)!=len(set(names)) or not names:raise ValueError('Empty or duplicate opponent names')
    return panel

def runtime_identity(arena):
    from cache_identity import engine_identity
    root=Path(__file__).resolve().parent.parent
    files=list(root.glob('*.py'))+list((root/'ppo').glob('*.py'))+list((Path(arena)/'tools/arena').glob('*.py'))
    return digest(dict(files={str(p):sha(p) for p in sorted(files)},engine=engine_identity()))

def initialize(arena):
    import torch
    torch.set_num_threads(1)
    sys.path.insert(0,arena)

def agent(spec,role='candidate'):
    from exact_decoder import ExactAgent
    key=(role,spec['sha256'])
    if key not in AGENTS:
        if sha(spec['path'])!=spec['sha256']:raise ValueError('Evaluation checkpoint changed')
        if len(AGENTS)>=3:AGENTS.pop(next(iter(AGENTS)))
        AGENTS[key]=ExactAgent.from_checkpoint(spec['path'],device='cpu',greedy=True)
    AGENTS[key].reset()
    return AGENTS[key]

def seed_pair(job):
    from tools.arena.worker import AgentProcess,official_game
    from tools.arena.sandbox import docker_args,cleanup
    candidate,opp,seed,cfg,output=job
    rows=[]
    for seat in (0,1):
        proc=None;name='league-eval-'+digest([str(output),opp['name'],seed,seat])[:16]
        try:
            policy=agent(candidate)
            if opp.get('kind')=='checkpoint':
                # Candidate and opponent can share weights, but never recurrent state.
                other_agent=agent(opp,'opponent')
                other=lambda obs,config:other_agent.act(obs)[0]
            elif opp.get('archive'):
                if sha(opp['file'])!=opp['archive']:raise ValueError('Opponent archive changed')
                manifest=Path(output)/(name+'.json');atomic(manifest,opp['manifest']);manifest.chmod(0o644)
                args=docker_args(cfg,name,[(Path(opp['file']),'/input.zip'),(manifest,'/manifest.json')],
                    ['python','/opt/arena/inside.py','run',str(seed)],512)
                args.insert(2,'-i');proc=AgentProcess(args,cfg['turn_timeout_seconds']);proc.ready(120);other=proc
            elif opp.get('kind')=='pass':other=lambda obs,config:dict(farmer=['PASS'],hands=[],market=[])
            else:raise ValueError('Unsupported opponent descriptor: '+opp['name'])
            players=[other,other];players[seat]=lambda obs,config:policy.act(obs)[0]
            start=time.monotonic()
            result=official_game(players,dict(seed=seed),dict(cfg['contract'],game_timeout_seconds=600))
            good=(result.get('terminal') and result.get('reason')=='terminal' and result.get('statuses')==['DONE','DONE']
                  and len(result.get('cash',[]))==2 and all(math.isfinite(x) for x in result['cash']))
            cash=result.get('cash')
            score=(.5 if cash[0]==cash[1] else float(cash[seat]>cash[1-seat])) if good else None
            result.update(opponent=opp['name'],seed=seed,seat=seat,score=score,valid=bool(good),
                checkpoint_sha256=candidate['sha256'],opponent_sha256=opp.get('sha256',opp.get('archive')),
                seconds=time.monotonic()-start)
            rows.append(result)
        except Exception as exc:
            rows.append(dict(opponent=opp['name'],seed=seed,seat=seat,valid=False,score=None,error=repr(exc)))
        finally:
            if proc:proc.close()
            cleanup(name)
    return rows

def summarize(rows,panel):
    expected={(o['name'],s,t) for o in panel['opponents'] for s in o.get('seeds',panel.get('seeds',[])) for t in (0,1)}
    keys=[(r['opponent'],r['seed'],r['seat']) for r in rows]
    if len(keys)!=len(set(keys)) or set(keys)!=expected:raise ValueError('Duplicate/missing/unexpected evaluation assignments')
    result={}
    for o in panel['opponents']:
        rr=[r for r in rows if r['opponent']==o['name']];good=[r for r in rr if r['valid']]
        result[o['name']]=dict(games=len(rr),valid=len(good),failures=len(rr)-len(good),
            wins=sum(r['score']==1 for r in good),draws=sum(r['score']==.5 for r in good),losses=sum(r['score']==0 for r in good),
            mean_cash=float(np.mean([r['cash'][r['seat']] for r in good])) if good else None,
            mean_opponent_cash=float(np.mean([r['cash'][1-r['seat']] for r in good])) if good else None,
            mean_margin=float(np.mean([r['cash'][r['seat']]-r['cash'][1-r['seat']] for r in good])) if good else None,
            score=float(np.mean([r['score'] for r in good])) if good else None)
    return dict(opponents=result,failures=sum(not r['valid'] for r in rows))

def evaluate(candidate,panel,output,arena,stop=None):
    validate_panel(panel)
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    request=dict(candidate=candidate,panel=panel,decoding='greedy',runtime_sha=runtime_identity(arena))
    request_id=digest(request)
    if (output/'request.json').exists():
        if json.loads((output/'request.json').read_text())!=request:raise ValueError('Evaluation request mismatch')
    else:atomic(output/'request.json',request)
    jobs=[];completed=[]
    for opp in panel['opponents']:
        for seed in opp.get('seeds',panel.get('seeds',[])):
            key=digest([opp['name'],seed]);receipt=output/(key+'.json')
            if receipt.exists():completed.extend(json.loads(receipt.read_text()))
            else:jobs.append((candidate,opp,seed,panel['config'],str(output)))
    if jobs:
        pool=mp.get_context('spawn').Pool(4,initialize,(arena,))
        active_jobs=[]
        try:
            for start in range(0,len(jobs),4):
                if stop and Path(stop).exists():raise InterruptedError('Evaluation explicitly stopped')
                # At most four pair jobs are queued; timeout kills the isolated pool.
                active_jobs=jobs[start:start+4]
                pending=[pool.apply_async(seed_pair,(j,)) for j in active_jobs]
                for job,future in zip(active_jobs,pending):
                    rows=future.get(timeout=1500)
                    atomic(output/(digest([job[1]['name'],job[2]])+'.json'),rows);completed.extend(rows)
        finally:
            pool.terminate();pool.join()
            # A killed worker cannot execute its own finally block.
            sys.path.insert(0,arena)
            from tools.arena.sandbox import cleanup
            for job in active_jobs:
                for seat in (0,1):cleanup('league-eval-'+digest([job[4],job[1]['name'],job[2],seat])[:16])
    summary=summarize(completed,panel)
    summary.update(request_id=request_id,checkpoint_sha256=candidate['sha256'],panel=panel)
    atomic(output/'summary.json',summary)
    (output/'games.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in completed))
    return summary,completed

def paired_stats(rows,opponent):
    groups={}
    for r in rows:
        if r['opponent']!=opponent:continue
        if not r['valid']:raise ValueError('Invalid confirmation game')
        group=groups.setdefault(r['seed'],{})
        if r['seat'] in group:raise ValueError('Duplicate confirmation assignment')
        group[r['seat']]=r['score']
    if any(set(x)!={0,1} for x in groups.values()):raise ValueError('Unpaired seed')
    x=np.array([sum(v.values())/2 for v in groups.values()])
    if len(x)==0:raise ValueError('Empty confirmation')
    mean=float(x.mean());radius=math.sqrt(math.log(20)/(2*len(x)))
    return dict(pairs=len(x),score=mean,lower=max(0,mean-radius),bound='one_sided_lower',
                alpha=.05,scope='per_comparison',accepted=len(x)==256 and mean-radius>.5)

def regressions(a,b):
    if a['failures'] or b['failures']:return ['invalid_games']
    keys=set(a['opponents'])
    if keys!=set(b['opponents']):raise ValueError('Unmatched screen')
    differences={k:a['opponents'][k]['score']-b['opponents'][k]['score'] for k in keys}
    bad=[k for k,v in differences.items() if v<-.125]
    real=[v for k,v in differences.items() if k!='PASS']
    if real and sum(real)/len(real)<-.125:bad.extend(keys-{'PASS'})
    return sorted(set(bad))
