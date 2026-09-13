"""Exact paired local pilot; no existing sealed-panel results are consumed."""
import argparse
import copy
import gzip
import hashlib
import inspect
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor,as_completed
import multiprocessing as mp
import numpy as np
import torch
from . import native
from .runtime import HERE,REPO,RUNTIME,PACKAGE,LocalGame,ENGINE,clean_obs,sha,dump
from .plans import simple_plan,repair
from .policy import PolicyAgent,load
from .protocol import verify_freeze


def opponent(name):
    sys.path.insert(0,str(PACKAGE));import arena
    if name.startswith('native:'):
        directory=RUNTIME/'nt/latest_20260911_afs_workflow_repair_r1_r2'/name.split(':')[1]
        agent=arena.codec.Agent(config=json.loads((directory/'policy/config.json').read_text()),binary_path=directory/'policy/agent.so')
        return lambda obs,cfg:agent(obs),agent.close
    directory=RUNTIME/'research/robust90/opponents'/name.split(':')[1]
    mod=arena.module(directory/'main.py','independent_pilot_opponent');fn=mod.agent
    try:inspect.signature(fn).bind({}, {})
    except TypeError:return lambda obs,cfg:fn(obs),lambda:None
    return fn,lambda:None


def worker(job):
    candidate,seed,rival_name,seat,trace=job;start=time.perf_counter();close=lambda:None
    row={'candidate':candidate['id'],'seed':seed,'opponent':rival_name,'seat':seat,'valid':False}
    def timeout(signum,frame):raise TimeoutError('pilot game exceeded 90 seconds')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(90);torch.set_num_threads(1)
    try:
        call,close=opponent(rival_name)
        if candidate['kind']=='ppo':model,_=load(REPO/candidate['path']);own=PolicyAgent(model)
        else:
            plan=repair(simple_plan()) if candidate['kind']=='control' else json.loads((REPO/candidate['path']).read_text())['plan']
            own=lambda obs:native.plan_action(obs,plan)
        n=native.Batch([seed],1);ref=LocalGame(seed,ENGINE) if trace else None
        actions=[];days=[];lat=[];hashes=hashlib.sha256()
        for t in range(719):
            obs=[n.observation(0,s) for s in (0,1)]
            if ref:
                for s in (0,1):assert obs[s]==clean_obs(ref.observation(s)),(t,s)
                if t%24==0:days.append({'step':t,'observations':obs})
            began=time.perf_counter();a=own(obs[seat]);lat.append(time.perf_counter()-began)
            b=call(obs[1-seat],copy.deepcopy(LocalConfig))
            pair=[None,None];pair[seat]=a;pair[1-seat]=b
            packed=json.dumps(pair,separators=(',',':')).encode();hashes.update(packed+b'\n')
            if trace:actions.append(pair)
            n.primitive_step(0,pair)
            if ref:ref.advance(pair)
        final=n.observation(0,0)
        if ref:
            for s in (0,1):assert n.observation(0,s)==clean_obs(ref.observation(s))
            assert ref.done and ref.t==719
        cash=[f['money'] for f in final['farms']];margin=cash[seat]-cash[1-seat]
        row.update(valid=True,steps=719,own_cash=cash[seat],opponent_cash=cash[1-seat],margin=margin,
                   win=margin>0,tie=margin==0,match_score=float(margin>0)+.5*float(margin==0),
                   latency_max=max(lat),latency_p99=float(np.quantile(lat,.99)),action_hash=hashes.hexdigest(),
                   official_comparisons=1440 if trace else 0,terminal_shops=final['town']['unlocked_shops'])
        if trace:
            path=Path(trace);path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(gzip.compress(json.dumps({'result':row,'days':days,'actions':actions,'terminal':final}).encode(),mtime=0))
            row.update(trace=str(path.relative_to(REPO)),trace_sha256=sha(path))
    except Exception:row['error']=traceback.format_exc()
    finally:signal.alarm(0);close()
    row['seconds']=time.perf_counter()-start;return row


LocalConfig=LocalGame(0,ENGINE).configuration


def score(rows):
    return (np.mean([r['win'] for r in rows]),np.mean([r['match_score'] for r in rows]),np.mean([r['margin'] for r in rows]))


def roster(root):
    out=[{'id':'control','kind':'control','family':'control','config':'control','training_seed':None}]
    for p in sorted((root/'trials').glob('*/candidate.json')):
        meta=json.loads(p.read_text());out.append({'id':p.parent.name,'kind':'plan','family':'search','config':meta['method'],
          'training_seed':meta['seed'],'path':str(p.relative_to(REPO)),'sha256':sha(p)})
    for p in sorted((root/'trials').glob('*/candidate.pt')):
        _,meta=load(p);out.append({'id':p.parent.name,'kind':'ppo','family':'ppo','config':str(meta['entropy']),
          'training_seed':meta['seed'],'path':str(p.relative_to(REPO)),'sha256':sha(p)})
        if meta['entropy']==.003:
            u=p.with_name('untrained.pt');out.append({'id':'untrained-'+str(meta['seed']),'kind':'ppo','family':'untrained','config':'untrained',
              'training_seed':meta['seed'],'path':str(u.relative_to(REPO)),'sha256':sha(u)})
    assert len(out)==16,('expected all twelve trials, three untrained controls, one rule control',len(out))
    return out


def run_panel(root,phase,candidates,seeds,opponents,workers):
    out=root/phase;out.mkdir(exist_ok=False);jobs=[]
    for c in candidates:
        for seed in seeds:
            for rival in opponents:
                for seat in (0,1):
                    trace=str(out/'replays'/f'{c["id"]}-{rival.replace(":","_")}-{seat}.json.gz') if seed==seeds[0] else None
                    jobs.append((c,seed,rival,seat,trace))
    dump(out/'MANIFEST.json',{'phase':phase,'candidates':candidates,'seeds':seeds,'opponents':opponents,'scheduled':len(jobs)})
    rows=[]
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn')) as pool:
        futures={pool.submit(worker,j):j for j in jobs}
        for f in as_completed(futures):
            r=f.result();rows.append(r)
            with (out/'rows.jsonl').open('a') as stream:stream.write(json.dumps(r)+'\n')
            status={'complete':False,'scheduled':len(jobs),'completed':len(rows),'invalid':sum(not r['valid'] for r in rows)}
            dump(out/'STATUS.json',status)
            if not r['valid']:raise RuntimeError(r['error'])
            if len(rows)%32==0:print(json.dumps(status),flush=True)
    keys={(r['candidate'],r['seed'],r['opponent'],r['seat']) for r in rows};assert len(keys)==len(jobs)
    verify_freeze(root/'inputs/FREEZE.json')
    for c in candidates:
        if 'path' in c:assert sha(REPO/c['path'])==c['sha256'],'candidate changed during evaluation'
    dump(out/'STATUS.json',{'complete':True,'scheduled':len(jobs),'completed':len(rows),'invalid':0})
    return rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--workers',type=int,default=2)
    p.add_argument('--phase',choices=('selection','pilot_test'),required=True);args=p.parse_args();root=args.root.resolve()
    assert 1<=args.workers<=2;verify_freeze(root/'inputs/FREEZE.json')
    seeds=json.loads((root/'inputs/seeds.json').read_text());protocol=json.loads((root/'inputs/PROTOCOL.json').read_text())
    candidates=roster(root)
    if args.phase=='selection':
        rows=run_panel(root,'selection',candidates,seeds['selection'],protocol['pilot_opponents'],args.workers)
        selected={}
        for family in ('search','ppo'):
            configs=sorted({c['config'] for c in candidates if c['family']==family})
            scores={config:score([r for r in rows if r['candidate'] in {c['id'] for c in candidates if c['family']==family and c['config']==config}]) for config in configs}
            selected[family]=max(configs,key=lambda c:scores[c])
        keep=[c for c in candidates if c['family'] in ('control','untrained') or c['config']==selected.get(c['family'])]
        dump(root/'SELECTED.json',{'configs':selected,'candidates':keep,'selection_rows_sha256':sha(root/'selection/rows.jsonl')})
    else:
        selected=json.loads((root/'SELECTED.json').read_text());assert selected['selection_rows_sha256']==sha(root/'selection/rows.jsonl')
        # Reconstruct eligibility from current immutable candidate files before test.
        candidates=selected['candidates']
        for c in candidates:
            if 'path' in c:assert sha(REPO/c['path'])==c['sha256']
        run_panel(root,'pilot_test',candidates,seeds['pilot_test'],protocol['pilot_opponents'],args.workers)


if __name__=='__main__':main()
