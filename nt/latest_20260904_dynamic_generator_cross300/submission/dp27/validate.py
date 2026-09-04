"""Verify the deployable entrypoint against the exact frozen DP27 trajectory."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,sys,time
from concurrent.futures import ProcessPoolExecutor

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];EXP=ROOT/'experiments/daily_dp_v7_20260903'
sys.path.insert(0,str(EXP/'search8_multiseed'))
import run as old
read,write,sha=old.read,old.write,old.sha
SOURCE='R03_F001_E104234696_P0'

def initialize():
    global f,n,pool,entry,selected
    f,n,pool=old.runtime(EXP/'search8_multiseed/receipts_v1')
    spec=importlib.util.spec_from_file_location('dp27_submit',HERE/'main.py');entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)
    selected=read(HERE/'frozen_plan.json')
    assert selected['source']==SOURCE and selected['index']==27

def one(job):
    rid,seed,seat,use_official=job;entry._load().dp27_reset()
    fixture=n.replay8(pool,f['route_ids'].index(rid),f['config'],seed,seat,list(range(29)),selected['plan'],True)
    refpath=EXP/'search29_cross300/receipts_v1/games'/SOURCE/'027'/(rid+'.json')
    oldrow=next(r for r in read(refpath)['rows'] if r['seed']==seed and r['seat']==seat)
    assert fixture['state_action_hash']==oldrow['state_action_hash']
    env=n.Env(seed);official=None
    if use_official:
        sys.path.insert(0,str(ROOT/'gpt_review/codex/G001_CPU_FOR_GPT_20260903'))
        from cpu_runtime import LocalGame
        official=LocalGame(seed)
    walls=[];cpus=[];start=time.perf_counter()
    for step,pair in enumerate(fixture['trace']):
        obs=official.observation(seat) if official else env.observation(seat)
        wall=time.perf_counter();cpu=time.process_time();action=entry.agent(obs,{'episodeSteps':720})
        cpus.append(time.process_time()-cpu);walls.append(time.perf_counter()-wall)
        if action!=pair[seat]:
            detail=dict(status='FAIL_ACTION_PARITY',route=rid,seed=seed,seat=seat,step=step,expected=pair[seat],actual=action,observation=obs)
            write(HERE/f'failure_{rid}_{seed}_{seat}_{time.time_ns()}.json',detail)
            raise RuntimeError(f'Action mismatch {rid} {seed} {seat} step={step}')
        actual=list(pair);actual[seat]=action;env.step(actual)
        if official:
            official.advance(actual)
            for p in (0,1):
                a=env.observation(p);b=official.observation(p)
                for key in ('farms','private','market','town'):
                    assert json.loads(json.dumps(a[key]))==json.loads(json.dumps(b[key])),(rid,seed,seat,step,key)
    assert env.done and env.step_count==719
    ending=env.observation(seat)['farms']
    assert ending[seat]['money']==oldrow['cash'] and ending[1-seat]['money']==oldrow['opponent_cash']
    if official:assert official.done
    return dict(route=rid,seed=seed,seat=seat,steps=719,official=use_official,action_equal=True,seconds=time.perf_counter()-start,
        max_wall=max(walls),max_cpu=max(cpus),sum_wall=sum(walls),sum_cpu=sum(cpus),
        max_wall_step=walls.index(max(walls)),max_cpu_step=cpus.index(max(cpus)),
        cash=oldrow['cash'],opponent_cash=oldrow['opponent_cash'],win=oldrow['win'])

def main(mode,version):
    build=read(HERE/'build_receipt.json');assert build['status']=='PASS'
    assert sha(HERE/'main.py')==build['entry_sha256'] and sha(HERE/'agent.so')==build['binary_sha256']
    frozen=read(EXP/'search29_cross300/receipts_v1/inputs.json');rids=[x['route_id'] for x in frozen['opponents']]
    seeds=frozen['seeds'][:1] if mode=='smoke' else frozen['seeds']
    jobs=[(r,s,p,mode=='smoke') for r in rids for s in seeds for p in (0,1)]
    out=HERE/f'validation_{mode}_{version}.json';assert not out.exists(),'Preserve existing validation receipt'
    rows=[];tic=time.perf_counter()
    if mode=='smoke':
        initialize()
        for job in jobs:
            r=one(job);rows.append(r);print(json.dumps(r),flush=True)
    else:
        with ProcessPoolExecutor(max_workers=8,initializer=initialize) as workers:
            for r in workers.map(one,jobs):
                rows.append(r);print(json.dumps(dict(done=len(rows),games=len(jobs),**r)),flush=True)
    assert len(rows)==len(jobs)
    write(out,dict(status='PASS_ACTION_PARITY',mode=mode,games=len(rows),rows=rows,seconds=time.perf_counter()-tic,
        max_wall=max(r['max_wall'] for r in rows),max_cpu=max(r['max_cpu'] for r in rows),
        entry_sha256=sha(HERE/'main.py'),binary_sha256=sha(HERE/'agent.so'),
        caveat='Local timing only; official Kaggle hardware/sandbox acceptance still required.'))
    print(json.dumps(dict(status='PASS',games=len(rows),max_wall=max(r['max_wall'] for r in rows),seconds=time.perf_counter()-tic)),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['smoke','parity']);parser.add_argument('--version',default='compat_v1');args=parser.parse_args();main(args.mode,args.version)
