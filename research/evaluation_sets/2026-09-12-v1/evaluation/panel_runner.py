"""Matched manifest runner with full-game parity and append-only result journals."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import platform
import sys
import time
import traceback

HERE=Path(__file__).resolve().parent
RUNTIME=HERE/'runtime'
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
sys.dont_write_bytecode=True
sys.path.insert(0,str(RUNTIME/'research/robust90'))
sys.path.insert(0,str(RUNTIME/'nt/latest_20260911_p16_jointafs_r1/agent'))
import evaluate
import native_env

ACTIVE_ECONOMY=[]
DirectBase=native_env.DirectGame

class MeasuredDirectGame(DirectBase):
    """Read the public market every four turns, before actions execute."""
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.economy=dict(sample_steps=[],prices={},inventory={},unlock_events=[])
        self.last_shops=[]
        ACTIVE_ECONOMY.append(self.economy)
    def advance(self,actions):
        if self.t%4==0:
            obs=self.observation(0)
            e=self.economy;e['sample_steps'].append(self.t)
            for kind in ('prices','inventory'):
                for item,value in obs['market'][kind].items():e[kind].setdefault(item,[]).append(value)
            shops=obs['town']['unlocked_shops']
            if shops!=self.last_shops:
                assert shops[:len(self.last_shops)]==self.last_shops
                for shop in shops[len(self.last_shops):]:e['unlock_events'].append(dict(step=self.t,shop=shop))
                self.last_shops=shops[:]
        super().advance(actions)

native_env.DirectGame=MeasuredDirectGame

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,d):
    p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_suffix(p.suffix+'.tmp')
    temp.write_text(json.dumps(d,indent=2,sort_keys=True)+'\n',encoding='utf8')
    temp.replace(p)

def resolve(value):
    if isinstance(value,dict):
        if set(value)=={'runtime_path'}:return str(RUNTIME/value['runtime_path'])
        return {k:resolve(v) for k,v in value.items()}
    if isinstance(value,list):return [resolve(v) for v in value]
    return value

def compact_economy(e):
    n=len(e['sample_steps']);assert e['sample_steps']==list(range(0,719,4))
    assert len(e['unlock_events'])==8
    out=dict(sampling='public pre-action observation every four turns, steps 0 through 716',
             samples=n,unlock_events=e['unlock_events'],market={})
    for item,prices in e['prices'].items():
        inv=e['inventory'][item]
        out['market'][item]=dict(price_min=min(prices),price_max=max(prices),price_sum=sum(prices),
          floor_samples=sum(x==1 for x in prices),inventory_min=min(inv),inventory_max=max(inv),
          inventory_sum=sum(inv),inventory_first=inv[0],inventory_last=inv[-1])
    return out

def play(c,seed,opponent,seat,mode='direct',adapter=True,measure=True):
    ACTIVE_ECONOMY.clear()
    cfg=resolve(c.get('native_adapter_config',c['config']) if adapter else c['config'])
    cfg.update(__engine=mode,__sparse_observations=mode=='direct')
    row=evaluate.worker((opponent,seed,seat,str(RUNTIME/c['binary']),cfg,None))
    if not row.get('runtime_error') and row.get('steps')==719 and measure:
        assert len(ACTIVE_ECONOMY)==1
        row['economy']=compact_economy(ACTIVE_ECONOMY[0])
    allowed=('opponent','seed','opponent_seat','runtime_error','engine','steps','own_cash','opponent_cash',
             'margin','win','tie','action_hash','latency_max','terminal_shops','seconds','observations_checked',
             'selected_profile','prefix_profiles','market_order_changes','economy')
    result={k:row[k] for k in allowed if k in row}
    result.update(candidate_id=c['id'],candidate_name=c.get('display_name',c['name']),candidate_seat=1-seat,
                  entry_artifact_sha256=c['binary_sha256'],native_adapter=bool(adapter and 'native_adapter_config' in c))
    return result

def equivalent(a,b):
    fields=('steps','own_cash','opponent_cash','margin','win','tie','action_hash','terminal_shops','economy')
    return {k:[a.get(k),b.get(k)] for k in fields if a.get(k)!=b.get(k)}

def execute(job):
    c,panel,seed,opponent,seat,phase=job
    try:
        if phase=='preflight':
            # All observations in both seats match the official interpreter during this game.
            parity=play(c,seed,opponent,seat,'direct_parity')
            fast=play(c,seed,opponent,seat,'direct')
            package=play(c,seed,opponent,seat,'direct',False) if 'native_adapter_config' in c else None
            comparisons=dict(parity_to_direct=equivalent(parity,fast))
            if package:comparisons['package_to_native_adapter']=equivalent(package,fast)
            valid=all(not r.get('runtime_error') and r.get('steps')==719 for r in (parity,fast,package) if r)
            result=dict(candidate_id=c['id'],panel=panel,seed=seed,opponent=opponent,opponent_seat=seat,
                        valid=valid and not any(comparisons.values()),comparisons=comparisons,
                        parity=parity,direct=fast,package=package)
        else:
            result=play(c,seed,opponent,seat)
            result['panel']=panel
            result['valid']=not result.get('runtime_error') and result.get('steps')==719
        return result
    except BaseException:
        return dict(candidate_id=c['id'],panel=panel,seed=seed,opponent=opponent,opponent_seat=seat,
                    valid=False,runtime_error=traceback.format_exc())

def key(row):return (row['candidate_id'],row['panel'],row['seed'],row['opponent'],row['opponent_seat'])

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--roster',type=Path,required=True)
    p.add_argument('--panels',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--phase',choices=('preflight','development','holdout'),required=True)
    p.add_argument('--workers',type=int,default=16)
    p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--preflight',type=Path)
    p.add_argument('--release',type=Path)
    args=p.parse_args()
    assert 1<=args.workers<=16
    audit=json.loads(args.audit.read_text());assert audit['status']=='PASS'
    roster=json.loads(args.roster.read_text())
    opponents=json.loads((args.panels/'opponents.json').read_text())['opponents']
    assert len(opponents)==16 and len({r['id'] for r in roster['candidates']})==len(roster['candidates'])
    for file,digest in roster['runtime_files'].items():
        assert sha(RUNTIME/file)==digest, f'Runtime hash changed: {file}'
    if args.phase!='preflight':
        preflight=json.loads(args.preflight.read_text());assert preflight['complete'] and preflight['invalid']==0
        assert preflight['roster_sha256']==sha(args.roster)
    if args.phase=='holdout':
        release=json.loads(args.release.read_text())
        assert release['status']=='released_for_one_frozen_comparison'
        assert release['roster_sha256']==sha(args.roster) and release['runner_sha256']==sha(Path(__file__))
        assert release['holdout_sha256']==sha(args.panels/'sealed/holdout.json')
    names=['representative'] if args.phase=='preflight' else ['holdout'] if args.phase=='holdout' else ['representative','stress']
    jobs=[]
    for name in names:
        path=args.panels/('sealed' if name=='holdout' else 'manifests')/(name+'.json')
        seeds=json.loads(path.read_text())['seeds']
        if args.phase=='preflight':seeds=seeds[:1]
        for seed in seeds:
            for opponent in opponents:
                for seat in (0,1):
                    for candidate in roster['candidates']:jobs.append((candidate,name,seed,opponent['id'],seat,args.phase))
    args.out.mkdir(parents=True,exist_ok=True)
    manifest=dict(roster_sha256=sha(args.roster),runner_sha256=sha(Path(__file__)),phase=args.phase,
                  audit_sha256=sha(args.audit),workers=args.workers,scheduled=len(jobs),
                  timeout_seconds_per_game=180,python=sys.version,platform=platform.platform(),
                  multiprocessing_start='fork',cpu_only=True,created_unix=time.time())
    if (args.out/'MANIFEST.json').exists():
        old=json.loads((args.out/'MANIFEST.json').read_text())
        for field in ('roster_sha256','runner_sha256','phase','scheduled','timeout_seconds_per_game'):
            assert old[field]==manifest[field],f'Resume differs: {field}'
    else:dump(args.out/'MANIFEST.json',manifest)
    journal=args.out/'rows.jsonl';seen={};invalid=0
    if journal.exists():
        for line in journal.read_text().splitlines():
            row=json.loads(line);k=key(row);assert k not in seen,'Duplicate result cell'
            seen[k]=row;invalid+=not row['valid']
    expected={(c['id'],n,s,o,t) for c,n,s,o,t,_ in jobs}
    assert set(seen)<=expected
    pending=[j for j in jobs if (j[0]['id'],j[1],j[2],j[3],j[4]) not in seen]
    started=time.time();completed_start=len(seen);last=0
    def status():
        elapsed=time.time()-started;new=len(seen)-completed_start
        record=dict(scheduled=len(jobs),completed=len(seen),invalid=invalid,missing=len(jobs)-len(seen),
                    complete=len(seen)==len(jobs) and invalid==0,elapsed_this_process=elapsed,
                    games_per_second=new/elapsed if elapsed else 0,roster_sha256=sha(args.roster),
                    phase=args.phase,updated_unix=time.time())
        dump(args.out/'STATUS.json',record);print(json.dumps(record),flush=True)
    with journal.open('a',encoding='utf8',buffering=1) as stream, ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('fork')) as pool:
        futures={};iterator=iter(pending)
        def fill():
            while len(futures)<args.workers*2:
                job=next(iterator,None)
                if job is None:break
                futures[pool.submit(execute,job)]=job
        fill()
        while futures:
            done,_=wait(futures,timeout=10,return_when=FIRST_COMPLETED)
            for f in done:
                job=futures.pop(f)
                try:row=f.result()
                except BaseException:
                    c,n,s,o,t,_=job;row=dict(candidate_id=c['id'],panel=n,seed=s,opponent=o,opponent_seat=t,valid=False,runtime_error=traceback.format_exc())
                k=key(row);assert k not in seen
                seen[k]=row;invalid+=not row['valid'];stream.write(json.dumps(row,separators=(',',':'))+'\n')
            fill()
            if time.time()-last>30:status();last=time.time()
    status()
    if invalid:raise SystemExit('Invalid games retained; comparison is incomplete. Inspect errors before a documented repair.')

if __name__=='__main__':main()
