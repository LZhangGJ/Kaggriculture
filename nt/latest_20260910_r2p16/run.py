"""Official local referee + original realtime opponents; no online actions."""
from pathlib import Path
import argparse
import concurrent.futures
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

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'referee'))
from cpu_runtime import LocalGame, load_engine

os.environ['OMP_NUM_THREADS']='1'
def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec)
    sys.modules[name]=mod
    spec.loader.exec_module(mod)
    return mod

class FreshPublic:
    # Same per-game module isolation as the original run_panel.py.
    def __init__(self,directory):
        self.directory=Path(directory).resolve()
        self.previous_cwd=Path.cwd();self.previous_path=sys.path[:]
        self.names={p.stem for p in self.directory.glob('*.py')}
        self.displaced={n:sys.modules.pop(n) for n in self.names if n in sys.modules}
        sys.path.insert(0,str(self.directory));os.chdir(self.directory)
        try:
            self.mod=module(self.directory/'main.py','main');self.call=self.mod.agent
        except Exception:
            self.close();raise
    def close(self):
        for name,mod in list(sys.modules.items()):
            fn=getattr(mod,'__file__',None)
            if name in self.names or (fn and Path(fn).resolve().is_relative_to(self.directory)):
                sys.modules.pop(name,None)
        sys.modules.update(self.displaced);sys.path[:]=self.previous_path;os.chdir(self.previous_cwd)

def game(job):
    opponent,seed,seat,binary=job
    row=dict(opponent=opponent,seed=seed,opponent_seat=seat)
    public=policy=None
    tick=time.perf_counter()
    try:
        runtime=module(ROOT/'main.py','p16_handoff_runtime')
        policy=runtime.create_agent(binary)
        public=FreshPublic(ROOT/'opponents'/opponent)
        env=LocalGame(seed,load_engine());actions=[];latency=[]
        while not env.done:
            obs=[env.observation(0),env.observation(1)]
            assert env.configuration.seed is None
            out=[None,None]
            out[seat]=public.call(obs[seat],copy.deepcopy(env.configuration))
            start=time.perf_counter();out[1-seat]=policy(obs[1-seat]);latency.append(time.perf_counter()-start)
            assert all(isinstance(a,dict) and len(a.get('market',[]))<=10 for a in out)
            actions.append(copy.deepcopy(out));env.advance(out)
            assert env.t<=719
        assert env.t==719 and all(s.status=='DONE' for s in env.state)
        final=env.observation(0)
        cash=[final['farms'][s]['money'] for s in (0,1)]
        margin=cash[1-seat]-cash[seat]
        row.update(steps=env.t,r2_cash=cash[1-seat],opponent_cash=cash[seat],r2_margin=margin,
                   r2_win=margin>0,opponent_win=margin<0,tie=margin==0,runtime_error=None,
                   joint_action_sha256=hashlib.sha256(json.dumps(actions,separators=(',',':')).encode()).hexdigest(),
                   max_policy_seconds=max(latency),over_1s=sum(x>1 for x in latency))
    except Exception:
        row['runtime_error']=traceback.format_exc()
    finally:
        if policy is not None:policy.close()
        if public is not None:public.close()
    row['seconds']=time.perf_counter()-tick
    return row

def summary(rows):
    good=[r for r in rows if not r['runtime_error']]
    out=dict(games=len(rows),errors=len(rows)-len(good),wins=sum(r['r2_win'] for r in good),
             losses=sum(r['opponent_win'] for r in good),ties=sum(r['tie'] for r in good))
    out['win_rate']=out['wins']/len(rows)
    if good:
        out.update(mean_cash=statistics.mean(r['r2_cash'] for r in good),mean_margin=statistics.mean(r['r2_margin'] for r in good),
                   max_policy_seconds=max(r['max_policy_seconds'] for r in good),over_1s=sum(r['over_1s'] for r in good))
    return out

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--version',choices=['p16','p12','original'],default='p16')
    p.add_argument('--seed-start',type=int,default=2609125000)
    p.add_argument('--seeds',type=int,default=1)
    p.add_argument('--names',default='all')
    p.add_argument('--workers',type=int,default=16)
    p.add_argument('--tag',required=True)
    p.add_argument('--reference',action='store_true',help='Verify existing block2 results, not new strength data')
    p.add_argument('--binary',type=Path,help='Optional rebuilt version; reference must match --version')
    p.add_argument('--repeat-reset',action='store_true',help='Repeat one match in the same host process')
    a=p.parse_args()
    assert 1<=a.workers<=16 and a.seeds>0
    assert a.tag and Path(a.tag).name==a.tag and a.tag not in {'.','..'}
    out=ROOT/'runs'/a.tag
    assert not out.exists(), f'Refusing overwrite: {out}'
    out.mkdir(parents=True)
    pool=[x['id'] for x in json.loads((ROOT/'POOL.json').read_text())]
    names=pool if a.names=='all' else a.names.split(',')
    assert names and len(set(names))==len(names) and set(names)<=set(pool)
    binaries={'p16':ROOT/'policy/startupsupply2.so','p12':ROOT/'baselines/p12.so','original':ROOT/'baselines/original.so'}
    binary=(a.binary or binaries[a.version]).resolve();assert binary.is_file()
    expected={}
    if a.reference:
        records=json.loads(gzip.decompress((ROOT/f'evidence/new32_block2/{a.version}/rows.json.gz').read_bytes()))
        expected={(r['opponent'],r['seed'],r['opponent_seat']):r for r in records}
    jobs=[(op,seed,seat,str(binary)) for seed in range(a.seed_start,a.seed_start+a.seeds) for op in names for seat in (0,1)]
    if a.reference:
        assert all(j[:3] in expected for j in jobs), 'Reference only covers the frozen block2 seeds'
    protocol=dict(version=a.version,seed_start=a.seed_start,seeds=a.seeds,opponents=names,both_seats=True,
                  binary=str(binary),binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                  config_sha256=hashlib.sha256((ROOT/'policy/config.json').read_bytes()).hexdigest(),
                  official_version='1.32.7',realtime_opponents=True,reference_regression=a.reference,workers=a.workers)
    (out/'PROTOCOL.json').write_text(json.dumps(protocol,indent=2)+'\n')
    started=time.perf_counter();rows=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('spawn')) as executor:
        for row in executor.map(game,jobs):
            rows.append(row)
            if len(rows)%22==0 or len(rows)==len(jobs):
                print(json.dumps(dict(done=len(rows),total=len(jobs),errors=sum(bool(r['runtime_error']) for r in rows))),flush=True)
    fields=['steps','joint_action_sha256','r2_cash','opponent_cash','r2_margin','r2_win','opponent_win','tie']
    mismatches=[]
    if a.reference:
        for r in rows:
            e=expected[(r['opponent'],r['seed'],r['opponent_seat'])]
            for k in fields:
                if r.get(k)!=e[k]:mismatches.append(dict(opponent=r['opponent'],seed=r['seed'],seat=r['opponent_seat'],field=k))
    reset=None
    if a.repeat_reset:
        r1=game(jobs[0]);r2=game(jobs[0]);base=rows[0]
        reset=dict(passed=not r1['runtime_error'] and not r2['runtime_error'] and all(r1.get(k)==r2.get(k)==base.get(k) for k in fields))
    result=dict(overall=summary(rows),by_opponent={op:summary([r for r in rows if r['opponent']==op]) for op in names},
                seconds=time.perf_counter()-started,mismatches=mismatches,reset=reset,
                limitation='Local official interpreter; not Kaggle sandbox timing/schema certification')
    result['status']='PASS' if not result['overall']['errors'] and not mismatches and (reset is None or reset['passed']) else 'FAIL'
    (out/'rows.json').write_text(json.dumps(rows,indent=2)+'\n')
    (out/'RESULTS.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='by_opponent'},indent=2))
    if result['status']!='PASS':raise SystemExit(1)
if __name__=='__main__':main()
