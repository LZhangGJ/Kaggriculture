"""All feasible candidates at outcome-blind stratified real states; two tails."""
from pathlib import Path
import os,sys,time,json,zlib,hashlib
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import numpy as np
P=Path(__file__).resolve().parent;ROOT=P.parents[1]
OLD=P.parent/'economic_rl_three_arm_20260906';FIX=P.parent/'economic_rl_f3_ledger_fix_20260906'
NEW=P.parent/'economic_rl_f3_ledger_rerun_20260906'
sys.path.insert(0,str(OLD));sys.path.insert(0,str(P/'build'))
import _economic_candidate_native as native
import runtime as oldrt
from runtime import read,save,digest,NAMES,store_result
GOLDEN=0x9e3779b97f4a7c15

def data():
    reg=ROOT/'experiments/daily_dp_v7_20260903/opponents/registry.json';meta=read(reg)['opponents'];out={}
    for name in NAMES[:5]:
        path=reg.parent.parent/meta[name]['asset'];out[name]=json.loads(zlib.decompress(path.read_bytes()))
        assert out[name]['source_sha256']==meta[name]['source_sha256']==digest(reg.parent.parent/meta[name]['source'])
    return out

def call(pool,source,jobs,trace=False,threads=16,library=None):
    t=time.perf_counter()
    r=pool.batch(str(library or P/'build/audit.so'),source['model'],[j['mode']for j in jobs],
        [j['seed']for j in jobs],[j['seat']for j in jobs],[j['opponent']for j in jobs],
        [j['rng']for j in jobs],threads,trace)
    r['call_seconds']=time.perf_counter()-t;r['bridge_seconds']=r['call_seconds']-r['wall_seconds'];r['mode']=2
    for row in r['rows']:
        row['opponent']=NAMES[row['opponent']]
        assert not row['error'] and row['steps']==719 and row['plan_calls']==30 and row['reference_calls']==0,row
    for k in ('global','candidates','mask','probability','logp','value','reward'):
        assert np.isfinite(r[k]).all(),k
    return r

def make_sources():
    sources=[]
    for rep in (0,1):
        for step in (1,20):
            sources.append(dict(id=f'train_r{rep}_b{step:02}',cohort='train',run=rep,
                model=str(NEW/'training'/f'f3_r{rep}'/f'step{step-1:03}.bin'),
                start=61000000+rep*100000+(step-1)*16,count=1,sample=70000+rep*10000+step,
                original=str(NEW/'training'/f'f3_r{rep}'/f'rollout_{step:03}')))
    for rep in (0,1):
        sources.append(dict(id=f'new_r{rep}',cohort='new',run=rep,
            model=str(NEW/'training'/f'f3_r{rep}'/'step020.bin'),
            start=67000000+rep*2,count=2,sample=9917,original=None))
    for source in sources:
        source['model_sha256']=digest(source['model'])
        source['jobs']=[dict(seed=seed,seat=seat,opponent=opp,mode=2,
            rng=(source['sample']+GOLDEN*(n+1))%(1<<64))
            for n,(seed,seat,opp)in enumerate(oldrt.jobs(source['start'],source['count']))]
    return sources

def preflight(pool,source):
    jj=source['jobs'];original=oldrt.make_pool()
    r=original.batch(str(FIX/'build/f3.so'),source['model'],2,[j['seed']for j in jj],
        [j['seat']for j in jj],[j['opponent']for j in jj],source['sample'],16,True)
    plain=call(pool,source,jj,trace=True)
    assert r['traces']==plain['traces']
    for k in ('global','candidates','mask','probability','choice','reward','value'):
        assert np.array_equal(r[k],plain[k]),k
    forced=[]
    for i,j in enumerate(jj):
        day=(i*2)%29;a=int(r['choice'][i*30+day]);forced.append(dict(j,mode=1000+600+day*10+a))
    replay=call(pool,source,forced,trace=True)
    one=call(pool,source,forced,trace=True,threads=1)
    assert r['traces']==replay['traces']==one['traces']
    for k in ('global','candidates','mask','probability','choice','reward','value'):
        assert np.array_equal(r[k],replay[k])and np.array_equal(r[k],one[k]),k
    receipt=dict(status='PASS',games=len(jj),steps=len(jj)*719,
        unmodified_wrapper_exact=True,forced_original_choice_exact=True,threads_1_16_exact=True)
    save(P/'PREFLIGHT.json',receipt);print('PREFLIGHT',receipt,flush=True)

def main():
    assert read(P/'BUILD_RECEIPT.json')['status']=='PASS'
    if (P/'PROTOCOL.json').exists():raise FileExistsError('Audit already started')
    sources=make_sources()
    hashes=dict(read(P/'BUILD_RECEIPT.json')['preserved']);hashes.update(read(P/'BUILD_RECEIPT.json')['built'])
    hashes.update({s['model']:s['model_sha256']for s in sources})
    hashes.update({str(f):digest(f)for f in P.glob('*.py')})
    hashes[str(P/'intervention.cpp')]=digest(P/'intervention.cpp');hashes[str(P/'PLAN_ZH.md')]=digest(P/'PLAN_ZH.md')
    for f,h in hashes.items():assert digest(f)==h,f
    save(P/'PROTOCOL.json',dict(sources=sources,hashes=hashes,day_strata=[[0,2],[3,6],[7,10],[11,14],[15,18],[19,22],[23,26],[27,28]],
                              sampling_rng=20260906,tail_modes=[0,2],no_learning=True))
    pool=native.RLPool(data());tic=time.perf_counter();preflight(pool,sources[0])
    baseline={};states=[];rng=np.random.default_rng(20260906)
    bins=[(0,2),(3,6),(7,10),(11,14),(15,18),(19,22),(23,26),(27,28)]
    for source in sources:
        r=call(pool,source,source['jobs']);baseline[source['id']]=r
        store_result(P/'baselines'/source['id'],r)
        if source['original']:
            with np.load(Path(source['original'])/'decisions.npz')as old:
                for key in ('global','candidates','mask','choice','probability','reward','value'):
                    assert np.array_equal(old[key][:len(r[key])],r[key]),(source['id'],key)
        for gi,j in enumerate(source['jobs']):
            for lo,hi in bins:
                day=int(rng.integers(lo,hi+1));ix=gi*30+day
                states.append(dict(id=len(states),source=source['id'],cohort=source['cohort'],game=gi,
                    seed=j['seed'],seat=j['seat'],opponent=NAMES[j['opponent']],day=day,
                    available=np.flatnonzero(r['mask'][ix]).tolist(),probability=r['probability'][ix].tolist(),
                    original_choice=int(r['choice'][ix]),base_game_win=bool(r['rows'][gi]['win'])))
    save(P/'STATES.json',states);save(P/'BASELINES_COMPLETE.json',dict(status='PASS',source_games=112,states=len(states)))
    # Each state's complete feasible pool and both tails are fixed before branch scores are known.
    alljobs={}
    for source in sources:
        jobs=[]
        for state in states:
            if state['source']!=source['id']:continue
            base=source['jobs'][state['game']]
            for tail in (0,2):
                for action in state['available']:
                    jobs.append(dict(base,state_id=state['id'],day=state['day'],choice=action,tail=tail,
                                     mode=1000+tail*300+state['day']*10+action))
        alljobs[source['id']]=jobs
    save(P/'JOB_MANIFEST.json',alljobs)
    total=sum(len(j)for j in alljobs.values());finished=identity=0;native_s=call_s=0.;chunk_id=0
    (P/'branches').mkdir(exist_ok=False)
    for source in sources:
        ref=baseline[source['id']];jobs=alljobs[source['id']]
        for offset in range(0,len(jobs),224):
            chunk=jobs[offset:offset+224];r=call(pool,source,chunk);rows=[]
            native_s+=r['wall_seconds'];call_s+=r['call_seconds']
            for n,(job,row)in enumerate(zip(chunk,r['rows'])):
                state=states[job['state_id']];day=job['day'];a=state['game']*30;b=n*30
                for key in ('global','candidates','mask','probability'):
                    assert np.array_equal(ref[key][a:a+day+1],r[key][b:b+day+1]),(source['id'],state['id'],key)
                assert np.array_equal(ref['choice'][a:a+day],r['choice'][b:b+day])
                assert r['choice'][b+day]==job['choice']
                if job['tail']==0:assert (r['choice'][b+day+1:b+30]==0).all()
                if job['tail']==2 and job['choice']==state['original_choice']:
                    for key in ('global','candidates','mask','probability','choice','reward','value'):
                        assert np.array_equal(ref[key][a:a+30],r[key][b:b+30]),(state['id'],key)
                    assert row['cash']==ref['rows'][state['game']]['cash']and row['opponent_cash']==ref['rows'][state['game']]['opponent_cash']
                    identity+=1
                rows.append(dict(row,state_id=state['id'],source=source['id'],cohort=source['cohort'],day=day,
                    choice=job['choice'],tail=job['tail'],reward=float(r['reward'][b+29]),
                    edited=int(r['changed'][b+day]),target_pos=int(r['pos'][b+day]),target_kind=int(r['kind'][b+day])))
            save(P/'branches'/f'{chunk_id:04}.json',rows);chunk_id+=1;finished+=len(chunk)
            progress=dict(status='RUNNING',completed=finished,total=total,native_seconds=native_s,
                call_seconds=call_s,elapsed=time.perf_counter()-tic,identity_suffix_checks=identity)
            save(P/'PROGRESS.json',progress)
            print('AUDIT',finished,'/',total,source['id'],'games/s',round(finished/native_s,1),flush=True)
    for f,h in hashes.items():assert digest(f)==h,f
    save(P/'RUN_COMPLETE.json',dict(status='PASS',source_games=112,states=len(states),branch_games=finished,
        native_seconds=native_s,call_seconds=call_s,seconds=time.perf_counter()-tic,identity_suffix_checks=identity,
        preserved_inputs=True))
    print('AUDIT_COMPLETE',finished,flush=True)

if __name__=='__main__':main()
