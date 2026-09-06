"""Fixed-matrix correctness/KEEP regression only; no training or model selection."""
from pathlib import Path
import os,sys,time,json,multiprocessing
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
P=Path(__file__).resolve().parent;OLD=P.parent/'economic_rl_three_arm_20260906'
sys.path.insert(0,str(OLD))
from runtime import make_pool,jobs,NAMES,digest,read,save,summary,store_result
from checks import check_game

def roll(pool,library,mode,count=4,threads=16,trace=True):
    jj=jobs(64002000,count);tic=time.perf_counter()
    r=pool.batch(str(library),'',mode,[x[0]for x in jj],[x[1]for x in jj],[x[2]for x in jj],1719,threads,trace)
    r['call_seconds']=time.perf_counter()-tic;r['bridge_seconds']=r['call_seconds']-r['wall_seconds'];r['mode']=mode
    for row in r['rows']:row['opponent']=NAMES[row['opponent']]
    return r

def main():
    out=P/'regression';out.mkdir(exist_ok=False)
    assert read(P/'UNIT_TEST_RECEIPT.json')['status']=='PASS'
    preserved=read(P/'BUILD_RECEIPT.json')['preserved_original']
    frozen={str(p):digest(p)for p in [P/'f3_policy.cpp',P/'stage/f3/agent.cpp',P/'stage/f3/project_forecast_ledger.hpp',P/'build/f3.so']}
    pool=make_pool();newlib=P/'build/f3.so';oldlib=OLD/'build/f3.so'
    old=roll(pool,oldlib,0);new=roll(pool,newlib,0)
    store_result(out/'old_keep',old);store_result(out/'fixed_keep',new)
    assert summary(old)['status']==summary(new)['status']=='PASS'
    assert old['traces']==new['traces'],'KEEP full action trace changed'
    assert np.array_equal(old['global'],new['global']),'KEEP model input changed'
    assert np.array_equal(old['mask'],new['mask']),'KEEP candidate availability changed'
    assert [x['cash']for x in old['rows']]==[x['cash']for x in new['rows']]
    print('KEEP 56/56 full traces and cash equal',flush=True)
    random=roll(pool,newlib,3);store_result(out/'fixed_random',random)
    assert summary(random)['status']=='PASS'and summary(random)['changed']>0
    assert np.isfinite(random['global']).all()and np.isfinite(random['candidates']).all()
    assert (random['mask'][np.arange(len(random['choice'])),random['choice']]>0).all()
    one=roll(pool,newlib,3,count=1,threads=1)
    assert one['traces']==random['traces'][:14],'thread/state isolation'
    print('RANDOM 56 full games; 1/16 thread 14 games identical',summary(random)['changed'],flush=True)
    save(P/'NATIVE_REGRESSION.json',dict(status='PASS',keep_equal_games=56,keep_equal_steps=56*719,
        random_games=56,thread_equivalence_games=14,summaries={name:summary(r)for name,r in [('old_keep',old),('fixed_keep',new),('fixed_random',random)]}))
    # One seed, seven opponents, both seats; states of BOTH seats at EVERY step.
    rows=[];start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=14,mp_context=multiprocessing.get_context('spawn'))as workers:
        future=[workers.submit(check_game,('f3',3,n,str(out/'fixed_random')))for n in range(14)]
        for f in as_completed(future):
            try:rows.append(f.result())
            except Exception as exc:rows.append(dict(status='FAIL',error=repr(exc)))
            print('OFFICIAL',len(rows),14,rows[-1]['status'],flush=True)
    status='PASS'if all(r['status']=='PASS'for r in rows)else'FAIL'
    save(P/'OFFICIAL_REGRESSION.json',dict(status=status,rows=rows,seconds=time.perf_counter()-start,steps=sum(r.get('steps',0)for r in rows)))
    assert status=='PASS',rows
    # Alternating pair order, matching seeds, no trace serialization in timings.
    perf=[]
    for rep in range(4):
        order=[('old',oldlib),('fixed',newlib)]if rep%2==0 else [('fixed',newlib),('old',oldlib)]
        for label,lib in order:
            r=roll(pool,lib,0,count=8,trace=False);assert summary(r)['status']=='PASS'
            perf.append(dict(repetition=rep,version=label,seconds=r['call_seconds'],games=112))
    performance={v:dict(median_seconds=float(np.median([r['seconds']for r in perf if r['version']==v])),
                       games_per_second=112/float(np.median([r['seconds']for r in perf if r['version']==v])))for v in ('old','fixed')}
    save(P/'PERFORMANCE.json',dict(pairs=perf,summary=performance,note='KEEP workloads, 16 threads; not training or strength evaluation'))
    assert all(digest(p)==h for p,h in preserved.items())
    assert all(digest(p)==h for p,h in frozen.items())
    save(P/'ACCEPTANCE.json',dict(status='PASS',scope='F3 RL daily forecast ownership fix',training_performed=False,
        unit=read(P/'UNIT_TEST_RECEIPT.json'),native=read(P/'NATIVE_REGRESSION.json'),official=read(P/'OFFICIAL_REGRESSION.json'),
        performance=performance,preserved_original=preserved,fixed_hashes=frozen))
    print('ALL PASS',performance,flush=True)

if __name__=='__main__':main()
