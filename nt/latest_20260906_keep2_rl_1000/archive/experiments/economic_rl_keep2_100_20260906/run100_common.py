from pathlib import Path
import os,sys,time
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
P=Path(__file__).resolve().parent;AUX=P.parent/'economic_rl_future_aux_20260906';OLD=P.parent/'economic_rl_three_arm_20260906'
PRIOR=P.parent/'economic_rl_keep_bias_20260906'
sys.path.insert(0,str(AUX))
from common import read,save,digest,summary,store_result,NAMES,bootstrap,configure_torch
import runtime
import numpy as np

def check_hashes():
    for f,h in read(P/'PROTOCOL.json')['hashes'].items():assert digest(f)==h,f

def rollout(pool,model='',mode=0,start=69900000,count=100,sample=9901,threads=16,trace=False,prior=False):
    jobs=runtime.jobs(start,count);tic=time.perf_counter()
    lib=PRIOR/'build/keep.so' if prior else P/'build/keep2.so'
    r=pool.batch(str(lib),str(model),mode,[j[0]for j in jobs],[j[1]for j in jobs],[j[2]for j in jobs],sample,threads,trace)
    r['call_seconds']=time.perf_counter()-tic;r['bridge_seconds']=r['call_seconds']-r['wall_seconds'];r['mode']=mode
    for row in r['rows']:
        row['opponent']=NAMES[row['opponent']]
        assert not row['error'] and row['steps']==719 and row['reference_calls']==0,row
        assert row['plan_calls']==30 and row['execute_calls']==719
    assert len(r['choice'])==len(jobs)*30
    assert (r['mask'][np.arange(len(r['choice'])),r['choice']]>0).all()
    for k in ('global','candidates','mask','probability','value','reward','logp'):assert np.isfinite(r[k]).all(),k
    return r

def evaluate(pool,out,model='',mode=0,start=69900000,count=100,sample=9901):
    r=rollout(pool,model,mode,start,count,sample);store_result(out,r)
    s=summary(r);save(Path(out)/'provenance.json',dict(checkpoint=str(model),sha256=digest(model)if model else None,
         library=str(P/'build/keep2.so'),library_sha=digest(P/'build/keep2.so'),seed_start=start,seeds=count,mode=mode,keep_bonus=2.))
    print('EVAL',Path(out).name,s['overall'],'nonkeep',s['non_keep'],flush=True)
    return s
