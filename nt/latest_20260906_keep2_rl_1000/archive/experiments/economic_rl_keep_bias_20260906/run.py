"""Inference-only KEEP-prior ablation, paired fresh seeds, live opponents."""
from pathlib import Path
import os,sys,time
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
P=Path(__file__).resolve().parent;OLD=P.parent/'economic_rl_three_arm_20260906';FIX=P.parent/'economic_rl_f3_ledger_fix_20260906'
AUX=P.parent/'economic_rl_future_aux_20260906'
sys.path.insert(0,str(OLD));import runtime
from runtime import read,save,digest,summary,store_result,NAMES
import numpy as np

BONUSES=[3.58351893846,2.,1.,0.]

def rollout(pool,checkpoint,mode,start,count=100,threads=16,trace=False,original=False):
    jobs=runtime.jobs(start,count);tic=time.perf_counter()
    path=FIX/'build/f3.so' if original else P/'build/keep.so'
    r=pool.batch(str(path),str(checkpoint),mode,[j[0]for j in jobs],[j[1]for j in jobs],
                 [j[2]for j in jobs],9917,threads,trace)
    r['call_seconds']=time.perf_counter()-tic;r['bridge_seconds']=r['call_seconds']-r['wall_seconds'];r['mode']=1 if mode else 0
    for row in r['rows']:
        row['opponent']=NAMES[row['opponent']]
        assert not row['error'] and row['steps']==719 and row['reference_calls']==0,row
        assert row['plan_calls']==30 and row['execute_calls']==719
    for k in ('global','candidates','mask','probability','logp','value','reward'):assert np.isfinite(r[k]).all(),k
    assert (r['mask'][np.arange(len(r['choice'])),r['choice']]>0).all()
    assert (r['choice']==r['probability'].argmax(1)).all() or mode==0
    return r

def checks(pool,checkpoint):
    import torch
    from model import Model
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    out=P/'checks';out.mkdir(exist_ok=False)
    old=rollout(pool,checkpoint,1,69420000,1,trace=True,original=True)
    r=rollout(pool,checkpoint,100,69420000,1,trace=True)
    assert old['traces']==r['traces']
    for k in ('global','candidates','mask','probability','choice','value','reward'):assert np.array_equal(old[k],r[k]),k
    store_result(out/'original_bonus',r)
    m=Model().cuda();state=torch.load(Path(checkpoint).with_suffix('.pt'),weights_only=True)
    m.load_state_dict({k:v for k,v in state.items()if not k.startswith('future.')})
    errors=[]
    for bi in range(1,4):
        r=rollout(pool,checkpoint,100+bi,69420000,1,trace=True)
        x=torch.as_tensor(r['global'],device='cuda');c=torch.as_tensor(r['candidates'],device='cuda');mask=torch.as_tensor(r['mask'],device='cuda').bool()
        with torch.no_grad():
            scores=m.actor(torch.cat((x[:,None,:].expand(-1,10,-1),c),-1)).squeeze(-1)
            scores[:,0]+=BONUSES[bi];prob=scores.masked_fill(~mask,-1e30).softmax(-1).cpu().numpy()
        error=float(np.max(abs(prob-r['probability'])));assert error<2e-5,error
        assert np.array_equal(prob.argmax(1),r['choice'])
        store_result(out/f'bonus_{bi}',r);errors.append(dict(bonus=BONUSES[bi],probability_error=error,nonkeep=int((r['choice']!=0).sum())))
    r1=rollout(pool,checkpoint,103,69420000,1,threads=1,trace=True)
    assert r['traces']==r1['traces']
    # A subsequent original-bonus run on reused native threads must not inherit zero.
    again=rollout(pool,checkpoint,100,69420000,1,trace=True)
    assert again['traces']==old['traces']
    save(P/'G0_ACCEPTANCE.json',dict(status='PASS',original_bonus_full_trace_exact=True,cpp_torch_probabilities=errors,
        threads_1_16_full_trace_exact=True,no_threadlocal_stale_bonus=True,all_choices_legal=True,games=98))
    print('G0_PASS',errors,flush=True)

def main():
    receipt=read(P/'BUILD_RECEIPT.json');assert receipt['status']=='PASS'
    if (P/'PROTOCOL.json').exists():raise FileExistsError('Frozen ablation already started')
    models={key:value['last'] for key,value in read(AUX/'FINAL_SELECTION_FROZEN.json').items()}
    hashes=dict(receipt['preserved']);hashes.update(receipt['generated'])
    hashes.update({str(f):digest(f)for f in [*P.glob('*.py'),P/'policy.cpp',P/'PLAN_ZH.md']})
    hashes.update({f:digest(f) for f in models.values()})
    for f,h in hashes.items():assert digest(f)==h,f
    save(P/'PROTOCOL.json',dict(models=models,bonuses=BONUSES,seed_start=69510000,seed_count=100,hashes=hashes,
         retraining=False,greedy_only=True,all_configs_fixed_before_scores=True))
    pool=runtime.make_pool();checks(pool,models['aux_r0']);tic=time.perf_counter()
    final=P/'evaluation';final.mkdir(exist_ok=False)
    r=rollout(pool,'',0,69510000);store_result(final/'keep',r)
    overall={};n=1400
    for key,checkpoint in models.items():
        for bi,bonus in enumerate(BONUSES):
            r=rollout(pool,checkpoint,100+bi,69510000)
            name=f'{key}_b{bi}';store_result(final/name,r);s=summary(r)
            active=r['mask'].sum(1)>1
            s.update(bonus=bonus,active_nonkeep_rate=float((r['choice'][active]!=0).mean()),
                     checkpoint=checkpoint,checkpoint_sha=digest(checkpoint))
            save(final/name/'detail.json',s);overall[name]=s;n+=1400
            save(P/'PROGRESS.json',dict(status='RUNNING',completed_games=n,total=23800,latest=name,seconds=time.perf_counter()-tic))
            print('EVAL',name,'bonus',bonus,'win',s['overall']['win_rate'],'margin',s['overall']['mean_margin'],
                  'nonkeep',s['non_keep'],'rate',s['active_nonkeep_rate'],flush=True)
    for f,h in hashes.items():assert digest(f)==h,f
    save(P/'RESULTS.json',overall)
    save(P/'RUN_COMPLETE.json',dict(status='PASS',games=n,seconds=time.perf_counter()-tic,source_weights_unchanged=True,
         no_retraining=True,all_games_full_719=True,all_choices_legal=True))
    import subprocess
    subprocess.run([sys.executable,str(P/'report.py')],check=True)
    print('COMPLETE',flush=True)

if __name__=='__main__':main()
