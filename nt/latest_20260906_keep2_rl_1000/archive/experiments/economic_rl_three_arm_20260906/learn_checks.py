import os
for n in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[n]='1'
import time,json,hashlib
import numpy as np
import torch
from runtime import *
from model import Model,tensor_batch,advantages,update
def main():
 assert read(P/'G1_RECEIPT.json')['status']=='PASS'
 out=P/'learn_checks_v3';out.mkdir(exist_ok=False)
 torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.backends.cuda.matmul.allow_tf32=False
 torch.manual_seed(101);model=Model().cuda();count=model.export(out/'initial.bin');assert count==native.RL_NW
 pool=make_pool();receipts=[]
 for arm in ('c3auto','f3','c3j7'):
  r=rollout(pool,arm,out/'initial.bin',mode=2,start=64001000,count=2,sample=912,trace=False)
  store_result(out/arm,r);assert summary(r)['status']=='PASS'
  b=tensor_batch(r,'cuda')
  with torch.no_grad():d,v=model(b['global'],b['candidates'],b['mask']);lp=d.log_prob(b['choice'].long())
  errors=dict(probability=float(np.max(np.abs(d.probs.cpu().numpy()-r['probability']))),logp=float(torch.max(torch.abs(lp-b['logp']))),value=float(torch.max(torch.abs(v-b['value']))))
  assert max(errors.values())<2e-5,errors
  assert float(torch.max(torch.abs(torch.exp(lp-b['logp'])-1)))<2e-5
  assert np.all(r['reward'][r['day']<29]==0) and np.all(r['mask'][:,0]==1)
  assert np.all(np.sum(r['mask'],1)>=1) and np.isfinite(r['global']).all()
  # Consecutive episode boundary must break reward/advantage propagation.
  av,target=advantages(r);dummy={k:np.array(r[k],copy=True)for k in ('value','reward','game')};dummy['reward'][-1]+=100
  av2,_=advantages(dummy);assert np.array_equal(av[r['game']!=r['game'][-1]],av2[r['game']!=r['game'][-1]])
  fresh=Model().cuda();fresh.load_state_dict(model.state_dict());opt=torch.optim.Adam(fresh.parameters(),lr=3e-4)
  torch.cuda.synchronize();start=time.perf_counter();stats=update(fresh,opt,r,'cuda',101);torch.cuda.synchronize();seconds=time.perf_counter()-start
  with torch.no_grad():d2,_=fresh(b['global'],b['candidates'],b['mask'])
  delta=float(torch.max(torch.abs(d2.probs-d.probs)));assert delta>1e-6
  fresh.export(out/f'{arm}_after.bin')
  again=rollout(pool,arm,out/f'{arm}_after.bin',mode=2,start=64001000,count=2,sample=912)
  store_result(out/f'{arm}_after',again);assert summary(again)['status']=='PASS'
  different=int(np.sum(r['choice']!=again['choice']))
  receipts.append(dict(arm=arm,parameters=count,errors=errors,ppo=stats,ppo_seconds=seconds,probability_change=delta,trajectory_choice_difference=different,before=summary(r),after=summary(again)))
  print(arm,errors,stats,'update_seconds',seconds,'choice_change',different,flush=True)
 save(P/'G2_RECEIPT.json',dict(status='PASS',torch=torch.__version__,device=torch.cuda.get_device_name(0),receipts=receipts))
if __name__=='__main__':main()
