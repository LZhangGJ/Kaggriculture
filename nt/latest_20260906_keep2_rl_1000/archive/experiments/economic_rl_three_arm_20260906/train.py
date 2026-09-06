"""Frozen-budget synchronous PPO; independent developer and final seed sets."""
import os
for n in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[n]='1'
import time,json,argparse,hashlib
import numpy as np
import torch
from runtime import *
from model import Model,update
def evaluate(pool,arm,path,out,mode=1,count=16,start=62000000,sample=9301):
 r=rollout(pool,arm,path,mode=mode,start=start,count=count,sample=sample)
 store_result(out,r)
 if summary(r)['status']!='PASS':raise RuntimeError(summary(r)['errors'])
 return summary(r)
def main():
 p=argparse.ArgumentParser();p.add_argument('--arm',choices=['c3auto','f3','c3j7'],required=True);p.add_argument('--run',type=int,choices=[0,1],required=True);p.add_argument('--batches',type=int,default=20)
 args=p.parse_args();assert read(P/'G1_RECEIPT.json')['status']=='PASS' and read(P/'G2_RECEIPT.json')['status']=='PASS'
 out=P/'training'/f'{args.arm}_r{args.run}';out.mkdir(parents=True,exist_ok=False)
 frozen={str(f):digest(f)for f in [P/'train.py',P/'model.py',P/'runtime.py',P/'rl_common.hpp',*list((P/'build').glob('*.so'))]}
 save(out/'protocol.json',dict(arguments=vars(args),hashes=frozen,training_seed_start=61000000+args.run*100000,development_seed_start=62000000,final_seed_start=63000000))
 torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.backends.cuda.matmul.allow_tf32=False
 torch.manual_seed(701+args.run);device='cuda';model=Model().to(device);opt=torch.optim.Adam(model.parameters(),lr=3e-4);pool=make_pool()
 model.export(out/'step000.bin');torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict()),out/'step000.pt')
 history=[];selection=[];tic=time.perf_counter()
 def dev(step):
  s=evaluate(pool,args.arm,out/f'step{step:03}.bin',out/f'dev_{step:03}')
  score=(s['overall']['win_rate'],s['overall']['mean_margin'],-step)
  selection.append(dict(step=step,score=score,summary=s));save(out/'selection.json',selection)
  print('DEV',args.arm,args.run,step,s['overall'],flush=True)
 dev(0)
 for step in range(1,args.batches+1):
  path=out/f'step{step-1:03}.bin';start=61000000+args.run*100000+(step-1)*16
  r=rollout(pool,args.arm,path,mode=2,start=start,count=16,sample=70000+args.run*10000+step)
  store_result(out/f'rollout_{step:03}',r);assert summary(r)['status']=='PASS',summary(r)['errors']
  torch.cuda.synchronize();t=time.perf_counter();stats=update(model,opt,r,device,step+args.run*1000);torch.cuda.synchronize();update_seconds=time.perf_counter()-t
  model.export(out/f'step{step:03}.bin');torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),step=step),out/f'step{step:03}.pt')
  record=dict(step=step,training_games=step*224,rollout=summary(r),ppo=stats,update_seconds=update_seconds,elapsed=time.perf_counter()-tic)
  history.append(record);save(out/'progress.json',dict(status='RUNNING',arm=args.arm,run=args.run,history=history));print('TRAIN',args.arm,args.run,step,'wins',summary(r)['overall']['win_rate'],'actor',summary(r)['effective_actor_records'],'rollout_s',round(r['wall_seconds'],2),'ppo_s',round(update_seconds,2),'KL',round(stats['approx_kl'],4),flush=True)
  if step in (5,10,20):dev(step)
 best=max(selection,key=lambda z:tuple(z['score']));save(out/'best.json',best)
 bestpath=out/f'step{best["step"]:03}.bin'
 final=evaluate(pool,args.arm,bestpath,out/'final_best',count=100,start=63000000)
 last=evaluate(pool,args.arm,out/f'step{args.batches:03}.bin',out/'final_last',count=100,start=63000000)if best['step']!=args.batches else final
 # Final stochastic policy is reported separately, never selected using this set.
 stochastic=evaluate(pool,args.arm,out/f'step{args.batches:03}.bin',out/'final_last_stochastic',mode=2,count=100,start=63000000,sample=9901)
 assert all(digest(f)==h for f,h in frozen.items())
 save(out/'FINAL.json',dict(status='PASS',best_step=best['step'],best=final,last=last,stochastic=stochastic,seconds=time.perf_counter()-tic,training_games=args.batches*224,history=history))
 save(out/'progress.json',dict(status='COMPLETE',arm=args.arm,run=args.run,history=history))
 print('DONE',args.arm,args.run,'best',best['step'],final['overall'],flush=True)
if __name__=='__main__':main()
