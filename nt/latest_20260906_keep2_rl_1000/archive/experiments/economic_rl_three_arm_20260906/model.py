import numpy as np
import torch
import time
from torch import nn
G,C,A,H=128,32,10,128
class Model(nn.Module):
 def __init__(self):
  super().__init__()
  self.actor=nn.Sequential(nn.Linear(G+C,H),nn.Tanh(),nn.Linear(H,H),nn.Tanh(),nn.Linear(H,1))
  self.critic=nn.Sequential(nn.Linear(G,H),nn.Tanh(),nn.Linear(H,H),nn.Tanh(),nn.Linear(H,1))
  for n in self.modules():
   if isinstance(n,nn.Linear):nn.init.orthogonal_(n.weight,gain=1.);nn.init.zeros_(n.bias)
  nn.init.zeros_(self.actor[-1].weight)
 def forward(self,x,c,mask):
  z=torch.cat((x[:,None,:].expand(-1,A,-1),c),dim=-1)
  logits=self.actor(z).squeeze(-1)
  prior=torch.zeros_like(logits);prior[:,0]=3.58351893846
  logits=(logits+prior).masked_fill(~mask.bool(),-1e30)
  return torch.distributions.Categorical(logits=logits),self.critic(x).squeeze(-1)
 def export(self,path):
  arr=np.concatenate([p.detach().cpu().numpy().astype('<f4').ravel()for p in self.parameters()]);arr.tofile(path);return len(arr)
def tensor_batch(data,device):
 return {k:torch.as_tensor(data[k],device=device)for k in ('global','candidates','mask','choice','logp','value','reward','game')}
def advantages(data):
 v=np.asarray(data['value']);r=np.asarray(data['reward']);g=np.asarray(data['game']);out=np.zeros_like(v)
 for i in range(len(v)-1,-1,-1):
  end=i==len(v)-1 or g[i+1]!=g[i]
  delta=r[i]-v[i]+(0 if end else v[i+1]);out[i]=delta+(0 if end else .95*out[i+1])
 return out,out+v
def update(model,opt,data,device,seed):
 if data.get('mode')!=2:raise ValueError('PPO accepts only this model stochastic on-policy rollouts')
 tic=time.perf_counter();b=tensor_batch(data,device)
 if str(device).startswith('cuda'):torch.cuda.synchronize()
 transfer_seconds=time.perf_counter()-tic
 adv,ret=advantages(data);active=b['mask'].sum(1)>1
 if not active.any():raise RuntimeError('No genuine actor choices in batch')
 av=torch.as_tensor(adv,device=device);target=torch.as_tensor(ret,device=device)
 av=(av-av[active].mean())/(av[active].std(unbiased=False)+1e-8)
 gen=np.random.default_rng(seed);stats=[]
 for epoch in range(4):
  for ix in np.array_split(gen.permutation(len(adv)),max(1,int(np.ceil(len(adv)/512)))):
   j=torch.as_tensor(ix,device=device);d,value=model(b['global'][j],b['candidates'][j],b['mask'][j]);lp=d.log_prob(b['choice'][j].long());ratio=torch.exp(lp-b['logp'][j]);valid=active[j]
   policy=-torch.minimum(ratio*av[j],ratio.clamp(.8,1.2)*av[j])[valid].mean() if valid.any() else lp.sum()*0
   vloss=.5*(value-target[j]).square().mean();ent=d.entropy()[valid].mean() if valid.any()else lp.sum()*0
   loss=policy+.5*vloss-.01*ent
   if not torch.isfinite(loss):raise RuntimeError('PPO nonfinite loss')
   opt.zero_grad();loss.backward();grad=nn.utils.clip_grad_norm_(model.parameters(),.5);opt.step()
   with torch.no_grad():
    stats.append([loss.item(),vloss.item(),ent.item(),((ratio-1)-(lp-b['logp'][j])).mean().item(),((ratio-1).abs()>.2).float().mean().item(),float(grad)])
 if str(device).startswith('cuda'):torch.cuda.synchronize()
 result=dict(zip(('loss','value_loss','entropy','approx_kl','clip_fraction','gradient_norm'),np.mean(stats,axis=0).tolist()))
 result['host_to_device_seconds']=transfer_seconds
 result['peak_gpu_allocated_mib']=torch.cuda.max_memory_allocated()/2**20 if str(device).startswith('cuda') else 0
 return result
