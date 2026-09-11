"""Training-only candidate-conditioned future heads; native ABI is unchanged."""
from common import *
import torch
from torch import nn
from model import Model, advantages, tensor_batch

HORIZONS=(1,3,7)
LABELS=('cash','margin','standing_yield','water_due','feed_due')

def outcomes(data):
    x=np.asarray(data['global']); day=np.asarray(data['day']); game=np.asarray(data['game'])
    # Exact feature layout in frozen rl_common.hpp. These are observed snapshots,
    # not forecasts and not cumulative harvest/sales.
    measures=np.stack((x[:,2],x[:,4],x[:,72:81].sum(1),x[:,82],x[:,83]),-1)
    target=np.zeros((len(x),3,5),np.float32);mask=np.zeros_like(target,dtype=bool)
    for hi,h in enumerate(HORIZONS):
        i=np.arange(len(x)-h);good=(game[i]==game[i+h])&(day[i+h]==day[i]+h)
        i=i[good];target[i,hi]=measures[i+h]-measures[i];mask[i,hi]=True
    return target,mask

class AuxModel(Model):
    def __init__(self):
        super().__init__()
        self.future=nn.Linear(128,15)
        nn.init.zeros_(self.future.bias)
        nn.init.normal_(self.future.weight,std=.01)
    def forecast(self,x,c,choice):
        selected=c[torch.arange(len(c),device=c.device),choice.long()]
        hidden=self.actor[:-1](torch.cat((x,selected),-1))
        return self.future(hidden).reshape(-1,3,5)
    def export(self,path):
        # Never export auxiliary weights; C++ actor/critic sees exactly 70,402 floats.
        params=list(self.actor.parameters())+list(self.critic.parameters())
        arr=np.concatenate([p.detach().cpu().numpy().astype('<f4').ravel()for p in params])
        arr.tofile(path);assert len(arr)==70402
        return len(arr)

def update_aux(model,opt,data,device,seed,weight):
    if data.get('mode')!=2:raise ValueError('PPO requires on-policy stochastic rollouts')
    tic=time.perf_counter();b=tensor_batch(data,device)
    target_aux,mask_aux=outcomes(data)
    ya=torch.as_tensor(target_aux,device=device);ma=torch.as_tensor(mask_aux,device=device)
    if str(device).startswith('cuda'):torch.cuda.synchronize()
    transfer=time.perf_counter()-tic
    adv,ret=advantages(data);active=b['mask'].sum(1)>1
    assert active.any()
    av=torch.as_tensor(adv,device=device);target=torch.as_tensor(ret,device=device)
    av=(av-av[active].mean())/(av[active].std(unbiased=False)+1e-8)
    gen=np.random.default_rng(seed);stats=[]
    for epoch in range(4):
        for ix in np.array_split(gen.permutation(len(adv)),max(1,int(np.ceil(len(adv)/512)))):
            j=torch.as_tensor(ix,device=device)
            d,value=model(b['global'][j],b['candidates'][j],b['mask'][j])
            lp=d.log_prob(b['choice'][j].long());ratio=torch.exp(lp-b['logp'][j]);valid=active[j]
            policy=-torch.minimum(ratio*av[j],ratio.clamp(.8,1.2)*av[j])[valid].mean() if valid.any() else lp.sum()*0
            vloss=.5*(value-target[j]).square().mean();ent=d.entropy()[valid].mean()if valid.any()else lp.sum()*0
            loss=policy+.5*vloss-.01*ent
            aux=loss.detach()*0
            if weight and ma[j].any():
                pred=model.forecast(b['global'][j],b['candidates'][j],b['choice'][j])
                aux=nn.functional.smooth_l1_loss(pred[ma[j]],ya[j][ma[j]])
                loss=loss+weight*aux
            if not torch.isfinite(loss):raise RuntimeError('nonfinite loss')
            opt.zero_grad();loss.backward();grad=nn.utils.clip_grad_norm_(model.parameters(),.5);opt.step()
            with torch.no_grad():
                stats.append([loss.item(),vloss.item(),ent.item(),((ratio-1)-(lp-b['logp'][j])).mean().item(),
                 ((ratio-1).abs()>.2).float().mean().item(),float(grad),float(aux)])
    if str(device).startswith('cuda'):torch.cuda.synchronize()
    out=dict(zip(('loss','value_loss','entropy','approx_kl','clip_fraction','gradient_norm','aux_loss'),np.mean(stats,axis=0).tolist()))
    out.update(host_to_device_seconds=transfer,peak_gpu_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
    return out

def forecast_metrics(model,data):
    y,m=outcomes(data)
    with torch.no_grad():
        preds=[]
        for start in range(0,len(y),2048):
            sl=slice(start,start+2048)
            x=torch.as_tensor(data['global'][sl],device='cuda');c=torch.as_tensor(data['candidates'][sl],device='cuda')
            a=torch.as_tensor(data['choice'][sl],device='cuda');preds.append(model.forecast(x,c,a).cpu().numpy())
    pred=np.concatenate(preds);out={}
    for hi,h in enumerate(HORIZONS):
        for k,name in enumerate(LABELS):
            good=m[:,hi,k];truth=y[good,hi,k];guess=pred[good,hi,k]
            out[f'{h}d_{name}']=dict(mse=float(np.mean((guess-truth)**2)),persistence_mse=float(np.mean(truth**2)),samples=int(good.sum()))
    return out
