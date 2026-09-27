"""Connect numeric observations, model decisions, and worker execution."""
import numpy as np
import torch
from bc_runtime import encode,market_features,market_order,MARKET,ITEMS,CROPS,ANIMALS,JobExecutor,job_key,prune_pending,program_features

def job_catalog():
    choices=[None]
    types=[('PLANT',x) for x in CROPS]+[('BUILD_PASTURE',None),('BUILD_COOP',None)]
    types += [('PLACE',x) for x in ANIMALS]+[('DIG',None)]
    for op,item in types:
        for y in range(10):
            for x in range(10):choices.append(dict(op=op,item=item,x=x,y=y))
    features=np.zeros((len(choices),64),np.float32);features[:,32]=1;features[0,0]=1
    ops=('PLANT','BUILD_PASTURE','BUILD_COOP','PLACE','DIG')
    for i,j in enumerate(choices[1:],1):
        features[i,1+ops.index(j['op'])]=1
        if j['item']:features[i,8+ITEMS.index(j['item'])]=1
        features[i,30:32]=[j['x']/9,j['y']/9]
    return choices,features

def contextual_job_features(jobs,base,board,global_features):
    """Current own-tile and supply features; no replay targets or future state."""
    result=base.copy()
    x=np.rint(base[1:,30]*9).astype(np.int64);y=np.rint(base[1:,31]*9).astype(np.int64)
    result[1:,33:57]=board[:,y,x].T
    items=base[1:,8:20];present=items.sum(-1)>0;item_index=items.argmax(-1)
    result[1:,57]=np.where(present&(item_index<5),global_features[11+np.minimum(item_index,4)],0)
    result[1:,58]=np.where(present,global_features[16+item_index],0)
    result[1:,59]=np.where(present,global_features[28+item_index],0)
    return result

class ReplayPolicy:
    def __init__(self,model,quantities,device='cpu',goal_decode='joint',state_job_features=False):
        if goal_decode not in ('joint','grouped'):raise ValueError('Unknown goal decoding mode')
        self.goal_decode=goal_decode
        self.state_job_features=state_job_features
        if not quantities or any(type(x)!=int or x<0 for x in quantities) or not any(x>0 for x in quantities):raise ValueError('Nonnegative integer vocabulary with positive quantities required')
        if len(set(quantities))!=len(quantities):raise ValueError('Duplicate quantity vocabulary')
        if model.amount[-1].out_features!=len(quantities):raise ValueError('Checkpoint quantity dimension mismatch')
        self.model=model.to(device);self.device=device;self.quantities=list(quantities)
        # Keep checkpoint indices intact; zero-quantity replay requests had no amount loss.
        self.quantity_legal=torch.tensor([[x>0 for x in quantities]],device=device)
        self.market_features=torch.tensor(market_features(),device=device)[None]
        self.jobs,features=job_catalog();self.job_features=torch.tensor(features,device=device)[None]
        self.base_job_features=features
        self.reset()

    def reset(self):
        self.actor=torch.zeros(1,256,device=self.device);self.critic=self.actor.clone()
        self.executor=JobExecutor();self.last_step=None

    @torch.no_grad()
    def act(self,obs,greedy=True):
        step=obs.get('step',obs['day']*24+obs['hour']);obs=dict(obs,step=step)
        if step==0:self.reset()
        if self.last_step is not None and step!=self.last_step+1:raise ValueError('Recurrent policy requires consecutive turns or reset')
        x=encode(obs,self.executor.pending)
        self.executor.pending=prune_pending(self.executor.pending,x['boards'][0],step)
        x['programs'],x['program_valid']=program_features(self.executor.pending,step)
        tensors=[torch.as_tensor(x[k],device=self.device)[None] for k in ('boards','global_features','market','programs','program_valid')]
        self.actor,self.critic,_=self.model.encode(*tensors,self.actor,self.critic)
        def choose(logits):
            return int(logits.argmax(-1).item()) if greedy else int(torch.distributions.Categorical(logits=logits).sample().item())
        prefix=torch.zeros_like(self.actor);selected=[]
        job_features=self.job_features
        if self.state_job_features:
            job_features=torch.tensor(contextual_job_features(self.jobs,self.base_job_features,x['boards'][0],x['global_features']),device=self.device)[None]
        pending={job_key(j) for j in self.executor.pending};farm=obs['farms'][obs['player']]
        legal=torch.tensor([[True]+[farm['tiles'][j['y']][j['x']]!='LOCKED' and job_key(j) not in pending for j in self.jobs[1:]]],device=self.device)
        for _ in range(16-len(pending)):
            logits,encoded=self.model.score(self.actor,prefix,job_features,legal)
            idx=choose(logits)
            if greedy and self.goal_decode=='grouped':
                # Diagnostic: compare STOP with total probability of any job.
                idx=0 if logits[0,0]>=torch.logsumexp(logits[0,1:],dim=0) else int(logits[0,1:].argmax())+1
            if idx==0:break
            selected.append(self.jobs[idx]);legal[0,idx]=False
            prefix=self.model.advance(prefix,encoded[:,idx],torch.zeros(1,device=self.device))
        self.executor.add(selected,step)
        action=self.executor.workers(obs)
        # Market decisions condition on the jobs selected in this same turn.
        for _ in range(10):
            logits,encoded=self.model.score(self.actor,prefix,self.market_features,torch.ones(1,len(MARKET),dtype=torch.bool,device=self.device))
            idx=choose(logits);quantity=0
            if MARKET[idx][1] is not None:
                q=self.model.quantities(self.actor,encoded[:,idx],self.quantity_legal)
                quantity=self.quantities[choose(q)]
            order=market_order(idx,quantity)
            if order is None:break
            action['market'].append(order)
            prefix=self.model.advance(prefix,encoded[:,idx],torch.tensor([quantity],device=self.device))
        self.last_step=step
        return action
