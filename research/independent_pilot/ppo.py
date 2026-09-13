"""Terminal-reward self-play, full-game recurrence and ordered PPO replay."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from . import native
from .policy import Policy,Stage,save
from .runtime import buffers,dump


@torch.no_grad()
def collect(policy,seeds,rivals=None):
    n=len(seeds);bsize=2*n
    engine=native.Batch([int(s) for s in seeds],1);engine.action_modes([1]*bsize);b=buffers(n)
    memory=torch.zeros(bsize,policy.width);xs=[];lps=[];vs=[];stages=[[] for _ in range(28)]
    mask=torch.ones(bsize,dtype=torch.bool)
    # Half the games use the same frozen independent controls as E1; half use self-play.
    if rivals:
        for env in range(n//2): mask[env*2+(env%2)]=False
    start=time.perf_counter()
    for t in range(719):
        if rivals:
            for env in range(n//2):engine.planned(env,env%2,rivals[env%len(rivals)])
        engine.begin(b['tiles'],b['glob']);x=policy.inputs(b['tiles'],b['glob']);xs.append(x.clone())
        h,memory,value=policy.step(x,memory);vs.append(value);prefix=torch.zeros_like(h);lp=torch.zeros(bsize)
        for depth in range(29):
            count,rows=engine.next(b['features'],b['owners'],b['bounds'],b['offsets'],b['ids'])
            if not count:break
            if depth==28:raise RuntimeError('decoder depth')
            f=torch.from_numpy(b['features'][:rows].copy());owners=torch.from_numpy(b['owners'][:rows].copy())
            bounds=torch.from_numpy(b['bounds'][:rows].copy());off=torch.from_numpy(b['offsets'][:count+1].copy())
            ids=torch.from_numpy(b['ids'][:count].copy())
            selected,qty,logp,_,nxt=policy.score(h[ids],prefix[ids],f,owners,bounds,off)
            prefix[ids]=nxt;lp[ids]+=logp
            stages[depth].append(Stage(f,owners,bounds,off,ids+t*bsize,selected,qty))
            engine.apply(selected.numpy(),qty.numpy())
        lps.append(lp);result=engine.step()
    assert (result[:,2]==1).all() and (result[:,3]==719).all()
    rewards=np.sign(result[:,0]-result[:,1]);outcome=torch.tensor(np.stack((rewards,-rewards),1).reshape(-1),dtype=torch.float32)
    packed=[]
    for depth in stages:
        if not depth:continue
        f=[];own=[];bound=[];off=[0];samples=[];sel=[];qty=[];count=0
        for s in depth:
            f.append(s.features);own.append(s.owners+count);bound.append(s.bounds)
            off.extend((s.offsets[1:]+off[-1]).tolist());samples.append(s.samples);sel.append(s.selected);qty.append(s.quantity)
            count+=len(s.samples)
        packed.append(Stage(torch.cat(f),torch.cat(own),torch.cat(bound),torch.tensor(off),torch.cat(samples),torch.cat(sel),torch.cat(qty)))
    return {'x':torch.stack(xs),'logp':torch.stack(lps),'value':torch.stack(vs),'outcome':outcome,'stages':packed,'mask':mask[None].expand(719,-1).clone(),
            'games':n,'transitions':719*n,'player_turns':719*int(mask.sum()),'simulated_player_turns':719*bsize,'seconds':time.perf_counter()-start,
            'cash':result[:,:2].tolist(),'seeds':[int(s) for s in seeds]}


def replay(policy,h,stages):
    h=h.reshape(-1,policy.width);prefix=torch.zeros_like(h);lp=torch.zeros(len(h));ent=torch.zeros(len(h));counts=torch.zeros(len(h))
    for s in stages:
        _,_,logp,e,nxt=policy.score(h[s.samples],prefix[s.samples],s.features,s.owners,s.bounds,s.offsets,s.selected,s.quantity)
        prefix=prefix.index_copy(0,s.samples,nxt);lp=lp.index_add(0,s.samples,logp)
        ent=ent.index_add(0,s.samples,e);counts=counts.index_add(0,s.samples,torch.ones_like(e))
    return lp,ent/counts.clamp_min(1)


def update(policy,optimizer,rollout,epochs=2,entropy=.01):
    start=time.perf_counter();target=rollout['outcome'][None].expand_as(rollout['value'])
    mask=rollout['mask'].flatten()
    adv=(target-rollout['value']).flatten();adv=(adv-adv[mask].mean())/adv[mask].std(unbiased=False).clamp_min(1e-6)
    metrics=[]
    for epoch in range(epochs):
        h,v=policy.sequence(rollout['x']);lp,ent=replay(policy,h,rollout['stages'])
        ratio=(lp-rollout['logp'].flatten()).exp()
        objective=torch.minimum(ratio*adv,ratio.clamp(.8,1.2)*adv)[mask].mean()
        value_loss=(v-target).square().flatten()[mask].mean();loss=-objective+.5*value_loss-entropy*ent[mask].mean()
        if not torch.isfinite(loss):raise RuntimeError('nonfinite PPO loss')
        optimizer.zero_grad();loss.backward();norm=torch.nn.utils.clip_grad_norm_(policy.parameters(),1.,error_if_nonfinite=True)
        optimizer.step()
        metrics.append({'loss':float(loss.detach()),'value_loss':float(value_loss.detach()),'gradient_norm':float(norm),
                        'approx_kl':float(((ratio-1)-(lp-rollout['logp'].flatten())).mean().detach())})
    return {'seconds':time.perf_counter()-start,'epochs':metrics}


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--seconds',type=float,required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--envs',type=int,default=4);p.add_argument('--entropy',type=float,default=.01)
    p.add_argument('--lr',type=float,default=.0003);p.add_argument('--seed-manifest',type=Path,required=True);args=p.parse_args()
    torch.set_num_threads(1);torch.manual_seed(args.seed);rng=np.random.default_rng(args.seed)
    from .plans import simple_plan,repair
    from .protocol import verify_freeze
    verify_freeze(args.seed_manifest.parent/'FREEZE.json')
    args.out.mkdir(parents=True,exist_ok=False);policy=Policy();optimizer=torch.optim.Adam(policy.parameters(),lr=args.lr,eps=1e-5)
    seeds=json.loads(args.seed_manifest.read_text())['train'];journal=[];start=time.perf_counter();cpu_start=time.process_time();games=turns=0
    rivals=[repair(simple_plan(k)) for k in (0,4,6)]
    save(policy,args.out/'untrained.pt',{'seed':args.seed,'games':0})
    while time.process_time()-cpu_start<args.seconds:
        # Rotate the frozen controls across updates; actor never receives their ids.
        rivals=rivals[1:]+rivals[:1]
        r=collect(policy,rng.choice(seeds,size=args.envs,replace=False),rivals);m=update(policy,optimizer,r,entropy=args.entropy)
        games+=r['games'];turns+=r['player_turns']
        row={'update_index':len(journal)+1,'games':games,'player_turns':turns,'elapsed':time.perf_counter()-start,
             'cpu_seconds':time.process_time()-cpu_start,'collect_seconds':r['seconds'],'update':m,'outcomes':r['outcome'].tolist(),'cash':r['cash'],'seeds':r['seeds']}
        journal.append(row)
        with (args.out/'updates.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        dump(args.out/'STATUS.json',{'complete':False,'games':games,'player_turns':turns,'elapsed':row['elapsed']})
        print(json.dumps({'games':games,'player_turns':turns,'elapsed':row['elapsed']}),flush=True)
    meta={'seed':args.seed,'games':games,'player_turns':turns,'seconds':time.perf_counter()-start,'budget_seconds':args.seconds,
          'cpu_seconds':time.process_time()-cpu_start,'entropy':args.entropy,'lr':args.lr,'reward':'terminal win=1 draw=0 loss=-1','complete':True,
          'seed_manifest_sha256':hashlib.sha256(args.seed_manifest.read_bytes()).hexdigest(),'updates':len(journal)}
    verify_freeze(args.seed_manifest.parent/'FREEZE.json')
    save(policy,args.out/'candidate.pt',meta);dump(args.out/'STATUS.json',meta)


if __name__=='__main__':main()
