"""Sparse read-only diagnostics; no optimizer steps or RNG consumption."""
import time
from pathlib import Path
import torch
from torch.nn import functional as F
from exact_training import to_device
from ppo.rollout import batch_windows

def vectors(grads,params):
    return torch.cat([(torch.zeros_like(p) if g is None else g).detach().float().flatten() for g,p in zip(grads,params)])

def _diagnose(scorer,reference,probe_rows,episodes,args,device,stage):
    tick=time.perf_counter();model=scorer.model; rows=[]
    # autograd.grad does not write parameter .grad or step Adam.
    named=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    params=[p for _,p in named]
    choices=[]
    families=sorted({e['assignment']['family'] for e in episodes})
    seen_windows=set()
    for key,original in probe_rows.items():
        if key[:3] in seen_windows:continue
        seen_windows.add(key[:3])
        if not original:continue
        _,begin,end,burn=original[0]
        for family in families:
            ep=next(e for e in episodes if e['assignment']['family']==family)
            choices.append((ep,begin,end,burn))
    for ep,begin,end,burn in choices:
        batch,state,refstate=batch_windows([(ep,begin,end,burn)])
        batch=to_device(batch,device);state=tuple(x.to(device) for x in state);refstate=tuple(x.to(device) for x in refstate)
        bank=Path(args.output).parent/'retained-behavior'
        bank.mkdir(exist_ok=True)
        rank=torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        target=bank/f"rank-{rank}-{ep['assignment']['family']}-phase-{(begin+burn)//240}.pt"
        if stage=='before' and not target.exists():
            def cpu(x):
                if torch.is_tensor(x):return x.detach().cpu().clone()
                if isinstance(x,dict):return {k:cpu(v) for k,v in x.items()}
                if isinstance(x,(tuple,list)):return type(x)(cpu(v) for v in x)
                return x
            prefixes=[]
            for start in range(0,begin,48):
                prefix,_,_=batch_windows([(ep,start,min(begin,start+48),0)]);prefixes.append(cpu(prefix['x']))
            torch.save(cpu(dict(batch=batch,state=state,reference_state=refstate,burn=burn,prefixes=prefixes,
                update=args.update_number,reference_sha=batch.get('reference_sha'),assignment=ep['assignment'])),target)
        mask=batch['loss_mask'];result=scorer(batch,state,burn,reference,refstate)
        ratio=(result['logp']-batch['old_logp']).clamp(-20,20).exp()
        factors=result['factor_count'].clamp_min(1)
        terms=dict(policy=-torch.minimum(ratio*batch['advantages'],ratio.clamp(1-args.clip,1+args.clip)*batch['advantages']),
            value=.5*F.cross_entropy(result['logits'],batch['outcome_turn'],reduction='none'),
            entropy=-.001*result['entropy']/factors,reference=args.anchor*result['kl']/factors)
        gradients={n:vectors(torch.autograd.grad(t[mask].mean(),params,retain_graph=True,allow_unused=True),params) for n,t in terms.items()}
        norms={n:float(g.norm()) for n,g in gradients.items()}
        alignment={n:float(torch.dot(g,gradients['policy'])/(g.norm()*gradients['policy'].norm()).clamp_min(1e-20)) for n,g in gradients.items() if n!='policy'}
        with torch.no_grad():
            probs=result['logits'][mask].softmax(-1);labels=F.one_hot(batch['outcome_turn'][mask],3)
            item=dict(first=begin+burn,family=ep['assignment']['family'],opponent_sha=ep['assignment'].get('opponent_sha256'),
                weighted_gradient_norms=norms,cosine_to_policy=alignment,value_brier=float((probs-labels).square().sum(-1).mean()),
                value_ce=float(F.cross_entropy(result['logits'][mask],batch['outcome_turn'][mask])),
                teacher_factor_kl=float((result['kl']/factors)[mask].mean()),joint_density_error=float((result['logp']-batch['old_logp'])[mask].abs().max()))
        groups={}
        offset=0
        for name,param in named:
            group='critic' if name.startswith(('critic_memory.','value.')) else ('actor' if name.startswith(('actor_memory.','worker_head.','market_head.','post_worker_context.')) else 'shared')
            groups.setdefault(group,[]).append((offset,offset+param.numel()));offset+=param.numel()
        item['gradient_groups']={}
        for group,spans in groups.items():
            gs={n:torch.cat([g[a:b] for a,b in spans]) for n,g in gradients.items()}
            item['gradient_groups'][group]=dict(norms={n:float(g.norm()) for n,g in gs.items()},cosine_to_policy={n:float(torch.dot(g,gs['policy'])/(g.norm()*gs['policy'].norm()).clamp_min(1e-20)) for n,g in gs.items() if n!='policy'})
        del terms,gradients,ratio,result,gs
        if stage in ('before','after'):
            # Reconstruct current-weight hidden state from the full observable prefix.
            # Only the encoder/RNN is needed; no actions or future observations enter it.
            full=tuple(torch.zeros_like(x) for x in state)
            with torch.no_grad():
                for start in range(0,begin,48):
                    stop=min(begin,start+48)
                    prefix,_,_=batch_windows([(ep,start,stop,0)]);prefix=to_device(prefix,device)
                    token,_=model.encode_features(prefix['x']);token=token.reshape(stop-start,1,-1)
                    for token_t in token:
                        full=(model.actor_memory(token_t,full[0]),model.critic_memory(token_t,full[1]))
                old=scorer(batch,state,burn);fresh=scorer(batch,full,burn)
                d=(fresh['logp']-old['logp'])[mask]
                recorded=batch['old_logp'][mask];adv=batch['advantages'][mask]
                normal_ratio=(old['logp'][mask]-recorded).clamp(-20,20).exp()
                fresh_ratio=(fresh['logp'][mask]-recorded).clamp(-20,20).exp()
                def clipped_branch(r):return r*adv > r.clamp(1-args.clip,1+args.clip)*adv
                full_density_error=float((fresh['logp'][mask]-recorded).abs().max())
                if stage=='before' and full_density_error>.002 and not getattr(args,'async_collect',0):  # async-v1: recorded, not enforced (separate collector process)
                    raise RuntimeError('Full-prefix collection-weight density mismatch')
                item['recurrent_reconstruction']=dict(mean_abs_logp=float(d.abs().mean()),p95_abs_logp=float(torch.quantile(d.abs(),.95)),max_abs_logp=float(d.abs().max()),
                    full_prefix_vs_recorded_max=full_density_error,clipping_branch_change_fraction=float((clipped_branch(normal_ratio)!=clipped_branch(fresh_ratio)).float().mean()),
                    mean_value_difference=float((fresh['value']-old['value'])[mask].abs().mean()))
        if target.exists():
            retained=torch.load(target,map_location=device,weights_only=True)
            rb=retained['batch'];full=tuple(torch.zeros_like(x) for x in retained['state'])
            with torch.no_grad():
                for features in retained.get('prefixes',[]):
                    tokens,_=model.encode_features(features)
                    for token_t in tokens[:,None,:]:full=(model.actor_memory(token_t,full[0]),model.critic_memory(token_t,full[1]))
                rr=scorer(rb,full,retained['burn']);rm=rb['loss_mask']
                item['retained_bank']=dict(recorded_update=retained['update'],reference_sha=retained['reference_sha'],
                    mean_logp_change=float((rr['logp']-rb['old_logp'])[rm].mean()),
                    mean_abs_logp_change=float((rr['logp']-rb['old_logp'])[rm].abs().mean()),
                    value_ce=float(F.cross_entropy(rr['logits'][rm],rb['outcome_turn'][rm])))
        rows.append(item)
    return dict(stage=stage,rows=rows,seconds=time.perf_counter()-tick,
        scope='Same on-policy windows, not held-out value calibration; full-prefix diagnostic after update; no parameter mutation')


def diagnose(scorer,reference,probe_rows,episodes,args,device,stage):
    params=list(scorer.model.parameters());before=[p.detach().clone() for p in params]
    grads=[None if p.grad is None else p.grad.detach().clone() for p in params]
    cpu=torch.get_rng_state();cuda=torch.cuda.get_rng_state(device) if device.type=='cuda' else None
    result=_diagnose(scorer,reference,probe_rows,episodes,args,device,stage)
    if not all(torch.equal(p.detach(),v) for p,v in zip(params,before)):raise RuntimeError('Diagnostic mutated model')
    if not all((p.grad is None and g is None) or (p.grad is not None and g is not None and torch.equal(p.grad,g)) for p,g in zip(params,grads)):raise RuntimeError('Diagnostic mutated gradients')
    if not torch.equal(cpu,torch.get_rng_state()):raise RuntimeError('Diagnostic consumed CPU RNG')
    if cuda is not None and not torch.equal(cuda,torch.cuda.get_rng_state(device)):raise RuntimeError('Diagnostic consumed CUDA RNG')
    result['read_only_verified']=True
    return result
