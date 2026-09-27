"""Synchronous complete-game PPO. Run only after the deferred GPU smoke."""
import os
for _name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_name,'1')
if os.environ.get('PPO_JAX_ONE_DEVICE')=='1' and os.environ.get('LOCAL_RANK') is not None:  # scale-out: one CUDA context per process
    os.environ.setdefault('JAX_CUDA_VISIBLE_DEVICES',os.environ['LOCAL_RANK'])
import argparse
from contextlib import nullcontext
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch import nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from exact_decoder import ExactAgent
from exact_model import ExactWorkerMarketPolicyV1
from exact_identity import validate_identity
from exact_training import to_device
from ppo.replay import ExactPPOChunk, ppo_terms
from ppo.rollout import collect, windows, batch_windows
from ppo.league import load_league, assignments, sha, adapt_history, update_matchups


def now(): return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    temp=Path(str(path)+'.tmp');temp.write_text(json.dumps(value,indent=2));temp.replace(path)


def code_hash():
    h=hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob('*.py')):
        h.update(path.name.encode());h.update(path.read_bytes())
    for name in FEATURE_EXT_FILES:  # v39 feature-ext-v1
        path=Path(__file__).parent.parent/name;h.update(name.encode());h.update(path.read_bytes())
    return h.hexdigest()


HPARAM_KEYS={'lr','clip','critic_lr','ppo_passes','drift_kl','market_entropy_coef','anchor_min','family_focus','arena_bot_workers','family_weighting'}  # hparam-v1 (source-v74; market_entropy_coef added 2026-09-25 18:2xZ)
SCALE_KEYS={'games','arena_games','workers','minibatch','arena_bot_workers','official_workers'}
FEATURE_EXT_FILES=('feature_ext.py','quantity_ext.py','long_market.py','price_impact.py','ext_agent.py')
EXT_PREFIXES=('worker_head.q_extra.','worker_head.q_index','market_head.q_extra.','market_head.q_index','market_long.')


def feature_ext_identity():
    return dict(version='feature-ext-v1',files={n:sha(Path(__file__).parent.parent/n) for n in FEATURE_EXT_FILES},
                ext_lr_scale=float(os.environ.get('PPO_EXT_LR_SCALE','1')))


def load_model(path, device, identity=None):
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    validate_identity(checkpoint['cache_identity'])
    if identity is not None and checkpoint['cache_identity']!=identity:
        raise ValueError('Opponent representation/cache contract differs from learner')
    if (checkpoint['worker_quantities']!=checkpoint['cache_identity']['worker_quantities'] or
        checkpoint['market_quantities']!=checkpoint['cache_identity']['market_quantities']):
        raise ValueError('Quantity vocabulary mismatch')
    from ppo.shaped_reward import attach_shaped_head,load_exact_with_head
    model=attach_shaped_head(ExactWorkerMarketPolicyV1())
    from feature_ext import attach_all,fill_missing  # feature-ext-v1 (audit): quantity-ext + price-impact + long-market
    attach_all(model,checkpoint['worker_quantities'],checkpoint['market_quantities'])
    checkpoint=dict(checkpoint,model=fill_missing(checkpoint['model'],model));load_exact_with_head(model,checkpoint)
    return model.to(device).eval(),checkpoint


def reduce(tensor):
    if dist.is_initialized(): dist.all_reduce(tensor)
    return tensor


def gather(value):
    if not dist.is_initialized(): return [value]
    from ppo import league_runtime
    result=[None]*dist.get_world_size();dist.all_gather_object(result,value,group=league_runtime.CONTROL);return result


class LossModule(nn.Module):
    def __init__(self,model):
        super().__init__();self.scorer=ExactPPOChunk(model)

    def forward(self,batch,state,ref_state,reference,burn,scale,clip,anchor,policy_weight=1.):
        if batch.get("reference_sha")!=getattr(reference,"reference_sha",None):
            raise ValueError("Rollout/replay reference identity mismatch")
        result=self.scorer(batch,state,burn,reference,ref_state)
        scored=slice(burn*batch['batch'] if batch.get('burn_events_skipped') else 0,None)  # v40: burn-in rows carry no events
        if not torch.equal(result['factor_count'][scored],batch['factor_count'][scored].to(result['factor_count'].dtype)):
            raise RuntimeError('PPO factor-count replay mismatch')
        if not torch.isfinite(batch['old_logp']).all() or not torch.isfinite(result['logp']).all():
            raise FloatingPointError('Nonfinite PPO action density')
        if batch.get('check_initial_density',False):
            error=(result['logp'].detach()-batch['old_logp']).abs()[batch['loss_mask']].max()
            if error>.002:raise RuntimeError(f'Initial GPU replay density mismatch: {float(error)}')
        if batch.get('market_entropy_coef'):  # market-entropy-v1
            mc,mq=batch['market_entropy_coef']
            terms,stats=ppo_terms(result,batch['old_logp'],batch['advantages'],batch['outcome_turn'],clip=clip,anchor=anchor,returns=batch['return_turn'],policy_weight=policy_weight,value_loss=getattr(self,'value_loss','huber'),market_ent=mc,market_qent=mq)
        else:
            terms,stats=ppo_terms(result,batch['old_logp'],batch['advantages'],batch['outcome_turn'],clip=clip,anchor=anchor,returns=batch['return_turn'],policy_weight=policy_weight,value_loss=getattr(self,'value_loss','huber'))
        mask=batch['loss_mask']
        loss=terms[mask].sum()*scale
        stats={k:v[mask].sum().detach() for k,v in stats.items()}
        stats['count']=mask.sum().detach()
        coef=float(batch.get('distill_coef',0.) or 0.)
        if coef:  # distill-v1: coef * NLL of the sampled teacher's turn under the policy, labelled rows only
            t=result.get('teacher');keys=('agree_worker','events_worker','agree_market','events_market','agree_quantity','events_quantity','need_mismatch','nonfinite')
            if t is not None:
                valid=t['valid']&mask
                nll=torch.where(valid,-t['logp'],torch.zeros_like(t['logp']))
                loss=loss+coef*nll.sum()*scale
                stats['distill_nll']=nll.detach().sum();stats['distill_rows']=valid.sum().detach().float()
                for k in keys:stats['distill_'+k]=t['stats'][k].detach()
            else:
                z=loss.detach().new_zeros(())
                stats['distill_nll']=z;stats['distill_rows']=z
                for k in keys:stats['distill_'+k]=z
        # A detached diagnostic cannot cause DDP to mark unused parameters as used.
        return loss,stats


def anchor_at(args):
    """v39: --anchor * --anchor-decay**(update - --anchor-start-update), floored at --anchor-min (decay 1 = constant, as before)."""
    decay=float(getattr(args,'anchor_decay',1.) or 1.)
    if decay==1.:return args.anchor
    steps=max(0,int(getattr(args,'update_number',0))-int(getattr(args,'anchor_start_update',0)))
    return max(float(getattr(args,'anchor_min',0.)),args.anchor*decay**steps)


def distill_coef_at(args,update_number):
    """distill-v1: coef * decay**(update - start); 0 when off or below --distill-min-coef."""
    coef=float(getattr(args,'distill_coef',0.) or 0.)
    if not coef:return 0.
    value=coef*float(args.distill_decay)**max(0,update_number-int(args.distill_start_update))
    return value if value>=float(args.distill_min_coef) else 0.


def update(module,reference,episodes,optimizer,device,args,world):
    device=torch.device(device)
    distill_coef=distill_coef_at(args,args.update_number)
    market_coef=(float(getattr(args,'market_entropy_coef',0.) or 0.),float(getattr(args,'market_quantity_entropy_coef',0.) or 0.))  # market-entropy-v1
    if device.type=='cuda': torch.cuda.reset_peak_memory_stats(device)
    started=time.perf_counter()
    stage_events=[]
    def mark(name):
        if device.type=='cuda':
            event=torch.cuda.Event(enable_timing=True);event.record()
            stage_events.append((name,event))
    # start-state-v1: forced-prefix turns (before an episode's 'train_from') never enter the advantage moments
    all_adv=torch.cat([torch.as_tensor(ep['advantages'][ep['train_from']:] if ep.get('train_from') else ep['advantages'],device=device).flatten() for ep in episodes]).double() if episodes else torch.empty(0,device=device,dtype=torch.float64)
    moments=reduce(torch.stack((all_adv.sum(),all_adv.square().sum(),all_adv.new_tensor(all_adv.numel()))))
    mean=moments[0]/moments[2].clamp_min(1)
    std=(moments[1]/moments[2].clamp_min(1)-mean**2).clamp_min(0).sqrt().clamp_min(1e-8)
    mean,std=float(mean),float(std)
    if episodes and 'comparison_advantages' in episodes[0]:
        other=torch.cat([ep['comparison_advantages'] for ep in episodes]).double()
        om=reduce(torch.stack((other.sum(),other.square().sum(),other.new_tensor(other.numel()))))
        mu=float(om[0]/om[2]);sd=float((om[1]/om[2]-mu**2).clamp_min(0).sqrt().clamp_min(1e-8))
        diagnostic=[]
        for family in sorted(set(ep['assignment']['family'] for ep in episodes)):
            selected=[ep for ep in episodes if ep['assignment']['family']==family]
            for phase,(lo,hi) in enumerate([(0,240),(240,480),(480,719)]):
                one=torch.cat([ep['advantages'][lo:hi] for ep in selected]);old=torch.cat([ep['comparison_advantages'][lo:hi] for ep in selected])
                n1=(one-mean)/std;n0=(old-mu)/sd
                diagnostic.append(dict(family=family,phase=phase,count=one.numel(),raw_sign_agreement=float((one.sign()==old.sign()).float().mean()),normalized_sign_agreement=float((n1.sign()==n0.sign()).float().mean()),lambda1_raw_std=float(one.std()),lambda095_raw_std=float(old.std()),lambda1_normalized_std=float(n1.std()),lambda095_normalized_std=float(n0.std()),lambda1_raw_mean=float(one.mean()),lambda095_raw_mean=float(old.mean())))
        atomic_json(args.output/f'advantage-comparison-rank-{dist.get_rank() if dist.is_initialized() else 0}.json',dict(update=args.update_number,rows=diagnostic,normalization='global learner-turn mean/std separately for each lambda'))
    for ep in episodes: ep['advantages']=(ep['advantages']-mean)/std
    groups=windows(episodes,args.burn,args.sequence)
    keys=sorted(set(k for ks in gather(list(groups)) for k in ks))
    if not keys: raise RuntimeError('No valid learner data')
    # Every rank needs a real dummy graph for empty minibatches.
    if any(n==0 for n in gather(len(episodes))): raise RuntimeError('Rank has no learner trajectories; no update performed')
    rng=random.Random(args.seed+args.update_number)
    rng.shuffle(keys)
    # Fixed representative windows, same identities before and after this update.
    # This diagnostic measures final-policy KL rather than averaging intermediate minibatches.
    probe_keys=[]
    for representation in sorted({k[3] for k in keys}):
        available=sorted(k for k in keys if k[3]==representation)
        probe_keys.extend(available[i] for i in (0,len(available)//2,len(available)-1))
    probe_rows={key:tuple(groups.get(key,[])[:16]) for key in probe_keys}
    def audit_probe():
        probe=[]
        scorer=module.module.scorer if hasattr(module,'module') else module.scorer
        with torch.no_grad():
            for key in probe_keys:
                rows=probe_rows[key]
                if not rows:continue
                pb,ps,_=batch_windows(rows);pb=to_device(pb,device);ps=tuple(v.to(device) for v in ps)
                result=scorer(pb,ps,burn=key[2]);mask=pb['loss_mask']
                delta=(result['logp']-pb['old_logp'])[mask]
                kl=delta.clamp(-20,20).exp()-1-delta
                ce=torch.nn.functional.cross_entropy(result['logits'][mask],pb['outcome_turn'][mask],reduction='none')
                probe.append(dict(episode_ids=[str(ep['assignment']['game'])+':'+str(ep['seat']) for ep,_,_,_ in rows],first=key[0]+key[2],representation=key[3],turns=int(mask.sum()),
                    max_density_error=float(delta.abs().max()),mean_kl=float(kl.mean()),
                    p95_kl=float(torch.quantile(kl,.95)),value_ce=float(ce.mean()),
                    mean_factors=float(result['factor_count'][mask].mean())))
        return probe
    audit_before=audit_probe()
    # drift-guard-v1 (source-v35): under async the lagged KL jumps with game-phase data blocks; measure actual movement
    # since the start of this update on the fixed probe windows instead (checked every 10 steps, all ranks together).
    async_drift=bool(getattr(args,'async_collect',0));drift_log=[]
    def probe_logp():
        scorer=module.module.scorer if hasattr(module,'module') else module.scorer
        out=[]
        with torch.no_grad():
            for key in probe_keys:
                rows=probe_rows[key]
                if not rows:continue
                pb,ps,_=batch_windows(rows);pb=to_device(pb,device);ps=tuple(v.to(device) for v in ps)
                out.append(scorer(pb,ps,burn=key[2])['logp'][pb['loss_mask']].float())
        return out
    start_logp=probe_logp() if async_drift else None
    def drift_kl():
        tot=torch.zeros(2,device=device,dtype=torch.float64)
        for a,b in zip(start_logp,probe_logp()):
            d=(b-a).double().clamp(-20,20);tot[0]+=(d.exp()-1-d).sum();tot[1]+=d.numel()
        reduce(tot);return float(tot[0]/tot[1].clamp_min(1))
    diagnostics=None
    if not getattr(args,'async_collect',0) and (args.update_number%20==17 or getattr(args,'hybrid_first_update',False)):  # v24: retention diagnostics need every-turn anchors; off under async
        from ppo.retention_diagnostics import diagnose
        scorer=module.module.scorer if hasattr(module,'module') else module.scorer
        diagnostics=diagnose(scorer,reference,probe_rows,episodes,args,device,stage='before')
    if max((x['max_density_error'] for x in audit_before),default=0)>.002 and not getattr(args,'async_collect',0):
        raise RuntimeError('Pre-update fixed-window replay density mismatch')
    accepted_by_phase=[0,0,0]
    total={};steps=0;stopped_kl=False;stopping_kl=None;recent_kl=[]  # kl-guard-v2: windowed mean
    policy_frozen=args.update_number<getattr(args,'policy_warmup_until',0)  # shaped-reward-v3 value warm-up
    local_batch=max(1,args.minibatch//world)
    schedule=[]
    window_keys=sorted({k[:3] for k in keys});rng.shuffle(window_keys)
    # Every rank shares a schedule; sample order stays rank-local.
    schedule_rng=random.Random(f'ppo-mixed-v1:{args.seed}:{args.update_number}')
    for window_key in window_keys:
        block=[]
        for key in sorted(k for k in keys if k[:3]==window_key):
            rows=groups.get(key,[]);rng.shuffle(rows)
            batches=max(gather((len(rows)+local_batch-1)//local_batch))
            block.extend((key,i) for i in range(batches))
        schedule_rng.shuffle(block)
        schedule.extend(block)
    accepted_by_format={}
    passes=max(1,int(getattr(args,'ppo_passes',1) or 1))  # ppo-passes-v1
    pass_totals={}  # per-pass diagnostics (v18)
    full_schedule=[]
    for pass_index in range(passes):
        order=list(schedule)
        if pass_index:schedule_rng.shuffle(order)
        full_schedule.extend((pass_index,key,i) for key,i in order)
    # prefetch-v1 (source-v22): build the next minibatch on a CPU thread while the GPU runs the current one.
    from concurrent.futures import ThreadPoolExecutor
    def _prep(entry):
        _pi,_key,_i=entry;_rows=groups.get(_key,[])
        _sel=_rows[_i*local_batch:(_i+1)*local_batch];_real=bool(_sel)
        if not _real:_sel=groups[next(iter(groups))][:1]
        return (_sel,_real)+tuple(batch_windows(_sel,teacher=True) if distill_coef else batch_windows(_sel))
    _pool=ThreadPoolExecutor(1);_futs={};ref_cache={}
    def _launch(j):  # v31: GPU-format batches call torch.compile'd code; build those on the main thread (no concurrent dynamo compile)
        if j<len(full_schedule) and full_schedule[j][1][3]!='gpu':_futs[j]=_pool.submit(_prep,full_schedule[j])
    _launch(0)
    _prof=None;_ptrig=Path(args.output)/'PROFILE_UPDATE'  # profile-v1 (source-v57)
    for _idx,(pass_index,key,i) in enumerate(full_schedule):
        if _idx==8 and _prof is None and _ptrig.exists():
            torch.cuda.synchronize();_prof=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]);_prof.__enter__();_pt0=time.perf_counter()
        if _idx==18 and _prof is not None:
            torch.cuda.synchronize();_pw=time.perf_counter()-_pt0;_prof.__exit__(None,None,None)
            _profile_report(_prof,str(Path(args.output)/'profiles'/('update-r%s-u%d.txt'%(os.environ.get('RANK','0'),args.update_number))),_pw,'update steps 8-18, rank %s, update %d'%(os.environ.get('RANK','0'),args.update_number))
            _prof=False
            if os.environ.get('RANK','0')=='0':_ptrig.unlink(missing_ok=True)
        mark('start')
        selected,real,batch,state,refstate=_futs.pop(_idx).result() if _idx in _futs else _prep(full_schedule[_idx])
        _launch(_idx+1)
        batch['check_initial_density']=real and steps==0 and 'gpu_data' in selected[0][0] and not getattr(args,'async_collect',0)
        if not real: batch['loss_mask'].zero_()
        batch=to_device(batch,device);state=tuple(s.to(device) for s in state);refstate=tuple(s.to(device) for s in refstate)
        if passes>1 and real and os.environ.get('PPO_REFERENCE_CACHE','1')=='1':
            # v40 reference-cache: every pass replays the same (key, i) minibatch; the frozen reference's factor densities
            # are computed in the first pass and reused (pinned host copy) by the later ones.
            batch['reference_cache']=ref_cache.setdefault((key,i),{});batch['reference_cache_store']=pass_index<passes-1
            if pass_index==passes-1:ref_cache.pop((key,i),None)
        if distill_coef:batch['distill_coef']=distill_coef
        if market_coef[0] or market_coef[1]:batch['market_entropy_terms']=(bool(market_coef[0]),bool(market_coef[1]));batch['market_entropy_coef']=market_coef  # market-entropy-v1
        count=reduce(batch['loss_mask'].sum().clone()).item()
        if not count: continue
        optimizer.zero_grad(set_to_none=True)
        mark('batch')
        ctx=torch.autocast('cuda',dtype=torch.bfloat16) if args.bf16 else nullcontext()
        with ctx:
            loss,stats=module(batch,state,refstate,reference,selected[0][3],world/count,args.clip,anchor_at(args))
        mark('forward')
        if os.environ.get('PPO_PACK_STATS')=='1':
            names=list(stats);values=reduce(torch.stack([stats[k] for k in names]))
            stats=dict(zip(names,values.unbind()))
        else:
            stats={k:reduce(v.clone()) for k,v in stats.items()}
        kl=float(stats['approx_kl']/count);recent_kl.append(kl);guard_kl=sum(recent_kl[-6:])/len(recent_kl[-6:])
        # kl-guard-v2: stop on the mean of the last six minibatch estimates (one temporal window), or on a single 3x outlier.
        # async-v1: the batch was collected by a policy one update old, so measure movement relative to the first window.
        base_kl=(sum(recent_kl[:6])/len(recent_kl[:6])) if getattr(args,'async_collect',0) else 0.
        if async_drift:
            if kl-base_kl>.3:stopped_kl=True;stopping_kl=kl;break  # catastrophic single-minibatch jump only
            if steps and steps%10==0:
                dk=drift_kl();drift_log.append(round(dk,5))
                if dk>args.drift_kl:stopped_kl=True;stopping_kl=dk;break
        elif guard_kl-base_kl>args.max_kl or kl-base_kl>3*args.max_kl:
            stopped_kl=True;stopping_kl=guard_kl;break
        if not torch.isfinite(loss): raise FloatingPointError('Nonfinite PPO loss')
        mark('statistics')
        loss.backward()
        if policy_frozen:
            from ppo.shaped_reward import freeze_actor_gradients
            freeze_actor_gradients(module,optimizer)
        mark('backward')
        norm=torch.nn.utils.clip_grad_norm_(module.parameters(),.5,error_if_nonfinite=True)
        optimizer.step();steps+=1
        if pass_index==0:accepted_by_phase[min(2,(key[0]+key[2])//240)]+=int(count)
        mark('optimizer')
        for k,v in stats.items(): total[k]=total.get(k,0.)+float(v)
        total['last_gradient_norm']=float(norm)
        pt=pass_totals.setdefault(pass_index,{})
        for k,v in stats.items(): pt[k]=pt.get(k,0.)+float(v)
        pt['steps']=pt.get('steps',0)+1;pt['grad_norm_sum']=pt.get('grad_norm_sum',0.)+float(norm)
        if pass_index==0:accepted_by_format[key[3]]=accepted_by_format.get(key[3],0)+int(count)
        if steps%6==0 and (not dist.is_initialized() or dist.get_rank()==0):
            print(json.dumps(dict(stage='ppo_update',optimizer_steps=steps,
                learner_turns=total.get('count',0),seconds=time.perf_counter()-started)),flush=True)
    _pool.shutdown(wait=False,cancel_futures=True)
    count=total.get('count',0.)
    stage_seconds={}
    if stage_events:
        torch.cuda.synchronize(device)
        for (_,before),(name,after) in zip(stage_events,stage_events[1:]):
            if name!='start':
                try:  # v34: timing is diagnostic only; never fail an update on an unfinished event
                    before.synchronize();after.synchronize();stage_seconds[name]=stage_seconds.get(name,0.)+before.elapsed_time(after)/1000
                except RuntimeError:stage_seconds['timing_errors']=stage_seconds.get('timing_errors',0)+1
    audit_after=audit_probe()
    if diagnostics is not None:
        from ppo.retention_diagnostics import diagnose
        diagnostics['after']=diagnose(scorer,reference,probe_rows,episodes,args,device,stage='after')
        atomic_json(args.output/f"retention-{args.update_number:06d}-rank-{dist.get_rank() if dist.is_initialized() else 0}.json",diagnostics)
    if [x["episode_ids"] for x in audit_before]!=[x["episode_ids"] for x in audit_after]:
        raise RuntimeError("Audit sample identities changed")
    if [x["mean_factors"] for x in audit_before]!=[x["mean_factors"] for x in audit_after]:
        raise RuntimeError("Audit factor counts changed")
    distill=None
    if distill_coef:  # distill-v1 summary (means over labelled teacher rows / events)
        d={k[8:]:total.pop(k) for k in [k for k in total if k.startswith('distill_')]}
        for pt in pass_totals.values():
            for k in [k for k in pt if k.startswith('distill_')]:pt.pop(k)
        rows=max(1.,d.get('rows',0.))
        distill=dict(coef=distill_coef,loss=d.get('nll',0.)/rows,labelled_rows=d.get('rows',0.),labelled_fraction=d.get('rows',0.)/max(1.,total.get('count',0.)),
                     agree_worker=d.get('agree_worker',0.)/max(1.,d.get('events_worker',0.)),agree_market=d.get('agree_market',0.)/max(1.,d.get('events_market',0.)),
                     agree_quantity=d.get('agree_quantity',0.)/max(1.,d.get('events_quantity',0.)),need_mismatch_rows=d.get('need_mismatch',0.),nonfinite_rows=d.get('nonfinite',0.))
    return dict(distill=distill,drift_kl=drift_log,step_kl=[round(x,4) for x in recent_kl],policy_frozen=policy_frozen,accepted_turns_by_format=accepted_by_format,retention_diagnostics=gather(diagnostics),audit_before=gather(audit_before),audit_after=gather(audit_after),accepted_turns_by_phase=accepted_by_phase,update_seconds=time.perf_counter()-started,optimizer_steps=steps,stopped_kl=stopped_kl,stopping_kl=stopping_kl,
                rank_peak_allocated_gib=gather(torch.cuda.max_memory_allocated(device)/2**30 if device.type=='cuda' else 0.),
                rank_stage_seconds=gather(stage_seconds),
                pass_stats=[dict({k:(v/pt['count'] if pt.get('count') and k not in ('count','steps','grad_norm_sum') else v) for k,v in pt.items()},mean_grad_norm=pt['grad_norm_sum']/max(1,pt['steps'])) for _,pt in sorted(pass_totals.items())],
                **{k:(v/count if count and k not in ('count','last_gradient_norm') else v) for k,v in total.items()})


def _restore_rng(saved,rank,device,seed):
    """scale-out-v2 (from v48 scale-out-v1): ranks that existed at save time restore exactly; new ranks get a deterministic seed."""
    rngs=saved['rngs']
    if rank<len(rngs):
        torch.set_rng_state(rngs[rank]['cpu'])
        if device.type=='cuda':torch.cuda.set_rng_state(rngs[rank]['cuda'],device)
        return True
    torch.manual_seed(seed+7919*rank+int(saved.get('update',0)))
    return False


def save_checkpoint(path,model,optimizer,base,contract,update_number,rngs,matchups=None,league_state=None):
    value={k:base[k] for k in ('architecture','schema','cache_identity','worker_quantities','market_quantities')}
    value.update(model={k:v.detach().cpu() for k,v in model.state_dict().items()},optimizer=optimizer.state_dict(),
                 ppo_schema='exact-recurrent-ppo-v1',ppo_contract=contract,update=update_number,rngs=rngs,
                 updated_at=now(),matchups=matchups or {},league_state=league_state,provenance=base.get('provenance'))
    tmp=Path(str(path)+'.tmp');torch.save(value,tmp);tmp.replace(path)


def make_optimizer(model,actor_lr,critic_lr,head_lr_scale=100.):
    actor=[];critic=[];head=[];ext=[]
    for name,p in model.named_parameters():
        (head if name.startswith('value_shaped.') else critic if name.startswith(('critic_memory.','value.'))
         else ext if name.startswith(EXT_PREFIXES) else actor).append(p)
    # shaped-reward-v4: the residual linear head gets its own group at head_lr_scale x critic LR.
    groups=[{'params':actor,'lr':actor_lr},{'params':critic,'lr':critic_lr},{'params':head,'lr':critic_lr*head_lr_scale}]
    if ext:groups.append({'params':ext,'lr':actor_lr*float(os.environ.get('PPO_EXT_LR_SCALE','1'))})  # v39 feature-ext-v1
    return torch.optim.AdamW(groups,eps=1e-5,weight_decay=1e-4,fused=True if os.environ.get('PPO_FUSED_ADAM')=='1' else None)


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bc-reference',type=Path,required=True);p.add_argument('--league',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--resume',type=Path)
    p.add_argument('--learner-init',type=Path,help='New PPO stage: learner weights only, fresh optimizer; BC reference stays fixed')
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda');p.add_argument('--bf16',action='store_true')
    p.add_argument('--collector',choices=['official','gpu'],default='official')
    p.add_argument('--games',type=int,default=64);p.add_argument('--workers',type=int,default=16,help='Global CPU processes')
    p.add_argument('--arena-mib',type=int,default=32);p.add_argument('--minibatch',type=int,default=128)
    p.add_argument('--burn',type=int,default=16);p.add_argument('--sequence',type=int,default=48)
    p.add_argument('--gae-lambda',type=float,default=.95);p.add_argument('--lr',type=float,default=2e-5);p.add_argument('--critic-lr',type=float,default=5e-5)
    p.add_argument('--family-weighting',default=None,help='pfsp-v1 (source-v77): deficit (default, v76) or pfsp (win-rate (1-p)^2)')
    p.add_argument('--family-focus',default=None,help='family-focus-v1 (source-v75): JSON {arena family: multiplier} on the deficit weights')
    p.add_argument('--drift-kl',type=float,default=.02,help='hparam-v1 (source-v74): async drift-guard stop threshold (was hard-coded .02)')
    p.add_argument('--clip',type=float,default=.1);p.add_argument('--anchor',type=float,default=.01)
    p.add_argument('--anchor-decay',type=float,default=1.,help='v39: multiplicative per-update decay of --anchor from --anchor-start-update')
    p.add_argument('--anchor-min',type=float,default=0.,help='v39: floor for the decayed anchor');p.add_argument('--anchor-start-update',type=int,default=0)
    p.add_argument('--max-kl',type=float,default=.03);p.add_argument('--seed',type=int,default=9280000)
    p.add_argument('--updates',type=int,default=1,help='Updates this invocation; 0 means until STOP')
    p.add_argument('--eval-panel',type=Path);p.add_argument('--eval-every',type=int,default=10)
    p.add_argument("--live-league",type=Path)
    p.add_argument("--migration",type=Path)
    p.add_argument("--arena-pool",help="Immutable real arena training pool JSON")
    p.add_argument('--reward-beta',type=float,default=1.,help='shaped-reward-v1: weight of the clipped cash-margin term')
    p.add_argument('--reward-sigma',type=float,default=25000.,help='shaped-reward-v1: cash-margin scale for the clipped term')
    p.add_argument('--policy-warmup-until',type=int,default=0,help='shaped-reward-v2: updates below this index train value heads only (policy term weight 0)')
    p.add_argument('--reward-cash-weight',type=float,default=0.,help='shaped-reward-v5: weight of the clipped absolute-cash term (0 = off)')
    p.add_argument('--reward-cash-center',type=float,default=100000.,help='shaped-reward-v5: cash level that scores zero on the absolute-cash term')
    p.add_argument('--reward-shape',default='clip',choices=('clip','tanh','linear'),help='shaped-reward-v6/v7: squash for the margin and cash terms (linear = none)')
    p.add_argument('--reward-dense',type=int,default=0,help='shaped-reward-v6: 1 = per-turn potential-based cash shaping (telescopes to the terminal objective)')
    p.add_argument('--reward-potential',default='cash',choices=('cash','networth','networth-growing'),
        help='networth-shaping-v1: level inside the dense potential Phi (cash = shaped-reward-v6/v7; networth = cash + held goods at market price; networth-growing adds heuristic crop/animal value). Terminal Phi always uses final cash.')
    p.add_argument('--reward-potential-weight',type=float,default=None,
        help='networth-shaping-v2: weight of the PER-TURN dense potential only (unset = --reward-cash-weight, as source-v42). The terminal potential keeps --reward-cash-weight on final cash.')
    p.add_argument('--ppo-passes',type=int,default=1,help='ppo-passes-v1: PPO passes over each collected batch (1 = original)')
    p.add_argument('--official-workers',type=int,default=0,help='collection-overlap-v1: actor processes per rank for the official stage (0 = workers/world)')
    p.add_argument('--overlap-collection',type=int,default=0,help='collection-overlap-v1: 1 runs official collection concurrently with GPU collection')
    p.add_argument('--async-collect',type=int,default=0,help='async-v1: 1 = GPU collection in a child process with a one-update policy lag')
    p.add_argument('--arena-games',type=int,default=0,help='async-v1: explicit real-arena games per update (0 = fixed 2/8 share)')
    p.add_argument('--async-official',type=int,default=0,help='async-v2: 1 = official real-arena collection in its own child process, one update behind')
    p.add_argument('--arena-in-gpu',type=int,default=0,help='arena-in-gpu-v1: 1 = real-bot arena seats inside the GPU collector child (no official child); --arena-games sets the arena games')
    p.add_argument('--arena-bot-workers',type=int,default=16,help='arena-in-gpu-v1: bot sandbox workers per rank')
    p.add_argument('--arena-parity-games',type=int,default=2,help='arena-in-gpu-v1: arena games per rank per update replayed in the official engine (diagnostic)')
    p.add_argument('--distill-coef',type=float,default=0.,help='distill-v1: weight of the teacher-turn NLL (0 = off, identical to source-v35)')
    p.add_argument('--distill-decay',type=float,default=1.,help='distill-v1: multiplicative per-update decay of the coefficient')
    p.add_argument('--distill-start-update',type=int,default=-1,help='distill-v1: update at which the coefficient equals --distill-coef (required when on)')
    p.add_argument('--distill-min-coef',type=float,default=1e-4,help='distill-v1: below this the term and the teacher queries switch off')
    p.add_argument('--distill-games',type=int,default=128,help='distill-v1: learner games per rank whose learner seat is shadowed by the teachers')
    p.add_argument('--distill-teachers',default='a93964afb01c,b1ea30ce8ca6,e2dd3e9c6e5a,27dfff16b561,0b5eb9560f4e',help='distill-v1: teacher program id prefixes (pool members, no dev-panel family)')
    p.add_argument('--distill-workers',type=int,default=16,help='distill-v1: teacher sandbox workers per rank')
    p.add_argument('--distill-label-workers',type=int,default=4,help='distill-v1: host label-conversion processes per rank')
    p.add_argument('--start-state-frac',type=float,default=0.,help='start-state-v1: fraction of each rank\'s self-play games started from a recorded top-bot mid-game state (0 = off, identical to source-v43)')
    p.add_argument('--start-state-bank',default=None,help='start-state-v1: start-state-bank-v1 npz (hash-pinned in the contract)')
    p.add_argument('--start-state-days',default='7,24',help='start-state-v1: "lo,hi" inclusive start-day range; the start turn is uniform in [24*lo, 24*(hi+1)-1]')
    p.add_argument('--market-entropy-coef',type=float,default=0.,help='market-entropy-v1: extra entropy bonus on the market command factors only (0 = off); the all-head entropy coefficient stays .001')
    p.add_argument('--market-quantity-entropy-coef',type=float,default=0.,help='market-entropy-v1: extra entropy bonus on the market quantity factors only (0 = off)')
    from ppo.value_trunk import add_arguments as _critic_fix_arguments;_critic_fix_arguments(p)  # critic-fix-v1
    return p


EXPLORE_KEYS=('start_state_frac','start_state_bank','start_state_days','market_entropy_coef','market_quantity_entropy_coef')


def explore_contract(args,contract):
    """explore-v1 (source-v44): with every new flag at its default the contract is exactly source-v43's."""
    frac=float(getattr(args,'start_state_frac',0.) or 0.)
    if not frac:
        for k in ('start_state_frac','start_state_bank','start_state_days'):contract['config'].pop(k,None)
    else:
        from ppo.start_states import parse_days,identity
        if not 0<frac<=1 or not args.start_state_bank:raise ValueError('--start-state-frac must be in (0,1] with --start-state-bank')
        if args.collector!='gpu' or not getattr(args,'async_collect',0) or not getattr(args,'live_league',None):
            raise ValueError('start-state-v1 requires --collector gpu, --async-collect 1 and a live league (self-play jobs in the GPU child)')
        contract['start_states']=identity(args.start_state_bank,parse_days(args.start_state_days))
    for k in ('market_entropy_coef','market_quantity_entropy_coef'):
        v=float(getattr(args,k,0.) or 0.)
        if v<0:raise ValueError('--'+k.replace('_','-')+' must be >= 0')
        if not v:contract['config'].pop(k,None)


def explore_migration_allows(prior_config,current_config,migration):
    """explore-v1 migration: the config may differ ONLY in EXPLORE_KEYS, and both sides must equal the receipt's
    'explore' / 'old_explore' dicts (absent key = None = off)."""
    prior,current=dict(prior_config),dict(current_config)
    old={k:prior.pop(k,None) for k in EXPLORE_KEYS};new={k:current.pop(k,None) for k in EXPLORE_KEYS}
    return prior==current and new==migration.get('explore') and old==migration.get('old_explore')


def explore_critic_migration_allows(prior_config,current_config,migration):
    """explore-critic-v1 (source-v46): explore-v1 keys plus critic-fix-v1 keys (value_trunk_grad, value_loss, policy_warmup_until,
    critic_lr); each group must equal its receipt dicts ('explore'/'old_explore', 'new_values'/'old_values'), nothing else changes."""
    from ppo.value_trunk import MIGRATION_KEYS,DEFAULTS
    prior,current=dict(prior_config),dict(current_config)
    old={k:prior.pop(k,None) for k in EXPLORE_KEYS};new={k:current.pop(k,None) for k in EXPLORE_KEYS}
    old_c={k:prior.pop(k,DEFAULTS.get(k)) for k in MIGRATION_KEYS};new_c={k:current.pop(k,DEFAULTS.get(k)) for k in MIGRATION_KEYS}
    return (prior==current and new==migration.get('explore') and old==migration.get('old_explore')
            and new_c==migration.get('new_values') and old_c==migration.get('old_values'))


def install_termination_handler():
    import signal
    def terminate(signum,frame):raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,terminate)


def main():
    install_termination_handler()
    import faulthandler
    if os.environ.get('PPO_TRACE_TIMEOUT'):faulthandler.dump_traceback_later(int(os.environ['PPO_TRACE_TIMEOUT']),repeat=True)
    args=parser().parse_args()
    if args.bf16 and not args.async_collect:  # async-v2: bf16 replay allowed under lag (density recorded, not enforced)
        raise ValueError("BF16 PPO is disabled until full-turn collector/replay density parity passes")
    torch.set_num_threads(1)
    # Collected and replayed joint densities must agree before an update.
    # TF32 kernels differ across collection/replay batch shapes on Blackwell.
    torch.backends.cuda.matmul.allow_tf32=(os.environ.get('PPO_TF32','1')=='1')  # tf32-v1 (source-v58); was False
    torch.backends.cudnn.allow_tf32=(os.environ.get('PPO_TF32','1')=='1')
    if getattr(args,'family_weighting',None):os.environ['PPO_FAMILY_WEIGHTING']=args.family_weighting  # pfsp-v1
    if getattr(args,'family_focus',None):os.environ['PPO_FAMILY_FOCUS']=args.family_focus  # family-focus-v1: read by arena_opponents
    if args.resume and args.learner_init: raise ValueError('Choose exact resume or a new learner-init stage')
    if args.eval_every<=0: raise ValueError('eval-every must be positive')
    world=int(os.environ.get('WORLD_SIZE','1'));rank=int(os.environ.get('RANK','0'))
    local=int(os.environ.get('LOCAL_RANK','0'))
    if args.device=='cpu' and args.bf16: raise ValueError('Use CPU FP32 for correctness checks')
    if min(args.games,args.workers,args.minibatch,args.sequence,args.arena_mib)<=0 or args.burn<0: raise ValueError('Invalid batch configuration')
    if args.games%world or args.workers%world or args.minibatch%world: raise ValueError('Global batch/workers must divide world size')
    if args.device=='cuda': torch.cuda.set_device(local)
    device=torch.device(f'cuda:{local}' if args.device=='cuda' else 'cpu')
    if args.collector=='gpu' and device.type!='cuda':raise ValueError('GPU collector requires CUDA')
    if world>1: dist.init_process_group('nccl' if args.device=='cuda' else 'gloo',timeout=timedelta(minutes=30))
    failed=False;owner=None
    try:
        # Never share output ownership with another launch or an existing BC run.
        if rank==0:
            if not args.resume: args.output.mkdir(parents=True,exist_ok=False)
            else: args.output.mkdir(parents=True,exist_ok=True)
            lock=args.output/'trainer.lock'
            import fcntl
            owner=lock.open('a+');fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
            owner.seek(0);owner.truncate();owner.write(str(os.getpid()));owner.flush()
        if world>1: dist.barrier()
        from ppo.league_runtime import LiveLeague,PROTOCOL,broadcast,descriptor,atomic,init_control,snapshot,matchup_counts,finalize_checkpoint,validate_manifest,digest
        init_control()
        league=load_league(args.league) if not args.live_league else {"protocol":PROTOCOL,"opponents":[]}
        simulator_identity=None
        if args.collector=='gpu':
            if args.bf16 and not args.async_collect:raise ValueError('GPU collection requires FP32 replay; BF16 has not passed density checks')
            if any(r['kind']=='script' and r['weight']>0 for r in league['opponents']):
                raise ValueError('GPU collector requires neural opponents; script agents need the official collector')
            import sys
            for key in ('PPO_JAX_SITE_PACKAGES','PPO_JAX_SIM_SRC'):
                if os.environ.get(key):sys.path.append(os.environ[key])
            from ppo.gpu_env import runtime_identity
            simulator_identity=runtime_identity()
        panel=None
        if args.eval_panel:
            from ppo.evaluate import validate_panel
            panel=validate_panel(args.eval_panel)
        model,base=load_model(args.bc_reference,device)
        if args.learner_init:
            model,initial=load_model(args.learner_init,device,base['cache_identity'])
            if initial.get('ppo_schema')!='exact-recurrent-ppo-v1': raise ValueError('learner-init requires a PPO checkpoint')
            if initial['ppo_contract']['bc_sha256']!=sha(args.bc_reference): raise ValueError('Stage must retain the original BC reference')
        reference,_=load_model(args.bc_reference,device)
        reference.requires_grad_(False)
        ref_scorer=ExactPPOChunk(reference)
        models={'learner':model,'bc':reference}
        if args.live_league:
            if args.collector!='gpu' or not args.resume:raise ValueError('Live league requires GPU continuation')
            models={'learner':model}
            for slot in ('champion_slot','history_slot'):
                models[slot],_=load_model(args.bc_reference,device)
                models[slot].requires_grad_(False)
        for row in league['opponents']:
            if row['kind']=='checkpoint':
                models[row['id']],_=load_model(row['path'],device,base['cache_identity'])
                models[row['id']].requires_grad_(False)
        contract=dict(code_sha256=code_hash(),bc_sha256=sha(args.bc_reference),league=league,
                      simulator=simulator_identity,
                      collection_backend={name:os.environ.get(name,'0') for name in ('PPO_GPU_WORKERS','PPO_COMPILE_WORKERS','PPO_GRAPH_ENCODER','PPO_COMPILE_MARKET','PPO_COMPACT_MARKET','PPO_COMPACT_WORKERS','PPO_COMPILE_HEADS','PPO_COMPILE_FEEDBACK','PPO_WORKER_BUCKETS','PPO_DONATE_SIM_STATE','PPO_FAST_MARKET_HEAD','PPO_FAST_WORKER_HEAD','PPO_PACK_STATS','PPO_FUSED_ADAM','PPO_SEQUENCE_GRU','PPO_ROLE_WORKER_BUCKETS','PPO_FINE_WORKER_BUCKETS','PPO_SPLIT_BURN_ENCODER')},
                      learner_init_sha256=sha(args.learner_init) if args.learner_init else None,
                      world=world,config={k:v for k,v in vars(args).items() if k not in ('output','resume','updates','bc_reference','league','eval_panel','learner_init','live_league','migration')},
                      eval_panel=panel)
        arena_pool=None
        if args.arena_pool:
            if not args.live_league or args.collector!='gpu':raise ValueError('Arena mix requires live GPU continuation')
            from ppo.arena_opponents import load_pool
            arena_pool=load_pool(args.arena_pool)
            contract['arena_pool_sha256']=sha(args.arena_pool)
            contract['hybrid_collection']=dict(protocol='real-arena-training-v3',game_allocation=([.5,.25,.25] if not args.arena_games else [1-1/8-args.arena_games/args.games,1/8,args.arena_games/args.games]),family_sampling='deficit-weighted: weight=.5+deficit/mean(deficit), EMA .9 of per-family mean cash margin, uniform when no history',
                backends=['pinned GPU simulator','official kaggle-environments 1.32.7 stream adapter'],
                minibatches='homogeneous format; deterministic shuffle of five GPU plus one official per temporal window',
                opponents='uniform author then uniform member; source-native sandbox',failure='reject entire update')
            if getattr(args,'arena_in_gpu',0):  # arena-in-gpu-v1
                if not args.async_collect or not args.arena_games:raise ValueError('--arena-in-gpu requires --async-collect 1 and --arena-games')
                contract['hybrid_collection']=dict(contract['hybrid_collection'],protocol='real-arena-training-v4-arena-in-gpu',
                    backends=['pinned GPU simulator; arena opponent seats = real programs in arena sandboxes (ppo.arena_mux_worker, observations from ppo.arena_obs)'],
                    minibatches='homogeneous GPU format',failure='opponent faults discard that game; infrastructure errors raise',
                    worker_pins=dict(__import__('ppo.arena_gpu',fromlist=['verify_pins']).verify_pins()))
        else:
            contract['config'].pop('arena_pool',None)
        if not getattr(args,'arena_in_gpu',0):  # arena-in-gpu-v1: flag off leaves the contract exactly as source-v25
            for k in ('arena_in_gpu','arena_bot_workers','arena_parity_games'):contract['config'].pop(k,None)
        distill_keys=('distill_coef','distill_decay','distill_start_update','distill_min_coef','distill_games','distill_teachers','distill_workers','distill_label_workers')
        teacher_rows=None
        if not getattr(args,'distill_coef',0.):  # distill-v1: flag off leaves the contract exactly as source-v35
            for k in distill_keys:contract['config'].pop(k,None)
        else:
            if not getattr(args,'arena_in_gpu',0) or arena_pool is None:raise ValueError('--distill-coef requires --arena-in-gpu 1 and an arena pool')
            if args.distill_start_update<0 or not 0<args.distill_decay<=1:raise ValueError('--distill-start-update and 0<--distill-decay<=1 are required')
            from ppo.distill_gpu import select_teachers
            teacher_rows=select_teachers(arena_pool,args.distill_teachers.split(','),Path(args.live_league).parent/'panel.json')
            contract['distill']=dict(protocol='distill-v1',teachers=[dict(id=r['id'],family=r['family'],archive=r['archive']) for r in teacher_rows],
                labels='exact_decoder.prepare_turn semantics (ppo.distill_labels), whole-turn supervised, trailing no-op orders stripped',
                loss='coef*decay**(update-start) * NLL(one rotating teacher per labelled row) / PPO row count')
        if getattr(args,'reward_potential','cash')=='cash':  # networth-shaping-v1: the default leaves the contract exactly as source-v39
            contract['config'].pop('reward_potential',None)
        else:
            if not args.reward_dense:raise ValueError('--reward-potential networth requires --reward-dense 1')
            # Only GPU-simulator rollouts record the state needed for worth; the official CPU collector records cash only.
            if args.collector!='gpu' or (arena_pool is not None and not getattr(args,'arena_in_gpu',0)):
                raise ValueError('--reward-potential networth requires GPU collection for every learner seat (--collector gpu, and --arena-in-gpu 1 with an arena pool)')
        if getattr(args,'reward_potential_weight',None) is None:  # networth-shaping-v2: unset leaves the contract exactly as source-v42
            contract['config'].pop('reward_potential_weight',None)
        else:
            if not args.reward_dense or args.reward_potential_weight<0:raise ValueError('--reward-potential-weight requires --reward-dense 1 and a weight >= 0')
            if args.collector!='gpu' or (arena_pool is not None and not getattr(args,'arena_in_gpu',0)):
                raise ValueError('--reward-potential-weight requires GPU collection for every learner seat (the official CPU collector ignores it)')
        explore_contract(args,contract)  # explore-v1 (source-v44): start-state-v1 + market-entropy-v1; off = source-v43 contract
        from ppo.value_trunk import contract_config as _critic_fix_contract;_critic_fix_contract(contract,args)  # critic-fix-v1: defaults leave the contract as source-v43
        contract['feature_ext']=feature_ext_identity()  # v39
        if getattr(args,'async_official',0) and not getattr(args,'arena_in_gpu',0):
            raise ValueError('feature-ext-v1: the CPU official collector has no long-market history; use --arena-in-gpu 1')
        optimizer=make_optimizer(model,args.lr,args.critic_lr)
        start_update=0;matchups={};migration_applied=False
        if args.resume:
            saved=torch.load(args.resume,map_location='cpu',weights_only=True)
            contract['learner_init_sha256']=saved.get('ppo_contract',{}).get('learner_init_sha256')
            if saved.get('ppo_schema')!='exact-recurrent-ppo-v1':raise ValueError('PPO schema mismatch')
            if saved['ppo_contract']!=contract:
                if not args.migration:raise ValueError('PPO resume contract mismatch')
                migration=json.loads(args.migration.read_text())
                if (migration['source_sha256']!=sha(args.resume) or migration['old_contract']!=saved['ppo_contract']
                    or migration['new_contract']!=contract):raise ValueError('Migration does not match exact contracts')
                for key in contract:
                    if key not in ('code_sha256','league') and contract[key]!=saved['ppo_contract'].get(key):
                        if migration.get('kind') in ('arena-mix-v2','arena-mix-v3') and key=='hybrid_collection' and migration.get('hybrid_collection')==contract[key]:continue
                        if migration.get('kind')=='arena-pool-update-v1':
                            if key=='arena_pool_sha256' and migration.get('arena_pool_sha256')==contract[key]:continue
                            if key=='config':
                                prior=dict(saved['ppo_contract']['config']);current=dict(contract['config'])
                                if prior.pop('arena_pool',None)!=migration.get('old_arena_pool') or current.pop('arena_pool',None)!=migration.get('new_arena_pool'):raise ValueError('Migration pool paths do not match')
                                if prior==current:continue
                        if migration.get('kind')=='collection-overlap-v1' and key=='config':
                            prior=dict(saved['ppo_contract']['config']);current=dict(contract['config'])
                            added={k:current.pop(k) for k in ('official_workers','overlap_collection')}
                            if prior==current and added==migration.get('collection',{}):continue
                        if migration.get('kind')=='async-v1' and key in ('config','hybrid_collection'):
                            if key=='hybrid_collection' and migration.get('hybrid_collection')==contract[key]:continue
                            if key=='config':
                                prior=dict(saved['ppo_contract']['config']);current=dict(contract['config'])
                                keys=('games','minibatch','bf16','ppo_passes','async_collect','arena_games','official_workers','workers','overlap_collection','async_official','lr')
                                old_vals={k:prior.pop(k,None) for k in keys};new_vals={k:current.pop(k,None) for k in keys}
                                if prior==current and new_vals==migration.get('async',{}) and old_vals==migration.get('old_async',{}):continue
                        if migration.get('kind')=='distill-v1' and key in ('config','distill'):
                            if key=='distill' and migration.get('distill')==contract[key]:continue
                            if key=='config':
                                prior=dict(saved['ppo_contract']['config']);current=dict(contract['config'])
                                keys=('distill_coef','distill_decay','distill_start_update','distill_min_coef','distill_games','distill_teachers','distill_workers','distill_label_workers','lr')
                                old_vals={k:prior.pop(k,None) for k in keys};new_vals={k:current.pop(k,None) for k in keys}
                                if prior==current and new_vals==migration.get('distill_config',{}) and old_vals==migration.get('old_distill_config',{}):continue
                        if migration.get('kind')=='arena-in-gpu-v1' and key in ('config','hybrid_collection'):
                            if key=='hybrid_collection' and migration.get('hybrid_collection')==contract[key]:continue
                            if key=='config':
                                prior=dict(saved['ppo_contract']['config']);current=dict(contract['config'])
                                keys=('arena_in_gpu','arena_bot_workers','arena_parity_games','arena_games','async_official','official_workers','games','workers','lr')
                                old_vals={k:prior.pop(k,None) for k in keys};new_vals={k:current.pop(k,None) for k in keys}
                                if prior==current and new_vals==migration.get('arena',{}) and old_vals==migration.get('old_arena',{}):continue
                        if migration.get('kind')=='feature-ext-v1':  # v39: CP800 + zero-init extensions, fresh optimizer (see the receipt)
                            if key=='feature_ext' and migration.get('feature_ext')==contract[key]:continue
                            if key=='config':
                                prior=dict(saved['ppo_contract']['config']);current=dict(contract['config'])
                                keys=tuple(migration.get('config_changes',{}))
                                old_vals={k:prior.pop(k,None) for k in keys};new_vals={k:current.pop(k,None) for k in keys}
                                if prior==current and new_vals==migration['config_changes'] and old_vals==migration.get('old_config_values',{}):continue
                        if migration.get('kind')=='gae-lambda-v1' and key=='config':  # source-v66: only the GAE lambda may change
                            prior=dict(saved['ppo_contract']['config']);current=dict(contract['config'])
                            old_l=prior.pop('gae_lambda',None);new_l=current.pop('gae_lambda',None)
                            if prior==current and new_l==migration.get('gae_lambda') and old_l==migration.get('old_gae_lambda'):continue
                        if migration.get('kind')=='scale-out-v2':  # source-v73: more GPUs, same per-rank load; world + global counts scale
                            if key=='world' and migration.get('world')==contract[key]:continue
                            if key=='hybrid_collection' and migration.get('hybrid_collection')==contract[key]:continue
                            if key=='config':
                                prior=dict(saved['ppo_contract']['config']);current=dict(contract['config'])
                                keys=tuple(migration.get('config_changes',{}))
                                if set(keys)<={'games','arena_games','workers','minibatch','arena_bot_workers','official_workers'}:
                                    old_vals={k:prior.pop(k,None) for k in keys};new_vals={k:current.pop(k,None) for k in keys}
                                    if prior==current and new_vals==migration['config_changes'] and old_vals==migration.get('old_config_values',{}):continue
                        if migration.get('kind')=='mux-cpu-timer-v1' and key=='hybrid_collection':  # source-v76: only the mux worker pin
                            prior=json.loads(json.dumps(saved['ppo_contract'][key]));current=json.loads(json.dumps(contract[key]))
                            op=prior.get('worker_pins',{}).pop('mux',None);np_=current.get('worker_pins',{}).pop('mux',None)
                            if prior==current and op==migration.get('old_mux') and np_==migration.get('new_mux'):continue
                        if migration.get('kind') in ('hparam-v1','scale-out-v2+hparam','mux-cpu-timer-v1') and key=='config':  # source-v74 (+v76 mux kind)
                            prior=dict(saved['ppo_contract']['config']);current=dict(contract['config'])
                            keys=tuple(migration.get('config_changes',{}))
                            allowed=HPARAM_KEYS|(SCALE_KEYS if migration['kind']=='scale-out-v2+hparam' else set())
                            if keys and set(keys)<=allowed:
                                old_vals={k:prior.pop(k,None) for k in keys};new_vals={k:current.pop(k,None) for k in keys}
                                if prior==current and new_vals==migration['config_changes'] and old_vals==migration.get('old_config_values',{}):continue
                        if migration.get('kind')=='scale-out-v2+hparam' and key=='world' and migration.get('world')==contract[key]:continue
                        if migration.get('kind')=='scale-out-v2+hparam' and key=='hybrid_collection' and migration.get('hybrid_collection')==contract[key]:continue
                        if migration.get('kind')=='bot-workers-v1' and key=='config':  # source-v51: only the bot sandbox worker count may change
                            prior=dict(saved['ppo_contract']['config']);current=dict(contract['config'])
                            old_w=prior.pop('arena_bot_workers',None);new_w=current.pop('arena_bot_workers',None)
                            if prior==current and new_w==migration.get('arena_bot_workers') and old_w==migration.get('old_arena_bot_workers'):continue
                        if migration.get('kind')=='ppo-passes-v1' and key=='config':
                            prior=dict(saved['ppo_contract']['config']);current=dict(contract['config'])
                            old_p=prior.pop('ppo_passes',None);new_p=current.pop('ppo_passes',None)
                            if prior==current and new_p==migration.get('ppo_passes') and old_p==migration.get('old_ppo_passes'):continue
                        if migration.get('kind')=='explore-v1':  # source-v44: only the explore flags (and the start-state identity) may change
                            if key=='start_states' and migration.get('start_states')==contract[key]:continue
                            if key=='config' and explore_migration_allows(saved['ppo_contract']['config'],contract['config'],migration):continue
                        if migration.get('kind')=='explore-critic-v1':  # source-v46: explore-v1 + critic-fix-v1 keys together
                            if key=='start_states' and migration.get('start_states')==contract[key]:continue
                            if key=='config' and explore_critic_migration_allows(saved['ppo_contract']['config'],contract['config'],migration):continue
                        if migration.get('kind')=='critic-fix-v1' and key=='config':  # critic-fix-v1: only ppo.value_trunk.MIGRATION_KEYS may change
                            from ppo.value_trunk import migration_allows as _critic_fix_allows
                            if _critic_fix_allows(saved['ppo_contract']['config'],contract['config'],migration):continue
                        if migration.get('kind')=='networth-shaping-v2' and key=='config':  # only potential level and per-turn weight may change
                            from ppo.networth import migration_allows_v2
                            if migration_allows_v2(saved['ppo_contract']['config'],contract['config'],migration):continue
                        if migration.get('kind')=='networth-shaping-v1' and key=='config':  # only the potential level may change
                            from ppo.networth import migration_allows
                            if migration_allows(saved['ppo_contract']['config'],contract['config'],migration):continue
                        if migration.get('kind')=='shaped-reward-v6' and key=='config':
                            prior=dict(saved['ppo_contract']['config']);current=dict(contract['config'])
                            keys=('reward_beta','reward_sigma','policy_warmup_until','reward_cash_weight','reward_cash_center','reward_shape','reward_dense','gae_lambda')
                            old_vals={k:prior.pop(k,None) for k in keys};new_vals={k:current.pop(k,None) for k in keys}
                            if prior==current and new_vals==migration.get('reward',{}) and old_vals==migration.get('old_reward',{}):continue
                        if migration.get('kind')=='shaped-reward-v5' and key=='config':
                            prior=dict(saved['ppo_contract']['config']);current=dict(contract['config'])
                            keys=('reward_beta','reward_sigma','policy_warmup_until','reward_cash_weight','reward_cash_center')
                            old_vals={k:prior.pop(k,None) for k in keys};new_vals={k:current.pop(k,None) for k in keys}
                            if prior==current and new_vals==migration.get('reward',{}) and old_vals==migration.get('old_reward',{}):continue
                        if migration.get('kind') in ('shaped-reward-v1','shaped-reward-v2','shaped-reward-v3','shaped-reward-v4') and key=='config':
                            prior=dict(saved['ppo_contract']['config']);current=dict(contract['config'])
                            keys=('reward_beta','reward_sigma')+(('policy_warmup_until',) if migration.get('kind') in ('shaped-reward-v2','shaped-reward-v3','shaped-reward-v4') else ())
                            added={k:current.pop(k) for k in keys}
                            if prior==current and added==migration.get('reward',{}):continue
                        if migration.get('kind')=='real-arena-training-v1':
                            if key in ('arena_pool_sha256','hybrid_collection') and args.arena_pool and migration.get(key)==contract[key]:continue
                            prior=dict(saved['ppo_contract']['config']);current=dict(contract['config'])
                            added=current.pop('arena_pool',None)
                            if key=='config' and added==args.arena_pool and prior==current:continue
                        old_config=dict(saved['ppo_contract'].get('config',{}));new_config=dict(contract.get('config',{}))
                        old_lr=old_config.pop('lr',None);new_lr=new_config.pop('lr',None)
                        old_lam=old_config.pop('gae_lambda',.95);new_lam=new_config.pop('gae_lambda',.95)
                        old_seed=old_config.pop('seed');new_seed=new_config.pop('seed')
                        if not (key=='config' and migration.get('kind')=='restart20-lambda1-v1'
                                and old_config==new_config and old_lr==2e-5 and new_lr==1e-5 and old_lam==.95 and new_lam==1.0 and old_seed==190000000 and new_seed==290000000):
                            raise ValueError('Migration changes training settings: '+key)
                migration_applied=True
            if migration_applied and migration.get('kind') in ('shaped-reward-v1','shaped-reward-v2','shaped-reward-v3','shaped-reward-v4'):
                from ppo.shaped_reward import migrate_state
                receipt=migrate_state(saved,model)
                if rank==0:atomic_json(args.output/'shaped-reward-migration.json',dict(source_sha256=sha(args.resume),source_update=saved['update'],**receipt))
            model.load_state_dict(saved['model'],strict=True);optimizer.load_state_dict(saved['optimizer'])
            optimizer.param_groups[0]['lr']=args.lr  # v33: --lr always applies to the policy group (saved Adam state used to override it)
            if os.environ.get('PPO_HEAD_LR_SCALE') and len(optimizer.param_groups)>2:  # v38: value_shaped head lr = critic_lr * scale (critic diagnosis: 100x adds jitter)
                optimizer.param_groups[2]['lr']=args.critic_lr*float(os.environ['PPO_HEAD_LR_SCALE'])
            if len(optimizer.param_groups)>3:  # v39: feature-ext group follows --lr
                optimizer.param_groups[3]['lr']=args.lr*float(os.environ.get('PPO_EXT_LR_SCALE','1'))
            if migration_applied:  # critic-fix-v1: a migrated --critic-lr change reaches the critic group (explore-critic-v1 too)
                from ppo.value_trunk import apply_resume_learning_rate as _critic_fix_lr
                _critic_fix_lr(optimizer,args,dict(migration,kind='critic-fix-v1') if migration.get('kind')=='explore-critic-v1' else migration)
            if migration_applied and migration.get('kind') in ('hparam-v1','scale-out-v2+hparam') and 'critic_lr' in migration.get('config_changes',{}):
                optimizer.param_groups[1]['lr']=args.critic_lr  # source-v74: a migrated critic lr reaches the critic group
            if rank==0:print(json.dumps(dict(stage='lr_applied',learning_rates=[g['lr'] for g in optimizer.param_groups])),flush=True)
            if migration_applied and migration.get('kind')=='restart20-lambda1-v1':
                # Restore moments and steps first; override only the declared group LR.
                optimizer.param_groups[0]['lr']=args.lr
                optimizer.param_groups[1]['lr']=args.critic_lr
                if rank==0:atomic_json(args.output/'fork-restoration.json',dict(
                    source_sha256=sha(args.resume),source_update=saved['update'],
                    learning_rates=[g['lr'] for g in optimizer.param_groups],
                    optimizer_state_entries=len(optimizer.state),preserved=['model','Adam moments','optimizer counters','CPU/CUDA RNG','league state','reference state']))

            start_update=saved['update']+1
            matchups=saved.get('matchups',{})
            _restore_rng(saved,rank,device,args.seed)
        else:
            torch.manual_seed(args.seed+rank)
        loss_module=LossModule(model)
        from ppo.value_trunk import configure as _critic_fix_configure;_critic_fix_configure(loss_module,args)  # critic-fix-v1 (no-op at defaults)
        module=DDP(loss_module,device_ids=[local] if device.type=='cuda' else None,find_unused_parameters=True) if world>1 else loss_module
        if rank==0: atomic_json(args.output/'run-config.json',contract)
        iteration=start_update
        start_meta=start_days=None
        if contract.get('start_states'):  # start-state-v1
            from ppo.start_states import bank_meta,parse_days
            start_meta=bank_meta(args.start_state_bank);start_days=parse_days(args.start_state_days)
        from ppo.sampler import BatchedExactSampler
        official_generator=torch.Generator(device=device) if (args.arena_pool and device.type=='cuda') else None
        sampler=BatchedExactSampler(models,reference,base['worker_quantities'],base['market_quantities'],device,args.bf16 and not args.async_collect,generator=official_generator)
        gpu_backend=None;collector=None;official=None
        if args.collector=='gpu' and args.async_collect:
            import sys
            for key in ('PPO_JAX_SITE_PACKAGES','PPO_JAX_SIM_SRC'):
                if os.environ.get(key):sys.path.append(os.environ[key])
            from ppo.async_collect import AsyncCollector
            collector=AsyncCollector(device,args.bc_reference,args.workers//world,{k:os.environ[k] for k in os.environ if k.startswith(('PPO_','XLA_','PYTHONPATH','TORCHINDUCTOR','OMP_','MKL_','OPENBLAS_'))})
            reward_spec=dict(beta=args.reward_beta,sigma=args.reward_sigma,cash_weight=args.reward_cash_weight,cash_center=args.reward_cash_center,shape=args.reward_shape,dense=bool(args.reward_dense),potential=args.reward_potential,potential_weight=args.reward_potential_weight)
            async_pending={}
            if getattr(args,'async_official',0) and not getattr(args,'arena_in_gpu',0):  # async-v2 (arena-in-gpu-v1: no official child)
                from ppo.async_collect import AsyncOfficial
                official=AsyncOfficial(device,args.bc_reference,getattr(args,'official_workers',0) or args.workers//world,args.arena_mib,args.burn,args.sequence,{k:os.environ[k] for k in os.environ if k.startswith(('PPO_','XLA_','PYTHONPATH','TORCHINDUCTOR','OMP_','MKL_','OPENBLAS_'))})
        if args.collector=='gpu' and not args.async_collect:
            import sys
            for key in ('PPO_JAX_SITE_PACKAGES','PPO_JAX_SIM_SRC'):
                if os.environ.get(key):sys.path.append(os.environ[key])
            from ppo.gpu_backend import GPUBackend
            gpu_backend=GPUBackend(models,reference,base['worker_quantities'],base['market_quantities'],device,args.workers//world)
            gpu_backend.gae_lambda=args.gae_lambda
            gpu_backend.reward_beta=args.reward_beta;gpu_backend.reward_sigma=args.reward_sigma
            gpu_backend.reward_cash_weight=args.reward_cash_weight;gpu_backend.reward_cash_center=args.reward_cash_center
            gpu_backend.reward_shape=args.reward_shape;gpu_backend.reward_dense=bool(args.reward_dense)
            gpu_backend.reward_potential=args.reward_potential  # networth-shaping-v1
            gpu_backend.reward_potential_weight=args.reward_potential_weight  # networth-shaping-v2
        live=None
        previous=descriptor(args.resume) if args.resume else None
        if args.resume:base['provenance']=saved.get('provenance')
        if migration_applied:
            base['provenance']=dict(parent=previous,receipt_sha256=sha(args.migration))
            if rank==0:atomic(args.output/'migration-provenance.json',dict(**base['provenance'],receipt=json.loads(args.migration.read_text())))
        if args.live_league:
            league_state=saved.get('league_state')
            if migration_applied and migration.get('kind')=='broad-ppo-v1':
                new_manifest=validate_manifest(json.loads(args.live_league.read_text()))
                if digest(new_manifest)!=migration['initial_manifest_digest']:raise ValueError('Migration manifest changed')
                matchups={}  # New curriculum evidence starts empty in both saved locations.
                league_state=dict(manifest=new_manifest,reference=new_manifest['teacher'],reference_age=0,
                    reference_refreshes=0,rotation=0,applied={},matchups={})
            live=LiveLeague(args.live_league,models,reference,base['cache_identity'],previous,league_state)
            _restore_rng(saved,rank,device,args.seed)
        if args.resume and rank<len(saved['rngs']):
            # Verify the actual restored RNG on each rank, after graph/model construction.
            import hashlib as _hashlib
            def _rng_hash(x):return _hashlib.sha256(x.cpu().numpy().tobytes()).hexdigest()
            receipt=dict(rank=rank,cpu_rng=_rng_hash(torch.get_rng_state()),expected_cpu_rng=_rng_hash(saved['rngs'][rank]['cpu']),
                         cuda_rng=_rng_hash(torch.cuda.get_rng_state(device)),expected_cuda_rng=_rng_hash(saved['rngs'][rank]['cuda']),
                         learning_rates=[g['lr'] for g in optimizer.param_groups])
            if receipt['cpu_rng']!=receipt['expected_cpu_rng'] or receipt['cuda_rng']!=receipt['expected_cuda_rng']:
                raise RuntimeError('Fork RNG restore mismatch')
            atomic(args.output/f'restored-rank-{rank}.json',receipt)
        if live and args.resume and rank==0 and args.resume.resolve().parent==args.output.resolve():
            finalize_checkpoint(args.resume,args.output,start_update-1,live.state['reference'])
        if world>1:dist.barrier()
        while not args.updates or iteration<start_update+args.updates:
            if broadcast((args.output/'STOP').exists() if rank==0 else None): break
            tick=time.perf_counter();args.update_number=iteration
            _at={};_t=[time.perf_counter()]
            def _lap(name):
                now_=time.perf_counter();_at[name]=round(_at.get(name,0.)+now_-_t[0],3);_t[0]=now_
            args.hybrid_first_update=arena_pool is not None and iteration==start_update
            if live:
                refresh=previous
                if PROTOCOL['reference_updates'] and live.state['reference_age']>=PROTOCOL['reference_updates']:
                    if rank==0:
                        target=args.output/'references'/f'update-{iteration-1:06d}.pt'
                        refresh=snapshot(previous['path'],target)
                    refresh=broadcast(refresh if rank==0 else None)
                live.boundary(iteration,refresh);_lap('live_boundary')
                if gpu_backend is not None:gpu_backend.reference_sha=live.loaded['reference']
                ref_scorer.reference_sha=live.loaded['reference']
            if rank==0: atomic_json(args.output/'lifecycle.json',dict(phase='collecting',update=iteration,at=now(),pid=os.getpid()))
            if collector is not None:
                from ppo.arena_opponents import assign
                from ppo.rollout import collect as collect_official
                def split_jobs(it):
                    rows=live.jobs(args.seed,it,args.games,world,rank,arena=(args.arena_games or None))
                    fw=None;wp=Path(args.output)/'arena-family-weights.json'
                    if wp.exists():fw=json.loads(wp.read_text()).get('weights') or None
                    rows=assign(rows,arena_pool,args.seed,it,fw)
                    if start_meta is not None:  # start-state-v1: a fraction of this rank's self-play games start mid-game
                        from ppo.start_states import assign as assign_start
                        rows=assign_start(rows,start_meta,args.start_state_bank,contract['start_states']['bank_sha256'],args.start_state_frac,start_days,args.seed,it,rank)
                    return [j for j in rows if j['family']!='arena'],[j for j in rows if j['family']=='arena']
                def official_seed_for(it):return (args.seed+1000003*it+7919*rank)%(2**63-1)
                in_gpu=bool(getattr(args,'arena_in_gpu',0))  # arena-in-gpu-v1: arena jobs ride in the GPU batch
                def gpu_jobs(pair):return pair[0]+pair[1] if in_gpu else pair[0]
                arena_spec=dict(workers=args.arena_bot_workers,parity_games=args.arena_parity_games,parity_dir=str(args.output/'arena-parity'/f'rank-{rank}')) if in_gpu else None
                def distill_spec(it):  # distill-v1: teacher queries only while the coefficient for update `it` is on
                    if teacher_rows is None or not distill_coef_at(args,it):return None
                    return dict(games=args.distill_games,teachers=teacher_rows,workers=args.distill_workers,label_workers=args.distill_label_workers,
                                arena_config=arena_pool['config'],arena_repo=arena_pool['arena_repo'])
                if iteration not in async_pending:
                    async_pending[iteration]=split_jobs(iteration)
                    collector.submit(iteration,gpu_jobs(async_pending[iteration]),models,live.loaded['reference'],reward_spec,args.gae_lambda,arena=arena_spec,distill=distill_spec(iteration))
                    if official is not None:official.submit(iteration,async_pending[iteration][1],models,live.loaded['reference'],reward_spec,args.gae_lambda,official_seed_for(iteration))
                _lap('pre_wait')
                neural_jobs,arena_jobs=async_pending.pop(iteration)
                gpu_episodes,gpu_metrics=collector.wait(iteration);_lap('gpu_wait')
                batch_id=collector.last
                if os.environ.get('PPO_SYNC_CHILD')!='1':  # v37: PPO_SYNC_CHILD=1 -> next batch is submitted after this update (on-policy, no one-update lag)
                    async_pending[iteration+1]=split_jobs(iteration+1);_lap('split_jobs')
                    collector.submit(iteration+1,gpu_jobs(async_pending[iteration+1]),models,live.loaded['reference'],reward_spec,args.gae_lambda,arena=arena_spec,distill=distill_spec(iteration+1));_lap('gpu_submit')
                if in_gpu:
                    t_off=time.perf_counter();_lap('official_wait')
                    official_episodes,official_metrics=[],dict(games=[],valid_games=0,full_seasons=0,learner_turns=0,collection_seconds=0.)
                else:
                    t_off=time.perf_counter()
                    official_episodes,official_metrics=official.wait(iteration) if official is not None else (None,None);_lap('official_wait')
                    if official is not None:official.submit(iteration+1,async_pending[iteration+1][1],models,live.loaded['reference'],reward_spec,args.gae_lambda,official_seed_for(iteration+1))
                    official_seed=official_seed_for(iteration)
                    if sampler.generator is not None:sampler.generator.manual_seed(official_seed)
                    if device.type=="cuda":torch.cuda.set_device(device)  # the official sampler captures CUDA graphs on the current device
                    if official is None:t_off=time.perf_counter()
                    if official is None:official_episodes,official_metrics=collect_official(arena_jobs,models,reference,base['worker_quantities'],base['market_quantities'],device,
                        getattr(args,'official_workers',0) or args.workers//world,False,args.arena_mib,args.burn,args.sequence,sampler=sampler,gae_lambda=args.gae_lambda,
                        reference_sha=live.loaded['reference'],all_anchors=True,reward_beta=args.reward_beta,reward_sigma=args.reward_sigma,reward_cash_weight=args.reward_cash_weight,
                        reward_cash_center=args.reward_cash_center,reward_shape=args.reward_shape,reward_dense=bool(args.reward_dense)) if arena_jobs else ([],dict(games=[],valid_games=0,full_seasons=0,learner_turns=0,collection_seconds=0.,families={}))
                    _lap('official_submit')
                    if official_metrics['full_seasons']!=len(arena_jobs):raise RuntimeError('Real arena batch has invalid or incomplete games; no PPO update')
                episodes=gpu_episodes+official_episodes
                expected=sum(len(j['learner_seats']) for j in neural_jobs+arena_jobs)*719
                if in_gpu:expected-=719*sum(len(j['learner_seats']) for j,g in zip(neural_jobs+arena_jobs,gpu_metrics['games']) if any(g['faults']))  # opponent-fault games carry no episode
                if sum(len(e['turns']) for e in episodes)!=expected:raise RuntimeError('Async learner-turn coverage mismatch')
                from collections import Counter as _Counter
                games_all=gpu_metrics['games']+official_metrics['games']
                fams=_Counter(j['family'] for j in neural_jobs+arena_jobs)
                arena_fams=_Counter(j['arena_family'] for j in arena_jobs)
                metrics=dict(games=games_all,complete_games=len(games_all),valid_games=gpu_metrics['valid_games']+official_metrics['valid_games'],full_seasons=gpu_metrics['full_seasons']+official_metrics['full_seasons'],
                    learner_turns=gpu_metrics['learner_turns']+official_metrics['learner_turns'],collection_seconds=max(gpu_metrics.get('child_seconds',0),time.perf_counter()-t_off),
                    games_per_second=len(games_all)/max(1e-6,time.perf_counter()-tick),families=dict(fams),arena_families=dict(arena_fams),gpu_collection_seconds=gpu_metrics['collection_seconds'],
                    official_collection_seconds=official_metrics['collection_seconds'],gpu_peak_allocated_gib=gpu_metrics.get('memory',{}).get('allocated',0)/2**30,arena_family_weights=None,
                    async_child_seconds=gpu_metrics.get('child_seconds'),async_held_batches=gpu_metrics.get('held_batches'),protocol='async-v3-arena-in-gpu' if in_gpu else 'async-v2' if official is not None else 'async-v1',arena_in_gpu=gpu_metrics.get('arena_in_gpu'),distill_collection=gpu_metrics.get('distill'),gpu_phase_seconds=gpu_metrics.get('gpu_seconds'),event_preparation_seconds=gpu_metrics.get('event_preparation_seconds'),worker_bucket_turns=gpu_metrics.get('worker_bucket_turns'),
                    async_official_seconds=official_metrics.get('child_seconds') if official is not None else None,async_timing=_at)
            else:
              jobs=(live.jobs(args.seed,iteration,args.games,world,rank) if live else
                assignments(adapt_history(league,matchups),args.seed,iteration,args.games,world,rank,balanced=args.collector=='gpu'))
              if panel:
                reserved={s for p in panel['opponents'] for s in p['seeds']}
                if any(a['seed'] in reserved for a in jobs): raise ValueError('Collection seed collides with evaluation seed')
              if arena_pool is not None:
                from ppo.hybrid import collect_hybrid
                episodes,metrics=collect_hybrid(jobs,arena_pool,gpu_backend,models,reference,base,device,args,world,sampler)
              else:
                episodes,metrics=(gpu_backend.collect(jobs) if gpu_backend is not None else
                    collect(jobs,models,reference,base['worker_quantities'],base['market_quantities'],device,args.workers//world,args.bf16,args.arena_mib,args.burn,args.sequence,sampler=sampler,gae_lambda=args.gae_lambda,reward_beta=args.reward_beta,reward_sigma=args.reward_sigma))
            # PPO already synchronizes ranks on its first reduction. Put that
            # wait here so update_seconds does not include another rank's
            # unfinished collection or cold graph compilation.
            if world>1:dist.barrier()
            if collector is not None:_lap('barrier');metrics['async_timing']=dict(_at)
            _prep_ahead_done=False  # prep-ahead-v1 (source-v65): weights now, next collection's setup during the update
            if collector is not None and getattr(args,'arena_in_gpu',0) and os.environ.get('PPO_PREP_AHEAD','1')=='1':
                try:
                    _early=gather(dict(games=metrics['games']))
                    if rank==0:
                        from ppo.arena_opponents import update_family_weights
                        update_family_weights(args.output/'arena-family-weights.json',[g for r in _early for g in r['games']],iteration)
                    _prep_ahead_done=True
                    if world>1:dist.barrier()
                    collector.prepare(gpu_jobs(split_jobs(iteration+1)),arena_spec)
                except Exception as _pe:print(json.dumps(dict(stage='prep_ahead_failed',error=repr(_pe)[:300])),flush=True)
            if rank==0: atomic_json(args.output/'lifecycle.json',dict(phase='updating',update=iteration,at=now()))
            training=update(module,ref_scorer,episodes,optimizer,device,args,world)
            rngs=gather(dict(cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(device) if device.type=='cuda' else None))
            collected=gather(metrics)
            if live:
                live.complete()
                allowed={r['sha256'] for r in live.state['manifest']['roster']}|{live.loaded['champion_slot']}
                matchups=matchup_counts(matchups,[g for r in collected for g in r['games']],iteration,allowed)
                live.state['matchups']=matchups
            else:matchups=update_matchups(matchups,[g for r in collected for g in r['games']])
            if rank==0:
                ckpt=args.output/f'update-{iteration:06d}.pt'
                save_checkpoint(ckpt,model,optimizer,base,contract,iteration,rngs,matchups,live.state if live else None)
                previous=dict(path=str(ckpt.resolve()),sha256=sha(ckpt),update=iteration)
                atomic_json(args.output/'latest.json',previous)
                from ppo.arena_opponents import update_family_weights
                if not _prep_ahead_done:update_family_weights(args.output/'arena-family-weights.json',[g for r in collected for g in r['games']],iteration)  # prep-ahead-v1: applied once
                if live:
                    finalize_checkpoint(ckpt,args.output,iteration,live.state['reference'])
                seconds=time.perf_counter()-tick
                row=dict(update=iteration,at=now(),training=training,ranks=collected,wall_seconds=seconds,
                         league_state=live.state if live else None,league_error=live.error if live else None,
                         valid_games=sum(r['valid_games'] for r in collected),
                         end_to_end_games_per_second=sum(r['valid_games'] for r in collected)/seconds)
                log=args.output/(f'metrics-{iteration//1000:06d}.jsonl' if live else 'metrics.jsonl')
                with log.open('a') as f:f.write(json.dumps(row)+'\n')
                print(json.dumps({k:v for k,v in row.items() if k!='ranks'}),flush=True)
                if panel and (iteration+1)%args.eval_every==0:
                    atomic_json(args.output/f'eval-request-{iteration:06d}.json',
                                dict(checkpoint=str(ckpt.resolve()),checkpoint_sha256=sha(ckpt),
                                     panel=panel,output=str((args.output/f'eval-{iteration:06d}').resolve())))
            if collector is not None:
                del episodes,gpu_episodes
                import gc as _gc;_gc.collect()
                collector.release(batch_id)
            if world>1: dist.barrier()
            previous=broadcast(previous if rank==0 else None)
            iteration+=1
        if rank==0: atomic_json(args.output/'lifecycle.json',dict(phase='stopped' if (args.output/'STOP').exists() else 'complete',at=now(),next_update=iteration))
    except BaseException as exc:
        failed=True
        if args.output.exists():atomic_json(args.output/f'failure-rank-{rank}.json',dict(at=now(),error=repr(exc)))
        print(json.dumps(dict(stage='failure',rank=rank,error=repr(exc))),flush=True)
        if rank==0 and args.output.exists(): atomic_json(args.output/'failure.json',dict(at=now(),error=repr(exc)))
        raise
    finally:
        if owner is not None:owner.close()
        try:
            if 'collector' in dir() and collector is not None:collector.close()
        except Exception:pass
        try:
            if 'official' in dir() and official is not None:official.close()
        except Exception:pass
        # On failure another rank may still be collecting or reducing. Let
        # torchrun tear down peers instead of blocking exception reporting on
        # collective communicator destruction.
        if dist.is_initialized() and not failed: dist.destroy_process_group()


def _profile_report(prof, path, wall, title):
    """profile-v1 (source-v57): kernel table by CUDA time plus summed device time vs window wall time."""
    import os as _os
    ev = prof.key_averages()
    dev = sum(getattr(e, 'self_device_time_total', getattr(e, 'self_cuda_time_total', 0)) for e in ev) / 1e6
    _os.makedirs(_os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write('%s\nwindow wall %.3f s, summed self device time %.3f s (%.0f%% of wall)\n\n' % (title, wall, dev, 100 * dev / max(wall, 1e-9)))
        for key in ('self_cuda_time_total', 'cuda_time_total', 'cpu_time_total'):
            try: f.write(ev.table(sort_by=key, row_limit=45, max_name_column_width=90) + '\n\n')
            except Exception as exc: f.write('table %s failed: %r\n' % (key, exc))


if __name__=='__main__': main()

