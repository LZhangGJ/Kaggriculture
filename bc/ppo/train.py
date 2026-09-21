"""Synchronous complete-game PPO. Run only after the deferred GPU smoke."""
import os
for _name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_name,'1')
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
    return h.hexdigest()


def load_model(path, device, identity=None):
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    validate_identity(checkpoint['cache_identity'])
    if identity is not None and checkpoint['cache_identity']!=identity:
        raise ValueError('Opponent representation/cache contract differs from learner')
    if (checkpoint['worker_quantities']!=checkpoint['cache_identity']['worker_quantities'] or
        checkpoint['market_quantities']!=checkpoint['cache_identity']['market_quantities']):
        raise ValueError('Quantity vocabulary mismatch')
    from ppo.shaped_reward import attach_shaped_head,load_exact_with_head
    model=attach_shaped_head(ExactWorkerMarketPolicyV1());load_exact_with_head(model,checkpoint)
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
        if not torch.equal(result['factor_count'],batch['factor_count'].to(result['factor_count'].dtype)):
            raise RuntimeError('PPO factor-count replay mismatch')
        if not torch.isfinite(batch['old_logp']).all() or not torch.isfinite(result['logp']).all():
            raise FloatingPointError('Nonfinite PPO action density')
        if batch.get('check_initial_density',False):
            error=(result['logp'].detach()-batch['old_logp']).abs()[batch['loss_mask']].max()
            if error>.002:raise RuntimeError(f'Initial GPU replay density mismatch: {float(error)}')
        terms,stats=ppo_terms(result,batch['old_logp'],batch['advantages'],batch['outcome_turn'],clip=clip,anchor=anchor,returns=batch['return_turn'],policy_weight=policy_weight)
        mask=batch['loss_mask']
        loss=terms[mask].sum()*scale
        stats={k:v[mask].sum().detach() for k,v in stats.items()}
        stats['count']=mask.sum().detach()
        # A detached diagnostic cannot cause DDP to mark unused parameters as used.
        return loss,stats


def update(module,reference,episodes,optimizer,device,args,world):
    device=torch.device(device)
    if device.type=='cuda': torch.cuda.reset_peak_memory_stats(device)
    started=time.perf_counter()
    stage_events=[]
    def mark(name):
        if device.type=='cuda':
            event=torch.cuda.Event(enable_timing=True);event.record()
            stage_events.append((name,event))
    all_adv=torch.cat([torch.as_tensor(ep['advantages'],device=device).flatten() for ep in episodes]).double() if episodes else torch.empty(0,device=device,dtype=torch.float64)
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
    diagnostics=None
    if args.update_number%20==17 or getattr(args,'hybrid_first_update',False):
        from ppo.retention_diagnostics import diagnose
        scorer=module.module.scorer if hasattr(module,'module') else module.scorer
        diagnostics=diagnose(scorer,reference,probe_rows,episodes,args,device,stage='before')
    if max((x['max_density_error'] for x in audit_before),default=0)>.002:
        raise RuntimeError('Pre-update fixed-window replay density mismatch')
    accepted_by_phase=[0,0,0]
    total={};steps=0;stopped_kl=False;stopping_kl=None
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
    for key,i in schedule:
        rows=groups.get(key,[])
        mark('start')
        selected=rows[i*local_batch:(i+1)*local_batch]
        real=bool(selected)
        if not real:
            fallback_key=next(iter(groups));selected=groups[fallback_key][:1]
        batch,state,refstate=batch_windows(selected)
        batch['check_initial_density']=real and steps==0 and 'gpu_data' in selected[0][0]
        if not real: batch['loss_mask'].zero_()
        batch=to_device(batch,device);state=tuple(s.to(device) for s in state);refstate=tuple(s.to(device) for s in refstate)
        count=reduce(batch['loss_mask'].sum().clone()).item()
        if not count: continue
        optimizer.zero_grad(set_to_none=True)
        mark('batch')
        ctx=torch.autocast('cuda',dtype=torch.bfloat16) if args.bf16 else nullcontext()
        with ctx:
            loss,stats=module(batch,state,refstate,reference,selected[0][3],world/count,args.clip,args.anchor)
        mark('forward')
        if os.environ.get('PPO_PACK_STATS')=='1':
            names=list(stats);values=reduce(torch.stack([stats[k] for k in names]))
            stats=dict(zip(names,values.unbind()))
        else:
            stats={k:reduce(v.clone()) for k,v in stats.items()}
        kl=float(stats['approx_kl']/count)
        if kl>args.max_kl:
            stopped_kl=True;stopping_kl=kl;break
        if not torch.isfinite(loss): raise FloatingPointError('Nonfinite PPO loss')
        mark('statistics')
        loss.backward()
        if policy_frozen:
            from ppo.shaped_reward import freeze_actor_gradients
            freeze_actor_gradients(module,optimizer)
        mark('backward')
        norm=torch.nn.utils.clip_grad_norm_(module.parameters(),.5,error_if_nonfinite=True)
        optimizer.step();steps+=1
        accepted_by_phase[min(2,(key[0]+key[2])//240)]+=int(count)
        mark('optimizer')
        for k,v in stats.items(): total[k]=total.get(k,0.)+float(v)
        total['last_gradient_norm']=float(norm)
        accepted_by_format[key[3]]=accepted_by_format.get(key[3],0)+int(count)
        if steps%6==0 and (not dist.is_initialized() or dist.get_rank()==0):
            print(json.dumps(dict(stage='ppo_update',optimizer_steps=steps,
                learner_turns=total.get('count',0),seconds=time.perf_counter()-started)),flush=True)
    count=total.get('count',0.)
    stage_seconds={}
    if stage_events:
        torch.cuda.synchronize(device)
        for (_,before),(name,after) in zip(stage_events,stage_events[1:]):
            if name!='start':stage_seconds[name]=stage_seconds.get(name,0.)+before.elapsed_time(after)/1000
    audit_after=audit_probe()
    if diagnostics is not None:
        from ppo.retention_diagnostics import diagnose
        diagnostics['after']=diagnose(scorer,reference,probe_rows,episodes,args,device,stage='after')
        atomic_json(args.output/f"retention-{args.update_number:06d}-rank-{dist.get_rank() if dist.is_initialized() else 0}.json",diagnostics)
    if [x["episode_ids"] for x in audit_before]!=[x["episode_ids"] for x in audit_after]:
        raise RuntimeError("Audit sample identities changed")
    if [x["mean_factors"] for x in audit_before]!=[x["mean_factors"] for x in audit_after]:
        raise RuntimeError("Audit factor counts changed")
    return dict(policy_frozen=policy_frozen,accepted_turns_by_format=accepted_by_format,retention_diagnostics=gather(diagnostics),audit_before=gather(audit_before),audit_after=gather(audit_after),accepted_turns_by_phase=accepted_by_phase,update_seconds=time.perf_counter()-started,optimizer_steps=steps,stopped_kl=stopped_kl,stopping_kl=stopping_kl,
                rank_peak_allocated_gib=gather(torch.cuda.max_memory_allocated(device)/2**30 if device.type=='cuda' else 0.),
                rank_stage_seconds=gather(stage_seconds),
                **{k:(v/count if count and k not in ('count','last_gradient_norm') else v) for k,v in total.items()})


def save_checkpoint(path,model,optimizer,base,contract,update_number,rngs,matchups=None,league_state=None):
    value={k:base[k] for k in ('architecture','schema','cache_identity','worker_quantities','market_quantities')}
    value.update(model={k:v.detach().cpu() for k,v in model.state_dict().items()},optimizer=optimizer.state_dict(),
                 ppo_schema='exact-recurrent-ppo-v1',ppo_contract=contract,update=update_number,rngs=rngs,
                 updated_at=now(),matchups=matchups or {},league_state=league_state,provenance=base.get('provenance'))
    tmp=Path(str(path)+'.tmp');torch.save(value,tmp);tmp.replace(path)


def make_optimizer(model,actor_lr,critic_lr,head_lr_scale=100.):
    actor=[];critic=[];head=[]
    for name,p in model.named_parameters():
        (head if name.startswith('value_shaped.') else critic if name.startswith(('critic_memory.','value.')) else actor).append(p)
    # shaped-reward-v4: the residual linear head gets its own group at head_lr_scale x critic LR.
    return torch.optim.AdamW([{'params':actor,'lr':actor_lr},{'params':critic,'lr':critic_lr},{'params':head,'lr':critic_lr*head_lr_scale}],eps=1e-5,weight_decay=1e-4,
                            fused=True if os.environ.get('PPO_FUSED_ADAM')=='1' else None)


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
    p.add_argument('--clip',type=float,default=.1);p.add_argument('--anchor',type=float,default=.01)
    p.add_argument('--max-kl',type=float,default=.03);p.add_argument('--seed',type=int,default=9280000)
    p.add_argument('--updates',type=int,default=1,help='Updates this invocation; 0 means until STOP')
    p.add_argument('--eval-panel',type=Path);p.add_argument('--eval-every',type=int,default=10)
    p.add_argument("--live-league",type=Path)
    p.add_argument("--migration",type=Path)
    p.add_argument("--arena-pool",help="Immutable real arena training pool JSON")
    p.add_argument('--reward-beta',type=float,default=1.,help='shaped-reward-v1: weight of the clipped cash-margin term')
    p.add_argument('--reward-sigma',type=float,default=25000.,help='shaped-reward-v1: cash-margin scale for the clipped term')
    p.add_argument('--policy-warmup-until',type=int,default=0,help='shaped-reward-v2: updates below this index train value heads only (policy term weight 0)')
    p.add_argument('--official-workers',type=int,default=0,help='collection-overlap-v1: actor processes per rank for the official stage (0 = workers/world)')
    p.add_argument('--overlap-collection',type=int,default=0,help='collection-overlap-v1: 1 runs official collection concurrently with GPU collection')
    return p


def install_termination_handler():
    import signal
    def terminate(signum,frame):raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,terminate)


def main():
    install_termination_handler()
    import faulthandler
    if os.environ.get('PPO_TRACE_TIMEOUT'):faulthandler.dump_traceback_later(int(os.environ['PPO_TRACE_TIMEOUT']),repeat=True)
    args=parser().parse_args()
    if args.bf16:
        raise ValueError("BF16 PPO is disabled until full-turn collector/replay density parity passes")
    torch.set_num_threads(1)
    # Collected and replayed joint densities must agree before an update.
    # TF32 kernels differ across collection/replay batch shapes on Blackwell.
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
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
            if args.bf16:raise ValueError('GPU collection requires FP32 replay; BF16 has not passed density checks')
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
            contract['hybrid_collection']=dict(protocol='real-arena-training-v2',game_allocation=[.5,.125,.375],family_sampling='deficit-weighted: weight=.5+deficit/mean(deficit), EMA .9 of per-family mean cash margin, uniform when no history',
                backends=['pinned GPU simulator','official kaggle-environments 1.32.7 stream adapter'],
                minibatches='homogeneous format; deterministic shuffle of five GPU plus one official per temporal window',
                opponents='uniform author then uniform member; source-native sandbox',failure='reject entire update')
        else:
            contract['config'].pop('arena_pool',None)
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
                        if migration.get('kind')=='arena-mix-v2' and key=='hybrid_collection' and migration.get('hybrid_collection')==contract[key]:continue
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
            torch.set_rng_state(saved['rngs'][rank]['cpu'])
            if device.type=='cuda': torch.cuda.set_rng_state(saved['rngs'][rank]['cuda'],device)
        else:
            torch.manual_seed(args.seed+rank)
        loss_module=LossModule(model)
        module=DDP(loss_module,device_ids=[local] if device.type=='cuda' else None,find_unused_parameters=True) if world>1 else loss_module
        if rank==0: atomic_json(args.output/'run-config.json',contract)
        iteration=start_update
        from ppo.sampler import BatchedExactSampler
        official_generator=torch.Generator(device=device) if (args.arena_pool and device.type=='cuda') else None
        sampler=BatchedExactSampler(models,reference,base['worker_quantities'],base['market_quantities'],device,args.bf16,generator=official_generator)
        gpu_backend=None
        if args.collector=='gpu':
            import sys
            for key in ('PPO_JAX_SITE_PACKAGES','PPO_JAX_SIM_SRC'):
                if os.environ.get(key):sys.path.append(os.environ[key])
            from ppo.gpu_backend import GPUBackend
            gpu_backend=GPUBackend(models,reference,base['worker_quantities'],base['market_quantities'],device,args.workers//world)
            gpu_backend.gae_lambda=args.gae_lambda
            gpu_backend.reward_beta=args.reward_beta;gpu_backend.reward_sigma=args.reward_sigma
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
            torch.set_rng_state(saved['rngs'][rank]['cpu'])
            torch.cuda.set_rng_state(saved['rngs'][rank]['cuda'],device)
        if args.resume:
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
            args.hybrid_first_update=arena_pool is not None and iteration==start_update
            if live:
                refresh=previous
                if PROTOCOL['reference_updates'] and live.state['reference_age']>=PROTOCOL['reference_updates']:
                    if rank==0:
                        target=args.output/'references'/f'update-{iteration-1:06d}.pt'
                        refresh=snapshot(previous['path'],target)
                    refresh=broadcast(refresh if rank==0 else None)
                live.boundary(iteration,refresh)
                gpu_backend.reference_sha=live.loaded['reference']
                ref_scorer.reference_sha=live.loaded['reference']
            if rank==0: atomic_json(args.output/'lifecycle.json',dict(phase='collecting',update=iteration,at=now(),pid=os.getpid()))
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
                update_family_weights(args.output/'arena-family-weights.json',[g for r in collected for g in r['games']],iteration)
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
        # On failure another rank may still be collecting or reducing. Let
        # torchrun tear down peers instead of blocking exception reporting on
        # collective communicator destruction.
        if dist.is_initialized() and not failed: dist.destroy_process_group()


if __name__=='__main__': main()
