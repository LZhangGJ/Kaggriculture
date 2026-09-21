"""Bounded live league state; immutable snapshots remain outside the hot path."""
import copy
import hashlib
import json
import uuid
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from ppo.league import sha
CONTROL=None

def init_control():
    global CONTROL
    if dist.is_initialized():CONTROL=dist.new_group(backend='gloo')

PROTOCOL = dict(schema='broad-ppo-v1', weights=[.5,.25,.25], roster_size=8,
                rotation_updates=2, reference_updates=0, snapshot_updates=10,
                nomination_updates=50, confirmation_pairs=64, support_pairs=32, bootstrap_samples=20000, family_regression=.20, lower_quartile_regression=.05, confirmation_alpha=.05, proxies_enabled=False)

def atomic(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.'+uuid.uuid4().hex+'.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)

def digest(value): return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()

def descriptor(path): return dict(path=str(Path(path).resolve()),sha256=sha(path))

def validate_manifest(value):
    if value['protocol']!=PROTOCOL: raise ValueError('League protocol mismatch')
    rows=value['roster']
    if not 1<=len(rows)<=8 or len({r['sha256'] for r in rows})!=len(rows):
        raise ValueError('Invalid bounded roster')
    approved=value['approved']
    if len(approved)>12:raise ValueError('Unbounded approved bindings')
    bindings=[value['champion'],value['teacher'],*rows,*value['retention'],*value.get('coverage',[]),*value.get('recent',[])]
    for r in bindings:
        if len(r['sha256'])!=64 or not Path(r['path']).is_absolute():raise ValueError('Invalid descriptor')
        if r!=approved.get(r['sha256']):raise ValueError('Conflicting or unapproved binding')
        if r.get('pool')!='practice' or not r.get('eligible') or r.get('decoding')!='sampled' or not r.get('family'):raise ValueError('Unsafe binding')
        if r.get('proxy'):raise ValueError('Proxy admission disabled pending separate qualification review')
    if len(value['retention'])!=2 or len(value.get('coverage',[]))>4:raise ValueError('Invalid retention/coverage')
    return value

def broadcast(value):
    box=[value]
    if dist.is_initialized():dist.broadcast_object_list(box,src=0,group=CONTROL)
    return box[0]

def storage(model):return {k:v.data_ptr() for k,v in model.state_dict().items()}

def stage(model, spec, identity):
    if sha(spec['path'])!=spec['sha256']:raise ValueError('Snapshot checksum mismatch')
    ck=torch.load(spec['path'],map_location='cpu',weights_only=True)
    if ck['cache_identity']!=identity:raise ValueError('Snapshot representation mismatch')
    from ppo.shaped_reward import with_shaped_head
    target=model.state_dict();weights=with_shaped_head(ck['model'],model)
    if target.keys()!=weights.keys():raise ValueError('Snapshot keys mismatch')
    for k,v in weights.items():
        if v.shape!=target[k].shape or v.dtype!=target[k].dtype or not torch.isfinite(v).all():
            raise ValueError('Invalid snapshot tensor '+k)
    return weights

def copy_weights(model, weights):
    addresses=storage(model)
    with torch.no_grad():model.load_state_dict(weights,strict=True,assign=False)
    if storage(model)!=addresses:raise RuntimeError('Graph storage changed')

class LiveLeague:
    def __init__(self, manifest, models, reference, identity, initial, state=None):
        self.path=Path(manifest);self.models=models;self.reference=reference;self.identity=identity
        self.state=copy.deepcopy(state) if state else dict(manifest=validate_manifest(json.loads(self.path.read_text())),
            reference=initial,reference_age=0,reference_refreshes=0,rotation=0,applied={})
        validate_manifest(self.state['manifest'])
        for r in self.state.get('applied',{}).values():
            if r.get('proxy') or not r.get('eligible') or r.get('pool')!='practice' or r.get('decoding')!='sampled':raise ValueError('Unsafe restored binding')
        self.loaded={};self.error=None

    def boundary(self, iteration, previous=None):
        rank=dist.get_rank() if dist.is_initialized() else 0
        plan=copy.deepcopy(self.state)
        # Resume starts from saved state. Later boundaries may adopt a newer manifest.
        if rank==0 and self.loaded:
            try:
                desired=validate_manifest(json.loads(self.path.read_text()))
                if desired['version']>plan['manifest']['version']:plan['manifest']=desired
                self.error=None
            except Exception as exc:self.error=repr(exc)
        plan=broadcast(plan if rank==0 else None)
        from ppo.broad_curriculum import choose
        teacher=plan['manifest']['teacher']
        if plan['reference']['sha256']!=teacher['sha256']:
            plan['reference']=teacher;plan['reference_age']=0;plan['reference_refreshes']+=1
        block=plan['rotation']//2
        if plan.get('selection',{}).get('block')!=block or not plan.get('applied'):
            keeper,active,selection=choose(plan['manifest'],plan['rotation'],plan.get('matchups',{}))
            specs=dict(champion_slot=keeper,history_slot=active,reference=plan['reference'])
            plan['selection']=selection
        else:
            specs=dict(plan['applied'],reference=plan['reference'])
        def prepare(bindings):
            staged={};error=None
            try:
                for key,spec in bindings.items():
                    if self.loaded.get(key)!=spec['sha256']:
                        staged[key]=stage(self.reference if key=='reference' else self.models[key],spec,self.identity)
            except Exception as exc:error=repr(exc)
            errors=[error]
            if dist.is_initialized():
                errors=[None]*dist.get_world_size();dist.all_gather_object(errors,error,group=CONTROL)
            return staged,errors
        staged,errors=prepare(specs)
        accepted=not any(errors)
        if not accepted:
            if not self.loaded:raise ValueError('Initial league loading failed: '+str(errors))
            self.error=str(errors)
            plan=copy.deepcopy(self.state)
            specs=plan['applied']
            staged,errors=prepare(specs)
            if any(errors):raise ValueError('Retained binding failed: '+str(errors))
        device=next(self.reference.parameters()).device
        if device.type=='cuda':torch.cuda.synchronize(device)
        for key,weights in staged.items():
            copy_weights(self.reference if key=='reference' else self.models[key],weights)
        if dist.is_initialized():dist.barrier()
        self.loaded={k:v['sha256'] for k,v in specs.items()}
        plan['applied']=specs;self.state=plan
        return accepted

    def jobs(self, seed, iteration, games, world, rank):
        if games%(8*world):raise ValueError('Fixed league requires games divisible by 8*world')
        if seed<190000000 or seed+iteration*games+games>=2**31:raise ValueError('Training seed namespace exhausted/invalid')
        rows=[];offset=0
        for family,count in [('selfplay',games//2),('champion_slot',games//4),('history_slot',games//4)]:
            for j in range(count):
                owner=j%world;seat=(j//world+iteration)%2
                policies=['learner','learner']
                if family!='selfplay':policies[1-seat]=family
                idx=iteration*games+offset+j
                if owner==rank:
                    rows.append(dict(game=idx,seed=seed+idx,policies=policies,family=family,
                        opponent_sha256=self.loaded.get(family),reference_sha256=self.loaded['reference'],
                        learner_seats=[s for s,p in enumerate(policies) if p=='learner']))
            offset+=count
        np.random.default_rng(seed+iteration).shuffle(rows)
        return rows

    def complete(self):
        self.state['reference_age']+=1;self.state['rotation']+=1

def admit(manifest, entry):
    from ppo.broad_curriculum import admit as broad_admit
    return validate_manifest(broad_admit(manifest,entry))

def rebuild_roster(value):
    from ppo.broad_curriculum import rebuild
    return validate_manifest(rebuild(value))


def snapshot(checkpoint, destination):
    """Model-only immutable history; never prune user checkpoints here."""
    source=torch.load(checkpoint,map_location='cpu',weights_only=True)
    kept={k:source[k] for k in ('architecture','schema','cache_identity','worker_quantities','market_quantities','model')}
    from ppo.shaped_reward import without_shaped_head
    kept['model']=without_shaped_head(kept['model'])  # pinned evaluators/controller keep the pre-shaped-reward contract
    kept['update']=source.get('update',-1)
    kept['source_sha256']=sha(checkpoint)
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists():
        existing=torch.load(destination,map_location='cpu',weights_only=True)
        if any(existing.get(k)!=v for k,v in kept.items() if k!='model') or set(existing['model'])!=set(kept['model']):
            raise ValueError('Existing snapshot provenance mismatch')
        if any(not torch.equal(existing['model'][k],v) for k,v in kept['model'].items()):raise ValueError('Existing snapshot weights changed')
        return dict(**descriptor(destination),update=kept['update'])
    tmp=destination.with_suffix('.tmp');torch.save(kept,tmp);tmp.replace(destination)
    return dict(**descriptor(destination),update=source.get('update',-1))

def finalize_checkpoint(checkpoint,output,update,reference):
    output=Path(output);checkpoint=Path(checkpoint)
    if (update+1)%10==0:
        item=snapshot(checkpoint,output/'snapshots'/f'update-{update:06d}.pt')
        request=output/'snapshot-request.json'
        if not request.exists() or json.loads(request.read_text())['update']<=update:atomic(request,item)
    obsolete=output/f'update-{update-3:06d}.pt'
    if obsolete.exists() and obsolete.resolve()!=Path(reference['path']).resolve():obsolete.unlink()

def matchup_counts(previous,games,iteration,allowed):
    value={k:copy.deepcopy(v) for k,v in previous.items() if k in allowed and isinstance(v,dict)}
    for row in value.values():
        count=row.get('effective_games',0.);score=row.get('effective_score_sum',0.)
        scale=min(1.,1024/max(1.,count));row['effective_games']=count*scale;row['effective_score_sum']=score*scale
    for game in games:
        key=game.get('opponent_sha256')
        if key not in allowed or len(game['learner_seats'])!=1 or any(game['faults']):continue
        seat=game['learner_seats'][0];cash=game['cash']
        score=.5 if cash[0]==cash[1] else float(cash[seat]>cash[1-seat])
        row=value.setdefault(key,dict(games=0,score_sum=0.,decoding='stochastic'))
        row['games']+=1;row['score_sum']+=score;row['last_update']=iteration
        row['effective_games']=row.get('effective_games',0.)+1;row['effective_score_sum']=row.get('effective_score_sum',0.)+score
    return value
