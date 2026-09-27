"""async-v1: GPU self-play/older-policy collection in a child process on the SAME GPU as its trainer rank.

Protocol (torch.multiprocessing, spawn context; CUDA tensors cross the queue by IPC, no copies):
  parent -> child  request: dict(iteration, jobs, weights={role: state_dict on cuda}, reference_sha, reward=dict(...), gae_lambda)
  child  -> parent result:  dict(iteration, episodes=[gpu episode dicts], metrics, batch_id)
  parent -> child  release: dict(release=batch_id)   # the child may then drop its references to that rollout
  parent -> child  None     # shutdown
The child keeps every unreleased rollout alive (IPC-shared memory belongs to the exporting process).
The trainer consumes batch k while the child collects batch k+1 with weights that are one update old.
"""
import os,sys,time,json,traceback
import torch


def child_main(device_index,bc_reference,cache_identity_check,workers,request_queue,result_queue,settings):
    for k,v in settings.items():os.environ.setdefault(k,v)
    if os.environ.get('PPO_CHILD_ALLOC_CONF'):  # scale-out-v2: the child's CUDA tensors cross to the parent by IPC; with
        os.environ['PYTORCH_CUDA_ALLOC_CONF']=os.environ['PPO_CHILD_ALLOC_CONF']  # expandable segments that needs pidfd_getfd (blocked by container seccomp)
    for key in ('PPO_JAX_SITE_PACKAGES','PPO_JAX_SIM_SRC'):
        if os.environ.get(key) and os.environ[key] not in sys.path:sys.path.append(os.environ[key])
    import threading
    parent=os.getppid()
    def watchdog():
        while True:
            time.sleep(5)
            if os.getppid()!=parent:os._exit(0)
    threading.Thread(target=watchdog,daemon=True).start()
    torch.cuda.set_device(device_index);device=torch.device(f'cuda:{device_index}')
    torch.backends.cuda.matmul.allow_tf32=(os.environ.get('PPO_TF32','1')=='1');torch.backends.cudnn.allow_tf32=(os.environ.get('PPO_TF32','1')=='1')  # tf32-v1 (source-v58): match the trainer's replay numerics
    from ppo.train import load_model
    from ppo.gpu_backend import GPUBackend
    reference,base=load_model(bc_reference,device)
    models={};backend=None;held={};batch_id=0
    def role_model(name):
        if name not in models:
            m,_=load_model(bc_reference,device);m.requires_grad_(False);models[name]=m
        return models[name]
    while True:
        req=request_queue.get()
        if req is None:break
        if 'prepare' in req:  # prep-ahead-v1 (source-v65)
            try:
                if backend is not None:backend.arena=req.get('arena');backend.prepare_ahead(req['prepare'])
            except Exception as exc:print(json.dumps(dict(stage='prepare_ahead_request_error',error=repr(exc)[:300])),flush=True)
            continue
        if 'release' in req:
            held.pop(req['release'],None);torch.cuda.empty_cache();continue  # child-cache-v1 (source-v59)
        try:
            t0=time.perf_counter()
            weights=torch.load(req['weights_path'],map_location='cpu',weights_only=True)
            try:os.unlink(req['weights_path'])
            except OSError:pass
            for name,state in weights.items():
                role_model(name).load_state_dict({k:v.to(device) for k,v in state.items()},strict=True)
            del weights
            needed={p for j in req['jobs'] for p in j['policies'] if not (req.get('arena') and p.startswith('script:arena:'))}  # arena-in-gpu-v1: bots need no model
            for name in needed:role_model(name)
            if backend is None:
                backend=GPUBackend(models,reference,base['worker_quantities'],base['market_quantities'],device,workers)
            backend.models=models
            backend.arena=req.get('arena')  # arena-in-gpu-v1: None keeps the neural-only GPU collector
            backend.distill=req.get('distill')  # distill-v1: None = no teacher queries
            rw=req['reward'];backend.gae_lambda=req['gae_lambda']
            backend.reward_beta=rw['beta'];backend.reward_sigma=rw['sigma'];backend.reward_cash_weight=rw['cash_weight'];backend.reward_cash_center=rw['cash_center']
            backend.reward_shape=rw['shape'];backend.reward_dense=rw['dense'];backend.reference_sha=req['reference_sha']
            backend.reward_potential=rw.get('potential','cash')  # networth-shaping-v1
            backend.reward_potential_weight=rw.get('potential_weight')  # networth-shaping-v2
            backend.rollout=None  # fresh storage: the previous rollout may still be in use by the trainer
            backend.iteration=req['iteration']  # surrogate-capture-v1 (source-v67): shard naming
            episodes,metrics=backend.collect(req['jobs'])
            for ep in episodes:ep.pop('comparison_advantages',None)
            batch_id+=1;held[batch_id]=(backend.rollout,episodes)
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()  # child-cache-v1 (source-v59): idle cache back to the trainer during its update
            result_queue.put(dict(iteration=req['iteration'],batch_id=batch_id,episodes=episodes,metrics=dict(metrics,child_seconds=time.perf_counter()-t0,held_batches=len(held))))
        except BaseException as exc:
            result_queue.put(dict(iteration=req.get('iteration'),error=repr(exc),trace=traceback.format_exc()[-2000:]))
            if not isinstance(exc,Exception):raise


def official_child_main(device_index,bc_reference,workers,arena_mib,burn,sequence,request_queue,result_queue,settings):
    """async-v2: real-arena (official kaggle-environments + Docker programs) collection with the lagged learner."""
    for k,v in settings.items():os.environ.setdefault(k,v)
    if os.environ.get('PPO_CHILD_ALLOC_CONF'):  # scale-out-v2: the child's CUDA tensors cross to the parent by IPC; with
        os.environ['PYTORCH_CUDA_ALLOC_CONF']=os.environ['PPO_CHILD_ALLOC_CONF']  # expandable segments that needs pidfd_getfd (blocked by container seccomp)
    for key in ('PPO_JAX_SITE_PACKAGES','PPO_JAX_SIM_SRC'):
        if os.environ.get(key) and os.environ[key] not in sys.path:sys.path.append(os.environ[key])
    import threading
    parent=os.getppid()
    def watchdog():
        while True:
            time.sleep(5)
            if os.getppid()!=parent:os._exit(0)
    threading.Thread(target=watchdog,daemon=True).start()
    torch.cuda.set_device(device_index);device=torch.device(f'cuda:{device_index}')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    from ppo.train import load_model
    from ppo.sampler import BatchedExactSampler
    from ppo.rollout import collect
    reference,base=load_model(bc_reference,device)
    models={};sampler=None
    def role_model(name):
        if name not in models:
            m,_=load_model(bc_reference,device);m.requires_grad_(False);models[name]=m
        return models[name]
    while True:
        req=request_queue.get()
        if req is None:break
        try:
            t0=time.perf_counter()
            weights=torch.load(req['weights_path'],map_location='cpu',weights_only=True)
            try:os.unlink(req['weights_path'])
            except OSError:pass
            for name,state in weights.items():
                role_model(name).load_state_dict({k:v.to(device) for k,v in state.items()},strict=True)
            del weights
            for j in req['jobs']:
                for p in j['policies']:
                    if not p.startswith('script:'):role_model(p)
            if sampler is None:
                sampler=BatchedExactSampler(models,reference,base['worker_quantities'],base['market_quantities'],device,False,generator=torch.Generator(device=device))
            sampler.generator.manual_seed(req['seed'])
            rw=req['reward']
            if req['jobs']:
                episodes,metrics=collect(req['jobs'],models,reference,base['worker_quantities'],base['market_quantities'],device,workers,False,arena_mib,burn,sequence,
                    sampler=sampler,gae_lambda=req['gae_lambda'],reference_sha=req['reference_sha'],all_anchors=False,reward_beta=rw['beta'],reward_sigma=rw['sigma'],
                    reward_cash_weight=rw['cash_weight'],reward_cash_center=rw['cash_center'],reward_shape=rw['shape'],reward_dense=rw['dense'])
            else:
                episodes,metrics=[],dict(games=[],valid_games=0,full_seasons=0,learner_turns=0,collection_seconds=0.,families={})
            out=req['result_path'];tmp=out+'.tmp'
            torch.save(dict(episodes=episodes,metrics=dict(metrics,child_seconds=time.perf_counter()-t0)),tmp);os.replace(tmp,out)
            result_queue.put(dict(iteration=req['iteration'],result_path=out))
        except BaseException as exc:
            result_queue.put(dict(iteration=req.get('iteration'),error=repr(exc),trace=traceback.format_exc()[-2000:]))
            if not isinstance(exc,Exception):raise


class AsyncOfficial:
    """Parent-side handle for the official-collection child (async-v2)."""
    def __init__(self,device,bc_reference,workers,arena_mib,burn,sequence,settings):
        import torch.multiprocessing as mp
        ctx=mp.get_context('spawn')
        self.request=ctx.Queue();self.result=ctx.Queue()
        self.proc=ctx.Process(target=official_child_main,args=(device.index if device.index is not None else 0,str(bc_reference),workers,arena_mib,burn,sequence,self.request,self.result,dict(settings)),daemon=False)
        self.proc.start();self.pending=None
    def submit(self,iteration,jobs,models,reference_sha,reward,gae_lambda,seed):
        weights={name:{k:v.detach().to('cpu',copy=True) for k,v in m.state_dict().items()} for name,m in models.items() if name!='bc'}
        path=f'/dev/shm/async-official-weights-{os.getpid()}-{iteration}.pt';tmp=path+'.tmp';torch.save(weights,tmp);os.replace(tmp,path)
        result=f'/dev/shm/async-official-result-{os.getpid()}-{iteration}.pt'
        self.request.put(dict(iteration=iteration,jobs=jobs,weights_path=path,reference_sha=reference_sha,reward=reward,gae_lambda=gae_lambda,seed=seed,result_path=result));self.pending=iteration
    def wait(self,iteration,timeout=3600):
        res=self.result.get(timeout=timeout)
        if 'error' in res:raise RuntimeError('async official collector failed at iteration %s: %s\n%s'%(res.get('iteration'),res['error'],res.get('trace','')))
        if res['iteration']!=iteration:raise RuntimeError('async official collector returned iteration %s, expected %s'%(res['iteration'],iteration))
        payload=torch.load(res['result_path'],map_location='cpu',weights_only=False)
        try:os.unlink(res['result_path'])
        except OSError:pass
        self.pending=None
        return payload['episodes'],payload['metrics']
    def close(self):
        try:self.request.put(None);self.proc.join(30)
        finally:
            if self.proc.is_alive():self.proc.terminate()


class AsyncCollector:
    """Parent-side handle. One per rank; lives on the rank's GPU."""
    def __init__(self,device,bc_reference,workers,settings):
        import torch.multiprocessing as mp
        ctx=mp.get_context('spawn')
        self.request=ctx.Queue();self.result=ctx.Queue()
        # not a daemon: the collector's event preparation uses its own process pool; a watchdog thread exits with the parent instead.
        self.proc=ctx.Process(target=child_main,args=(device.index if device.index is not None else 0,str(bc_reference),None,workers,self.request,self.result,dict(settings)),daemon=False)
        self.proc.start();self.pending=None;self.last=None
    def submit(self,iteration,jobs,models,reference_sha,reward,gae_lambda,arena=None,distill=None):
        # Weights go through a file in /dev/shm: parent->child CUDA IPC needs pidfd_getfd() on the parent (forbidden by
        # ptrace_scope=1) and fd-passing of CPU shared tensors deadlocked one rank; a plain file is race-free.
        weights={name:{k:v.detach().to('cpu',copy=True) for k,v in m.state_dict().items()} for name,m in models.items() if name!='bc'}
        path=f'/dev/shm/async-weights-{os.getpid()}-{iteration}.pt';tmp=path+'.tmp';torch.save(weights,tmp);os.replace(tmp,path)
        request=dict(iteration=iteration,jobs=jobs,weights_path=path,reference_sha=reference_sha,reward=reward,gae_lambda=gae_lambda)
        if arena is not None:request['arena']=arena  # arena-in-gpu-v1
        if distill is not None:request['distill']=distill  # distill-v1
        self.request.put(request);self.pending=iteration
    def wait(self,iteration,timeout=3600):
        res=self.result.get(timeout=timeout)
        if 'error' in res:raise RuntimeError('async collector failed at iteration %s: %s\n%s'%(res.get('iteration'),res['error'],res.get('trace','')))
        if res['iteration']!=iteration:raise RuntimeError('async collector returned iteration %s, expected %s'%(res['iteration'],iteration))
        self.last=res['batch_id'];self.pending=None
        return res['episodes'],res['metrics']
    def release(self,batch_id):self.request.put(dict(release=batch_id))
    def prepare(self,jobs,arena):self.request.put(dict(prepare=jobs,arena=arena))  # prep-ahead-v1 (source-v65)
    def close(self):
        try:self.request.put(None);self.proc.join(30)
        finally:
            if self.proc.is_alive():self.proc.terminate()
