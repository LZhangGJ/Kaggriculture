"""One- or two-GPU recurrent BC over every cached trajectory, no holdouts."""
import argparse,json,os,pickle,time,resource
from contextlib import ExitStack
from pathlib import Path
from compression import zstd
import numpy as np
import torch
from torch import nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import IterableDataset,DataLoader,get_worker_info
from exact_model import ExactWorkerMarketPolicyV1,SCHEMA
from exact_training import ExactChunk,merge_chunks,to_device
from exact_identity import validate_identity
from cache_identity import file_sha


class CachedChunks(IterableDataset):
    def __init__(self,groups):self.groups=groups
    def __iter__(self):
        wi=get_worker_info();wid=wi.id if wi else 0;nw=wi.num_workers if wi else 1
        torch.set_num_threads(1)
        for group in range(wid,len(self.groups),nw):
            with ExitStack() as stack:
                files=[stack.enter_context(zstd.open(e['file'],'rb')) for e in self.groups[group]]
                while True:
                    chunks=[pickle.load(f) for f in files]
                    batch=merge_chunks(chunks);batch['group']=group;yield batch
                    if batch['last']:break


def identity(x):return x


def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--batch-games',type=int,default=64);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--epochs',type=int,default=1);p.add_argument('--max-chunks',type=int)
    p.add_argument('--fp32',action='store_true');p.add_argument('--initialize-encoder',type=Path)
    p.add_argument('--resume',type=Path,help='Continue a completed epoch, including optimizer state')
    a=p.parse_args();world=int(os.environ.get('WORLD_SIZE',1));rank=int(os.environ.get('RANK',0));local=int(os.environ.get('LOCAL_RANK',0))
    # Torch's file_descriptor sharing uses an FD per live CPU tensor storage.
    # Ragged event batches exceed SSH's default 1024; spawned loaders inherit
    # this process-local limit. Keep the kernel's existing hard limit unchanged.
    fd_soft,fd_hard=resource.getrlimit(resource.RLIMIT_NOFILE)
    fd_target=65536 if fd_hard==resource.RLIM_INFINITY else min(65536,fd_hard)
    if fd_soft<fd_target:resource.setrlimit(resource.RLIMIT_NOFILE,(fd_target,fd_hard))
    if a.batch_games<world or a.batch_games%world:raise ValueError('Global batch must divide by world size')
    ci=json.loads((a.cache/'identity.json').read_text());cache_digest=validate_identity(ci)
    status=json.loads((a.cache/'status.json').read_text())
    if not status['complete'] or status.get('cache_digest')!=cache_digest or status.get('manifest_sha')!=file_sha(a.cache/'manifest.jsonl'):
        raise ValueError('Incomplete or mismatched numeric cache')
    if not ci['all_replays'] and not a.max_chunks:raise ValueError('Subset cache is allowed only for bounded diagnostics')
    entries=[json.loads(l) for l in (a.cache/'manifest.jsonl').read_text().splitlines()]
    if len(entries)!=status['trajectories'] or any(e['cache_digest']!=cache_digest for e in entries):raise ValueError('Cache coverage mismatch')
    if status.get('turns')!=719*len(entries) or any(e.get('turns')!=719 for e in entries):raise ValueError('Incomplete season coverage')
    if (a.output/'run-config.json').exists():raise ValueError('Use a new run directory; never overwrite a run')
    torch.set_num_threads(2);torch.cuda.set_device(local);device=f'cuda:{local}'
    if world>1:dist.init_process_group('nccl')
    torch.manual_seed(1720);torch.backends.cuda.matmul.allow_tf32=True
    if a.resume and a.initialize_encoder:raise ValueError('Resume and encoder initialization are mutually exclusive')
    model=ExactWorkerMarketPolicyV1();initialization=None;resume=None;first_epoch=0
    if a.initialize_encoder:initialization=model.warm_start_encoder_file(a.initialize_encoder)
    if a.resume:
        resume=torch.load(a.resume,map_location='cpu',weights_only=True)
        if not resume.get('epoch_complete'):raise ValueError('Resume requires a completed epoch checkpoint')
        if resume['cache_identity']['digest']!=cache_digest:raise ValueError('Resume cache mismatch')
        model.load_exact(resume);first_epoch=resume['epoch']+1
        initialization=dict(resume_checkpoint=str(a.resume),sha256=file_sha(a.resume),optimizer_loaded=True)
    model=model.to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4)
    if resume:
        optimizer.load_state_dict(resume['optimizer'])
        del resume
    module=ExactChunk(model);wrapped=DDP(module,device_ids=[local],find_unused_parameters=True) if world>1 else module
    a.output.mkdir(parents=True,exist_ok=True)
    config=dict(cache_digest=cache_digest,cache=str(a.cache),schema=SCHEMA,architecture=model.architecture,
                trainer_sha=file_sha(__file__),manifest_sha=file_sha(a.cache/'manifest.jsonl'),
                all_replays=ci['all_replays'],seat_trajectories=len(entries),world=world,batch_games=a.batch_games,
                workers_per_rank=a.workers,chunk=ci['chunk'],bf16=not a.fp32,epochs=a.epochs,
                file_descriptor_limit=resource.getrlimit(resource.RLIMIT_NOFILE)[0],
                diagnostic=bool(a.max_chunks),initialization=initialization,first_epoch=first_epoch,
                optimizer='resumed AdamW' if a.resume else 'fresh AdamW lr=0.0001',
                recurrence='whole season; detach at chunk boundaries',loss=ci['loss'])
    if rank==0:(a.output/'run-config.json').write_text(json.dumps(config,indent=2))
    start=time.monotonic();last_save=start;seen=0;turns_seen=0;completed_seats=0;states={}
    def save(epoch,epoch_complete=False):
        if rank:return
        tmp=a.output/'latest.tmp'
        torch.save(dict(schema=SCHEMA,architecture=model.architecture,model=model.state_dict(),optimizer=optimizer.state_dict(),
            worker_quantities=ci['worker_quantities'],market_quantities=ci['market_quantities'],cache_identity=ci,
            run_config=config,epoch=epoch,epoch_complete=epoch_complete,chunks=seen),tmp);tmp.replace(a.output/'latest.pt')
    for epoch in range(first_epoch,first_epoch+a.epochs):
        epoch_turns=turns_seen;epoch_seats=completed_seats
        order=np.random.default_rng(1720+epoch).permutation(len(entries))
        global_groups=[order[s:s+a.batch_games] for s in range(0,len(order),a.batch_games)]
        # Merge a tiny remainder into the prior group, never duplicate/drop seats.
        if len(global_groups[-1])<world:
            if len(global_groups)==1:raise ValueError('Fewer seats than ranks')
            global_groups[-2]=np.concatenate(global_groups[-2:]);global_groups.pop()
        groups=[[entries[i] for i in group[rank::world]] for group in global_groups]
        opts=dict(batch_size=None,collate_fn=identity,num_workers=a.workers,pin_memory=True)
        if a.workers:opts.update(prefetch_factor=1,multiprocessing_context='spawn',timeout=120)
        loader=DataLoader(CachedChunks(groups),**opts);wait_start=time.monotonic()
        for cpu in loader:
            wait=time.monotonic()-wait_start;compute=time.monotonic();g=cpu['group'];b=cpu['batch']
            if cpu['begin']==0:states[g]=tuple(torch.zeros(b,256,device=device) for _ in range(2))
            batch=to_device(cpu,device);optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=not a.fp32):objective_sum,state,metrics=wrapped(batch,states[g])
            count=torch.tensor(b*cpu['steps'],device=device,dtype=torch.float32)
            if world>1:dist.all_reduce(count)
            objective=objective_sum.float()*world/count
            if not torch.isfinite(objective):raise RuntimeError('Nonfinite training objective')
            objective.backward();nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
            states[g]=state;seen+=1;turns_seen+=int(count)
            if cpu['last']:states.pop(g);completed_seats+=int(count)//cpu['steps']
            # All-reduce actual component sums, not rank-zero approximations.
            report_values=torch.cat((metrics['components'],metrics['value'].reshape(1),metrics['correct']))
            if world>1:dist.all_reduce(report_values)
            torch.cuda.synchronize();elapsed=time.monotonic()-start
            if rank==0 and (seen<=3 or seen%10==0 or cpu['last']):
                report=dict(epoch=epoch,chunks=seen,begin=cpu['begin'],completed_seats=completed_seats,total_seats=len(entries),
                    elapsed_seconds=elapsed,loader_wait_seconds=wait,compute_seconds=time.monotonic()-compute,
                    mean_components=(report_values[:7]/count).tolist(),correct_counts=report_values[7:].tolist(),
                    processed_player_turns=turns_seen,player_turns_per_second=turns_seen/elapsed,
                    peak_memory_gb=torch.cuda.max_memory_allocated()/1e9)
                with (a.output/'metrics.jsonl').open('a') as stream:stream.write(json.dumps(report)+'\n')
                print(json.dumps(report),flush=True)
            if time.monotonic()-last_save>60:save(epoch);last_save=time.monotonic()
            if a.max_chunks and seen>=a.max_chunks:
                save(epoch)
                if rank==0:(a.output/'diagnostic-complete.json').write_text(json.dumps(dict(complete=True,chunks=seen,seconds=elapsed)))
                if world>1:dist.destroy_process_group()
                return
            wait_start=time.monotonic()
        if turns_seen-epoch_turns!=719*len(entries) or completed_seats-epoch_seats!=len(entries) or states:
            raise RuntimeError('Epoch did not cover every complete trajectory exactly once')
        save(epoch,epoch_complete=True)
    if rank==0:(a.output/'complete.json').write_text(json.dumps(dict(complete=True,completed_seats=completed_seats,seconds=time.monotonic()-start)))
    if world>1:dist.destroy_process_group()


if __name__=='__main__':main()
