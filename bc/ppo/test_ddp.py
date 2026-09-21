"""Two CPU/Gloo ranks, one short synthetic update; no GPUs or full games."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import copy
from datetime import timedelta
from types import SimpleNamespace
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from exact_model import ExactWorkerMarketPolicyV1
from ppo.evaluate import LocalBackend
from ppo.sampler import BatchedExactSampler
from ppo.replay import ExactPPOChunk
from ppo.train import LossModule,update


def main():
    torch.set_num_threads(1)
    dist.init_process_group('gloo',timeout=timedelta(seconds=60))
    try:
        rank=dist.get_rank();torch.manual_seed(47)
        model=ExactWorkerMarketPolicyV1().eval();ref=copy.deepcopy(model).requires_grad_(False)
        jobs=[dict(game=rank,seed=9388000+rank,policies=['learner','learner'],learner_seats=[0,1],family='self')]
        backend=LocalBackend(jobs,[1,2],[1,2]);keys=[f'{rank}:0',f'{rank}:1']
        sampler=BatchedExactSampler({'learner':model},ref,[1,2],[1,2])
        records=sampler.act('learner',keys,backend.call('start',keys),backend)
        eps=[dict(turns=[records[k]],outcome=2,advantages=np.array([1. if i else -1.],np.float32)) for i,k in enumerate(keys)]
        # Uneven data forces a zero-weight DDP graph on one rank's last minibatch.
        if rank==1: eps=eps[:1]
        module=DistributedDataParallel(LossModule(model),find_unused_parameters=True)
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5)
        args=SimpleNamespace(burn=0,sequence=1,minibatch=2,seed=1,update_number=0,bf16=False,clip=.1,anchor=.01,max_kl=.03)
        report=update(module,ExactPPOChunk(ref),eps,optimizer,'cpu',args,2)
        assert report['optimizer_steps']>0
        args.max_kl=-1
        report=update(module,ExactPPOChunk(ref),eps,optimizer,'cpu',args,2)
        assert report['stopped_kl'] and report['optimizer_steps']==0
        args.max_kl=100
        report=update(module,ExactPPOChunk(ref),eps,optimizer,'cpu',args,2)
        assert report['optimizer_steps']>0
        # Verify every parameter, not just a checksum, agrees across ranks.
        for parameter in model.parameters():
            expected=parameter.detach().clone();dist.broadcast(expected,0)
            torch.testing.assert_close(parameter,expected,rtol=0,atol=0)
        if rank==0: print('CPU DDP uneven-batch update and identical parameters: PASS',flush=True)
    finally:dist.destroy_process_group()


if __name__=='__main__':main()
