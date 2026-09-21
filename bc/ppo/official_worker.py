"""collection-overlap-v1 child: official (CPU arena) collection in its own process/CUDA context.

Invoked by ppo.hybrid with a spool directory. Reads spec.pt (frozen learner weights, jobs, settings,
generator seed), runs ppo.rollout.collect exactly as the in-process path does, and writes result.pt.
"""
import os,sys,json,traceback
from pathlib import Path


def main(spool):
    spool=Path(spool)
    import torch
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    spec=torch.load(spool/'spec.pt',map_location='cpu',weights_only=False)
    device=torch.device(spec['device'])
    if device.type=='cuda':torch.cuda.set_device(device)
    from ppo.train import load_model
    from ppo.replay import ExactPPOChunk
    from ppo.sampler import BatchedExactSampler
    from ppo.rollout import collect
    model,base=load_model(spec['bc_reference'],device)
    model.load_state_dict(spec['learner_state'],strict=True);model.eval()
    reference,_=load_model(spec['bc_reference'],device)
    reference.requires_grad_(False)
    generator=torch.Generator(device=device);generator.manual_seed(spec['generator_seed'])
    sampler=BatchedExactSampler({'learner':model},reference,base['worker_quantities'],base['market_quantities'],device,False,generator=generator)
    a=spec['args']
    episodes,metrics=collect(spec['jobs'],{'learner':model},reference,base['worker_quantities'],base['market_quantities'],device,
        spec['official_workers'],a['bf16'],a['arena_mib'],a['burn'],a['sequence'],sampler=sampler,gae_lambda=a['gae_lambda'],
        reference_sha=spec['reference_sha'],all_anchors=True,reward_beta=a['reward_beta'],reward_sigma=a['reward_sigma'])
    tmp=spool/'result.pt.tmp';torch.save(dict(episodes=episodes,metrics=metrics),tmp);tmp.replace(spool/'result.pt')
    print(json.dumps(dict(stage='official_child_done',games=len(metrics['games']),seconds=metrics['collection_seconds'])),flush=True)


if __name__=='__main__':
    try:main(sys.argv[1])
    except BaseException:
        traceback.print_exc();sys.stdout.flush();sys.stderr.flush();os._exit(3)
