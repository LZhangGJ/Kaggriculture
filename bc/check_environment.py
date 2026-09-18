"""Check engine/source identity and optionally load a published checkpoint."""
import argparse,json,sys
import torch
from cache_identity import engine_identity
from exact_identity import sources
from exact_decoder import ExactAgent

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint');a=p.parse_args()
    if sys.version_info<(3,14):raise RuntimeError('Use Python 3.14: numeric caches use compression.zstd')
    result=dict(python=sys.version.split()[0],torch=torch.__version__,engine=engine_identity(),sources=sources(),cuda=torch.cuda.is_available())
    if a.checkpoint:
        agent=ExactAgent.from_checkpoint(a.checkpoint)
        result['parameters']=sum(x.numel() for x in agent.model.parameters())
        result['worker_quantities']=len(agent.wq);result['market_quantities']=len(agent.mq)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
