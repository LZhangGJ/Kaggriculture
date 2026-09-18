"""Run one real cached recurrent chunk through forward/backward on CPU."""
import argparse,json,pickle
from pathlib import Path
from compression import zstd
import torch
from exact_identity import validate_identity
from exact_model import ExactWorkerMarketPolicyV1
from exact_training import ExactChunk,merge_chunks

def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);a=p.parse_args()
    ci=json.loads((a.cache/'identity.json').read_text());validate_identity(ci)
    status=json.loads((a.cache/'status.json').read_text())
    if not status['complete']:raise ValueError('Incomplete cache')
    entry=json.loads((a.cache/'manifest.jsonl').open().readline())
    with zstd.open(entry['file'],'rb') as f:chunk=pickle.load(f)
    torch.set_num_threads(2);torch.manual_seed(1720)
    model=ExactWorkerMarketPolicyV1();module=ExactChunk(model)
    batch=merge_chunks([chunk]);state=(torch.zeros(1,256),torch.zeros(1,256))
    objective,_,metrics=module(batch,state);loss=objective/batch['steps']
    if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
    loss.backward()
    grads=[p.grad for p in model.parameters() if p.grad is not None]
    if not grads or not all(torch.isfinite(g).all() for g in grads):raise ValueError('Invalid gradients')
    print(json.dumps(dict(passed=True,steps=batch['steps'],loss=float(loss.detach()),gradient_tensors=len(grads))))

if __name__=='__main__':main()
