"""Standalone frozen-checkpoint evaluation; no training or oracle continuation."""
import argparse
from runtime import *

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--arm',choices=['c3auto','f3','c3j7'],required=True)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--mode',choices=['keep','greedy','sample','random'],default='greedy')
    p.add_argument('--seed-start',type=int,required=True)
    p.add_argument('--seeds',type=int,default=16)
    p.add_argument('--sample-seed',type=int,default=123456)
    p.add_argument('--threads',type=int,default=16)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    assert a.seeds>0 and 1<=a.threads<=16
    if a.out.exists():raise FileExistsError(a.out)
    if a.mode in ('greedy','sample'):
        if a.checkpoint is None or not a.checkpoint.is_file():p.error('greedy/sample requires an existing --checkpoint')
    r=rollout(make_pool(),a.arm,str(a.checkpoint)if a.checkpoint else '',
        mode={'keep':0,'greedy':1,'sample':2,'random':3}[a.mode],
        start=a.seed_start,count=a.seeds,sample=a.sample_seed,threads=a.threads)
    store_result(a.out,r)
    print(json.dumps(summary(r),indent=2))
    if summary(r)['status']!='PASS':raise SystemExit(1)

if __name__=='__main__':main()
