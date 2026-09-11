import argparse
from pathlib import Path
from runtime import ARMS,NAMES,make_pool,jobs,batch,save_result,summary
import json

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--arm',choices=ARMS,required=True)
    ap.add_argument('--seed-start',type=int,default=67000000)
    ap.add_argument('--seeds',type=int,default=4)
    ap.add_argument('--opponents',nargs='+',choices=NAMES,default=NAMES)
    ap.add_argument('--plan',type=Path)
    ap.add_argument('--threads',type=int,default=16)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    if args.out.exists(): ap.error('output already exists')
    result=batch(make_pool(),args.arm,jobs(args.seed_start,args.seeds,args.opponents),args.plan,args.threads)
    save_result(args.out,result)
    s=summary(result);print(json.dumps(s,ensure_ascii=False,indent=2))
    if s['status']!='PASS':raise SystemExit(1)
if __name__=='__main__':main()
