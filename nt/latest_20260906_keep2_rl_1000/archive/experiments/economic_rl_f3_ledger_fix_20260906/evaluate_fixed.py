"""Evaluate the fixed F3 runner explicitly; never silently load the old SO."""
from pathlib import Path
import os,sys,argparse,time
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
P=Path(__file__).resolve().parent;OLD=P.parent/'economic_rl_three_arm_20260906'
sys.path.insert(0,str(OLD))
from runtime import make_pool,jobs,NAMES,read,summary,store_result

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=['keep','random','greedy','sample'],default='keep')
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--seed-start',type=int,required=True)
    parser.add_argument('--seeds',type=int,default=4)
    parser.add_argument('--sample-seed',type=int,default=17)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():raise FileExistsError(args.out)
    if args.seeds<1:parser.error('--seeds must be positive')
    if args.mode in ('greedy','sample')and (args.checkpoint is None or not args.checkpoint.is_file()):
        parser.error('greedy/sample requires an existing checkpoint')
    assert read(P/'ACCEPTANCE.json')['status']=='PASS'
    jj=jobs(args.seed_start,args.seeds);pool=make_pool();tic=time.perf_counter()
    mode={'keep':0,'greedy':1,'sample':2,'random':3}[args.mode]
    r=pool.batch(str(P/'build/f3.so'),str(args.checkpoint)if args.checkpoint else '',mode,
        [j[0]for j in jj],[j[1]for j in jj],[j[2]for j in jj],args.sample_seed,16,False)
    r['call_seconds']=time.perf_counter()-tic;r['bridge_seconds']=r['call_seconds']-r['wall_seconds'];r['mode']=mode
    for row in r['rows']:row['opponent']=NAMES[row['opponent']]
    store_result(args.out,r);print(summary(r))
    if summary(r)['status']!='PASS':raise SystemExit(1)

if __name__=='__main__':main()
