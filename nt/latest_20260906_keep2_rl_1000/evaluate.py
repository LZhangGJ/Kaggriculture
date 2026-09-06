"""CPU inference only; no policy update, opponent identity or future information."""
from pathlib import Path
import argparse,sys,json,time
P=Path(__file__).resolve().parent
BASE=P.parent/'latest_20260906_c3_f3_j7c3_search'
sys.path.insert(0,str(BASE))
import runtime

def evaluate(pool,checkpoint,mode,start,count,threads=16,sample=9931):
    jobs=runtime.jobs(start,count);t=time.perf_counter()
    result=pool.batch(str(P/'build/keep2.so'),str(checkpoint)if checkpoint else '',mode,
        [j[0]for j in jobs],[j[1]for j in jobs],[j[2]for j in jobs],sample,threads,False)
    result['call_seconds']=time.perf_counter()-t
    result['library_sha256']=runtime.sha(P/'build/keep2.so');result['plan_sha256']=None
    for row in result['rows']:
        row['opponent_name']=runtime.NAMES[row['opponent']]
        assert not row['error'] and row['steps']==719 and row['reference_calls']==0
        assert row['plan_calls']==30 and row['execute_calls']==719
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',default='aux_r0_step1000')
    ap.add_argument('--mode',choices=['keep','greedy','sample'],default='greedy')
    ap.add_argument('--seed-start',type=int,default=71100000);ap.add_argument('--seeds',type=int,default=2)
    ap.add_argument('--threads',type=int,default=16);ap.add_argument('--sample-seed',type=int,default=9931)
    ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    if args.out.exists():ap.error('Output exists; use a fresh directory.')
    models={r['id']:r for r in json.loads((P/'MODEL_INDEX.json').read_text())}
    if args.model not in models:ap.error('Unknown model id; see MODEL_INDEX.json.')
    path=P/models[args.model]['bin'];assert runtime.sha(path)==models[args.model]['bin_sha256']
    result=evaluate(runtime.make_pool(),None if args.mode=='keep'else path,{'keep':0,'greedy':1,'sample':2}[args.mode],args.seed_start,args.seeds,args.threads,args.sample_seed)
    runtime.save_result(args.out,result);print(json.dumps(runtime.summary(result),indent=2))
if __name__=='__main__':main()
