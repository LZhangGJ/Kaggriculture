"""Bounded Beam example over EXISTING single-project daily choices, not Candidate8.

Every plan is replayed from reset against live opponents. No partial-state cache.
The example is an integration aid, NOT a claim that this narrow space can win all.
"""
import argparse, json
from pathlib import Path
from runtime import ARMS,NAMES,make_pool,jobs,batch,rank,save_json,save_result

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--arm',choices=ARMS,default='f3')
    ap.add_argument('--width',type=int,default=4)
    ap.add_argument('--last-day',type=int,default=28)
    ap.add_argument('--train-seed-start',type=int,default=68000000)
    ap.add_argument('--train-seeds',type=int,default=4)
    ap.add_argument('--test-seed-start',type=int,default=69000000)
    ap.add_argument('--test-seeds',type=int,default=32)
    ap.add_argument('--opponents',nargs='+',choices=NAMES,default=NAMES)
    ap.add_argument('--threads',type=int,default=16)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    if not 1<=args.width<=12 or not 0<=args.last_day<=28:
        ap.error('width 1..12 and last-day 0..28 required')
    train=jobs(args.train_seed_start,args.train_seeds,args.opponents)
    test=jobs(args.test_seed_start,args.test_seeds,args.opponents)
    if {x[0] for x in train}&{x[0] for x in test}:ap.error('train/test seeds overlap')
    args.out.mkdir(parents=True,exist_ok=False)
    (args.out/'plans').mkdir();pool=make_pool();cache={};beam=[tuple([0]*29)];logs=[]
    def evaluate(plan):
        if plan not in cache:
            path=args.out/'plans'/f'p{len(cache):06d}.txt'
            path.write_text(' '.join(map(str,plan)))
            r=batch(pool,args.arm,train,path,args.threads)
            info=dict(plan=list(plan),file=str(path.relative_to(args.out)),score=rank(r),
                train_games=r['rows'],fallbacks=int(r['fallback'].sum()))
            cache[plan]=info
        return cache[plan]
    for day in range(args.last_day+1):
        children=set()
        for parent in beam:
            for choice in range(10):
                child=list(parent);child[day]=choice;children.add(tuple(child))
        ranked=[evaluate(plan) for plan in sorted(children)]
        ranked.sort(key=lambda x:(tuple(-v for v in x['score']),tuple(x['plan'])))
        beam=[tuple(x['plan']) for x in ranked[:args.width]]
        logs.append(dict(day=day,evaluations=len(cache),best=ranked[0]))
        save_json(args.out/'progress.json',logs)
        print('day',day,'unique_plans',len(cache),'best',ranked[0]['score'],flush=True)
    # Freeze the training winner BEFORE looking at independent seeds.
    best=evaluate(beam[0]);frozen=args.out/'frozen_best.txt'
    frozen.write_text(' '.join(map(str,best['plan'])))
    save_json(args.out/'search.json',dict(args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        unique_plans=len(cache),search_games=len(cache)*len(train),best=best,
        caveat='Only 10 single-project RL overlay choices; numeric-plan dedup, not behavior clustering. Offline hindsight labels, not online decisions.'))
    result=batch(pool,args.arm,test,frozen,args.threads)
    save_result(args.out/'unseen_test',result)
    save_result(args.out/'unseen_keep',batch(pool,args.arm,test,threads=args.threads))
    print('Frozen candidate and matched KEEP tested. Do not tune using this test set.')

if __name__=='__main__':main()
