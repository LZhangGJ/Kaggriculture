"""Full-season beam construction and large-neighborhood search on our own plans."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from . import native
from .plans import simple_plan,repair,random_plan,mutate,solver_repair
from .runtime import dump


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--seconds',type=float,required=True)
    p.add_argument('--method',choices=('beam','lns'),required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--seed-manifest',type=Path,required=True);args=p.parse_args()
    from .protocol import verify_freeze
    verify_freeze(args.seed_manifest.parent/'FREEZE.json')
    args.out.mkdir(parents=True,exist_ok=False);rng=np.random.default_rng(args.seed)
    training=json.loads(args.seed_manifest.read_text())['train'];seeds=[int(s) for s in rng.choice(training,4,replace=False)]
    rivals=[repair(simple_plan(k)) for k in (0,4,6)]
    start=time.perf_counter();cpu_start=time.process_time();games=0;evaluated={};history=[]
    best=None;best_score=None
    def evaluate(plan):
        nonlocal games,best,best_score
        key=json.dumps(plan,separators=(',',':'))
        if key in evaluated:return evaluated[key]
        results=native.plan_games(plan,seeds,rivals);assert (results[:,2]==719).all()
        games+=len(results);margin=results[:,0]-results[:,1]
        # The selection order is frozen before any candidate is evaluated.
        score=(float((margin>0).mean()),float(((margin>0)+.5*(margin==0)).mean()),float(margin.mean()))
        evaluated[key]=score
        if best_score is None or score>best_score:best,best_score=plan,score
        row={'candidate':len(evaluated),'plan':plan,'win_rate':score[0],'match_score':score[1],'margin':score[2],
             'games':games,'cpu_seconds':time.process_time()-cpu_start,'wall_seconds':time.perf_counter()-start,'best':list(best_score)}
        history.append(row)
        with (args.out/'proposals.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        return score
    base=repair(simple_plan());evaluate(base)
    beam=[(best_score,base)];stage=0;counts=np.zeros(4);gains=np.zeros(4)
    while time.process_time()-cpu_start<args.seconds:
        if args.method=='beam':
            # Construct new six-day stages from free resource allocations.
            parent=beam[int(rng.integers(len(beam)))][1]
            proposal=[row[:] for row in parent];proposal[stage]=random_plan(rng)[stage]
            proposal=repair(proposal);score=evaluate(proposal)
            beam=sorted(beam+[(score,proposal)],key=lambda x:x[0],reverse=True)[:4]
            stage=(stage+1)%5
            if len(evaluated)%20==0:
                fresh=random_plan(rng);beam[-1]=(evaluate(fresh),fresh)
        else:
            # Uniform and UCB selection are alternated and labeled for diagnostics.
            bandit=len(evaluated)%2==0
            operator=int(np.argmax(gains/np.maximum(1,counts)+np.sqrt(2*np.log(counts.sum()+2)/np.maximum(1,counts)))) if bandit else int(rng.integers(4))
            before=best_score[0]
            proposal=solver_repair(mutate(best,rng,large=True,operator=operator))
            score=evaluate(proposal);counts[operator]+=1;gains[operator]+=max(0,score[0]-before)
        if len(evaluated)%25==0:
            dump(args.out/'STATUS.json',{'complete':False,'games':games,'proposals':len(evaluated),'cpu_seconds':time.process_time()-cpu_start,'best':best_score})
    verify_freeze(args.seed_manifest.parent/'FREEZE.json')
    meta={'complete':True,'method':args.method,'seed':args.seed,'training_seeds':seeds,'games':games,'proposals':len(evaluated),
          'cpu_seconds':time.process_time()-cpu_start,'wall_seconds':time.perf_counter()-start,'budget_seconds':args.seconds,
          'selection_score':best_score,'plan':best,'operator_counts':counts.tolist(),'operator_gains':gains.tolist()}
    dump(args.out/'candidate.json',meta);dump(args.out/'STATUS.json',meta);print(json.dumps(meta),flush=True)


if __name__=='__main__':main()
