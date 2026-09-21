"""Excluded official games for sampled-policy value calibration and anchor traces."""
import argparse,copy,gzip,json,math,sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

def one(job):
    candidate,opponent,seed,seat,plan,output,trace=job
    import torch
    torch.set_num_threads(1);sys.path.insert(0,plan['arena'])
    from exact_decoder import ExactAgent
    from ppo.league_eval import agent
    from ppo.league_runtime import digest,atomic
    from tools.arena.worker import official_game
    root=Path(output);root.mkdir(parents=True,exist_ok=True)
    key=digest([candidate['sha256'],opponent['sha256'],seed,seat]);receipt=root/(key+'.json')
    if receipt.exists():return json.loads(receipt.read_text())
    learner=ExactAgent.from_checkpoint(candidate['path'],device='cpu',greedy=False);other=agent(opponent)
    torch.manual_seed(int(digest([seed,seat,'value-calibration'])[:15],16));values=[];records=[]
    def act(obs,cfg):
        action,_=learner.act(obs)
        with torch.no_grad():values.append(learner.model.value(learner.state[1]).softmax(-1)[0].tolist())
        if trace:records.append(dict(observation=copy.deepcopy(obs),action=copy.deepcopy(action)))
        return action
    players=[lambda o,c:other.act(o)[0]]*2;players[seat]=act
    result=official_game(players,dict(seed=seed),dict(plan['config']['contract'],game_timeout_seconds=600))
    valid=result.get('terminal') and result.get('reason')=='terminal' and result.get('statuses')==['DONE','DONE'] and len(values)==719 and all(math.isfinite(x) for x in result.get('cash',[]))
    if not valid:raise ValueError('Invalid calibration game')
    cash=result['cash'];outcome=1 if cash[seat]==cash[1-seat] else (2 if cash[seat]>cash[1-seat] else 0)
    result.update(seed=seed,seat=seat,outcome=outcome,probabilities=values,checkpoint_sha=candidate['sha256'],opponent_sha=opponent['sha256'],valid=True,protocol='excluded-sampled-calibration-v1')
    if trace:
        with gzip.open(root/(key+'.jsonl.gz'),'wt') as f:
            for row in records:f.write(json.dumps(row)+'\n')
    atomic(receipt,result);return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);p.add_argument('--trace',action='store_true');a=p.parse_args()
    import numpy as np
    from ppo.league import sha
    from ppo.league_runtime import atomic
    plan=json.loads(Path(a.plan).read_text());candidate=dict(path=a.checkpoint,sha256=sha(a.checkpoint));seeds=list(range(942220000,942220004));out=Path(a.output)
    jobs=[(candidate,o,s,t,plan,str(out/'games'),a.trace) for o in plan['opponents'] for s in seeds for t in (0,1)]
    with ProcessPoolExecutor(max_workers=4) as pool:rows=list(pool.map(one,jobs))
    summary=[]
    for phase,(lo,hi) in enumerate([(0,240),(240,480),(480,719)]):
        probs=np.array([r['probabilities'][lo:hi] for r in rows]);y=np.array([r['outcome'] for r in rows]);labels=np.eye(3)[y][:,None,:]
        summary.append(dict(phase=phase,games=len(rows),turns=probs.shape[0]*probs.shape[1],brier=float(np.square(probs-labels).sum(-1).mean()),ce=float(-np.log(np.clip(probs[np.arange(len(rows)),:,y],1e-12,1)).mean()),predicted_outcome_mean=probs.mean((0,1)).tolist(),observed_outcome_frequency=np.eye(3)[y].mean(0).tolist()))
    atomic(out/'summary.json',dict(protocol='excluded-sampled-calibration-v1',checkpoint=candidate,games=len(rows),valid=len(rows),seeds=seeds,rows=summary,training_excluded=True,agent_held_out=False,uncertainty_unit='full game, not individual turns',trace=a.trace))
if __name__=='__main__':main()
