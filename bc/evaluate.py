"""Full official games against a built-in or supplied Python agent."""
import argparse,hashlib,importlib.util,json,time
from pathlib import Path
import torch
from exact_decoder import ExactAgent
from kaggle_environments import make
from kaggle_environments.envs.kaggriculture.kaggriculture import agents

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--opponent',default='starter',help='pass, starter, random, or trusted .py file exporting agent(obs, config)')
    p.add_argument('--seeds',type=int,nargs='+',default=list(range(9182201,9182209)))
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(2);policy=ExactAgent.from_checkpoint(a.checkpoint,device='cpu',greedy=True)
    if a.opponent in agents:
        builtin=agents[a.opponent]
        def other(obs,config):return builtin(obs)
        opponent_sha=None
    else:
        source=Path(a.opponent);opponent_sha=hashlib.sha256(source.read_bytes()).hexdigest()
        spec=importlib.util.spec_from_file_location('opponent',source);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        other=module.agent
    a.output.mkdir(parents=True,exist_ok=False);rows=[]
    sha=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()
    for seed in a.seeds:
        for seat in (0,1):
            policy.reset()
            def act(obs,config):return policy.act(obs)[0]
            players=[other,other];players[seat]=act
            env=make('kaggriculture',configuration=dict(seed=seed,episodeSteps=720),debug=False)
            start=time.monotonic();env.run(players);status=[s.status for s in env.state]
            valid=status==['DONE','DONE'] and len(env.steps)==720
            cash=[s.reward for s in env.state] if valid else None
            score=None if not valid else .5 if cash[0]==cash[1] else float(cash[seat]>cash[1-seat])
            row=dict(seed=seed,seat=seat,statuses=status,terminal=valid,cash=cash,score=score,seconds=time.monotonic()-start)
            rows.append(row)
            with (a.output/'games.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            print(json.dumps(row),flush=True)
    valid=[r for r in rows if r['terminal']]
    summary=dict(checkpoint_sha256=sha,opponent=a.opponent,opponent_sha256=opponent_sha,games=len(rows),terminal_games=len(valid),
        wins=sum(r['score']==1 for r in valid),draws=sum(r['score']==.5 for r in valid),losses=sum(r['score']==0 for r in valid),
        mean_cash=sum(r['cash'][r['seat']] for r in valid)/len(valid) if valid else None,
        mean_opponent_cash=sum(r['cash'][1-r['seat']] for r in valid)/len(valid) if valid else None)
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary))
    if len(valid)!=len(rows):raise RuntimeError('Evaluation had failed or unfinished games')

if __name__=='__main__':main()
