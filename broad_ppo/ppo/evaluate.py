"""Explicit full-game milestone panels and matched checkpoint comparison."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from ppo.league import sha
from ppo.actors import Games
from ppo.sampler import BatchedExactSampler


def validate_panel(path):
    panel=json.loads(Path(path).read_text()) if not isinstance(path,dict) else json.loads(json.dumps(path))
    if not panel.get('opponents'): raise ValueError('Empty evaluation panel')
    if len({r['id'] for r in panel['opponents']})!=len(panel['opponents']): raise ValueError('Duplicate evaluation IDs')
    for row in panel['opponents']:
        if not row.get('seeds') or len(set(row['seeds']))!=len(row['seeds']): raise ValueError('Missing/duplicate evaluation seeds')
        if row['kind'] not in ('script','checkpoint'): raise ValueError('Invalid evaluation opponent')
        if row['path'] not in ('starter','random','pass'):
            row['path']=str(Path(row['path']).resolve())
            if sha(row['path'])!=row['sha256']: raise ValueError('Evaluation opponent hash mismatch')
    return panel


class LocalBackend:
    def __init__(self,*args): self.games=Games(*args)
    def call(self,command,payload): return self.games.run(command,payload)


def run_panel(checkpoint,panel,output):
    # Explicit caller only. Never invoked by import or CPU unit checks.
    from ppo.train import load_model
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    model,base=load_model(checkpoint,'cpu')
    wq,mq=base['worker_quantities'],base['market_quantities']
    rows=[]
    for opponent in panel['opponents']:
        models={'candidate':model}
        policy='script:'+opponent['path']
        if opponent['kind']=='checkpoint':
            models['opponent'],_=load_model(opponent['path'],'cpu',base['cache_identity'])
            policy='opponent'
        for seed in opponent['seeds']:
            for seat in (0,1):
                policies=[policy,policy];policies[seat]='candidate'
                assignment=dict(game=0,seed=seed,policies=policies,learner_seats=[],family=opponent['id'])
                tick=time.perf_counter()
                try:
                    backend=LocalBackend([assignment],wq,mq)
                    sampler=BatchedExactSampler(models,model,wq,mq,'cpu')
                    while True:
                        keys=[f'0:{s}' for s,p in enumerate(policies) if not p.startswith('script:')]
                        starts=backend.call('start',keys)
                        for p in models:
                            group=[k for k in keys if policies[int(k[-1])]==p]
                            if group: sampler.act(p,group,starts,backend,greedy=True,retain=False)
                        result=backend.call('step',[0])[0]
                        if result['done']: break
                    valid=not any(result['faults']) and result['statuses']==['DONE','DONE']
                    cash=result['cash'] if valid else None
                    score=None if not valid else .5 if cash[seat]==cash[1-seat] else float(cash[seat]>cash[1-seat])
                    row=dict(result)
                    row.update(opponent=opponent['id'],seed=seed,seat=seat,valid=valid,cash=cash,score=score,
                               margin=None if not valid else cash[seat]-cash[1-seat])
                except Exception as exc:
                    row=dict(opponent=opponent['id'],seed=seed,seat=seat,valid=False,score=None,error=repr(exc))
                row['seconds']=time.perf_counter()-tick;rows.append(row)
                with (output/'games.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
    by={}
    for opponent in panel['opponents']:
        selected=[r for r in rows if r['opponent']==opponent['id']]
        good=[r for r in selected if r['valid']]
        by[opponent['id']]=dict(games=len(selected),valid=len(good),failures=len(selected)-len(good),
            wins=sum(r['score']==1 for r in good),draws=sum(r['score']==.5 for r in good),losses=sum(r['score']==0 for r in good),
            mean_cash=float(np.mean([r['cash'][r['seat']] for r in good])) if good else None,
            mean_margin=float(np.mean([r['margin'] for r in good])) if good else None)
    summary=dict(checkpoint_sha256=sha(checkpoint),panel=panel,opponents=by,failures=sum(not r['valid'] for r in rows))
    (output/'summary.json').write_text(json.dumps(summary,indent=2))
    return summary


def compare(candidate,incumbent):
    a=json.loads((Path(candidate)/'summary.json').read_text());b=json.loads((Path(incumbent)/'summary.json').read_text())
    if a['panel']!=b['panel']: raise ValueError('Panels are not matched')
    if a['failures'] or b['failures']: return dict(eligible=False,reason='failed_games')
    differences={k:(r['wins']+.5*r['draws'])-(b['opponents'][k]['wins']+.5*b['opponents'][k]['draws']) for k,r in a['opponents'].items()}
    return dict(eligible=sum(differences.values())>0 and min(differences.values())>=-2,
                point_changes=differences,reason='matched_screen_only',
                note='Small panels do not establish statistical superiority. No automatic promotion.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path);p.add_argument('--panel',type=Path)
    p.add_argument('--request',type=Path,help='Consume a saved evaluation request outside the DDP job')
    p.add_argument('--output',type=Path);p.add_argument('--compare',nargs=2)
    args=p.parse_args()
    if args.compare: print(json.dumps(compare(*args.compare),indent=2))
    elif args.request:
        request=json.loads(args.request.read_text())
        if sha(request['checkpoint'])!=request['checkpoint_sha256']: raise ValueError('Requested checkpoint hash mismatch')
        print(json.dumps(run_panel(request['checkpoint'],validate_panel(request['panel']),request['output']),indent=2))
    else:
        if not all((args.checkpoint,args.panel,args.output)): p.error('checkpoint, panel and output are required')
        print(json.dumps(run_panel(args.checkpoint,validate_panel(args.panel),args.output),indent=2))
