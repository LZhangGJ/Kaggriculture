"""Inspect fixed live regression decisions and compare recorded baseline trajectories."""
from pathlib import Path
import json
from run import ROOT,host,read

out=ROOT/'runs/regression_final';results=[]
for op,seed,seat,steps in [('submission_56140347',407296613,0,[240,241,242,243]),
                          ('submission_56140347',572753638,1,[529,530,531]),
                          ('submission_56146577',1508212750,1,[217,218,240,241,242,243]),
                          ('submission_56146577',347212678,1,[456])]:
    name=f'fixed_{op}_{seed}_seat{seat}'
    row=read(out/'matches'/(name+'.result.json'));replay=host.load_replay(out/'matches'/(name+'.json.gz'))
    audit=host.load_replay(out/'matches'/(name+'.audit.json.gz'));events=[]
    for step in steps:
        obs=replay['steps'][step][seat]['observation'];post=replay['steps'][step+1][seat]['observation']
        events.append(dict(step=step,day=step//24+1,cash_before=obs['farms'][seat]['money'],cash_after=post['farms'][seat]['money'],
                           action=replay['steps'][step+1][seat]['action'],seconds=audit['action_seconds'][step],
                           debug={k:v for k,v in audit['route'][step]['info'].items() if k.startswith('recovery_') or k in ['capacity_repairs','finance_repairs','budget_stops']}))
    prior=ROOT.parent/'r2p16_jointafs_diagnosis_20260911/runs'/('latest100' if op.endswith('56146577') else 'previous75')/'matches'
    if not (prior/f'recovery_{op}_{seed}_seat{seat}.json.gz').exists():prior=ROOT.parent/'r2p16_latest_submissions_20260910/runs/latest25/matches'
    comparisons=[]
    for arm in ['workflow','recovery']:
        oldname=f'{arm}_{op}_{seed}_seat{seat}';path=prior/(oldname+'.json.gz')
        if not path.exists():continue
        old=host.load_replay(path);oldrow=read(prior/(oldname+'.result.json'))
        first=next((t for t in range(719) if replay['steps'][t+1][seat]['action']!=old['steps'][t+1][seat]['action']),None)
        comparisons.append(dict(arm=arm,first_different_action_step=first,old_cash=oldrow['cash'],old_margin=oldrow['margin'],
                                note='Full live trajectories; final cash differences cannot be attributed to one repair if earlier decisions differ.'))
    results.append(dict(name=name,events=events,comparisons=comparisons,final_cash=row['cash'],final_margin=row['margin']))
(ROOT/'REGRESSION_EVIDENCE.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps([dict(name=r['name'],comparisons=r['comparisons'],timings=[(e['step'],e['seconds']) for e in r['events']]) for r in results],indent=2))
