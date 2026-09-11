"""Recheck every recorded transition, then calculate paired seed-cluster statistics."""
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import argparse, json, multiprocessing, random, statistics
from run import ROOT, WORKSPACE, host, read, digest

def check(path):
    host.OLD = path.parent
    result = host.calibrate_one(path)
    audit = host.load_replay(path.parent / (result['name'] + '.audit.json.gz'))
    unexpected, deferred = 0, 0
    days = []
    for day in audit['daily']:
        missing = Counter()
        for *key, count in day.get('uncompleted_required_tasks', []): missing[tuple(key)] += count
        waived = Counter(tuple(x) for x in day.get('route', {}).get('deferred_tasks', []))
        left = missing - waived
        unexpected += sum(left.values()); deferred += sum((missing & waived).values())
        if missing:
            days.append(dict(day=day['day'], missing=sum(missing.values()), explicit_deferrals=sum((missing & waived).values()),
                             unexpected=sum(left.values()), tasks=[list(k)+[n] for k,n in left.items()]))
    result.update(unexpected=unexpected, explicit_deferrals=deferred, days=days)
    return result

def describe(rows, qa):
    mean = statistics.fmean
    return dict(games=len(rows), seeds=len({r['seed'] for r in rows}), wins=sum(r['win'] for r in rows),
                draws=sum(r['margin']==0 for r in rows), win_rate=100*mean(r['win'] for r in rows),
                **{'mean_'+k:mean(r[k] for r in rows) for k in ['cash','opponent_cash','margin','wages','hires','overflow_units']},
                invalid_actions=sum(r['invalid_actions'] for r in rows),
                recorded_uncompleted=sum(r['uncompleted_required_tasks'] for r in rows),
                unexpected_uncompleted=sum(qa[r['name']]['unexpected'] for r in rows),
                explicit_deferrals=sum(qa[r['name']]['explicit_deferrals'] for r in rows),
                max_seconds=max(r['max_seconds'] for r in rows), over_one_second=sum(r['over_one_second'] for r in rows),
                **{k:sum(r['route'].get(k,0) for r in rows) for k in ['recovery_events','recovery_kept','recovery_retries','recovery_joint','recovery_partial','budget_stops']})

def paired(rows, old):
    pairs = defaultdict(dict)
    for r in rows:
        if r['arm'] in [old,'fixed']: pairs[r['opponent'],r['seed'],r['seat']][r['arm']] = r
    clusters = defaultdict(list); deltas=[]
    keys = ['win','cash','margin','wages','hires']
    for (op, seed, seat), group in sorted(pairs.items()):
        assert set(group)=={old,'fixed'}
        a,b=group[old],group['fixed']
        delta={k:float(b[k])-float(a[k]) for k in keys};delta['win']*=100
        deltas.append(dict(opponent=op,seed=seed,seat=seat,**delta));clusters[seed].append(delta)
    means=[{k:statistics.fmean(d[k] for d in ds) for k in keys} for ds in clusters.values()]
    assert len({len(ds) for ds in clusters.values()})==1
    rng=random.Random(20260911);samples={k:[] for k in keys}
    for _ in range(10000):
        sample=rng.choices(means,k=len(means))
        for k in keys:samples[k].append(statistics.fmean(d[k] for d in sample))
    intervals={}
    for k in keys:
        sorted_values=sorted(samples[k]);intervals[k]=dict(mean=statistics.fmean(d[k] for d in deltas),ci95=[sorted_values[250],sorted_values[9749]])
    return dict(baseline=old,pairs=len(deltas),seeds=len(clusters),method='10000 paired bootstrap samples, clustered by seed; retain seats and opponents together',
                deltas=intervals,gained_wins=sum(d['win']>0 for d in deltas),lost_wins=sum(d['win']<0 for d in deltas))

def main():
    p=argparse.ArgumentParser();p.add_argument('tag');p.add_argument('--workers',type=int,default=6);a=p.parse_args()
    out=ROOT/'runs'/a.tag;protocol=read(out/'PROTOCOL.json');rows=read(out/'rows.json')
    for name,sha in protocol['hashes'].items():assert digest(WORKSPACE/name)==sha,name
    assert len(rows)==len(protocol['jobs'])
    assert len({r['name'] for r in rows})==len(rows)
    assert {(r['arm'],r['opponent'],r['seed'],r['seat']) for r in rows}=={tuple(j) for j in protocol['jobs']}
    assert all(not r['error'] and r['frames']==720 for r in rows)
    checks=[]
    with ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        for f in as_completed([pool.submit(check,out/'matches'/(r['name']+'.result.json')) for r in rows]):
            checks.append(f.result())
            if len(checks)%100==0 or len(checks)==len(rows):
                print(json.dumps(dict(verified=len(checks),planned=len(rows))),flush=True)
    qa={q['name']:q for q in checks};arms=protocol['arms']
    result=dict(stage=protocol['stage'],games=len(rows),observations=sum(q['observations'] for q in checks),
                arms={arm:describe([r for r in rows if r['arm']==arm],qa) for arm in arms},
                opponents={op:{arm:describe([r for r in rows if r['arm']==arm and r['opponent']==op],qa) for arm in arms} for op in sorted({r['opponent'] for r in rows})},
                comparisons=[paired(rows,arm) for arm in arms if arm!='fixed'] if 'fixed' in arms else [],
                caveats=['Local terminal-cash wins; host logs but does not enforce timeout forfeits.',
                         'Task obligations are recorded when each policy intervenes. Omission counts describe their own commitments, not a common counterfactual task list.',
                         'Development seeds are excluded from fresh-seed conclusions. CPU budget can change actions near the cutoff.'])
    (out/'QA.json').write_text(json.dumps(dict(status='PASS',checks=checks),indent=2)+'\n')
    (out/'SUMMARY.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['opponents','caveats']},indent=2),flush=True)

if __name__=='__main__':main()
