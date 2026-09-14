"""Discover public outputs daily and admit stronger notebooks on fresh paired games."""
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import subprocess
import sys

import numpy as np
from . import intake, schedule
from .store import digest, event, now, read, records, write


def scan(root):
    from kaggle.api.kaggle_api_extended import KaggleApi
    root=Path(root);cfg=read(root/'config.json')
    date=datetime.now(ZoneInfo(cfg['timezone'])).date().isoformat()
    path=root/'private/discovery.json';state=read(path,{'catalog':{}})
    if state.get('date')==date:
        return
    api=KaggleApi();api.authenticate()
    catalog=state['catalog'];seen=[];truncated=[]
    for order in ('dateCreated','hotness'):
        for page in range(1,11):
            rows=api.kernels_list(competition='kaggriculture',page=page,page_size=100,sort_by=order) or []
            for row in rows:
                ref=row.ref
                if ref not in catalog:catalog[ref]={'first_seen':now(),'status':'discovered'}
                if ref not in seen:seen.append(ref)
            if len(rows)<100:break
        else:truncated.append(order)
    tracked={s['notebook'] for s in cfg['public_sources']}
    pending=[ref for ref in seen if ref not in tracked]
    pending.sort(key=lambda ref:(catalog[ref].get('attempted',''),seen.index(ref)))
    for ref in pending[:cfg.get('discovery_downloads_per_day',4)]:
        sid='found-'+digest(ref)[:16];dest=(root/'exports'/f'{sid}.json').absolute()
        command=[sys.executable,'-m','tools.arena.kaggle_export',ref,str(dest)]
        catalog[ref]['attempted']=now()
        try:
            subprocess.run(command,check=True,timeout=120,capture_output=True)
            item=read(dest);accepted=intake.submit(root,item['archive_path'],item['manifest'])
            cfg['public_sources'].append(dict(id=sid,notebook=ref,export=str(dest),command=command))
            catalog[ref].update(status='downloaded',agent=accepted['agent'])
        except (OSError,ValueError,KeyError,subprocess.SubprocessError) as e:
            catalog[ref].update(status='download_failed',error=type(e).__name__)
    state.update(date=date,checked=now(),found=len(seen),truncated=truncated)
    write(path,state);write(root/'config.json',cfg)


def evaluate(root,m):
    c=m['public_challenge'];candidate,incumbent=c['candidate'],c['incumbent']
    seeds=sorted({g['seed'] for g in m['games']});lookup={};candidate_failures=0
    for g in m['games']:
        r=read(Path(root)/'runs'/m['id']/'games'/(g['id']+'.json'))
        schedule.validate_result(g,r)
        aid=candidate if candidate in g['agents'] else incumbent
        seat=g['agents'].index(aid)
        lookup[aid,g['agents'][1-seat],g['seed'],seat]=float(r['outcome']==f'win{seat}')
        if aid==candidate and not r.get('terminal'):candidate_failures+=1
    delta=np.array([[lookup[candidate,o,s,seat]-lookup[incumbent,o,s,seat]
                     for o in c['panel'] for seat in (0,1)] for s in seeds])
    blocks=delta.mean(axis=1);rng=np.random.default_rng(714)
    samples=blocks[rng.integers(len(blocks),size=(10000,len(blocks)))].mean(axis=1)
    lower=float(np.quantile(samples,.05));gain=float(blocks.mean())
    passed=gain>=.02 and lower>0 and candidate_failures==0
    return dict(candidate=candidate,incumbent=incumbent,run=m['id'],gain=gain,lower=lower,
                passed=bool(passed),candidate_failures=candidate_failures,at=now(),
                method='128 fresh paired seeds, both seats, equal opponent weights; approximate one-sided 95% bootstrap bound. Not a tournament-wide certificate.')


def advance(root):
    root=Path(root);cfg=read(root/'config.json')
    if not cfg.get('public_discovery_enabled'):return
    trials=[]
    for p in (root/'runs').glob('public-*/manifest.json'):
        m=read(p)
        if m.get('public_challenge'):trials.append(m)
    for m in trials:
        verdict=root/'runs'/m['id']/'public-decision.json'
        if verdict.exists():continue
        if not schedule.complete(root,m):return  # One challenger at a time.
        result=evaluate(root,m);roster=read(root/'roster.json')
        position=next((i for i,e in enumerate(roster) if e['agent']==result['incumbent']),None)
        unchanged=digest(roster)==m['public_challenge']['roster_hash']
        result['replaced']=bool(result['passed'] and unchanged and position is not None)
        if result['replaced']:
            roster[position]=dict(agent=result['candidate'],category='public',reason='Won paired public-roster comparison '+m['id'])
            intake.set_roster(root,roster)
        result['roster_changed']=not unchanged
        write(verdict,result)
        event(root,'public_roster_decision',m['id'],result)
    roster=read(root/'roster.json',[]);ids={e['agent'] for e in roster}
    # Choose a replacement from a completed daily panel with these exact versions.
    daily=[]
    for p in (root/'runs').glob('daily-*/manifest.json'):
        m=read(p)
        if set(m['agents'])==ids and schedule.complete(root,m):daily.append(m)
    if not daily:return
    from .reporting import run_report
    latest=max(daily,key=lambda m:m['created']);stats=run_report(root,latest,0)['stats']
    public=[e['agent'] for e in roster if e['category']=='public']
    if not public:return
    incumbent=min(public,key=lambda a:stats[a]['win_rate'])
    attempted={m['public_challenge']['candidate'] for m in trials
               if not read(root/'runs'/m['id']/'public-decision.json',{}).get('roster_changed')}
    candidates=[a for a in records(root,'agents') if a['id'] not in ids|attempted and a.get('build_verified')
                and a['manifest'].get('origin',{}).get('kind')=='public']
    # Earlier retired versions must not re-enter the challenge queue.
    candidates=[a for a in candidates if a['status']!='archived']
    if not candidates:return
    candidate=min(candidates,key=lambda a:a['created'])['id'];panel=sorted(ids-{incumbent})
    rid='public-'+candidate[:16]+'-'+digest(roster)[:8]
    m=schedule.plan(root,rid,'confirmation',128,candidate,panel)
    extra=[]
    for g in m['games']:
        h={k:v for k,v in g.items() if k!='id'}
        h['agents']=[incumbent if a==candidate else a for a in h['agents']]
        extra.append({**h,'id':digest(h)})
    m['games']+=extra;m['agents'][incumbent]=read(root/'agents'/f'{incumbent}.json')
    m['public_challenge']=dict(candidate=candidate,incumbent=incumbent,panel=panel,roster_hash=digest(roster))
    write(root/'runs'/rid/'manifest.json',m)


if __name__=='__main__':scan(sys.argv[1])
