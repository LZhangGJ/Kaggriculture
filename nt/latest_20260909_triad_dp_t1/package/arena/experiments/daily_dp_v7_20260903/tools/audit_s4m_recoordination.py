"""Current-state coverage diagnostic along unchanged full live policies."""
from pathlib import Path
from collections import Counter
import argparse,gzip,hashlib,json,statistics as st,sys,time,zlib
EXP=Path(__file__).resolve().parents[1];sys.path.insert(0,str(EXP/'native/build'))
import _dp7_native as native
build=json.loads((EXP/'native/build/build_receipt.json').read_text())
panel=json.loads((EXP/'receipts/s4l_eightway_N50_v1/results.json').read_text())
for rel,h in build['source_hashes'].items():assert hashlib.sha256((EXP/rel).read_bytes()).hexdigest()==h
# The explicit new switch is disabled in all reference configs. The only
# modified old runtime files are its implementation/binding; verify trajectories.
for rel,h in panel['build']['source_hashes'].items():
    if rel not in ('native/module.cpp','native/policy.hpp'):assert build['source_hashes'][rel]==h,rel
configs=panel['configurations'];refs={(r['variant'],r['opponent'],r['seed'],r['seat']):r for r in panel['rows']}
parser=argparse.ArgumentParser();parser.add_argument('--attempt',type=int,default=1);args=parser.parse_args()
out=EXP/f'receipts/s4m_recoordination_audit_v{args.attempt}';out.mkdir(exist_ok=False)
rows=[];examples=[];tic=time.perf_counter()
for opponent in ('g001','g003'):
    source=EXP/f'native/{opponent}_frozen.json.zlib'
    assert hashlib.sha256(source.read_bytes()).hexdigest()==panel['identities'][opponent]['asset_sha256']
    rival=native.G001(json.loads(zlib.decompress(source.read_bytes())))
    for label in ('all_intraday_causal_pickup','all_intraday_auto_portfolio_calendar_causal_pickup'):
        for seed in range(20262701,20262709):
            for seat in (0,1):
                assert not configs[label].get('shared_service_insertions',False)
                env=native.Env(seed);ctl=native.Controller(configs[label]);rs=native.G001State()
                counts=Counter();seen=set();sample=[]
                while not env.done:
                    before=ctl.debug();inspection=native.inspect_recoordination(ctl,env,seat)
                    assert before==ctl.debug(),'diagnostic mutated current controller'
                    if inspection['eligible']:
                        counts['eligible']+=1
                        counts['legacy_accepts']+=inspection['legacy_accepts']
                        counts['incomplete_full_assignment']+=inspection['assigned']<inspection['movable']
                        counts['insertion_improvement_steps']+=inspection['insertion_improves']
                        if inspection['insertion_improves']:
                            counts['max_peak_saving']=max(counts['max_peak_saving'],inspection['peak_saved'])
                            counts['max_total_saving']=max(counts['max_total_saving'],inspection['total_saved'])
                            obs=env.observation(seat)
                            key=('insert',obs['day'])
                            if key not in seen and len(sample)<24:
                                seen.add(key);sample.append(dict(kind='insertion',step=obs['step'],day=obs['day'],hour=obs['hour'],
                                    inspection=inspection,observation=obs,debug=before))
                        if inspection['extra']:
                            counts['extra_opportunity_steps']+=1
                            obs=env.observation(seat)
                            for e in inspection['extra']:
                                key=(obs['day'],e['pos'],e['op'])
                                if key in seen:continue
                                seen.add(key);counts['unique_extra_tasks']+=1;counts[f"unique_op_{e['op']}"]+=1
                                if len(sample)<24:sample.append(dict(step=obs['step'],day=obs['day'],hour=obs['hour'],
                                    inspection=inspection,observation=obs,debug=before))
                    own=ctl.act(env,seat);other=rival.act(env,1-seat,rs)
                    env.step([own,other] if seat==0 else [other,own])
                farms=env.observation(seat)['farms'];ref=refs[label,opponent,seed,seat]
                assert env.step_count==719 and farms[seat]['money']==ref['cash'] and farms[1-seat]['money']==ref['opponent_cash']
                row=dict(label=label,opponent=opponent,seed=seed,seat=seat,cash=ref['cash'],opponent_cash=ref['opponent_cash'],counts=dict(counts));rows.append(row)
                examples.append(dict(**row,samples=sample))
        group=[r for r in rows if r['label']==label and r['opponent']==opponent]
        print(json.dumps(dict(label=label,opponent=opponent,games=len(group),affected=sum(r['counts'].get('unique_extra_tasks',0)>0 for r in group),
            unique_tasks=sum(r['counts'].get('unique_extra_tasks',0) for r in group),
            insertion_affected=sum(r['counts'].get('insertion_improvement_steps',0)>0 for r in group))),flush=True)
        (out/'progress.json').write_text(json.dumps(rows,indent=2))
groups=[]
for label in sorted({r['label'] for r in rows}):
    for opponent in ('g001','g003'):
        rs=[r for r in rows if r['label']==label and r['opponent']==opponent];keys=set().union(*(r['counts'] for r in rs))
        groups.append(dict(label=label,opponent=opponent,games=len(rs),affected_games=sum(r['counts'].get('unique_extra_tasks',0)>0 for r in rs),
            mean_counts={k:st.fmean(r['counts'].get(k,0) for r in rs) for k in sorted(keys)}))
path=out/'examples.json.gz'
with gzip.open(path,'wt',encoding='utf8') as f:json.dump(examples,f)
(out/'acceptance.json').write_text(json.dumps(dict(status='COMPLETE_UNCHANGED_LIVE_RECOORDINATION_DIAGNOSTIC',build=build,groups=groups,rows=rows,
    unchanged_live_games=len(rows),examples_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),seconds=time.perf_counter()-tic,
    caveat='64 repeated development games. Potential extra tasks are feasibility evidence, not guaranteed cash or independent evaluation games. No continuation simulator or policy selection.'),indent=2))
print(json.dumps(dict(status='COMPLETE',unchanged_live_games=len(rows),seconds=time.perf_counter()-tic)))
