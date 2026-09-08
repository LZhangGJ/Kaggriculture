"""Offline C++ cash/production attribution. Never an input to the candidate."""
from pathlib import Path
import argparse
import gzip
import hashlib
import json
import statistics
import sys
import time
import zlib
from run_production_audit import total as production_total, means as mean_production
from run_cash_audit import totals as cash_total, mean_totals as mean_cash

EXP=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(EXP/'native/build'))
import _dp7_native as native

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--panel',required=True);p.add_argument('--labels',required=True)
    p.add_argument('--opponents',default='pass,g001,g003,boatlee_v29,kaito_v58,lynn_v5,yhay81_six_day,yhay81_three_day,ecobot_v7')
    p.add_argument('--out',required=True);a=p.parse_args()
    panel_path=Path(a.panel);panel=json.loads(panel_path.read_text());assert panel['status']=='COMPLETE_PANEL_NOT_GOAL_ACCEPTANCE'
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    build=json.loads((EXP/'native/build/build_receipt.json').read_text())
    for rel,h in build['source_hashes'].items(): assert sha(EXP/rel)==h, rel
    assert sha(next((EXP/'native/build').glob('_dp7_native*.so')))==build['binary_sha256']
    # Only this offline bridge changed; no simulation/policy/opponent source did.
    for rel,h in panel['build']['source_hashes'].items():
        if rel not in ('native/module.cpp','native/production_audit.hpp'): assert build['source_hashes'][rel]==h, rel
    mapping={'native_pass':0,'searched_route_native':1,'boatlee_v29_native':2,'kaito_v58_native':3,
             'lynn_v5_native':4,'fieldbook_native':5,'three_day_native':6,'ecobot_v7_native':7}
    constructors={1:native.G001,2:native.BoatleeV29,3:native.KaitoV58,4:native.LynnV5}
    opponents={}
    for key in a.opponents.split(','):
        entry=panel['identities'][key];kind=mapping[entry['runtime']];agent=None
        if key!='pass': assert sha(EXP/entry['source'])==entry['source_sha256']
        if kind in constructors:
            asset=EXP/entry['asset'];assert sha(asset)==entry['asset_sha256']
            payload=json.loads(zlib.decompress(asset.read_bytes()))
            if kind==4:
                for rel,h in payload['source_hashes'].items(): assert sha((EXP/entry['source']).parent/rel)==h
            agent=constructors[kind](payload)
        opponents[key]=(kind,agent)
    summary=[];started=time.perf_counter();compared=0
    receipt=dict(status='RUNNING_OFFLINE_AUDIT',args=vars(a),build=build,panel_sha256=sha(panel_path),
                 candidate_source_unchanged=True,all_ledgers_reconciled=False,final_goal_acceptance=False)
    (out/'plan.json').write_text(json.dumps(receipt,indent=2))
    for label in a.labels.split(','):
        config=panel['configurations'][label]
        for opponent,(kind,agent) in opponents.items():
            refs={(r['seed'],r['seat']):r for r in panel['rows'] if r['variant']==label and r['opponent']==opponent}
            keys=sorted(refs);start=time.perf_counter()
            rows=native.pool_audit_batch([s for s,_ in keys],[s for _,s in keys],config,kind,agent,16)
            assert len(rows)==len(keys)
            own_p=[];rival_p=[];own_c=[];rival_c=[]
            for r,k in zip(rows,keys):
                assert (r['seed'],r['seat'])==k;seat=r['seat'];ref=refs[k]
                assert r['money'][seat]==ref['cash'] and r['money'][1-seat]==ref['opponent_cash'],('audit changes result',label,opponent,k)
                assert r['overflow']==ref['overflow'];compared+=1
                own_p.append(production_total(r['production'],seat));rival_p.append(production_total(r['production'],1-seat))
                own_c.append(cash_total(r['cash_ledger'],seat));rival_c.append(cash_total(r['cash_ledger'],1-seat))
                assert own_c[-1]['cash']==ref['cash'] and rival_c[-1]['cash']==ref['opponent_cash']
            entry=dict(label=label,opponent=opponent,config=config,games=len(rows),
                own_production=mean_production(own_p),rival_production=mean_production(rival_p),
                own_cash=mean_cash(own_c),rival_cash=mean_cash(rival_c),wins=sum(refs[k]['win'] for k in keys),
                planner_mean_degraded_tasks=statistics.fmean(sum(d['degraded'] for d in r['planning']) for r in rows),
                planner_mean_unassigned_tasks=statistics.fmean(sum(d['compile_drop'] for d in r['planning']) for r in rows),
                planner_mean_resource_exchange_checks=statistics.fmean(r['planning'][-1].get('resource_exchange_checks',0) for r in rows),
                planner_mean_resource_exchange_applied=statistics.fmean(r['planning'][-1].get('resource_exchange_applied',0) for r in rows),
                planner_mean_resource_exchange_local_steps_saved=statistics.fmean(r['planning'][-1].get('resource_exchange_local_steps_saved',0) for r in rows),
                planner_mean_intraday_proposals=statistics.fmean(r['planning'][-1].get('intraday_proposals',0) for r in rows),
                planner_mean_intraday_activated=statistics.fmean(r['planning'][-1].get('intraday_activated',0) for r in rows),
                planner_mean_intraday_unfilled=statistics.fmean(r['planning'][-1].get('intraday_unfilled',0) for r in rows),
                planner_mean_intraday_cancelled=statistics.fmean(r['planning'][-1].get('intraday_cancelled',0) for r in rows),
                planner_mean_intraday_purchase_orders=statistics.fmean(r['planning'][-1].get('intraday_purchase_orders',0) for r in rows),
                planner_mean_intraday_by_kind=[statistics.fmean(r['planning'][-1].get('intraday_activated_by_kind',[0]*12)[k] for r in rows) for k in range(12)],
                seconds=time.perf_counter()-start)
            summary.append(entry)
            dest=out/f'{label}_{opponent}.json.gz'
            with gzip.open(dest,'wt',encoding='utf-8',compresslevel=1) as f:json.dump(dict(rows=rows,summary=entry),f,separators=(',',':'))
            print(json.dumps(dict(label=label,opponent=opponent,games=len(rows),cash=entry['own_cash']['cash'],wins=entry['wins'],seconds=entry['seconds'])),flush=True)
            (out/'summary.json').write_text(json.dumps(dict(**receipt,summary=summary),indent=2))
    receipt.update(status='PASS_OFFLINE_AUDIT_NOT_GOAL_ACCEPTANCE',all_ledgers_reconciled=True,
        unchanged_result_games=compared,seconds=time.perf_counter()-started,summary=summary,
        caveat='Opponent private accounting is offline evidence only; it is never passed to the candidate. No claim that a single accounting delta is an isolated causal gain.')
    (out/'summary.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(dict(status=receipt['status'],games=compared,seconds=receipt['seconds'])))

if __name__=='__main__': main()
