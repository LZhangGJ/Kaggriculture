"""Verify the original Three-Day adapter and unchanged old opponent results."""
from pathlib import Path
import argparse
import collections
import gzip
import hashlib
import json
import math
import re
import statistics

EXP=Path(__file__).resolve().parents[1]
def read(rel):return json.loads((EXP/rel).read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def signature(row):return json.dumps({k:v for k,v in row.items() if k!='seconds'},sort_keys=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    build=read('native/build/build_receipt.json')
    assert sha(next((EXP/'native/build').glob('_dp7_native*.so')))==build['binary_sha256']
    for rel,h in build['source_hashes'].items():assert sha(EXP/rel)==h,rel
    reg=read('opponents/registry.json');entry=reg['opponents']['yhay81_three_day']
    checks={k:read(entry[k]) for k in ('initial_parity_receipt','switched_parity_receipt','isolation_receipt')}
    for check in checks.values():assert check['status']=='PASS' and check['build']['binary_sha256']==build['binary_sha256']
    official=checks['initial_parity_receipt']['rows']+checks['switched_parity_receipt']['rows']
    assert len(official)==10 and {r['selected_route'] for r in official}=={0,1}
    for r in official:assert r['steps']==719 and r['action_mismatches']==r['official_state_mismatches']==0
    assert checks['isolation_receipt']['parallel_repeat_games']==1000
    for key in ('boatlee_v29','kaito_v58','lynn_v5','yhay81_six_day'):
        for field in ('initial_parity_receipt','isolation_receipt'):
            check=read(reg['opponents'][key][field])
            assert check['status']=='PASS' and check['build']['binary_sha256']==build['binary_sha256']
    mechanisms=read('receipts/three_day_unchanged_policy_mechanisms_v1/acceptance.json')
    assert mechanisms['status']=='PASS' and mechanisms['native_mechanism_checks']==546
    # Read-only decode of the frozen original route data, not a substitute agent.
    encoded=re.findall(r'"([0-9 ]+)"',(EXP/'opponents/yhay81_three_day/output/source/tape.inc').read_text())
    assert len(encoded)==1438
    items=['WHEAT','CARROT','TOMATO','STRAWBERRY','MELON','EGG','MILK','WOOL','FERTILIZER','GOOSE','COW','SHEEP']
    verbs=['NONE','HIRE','BUY_LAND','BUY_SEED','BUY_PRODUCT','BUY_ANIMAL','SELL']
    def market_line(route,step):
        n=list(map(int,encoded[route*719+step].split()));pos=2+3*n[0];count=n[1]
        orders=[]
        for i in range(count):
            op,item,q=n[pos+3*i:pos+3*i+3]
            if op:orders.append([verbs[op]] if op in (1,2) else [verbs[op],items[item],q])
        assert pos+3*count==len(n)
        return orders
    features=[];guard_frames=[]
    for field in ('initial_parity_receipt','switched_parity_receipt'):
        for replay in sorted((EXP/entry[field]).parent.glob('*.json.gz')):
            data=json.loads(gzip.decompress(replay.read_bytes()));seat=int(replay.stem.split('seat')[1].split('.')[0])
            for frame in data['trace']:
                step=frame['step'];obs=frame['observation'];r=frame['selected_route']
                if step==360:
                    shops=obs['town']['unlocked_shops'];first=shops[0] if shops else None
                    features.append(dict(replay=replay.name,first_shop=first,route=r))
                if step%72==0:
                    actual=frame['actions'][1-seat]['market'];raw=market_line(r,step)
                    if actual!=raw:guard_frames.append(dict(replay=replay.name,step=step,raw=raw,actual=actual))
    # Coverage claims are measured, never inferred just from branch existence.
    preserved=0;panels=[]
    for part in ('A','B'):
        before=read(f'receipts/pool_lynn_{part}50_v1/results.json')
        after=read(f'receipts/pool_three_day_{part}50_v1/results.json')
        assert after['status']=='COMPLETE_PANEL_NOT_GOAL_ACCEPTANCE' and after['build']['binary_sha256']==build['binary_sha256']
        old=[r for r in after['rows'] if r['opponent']!='yhay81_three_day']
        assert [signature(r) for r in old]==[signature(r) for r in before['rows']]
        preserved+=len(old);panels.append(after)
        for rel,h in before['build']['source_hashes'].items():
            if rel!='native/module.cpp':assert build['source_hashes'][rel]==h,rel
    assert preserved==5600
    lookup={(r['seed'],r['seat']):r for r in panels[0]['rows'] if r['opponent']=='yhay81_three_day' and r['variant']=='L3_base'}
    for ref in official:
        row=lookup[ref['seed'],ref['seat']]
        for key in ('cash','opponent_cash','selected_route'):assert row[key]==ref[key]
    aggregate={}
    for variant in panels[0]['summary']:
        unique={(r['seed'],r['seat']):r for panel in panels for r in panel['rows'] if r['variant']==variant and r['opponent']=='yhay81_three_day'}
        clusters=[(int(unique[s,0]['win'])+int(unique[s,1]['win']))/2 for s in sorted({s for s,_ in unique})]
        rate=statistics.fmean(clusters);se=statistics.stdev(clusters)/math.sqrt(len(clusters))
        aggregate[variant]=dict(games=len(unique),independent_seeds=len(clusters),wins=sum(r['win'] for r in unique.values()),
            win_rate=rate,approximate_seed_cluster_95pct_interval=[max(0,rate-1.96*se),min(1,rate+1.96*se)],
            mean_cash=statistics.fmean(r['cash'] for r in unique.values()),mean_opponent_cash=statistics.fmean(r['opponent_cash'] for r in unique.values()),
            mean_margin=statistics.fmean(r['margin'] for r in unique.values()),route_counts=dict(collections.Counter(r['selected_route'] for r in unique.values())))
    receipt=dict(status='PASS_ADAPTER_AND_REGRESSION_NOT_GOAL_ACCEPTANCE',build=build,old_panel_rows_unchanged=preserved,
        current_build_official_games=len(official),official_steps=719*len(official),parallel_repeat_games=1000,
        observed_route_conditions=features,budget_guard_changes=guard_frames,aggregate=aggregate,
        pending_opponents=[k for k,v in reg['opponents'].items() if v['runtime']=='pending_native'],
        final_goal_acceptance=False,final_holdout_used=False,caveat='Finite development evidence; no candidate promotion or exhaustive-state claim.')
    (out/'acceptance.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps({k:v for k,v in receipt.items() if k not in ('build','budget_guard_changes')}))
    print(json.dumps(dict(budget_guard_changed_frames=len(guard_frames))))

if __name__=='__main__':main()
