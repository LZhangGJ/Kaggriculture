"""Check EcoBot's source port and unchanged seven-opponent development panel."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import statistics

EXP = Path(__file__).resolve().parents[1]
def read(rel): return json.loads((EXP / rel).read_text())
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def signature(row): return json.dumps({k:v for k,v in row.items() if k != 'seconds'}, sort_keys=True)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--out', required=True); a=p.parse_args()
    out=Path(a.out); out.mkdir(parents=True, exist_ok=False)
    build=read('native/build/build_receipt.json')
    assert sha(next((EXP/'native/build').glob('_dp7_native*.so'))) == build['binary_sha256']
    for rel,h in build['source_hashes'].items(): assert sha(EXP/rel)==h, rel
    reg=read('opponents/registry.json'); entry=reg['opponents']['ecobot_v7']
    assert sha(EXP/entry['source'])==entry['source_sha256']
    checks={k:read(entry[k]) for k in ('initial_parity_receipt','isolation_receipt')}
    for c in checks.values(): assert c['status']=='PASS' and c['build']['binary_sha256']==build['binary_sha256']
    official=checks['initial_parity_receipt']['rows']; assert len(official)==8
    for r in official:
        assert r['steps']==719 and r['action_mismatches']==r['memory_mismatches']==r['official_state_mismatches']==0
    assert checks['isolation_receipt']['parallel_repeat_games']==1000
    for key in ('boatlee_v29','kaito_v58','lynn_v5','yhay81_six_day','yhay81_three_day'):
        for field in ('initial_parity_receipt','isolation_receipt'):
            c=read(reg['opponents'][key][field])
            assert c['status']=='PASS' and c['build']['binary_sha256']==build['binary_sha256']
    dispatch=read(entry['dispatch_receipt']); economy=read(entry['economy_receipt'])
    assert dispatch['mismatches']==economy['mismatches']==0
    assert dispatch['real_state_cases']+dispatch['synthetic_cases']==368
    assert economy['differential_calls']==730 and economy['price_comparisons']==144027
    probe=checks['initial_parity_receipt']['probe_build']
    for check in (dispatch,economy): assert check['build']['binary_sha256']==probe['binary_sha256']
    # The observation probe and arena must compile the very same policy sources.
    for rel in ('native/ecobot_v7_core.hpp','native/ecobot_v7_core.cpp','native/ecobot_v7.hpp','native/ecobot_v7.cpp'):
        assert build['source_hashes'][rel]==probe['source_hashes'][rel], rel
    mechanisms=read('receipts/ecobot_unchanged_policy_mechanisms_v1/acceptance.json')
    assert mechanisms['status']=='PASS' and mechanisms['native_mechanism_checks']==546
    assert mechanisms['source_sha256']==sha(EXP/'native/test_policy.cpp')
    assert mechanisms['build']['source_hashes']['native/policy.hpp']==build['source_hashes']['native/policy.hpp']
    panels=[]; preserved=0
    for part in ('A','B'):
        before=read(f'receipts/pool_three_day_{part}50_v1/results.json')
        after=read(f'receipts/pool_ecobot_{part}50_v1/results.json')
        assert after['status']=='COMPLETE_PANEL_NOT_GOAL_ACCEPTANCE'
        assert after['build']['binary_sha256']==build['binary_sha256'] and not after['pending_required_opponents']
        old=[r for r in after['rows'] if r['opponent']!='ecobot_v7']
        assert [signature(r) for r in old]==[signature(r) for r in before['rows']]
        for rel,h in before['build']['source_hashes'].items():
            if rel!='native/module.cpp': assert build['source_hashes'][rel]==h, rel
        preserved+=len(old); panels.append(after)
    assert preserved==6400
    lookup={(r['seed'],r['seat']):r for r in panels[0]['rows'] if r['opponent']=='ecobot_v7' and r['variant']=='L3_base'}
    for ref in official:
        for key in ('cash','opponent_cash'): assert lookup[ref['seed'],ref['seat']][key]==ref[key]
    aggregate={}
    for variant in panels[0]['summary']:
        aggregate[variant]={}
        for opponent in panels[0]['summary'][variant]:
            unique={(r['seed'],r['seat']):r for panel in panels for r in panel['rows'] if r['variant']==variant and r['opponent']==opponent}
            clusters=[(int(unique[s,0]['win'])+int(unique[s,1]['win']))/2 for s in sorted({s for s,_ in unique})]
            rate=statistics.fmean(clusters); se=statistics.stdev(clusters)/math.sqrt(len(clusters))
            aggregate[variant][opponent]=dict(games=len(unique),independent_seeds=len(clusters),wins=sum(r['win'] for r in unique.values()),
                win_rate=rate,approximate_seed_cluster_95pct_interval=[max(0,rate-1.96*se),min(1,rate+1.96*se)] if se>0 else None,
                conservative_seed_hoeffding_95pct_bound=[max(0,rate-math.sqrt(math.log(40)/(2*len(clusters)))),min(1,rate+math.sqrt(math.log(40)/(2*len(clusters))))],
                mean_cash=statistics.fmean(r['cash'] for r in unique.values()),mean_opponent_cash=statistics.fmean(r['opponent_cash'] for r in unique.values()),
                mean_margin=statistics.fmean(r['margin'] for r in unique.values()))
    receipt=dict(status='PASS_ADAPTER_AND_REGRESSION_NOT_GOAL_ACCEPTANCE',build=build,old_panel_rows_unchanged=preserved,
        current_build_official_games=len(official),official_steps=719*len(official),parallel_repeat_games=1000,
        dispatch_cases=368,economy_calls=730,price_comparisons=144027,aggregate=aggregate,
        pending_opponents=[],final_goal_acceptance=False,final_holdout_used=False,
        caveat='Finite development differential evidence. Dispatch-only receipt has stale remaining-scope text; full-agent evidence is the separate official receipt. No own candidate promotion.')
    (out/'acceptance.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps({k:v for k,v in receipt.items() if k not in ('build','aggregate')}))
    print(json.dumps(aggregate))

if __name__=='__main__': main()
