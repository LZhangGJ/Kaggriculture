"""Current-build evidence for complete Lynn port, old regression and new panel."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import statistics

EXP=Path(__file__).resolve().parents[1]
def read(rel):return json.loads((EXP/rel).read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def signature(row):
    return json.dumps({k:v for k,v in row.items() if k!='seconds'},sort_keys=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);args=p.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    build=read('native/build/build_receipt.json')
    assert sha(next((EXP/'native/build').glob('_dp7_native*.so')))==build['binary_sha256']
    for rel,h in build['source_hashes'].items():assert sha(EXP/rel)==h
    reg=read('opponents/registry.json');entry=reg['opponents']['lynn_v5']
    checks={k:read(entry[k]) for k in ('initial_parity_receipt','isolation_receipt','branches_receipt')}
    for check in checks.values():assert check['status']=='PASS' and check['build']['binary_sha256']==build['binary_sha256']
    assert checks['isolation_receipt']['parallel_repeat_games']==1000
    assert checks['branches_receipt']['deferred_cases']>0 and checks['branches_receipt']['mismatches']==0
    official=checks['initial_parity_receipt']['rows'];assert len(official)==10
    for r in official:assert r['steps']==719 and r['action_mismatches']==r['layer_action_mismatches']==r['memory_mismatches']==r['official_state_mismatches']==0
    assert any(r['weed_repair_frames'] for r in official)
    assert len({tuple(r['opponent_state']['assignments'].values()) for r in official})==4
    assert any(r['opponent_state']['delivery'][2]['intervention_steps']==[313,314,315,316] for r in official)
    for key in ('boatlee_v29','kaito_v58','yhay81_six_day'):
        for field in ('initial_parity_receipt','isolation_receipt'):
            check=read(reg['opponents'][key][field]);assert check['status']=='PASS' and check['build']['binary_sha256']==build['binary_sha256']
    mechanisms=read('receipts/lynn_unchanged_policy_mechanisms_v1/acceptance.json')
    assert mechanisms['status']=='PASS' and mechanisms['native_mechanism_checks']==546
    preserved=0;panels=[]
    for part in ('A','B'):
        before=read(f'receipts/pool_kaito_{part}50_v1/results.json')
        after=read(f'receipts/pool_lynn_{part}50_v1/results.json')
        assert after['status']=='COMPLETE_PANEL_NOT_GOAL_ACCEPTANCE' and after['build']['binary_sha256']==build['binary_sha256']
        old=[r for r in after['rows'] if r['opponent']!='lynn_v5']
        assert [signature(r) for r in old]==[signature(r) for r in before['rows']]
        preserved+=len(old);panels.append(after)
        for rel in ('native/policy.hpp','native/vendor/simulator.cpp','native/vendor/simulator.hpp',
                    'native/vendor/native_teammate.cpp','native/vendor/native_teammate.hpp',
                    'native/boatlee_v29.cpp','native/boatlee_v29.hpp','native/fieldbook_adapter.cpp','native/fieldbook_adapter.hpp',
                    'native/kaito_v58.cpp','native/kaito_v58.hpp'):
            assert before['build']['source_hashes'][rel]==after['build']['source_hashes'][rel],rel
    lookup={(r['seed'],r['seat']):r for r in panels[0]['rows'] if r['opponent']=='lynn_v5' and r['variant']=='L3_base'}
    for ref in official:
        actual=lookup[ref['seed'],ref['seat']]
        for key in ('cash','opponent_cash','weed_repair_frames'):assert actual[key]==ref[key]
    aggregate={}
    for variant in panels[0]['summary']:
        unique={(r['seed'],r['seat']):r for panel in panels for r in panel['rows'] if r['variant']==variant and r['opponent']=='lynn_v5'}
        clusters=[(int(unique[s,0]['win'])+int(unique[s,1]['win']))/2 for s in sorted({s for s,_ in unique})]
        rate=statistics.fmean(clusters);se=statistics.stdev(clusters)/math.sqrt(len(clusters))
        aggregate[variant]=dict(games=len(unique),independent_seeds=len(clusters),wins=sum(r['win'] for r in unique.values()),
                                win_rate=rate,approximate_seed_cluster_95pct_interval=[max(0,rate-1.96*se),min(1,rate+1.96*se)],
                                mean_cash=statistics.fmean(r['cash'] for r in unique.values()),
                                mean_opponent_cash=statistics.fmean(r['opponent_cash'] for r in unique.values()),
                                mean_margin=statistics.fmean(r['margin'] for r in unique.values()))
    acquisition=read('receipts/three_day_source_acquisition_v1.json')
    assert acquisition['status']=='PASS_ACQUISITION_NOT_NATIVE_ACCEPTANCE' and len(reg['required_opponents'])==8
    receipt=dict(status='PASS_ADAPTER_AND_REGRESSION_NOT_GOAL_ACCEPTANCE',build=build,
                 old_panel_rows_unchanged=preserved,current_build_official_games=len(official),
                 official_steps=719*len(official),synthetic_cases=checks['branches_receipt']['cases'],
                 synthetic_calls=checks['branches_receipt']['differential_calls'],parallel_repeat_games=1000,
                 aggregate=aggregate,native_opponents=[k for k,v in reg['opponents'].items() if v['runtime']!='pending_native'],
                 pending_opponents=[k for k,v in reg['opponents'].items() if v['runtime']=='pending_native'],
                 final_goal_acceptance=False,final_holdout_used=False,
                 caveat='Development evidence only. No candidate promotion or claim of exhaustive parity/90% wins.')
    (out/'acceptance.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps({k:v for k,v in receipt.items() if k!='build'}),flush=True)


if __name__=='__main__':main()
