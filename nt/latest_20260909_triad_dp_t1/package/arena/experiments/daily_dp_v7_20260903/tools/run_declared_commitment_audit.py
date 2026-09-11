"""Current-state, read-only declared-project valuation audit. No game suffixes."""
from pathlib import Path
import argparse
import collections
import gzip
import hashlib
import json
import shutil
import statistics
import sys
import time
import zlib

EXP=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(EXP/'native/build'))
import _dp7_native as native


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def changed(s):
    return tuple(s[k] for k in ('base_valid','base_kind','base_pos','base_unit')) != tuple(s[k] for k in ('alternative_valid','alternative_kind','alternative_pos','alternative_unit'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--panel',required=True)
    p.add_argument('--labels',default='intraday_stock,intraday_funded,all_intraday')
    p.add_argument('--out',required=True);a=p.parse_args()
    panel_path=Path(a.panel);panel=json.loads(panel_path.read_text())
    assert panel['status']=='COMPLETE_PANEL_NOT_GOAL_ACCEPTANCE'
    build=json.loads((EXP/'native/build/build_receipt.json').read_text())
    for rel,h in build['source_hashes'].items(): assert sha(EXP/rel)==h,rel
    assert sha(next((EXP/'native/build').glob('_dp7_native*.so')))==build['binary_sha256']
    permitted={'native/policy.hpp','native/intraday_admission.hpp','native/module.cpp'}
    for rel,h in panel['build']['source_hashes'].items():
        if rel not in permitted: assert build['source_hashes'][rel]==h,rel
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    for rel in build['source_hashes']:
        dst=out/'source'/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(EXP/rel,dst)
    kinds={'native_pass':0,'searched_route_native':1,'boatlee_v29_native':2,'kaito_v58_native':3,'lynn_v5_native':4,'fieldbook_native':5,'three_day_native':6,'ecobot_v7_native':7}
    constructors={1:native.G001,2:native.BoatleeV29,3:native.KaitoV58,4:native.LynnV5}
    opponents={}
    for key,entry in panel['identities'].items():
        kind=kinds[entry['runtime']];agent=None
        if key!='pass': assert sha(EXP/entry['source'])==entry['source_sha256']
        if kind in constructors:
            asset=EXP/entry['asset'];assert sha(asset)==entry['asset_sha256']
            payload=json.loads(zlib.decompress(asset.read_bytes()))
            if kind==4:
                for rel,h in payload['source_hashes'].items():assert sha((EXP/entry['source']).parent/rel)==h
            agent=constructors[kind](payload)
        opponents[key]=(kind,agent)
    result=dict(status='RUNNING_READ_ONLY_DECLARATION_AUDIT',build=build,args=vars(a),
                panel_sha256=sha(panel_path),suffix_rollouts=0,true_future_used=False,threads=16,summary=[])
    (out/'plan.json').write_text(json.dumps(result,indent=2));start=time.perf_counter();games=checks=0
    for label in a.labels.split(','):
        for opp,(kind,agent) in opponents.items():
            refs={(r['seed'],r['seat']):r for r in panel['rows'] if r['variant']==label and r['opponent']==opp}
            keys=sorted(refs);t=time.perf_counter()
            rows=native.declared_commitment_audit_batch([s for s,_ in keys],[s for _,s in keys],panel['configurations'][label],kind,agent,16)
            for r,k in zip(rows,keys):
                assert (r['seed'],r['seat'])==k and r['checked_actions']==719
                assert r['money'][k[1]]==refs[k]['cash'] and r['money'][1-k[1]]==refs[k]['opponent_cash']
                assert r['overflow']==refs[k]['overflow']
                games+=1;checks+=r['checked_actions']
            samples=[s for r in rows for s in r['samples']]
            omitted=[s for s in samples if sum(s['planned'])>0]
            selected=[s for s in omitted if s['base_valid']]
            altered=[s for s in samples if changed(s)]
            entry=dict(label=label,opponent=opp,games=len(rows),sampled_states=len(samples),
                       states_with_pending_projects=len(omitted),selected_states_with_pending_projects=len(selected),
                       score_changed_states=sum(abs(s['same_kind_before']-s['same_kind_after'])>1e-6 for s in selected),
                       feasible_choice_changed_states=len(altered),
                       games_with_choice_change=sum(any(changed(s) for s in r['samples']) for r in rows),
                       seeds_with_choice_change=len({r['seed'] for r in rows if any(changed(s) for s in r['samples'])}),
                       choice_pairs=dict(collections.Counter(f"{s['base_kind']}->{s['alternative_kind']}" for s in altered)),
                       planned_by_kind=[sum(s['planned'][i] for s in omitted) for i in range(12)],
                       mean_selected_value_delta=statistics.fmean(s['same_kind_after']-s['same_kind_before'] for s in selected) if selected else 0.,
                       ignored_duplicate_count=sum(s['duplicates'] for s in samples),
                       ignored_late_count=sum(s['late'] for s in samples),seconds=time.perf_counter()-t)
            with gzip.open(out/f'{label}_{opp}.json.gz','wt',encoding='utf8',compresslevel=1) as f:json.dump(dict(rows=rows,summary=entry),f,separators=(',',':'))
            result['summary'].append(entry);(out/'summary.json').write_text(json.dumps(result,indent=2))
            print(json.dumps({k:entry[k] for k in ('label','opponent','games','states_with_pending_projects','selected_states_with_pending_projects','feasible_choice_changed_states','games_with_choice_change')}),flush=True)
    result.update(status='PASS_READ_ONLY_DECLARATION_AUDIT_NOT_STRENGTH',unchanged_result_games=games,
                  identical_live_action_steps=checks,seconds=time.perf_counter()-start,
                  caveat='Alternative is rescoring the same feasible current proposals, not playing an alternative game. Conditional future owned-plan output is not a cash/resource credit or a guaranteed production forecast.')
    (out/'summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:result[k] for k in ('status','unchanged_result_games','identical_live_action_steps','seconds')}))


if __name__=='__main__':main()
