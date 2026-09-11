"""Public EcoBot v7 task/VRP differential tests. Partial port, not an agent gate."""
from pathlib import Path
import argparse
import copy
from dataclasses import asdict
import gzip
import hashlib
import importlib.util
import json
import random
import sys
import time

EXP=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(EXP/'native/eco7_probe_build'))
import _eco7_probe as native

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def canonical(x):
    if isinstance(x,dict):return {str(k):canonical(v) for k,v in x.items()}
    if isinstance(x,(set,frozenset)):return [canonical(v) for v in sorted(x)]
    if isinstance(x,(list,tuple)):return [canonical(v) for v in x]
    return x

def reference_kernel(ref,obs,hints,caps,pending):
    me=obs['farms'][obs['player']];priv=obs['private'];day=obs['day'];hour=obs['hour']
    units=[(tuple(pos),inv) for pos,inv in zip([me['farmer']]+me['hands'],priv['inventories'])]
    f=ref.parse_farm_state(me['tiles'],day);cc=ref.count_animal_census(f,priv['shed'],priv['inventories'])
    ps=ref.compute_needed_pastures(f,cc);cs=ref.compute_needed_coops(f,cc,set(ps))
    ts=ref.build_task_catalog(f,priv['shed'],priv['seeds'],ps,cs,units,day,hour,hints,me['unlocked_quadrants'],caps)
    rp=ref._ready_premium_bushes(f,day)
    budget=max(0,24-max(1,hour)-(2 if day==29 else 0))
    rs=ref._solve_day(ts,rp,units,priv['shed'],[budget]*len(units),day)
    routes=[asdict(r) for r in rs]
    for r in routes:r['carried']={k:v for k,v in r['carried'].items() if v>0}
    plan=ref.build_day_plan(units,f,priv['shed'],priv['seeds'],day,hour,ps,cs,hints,me['unlocked_quadrants'],caps,pending)
    return dict(farm_state=f,census=asdict(cc),needed_pastures=ps,needed_coops=cs,
        reserved=ref.live_reserved_structures(f,me['unlocked_quadrants'],hints['reserved_grazer_slots'],hints['reserved_geese_slots'],day),
        feed_thresholds=ref.wheat_feed_thresholds(cc,caps,day),fert_due_tomorrow=ref.count_fertilize_due_tomorrow(f,day),
        wheat_maturing=ref.count_maturing_tomorrow(f,'WHEAT'),carrot_maturing=ref.count_maturing_tomorrow(f,'CARROT'),
        catalog=[asdict(t) for t in ts],ready_premium=rp,routes=routes,
        required_hands=ref.required_hand_count(ts,rp,units,priv['shed'],min(13,11 if len(me['unlocked_quadrants'])<3 else 13),day),day_plan=plan.routes)

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    source=EXP/'opponents/ecobot_v7/output/main.py';source_hash=sha(source)
    assert source_hash=='0dc02e03c94ef60c06b5093efc2e2fd0530aa6eea20df507a90b90d6651bd067'
    spec=importlib.util.spec_from_file_location('eco7_frozen_core_reference',source);ref=importlib.util.module_from_spec(spec);sys.modules[spec.name]=ref;spec.loader.exec_module(ref)
    build=json.loads((EXP/'native/eco7_probe_build/build_receipt.json').read_text())
    assert sha(Path(native.__file__))==build['binary_sha256']
    for rel,h in build['source_hashes'].items():assert sha(EXP/rel)==h
    steps={0,1,2,12,23,24,25,120,121,216,217,360,361,480,481,648,649,672,673,696,697,708,716,718}
    cases=[];inputs={};rng=random.Random(7007123)
    for folder in ('three_day_original_official_parity_v1','three_day_switched_official_parity_v1'):
        for replay in sorted((EXP/'receipts'/folder).glob('*.json.gz')):
            inputs[str(replay.relative_to(EXP))]=sha(replay)
            data=json.loads(gzip.decompress(replay.read_bytes()))
            for row in data['trace']:
                if row['step'] not in steps:continue
                obs=row['observation']
                # Full original evaluator supplies its own reference hints. It is
                # not yet the C++ implementation or used in a native game loop.
                decision=ref.evaluate_turn(copy.deepcopy(obs),ref.EvalState())
                cases.append(dict(origin=f'{replay.name}:{row["step"]}',role='official_verified_state',obs=obs,
                    hints=asdict(decision.hints),caps=decision.retained_caps,pending=0))
    real_count=len(cases)
    for i in range(128):
        c=copy.deepcopy(cases[rng.randrange(real_count)]);c['origin']=f'synthetic:{i}';c['role']='synthetic_edge'
        priv=c['obs']['private'];priv['seeds']={k:rng.randrange(0,10) for k in ref.CROPS}
        priv['shed']={k:rng.randrange(0,6 if k in ref.ANIMALS else 20) for k in (*ref.SELLABLE_ITEMS,*ref.ANIMALS)}
        for inv in priv['inventories']:
            inv.clear()
            for k in ('WHEAT','FERTILIZER','COW','SHEEP','GOOSE','MELON','STRAWBERRY'):
                n=rng.randrange(0,3)
                if n:inv[k]=n
        c['hints']=dict(reserved_grazer_slots=rng.randrange(0,19),reserved_geese_slots=rng.randrange(0,7),crop_limits={k:rng.randrange(0,8) for k in ref.CROPS})
        c['caps']={k:rng.choice([0,1,2,4,999]) for k in ref.ANIMALS}
        remaining=24-max(1,c['obs']['hour'])-(2 if c['obs']['day']==29 else 0)
        c['pending']=rng.randrange(0,3) if remaining>=2 else 0
        cases.append(c)
    started=time.perf_counter();rows=[];coverage=dict(pickup=0,plant=0,feed=0,harvest=0,fertilize=0,build=0,premium=0,carried=0,pending=0)
    for i,c in enumerate(cases):
        expected=canonical(reference_kernel(ref,copy.deepcopy(c['obs']),copy.deepcopy(c['hints']),copy.deepcopy(c['caps']),c['pending']))
        actual=canonical(native.kernel(c['obs'],c['hints'],c['caps'],c['pending']))
        if actual!=expected:
            differences=[k for k in expected if actual.get(k)!=expected[k]]
            (out/'failure.json').write_text(json.dumps(dict(index=i,case=c,differing_fields=differences,expected=expected,actual=actual),indent=2))
            raise AssertionError((i,c['origin'],differences))
        flat=[a[0] for q in actual['day_plan'].values() for a in q]
        for k,op in [('pickup','PICKUP'),('plant','PLANT'),('feed','FEED'),('harvest','HARVEST'),('fertilize','FERTILIZE')]:coverage[k]+=op in flat
        coverage['build']+=any(op in flat for op in ('BUILD_COOP','BUILD_PASTURE'));coverage['premium']+=bool(actual['ready_premium'])
        coverage['carried']+=any(inv for inv in c['obs']['private']['inventories']);coverage['pending']+=c['pending']>0
        rows.append(dict(origin=c['origin'],role=c['role'],tasks=len(actual['catalog']),hands=actual['required_hands'],queue_actions=sum(map(len,actual['day_plan'].values())),output_sha256=hashlib.sha256(json.dumps(actual,sort_keys=True).encode()).hexdigest()))
        if (i+1)%50==0:print(json.dumps(dict(checked=i+1,total=len(cases))),flush=True)
    receipt=dict(status='PASS_PARTIAL_DISPATCH_NOT_FULL_AGENT',build=build,source_sha256=source_hash,input_replay_hashes=inputs,
        real_state_cases=real_count,synthetic_cases=len(cases)-real_count,cases=len(cases),mismatches=0,coverage=coverage,rows=rows,
        seconds=time.perf_counter()-started,remaining='Economic evaluator, stateful full agent, official complete-game parity and threaded arena not yet ported.',final_goal_acceptance=False)
    (out/'acceptance.json').write_text(json.dumps(receipt,indent=2));print(json.dumps({k:v for k,v in receipt.items() if k not in ('build','input_replay_hashes','rows')}))

if __name__=='__main__':main()
