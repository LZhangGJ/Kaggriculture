"""Frozen EcoBot economic/market/cull branch checks, not strength tuning."""
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
def canon(x):return json.loads(json.dumps(x))
def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    source=EXP/'opponents/ecobot_v7/output/main.py';assert sha(source)=='0dc02e03c94ef60c06b5093efc2e2fd0530aa6eea20df507a90b90d6651bd067'
    spec=importlib.util.spec_from_file_location('eco7_econ_ref',source);ref=importlib.util.module_from_spec(spec);sys.modules[spec.name]=ref;spec.loader.exec_module(ref)
    build=json.loads((EXP/'native/eco7_probe_build/build_receipt.json').read_text());assert sha(Path(native.__file__))==build['binary_sha256']
    for rel,h in build['source_hashes'].items():assert sha(EXP/rel)==h
    started=time.perf_counter();prices=0
    names=['WHEAT','CARROT','TOMATO','STRAWBERRY','MELON','EGG','MILK','WOOL','FERTILIZER']
    for item,name in enumerate(names):
        for inv in [*range(1,16001),20000,100000,1000000]:
            expected=ref.market_price(name,inv);actual=native.market_price(item,inv)
            if expected!=actual:
                (out/'failure.json').write_text(json.dumps(dict(kind='price',item=name,inventory=inv,expected=expected,actual=actual)));raise AssertionError((name,inv))
            prices+=1
    path=EXP/'receipts/three_day_original_official_parity_v1/20261404_seat0.json.gz';data=json.loads(gzip.decompress(path.read_bytes()))
    samples=[data['trace'][s]['observation'] for s in (0,121,237,361,481,649,697)]
    rng=random.Random(70703);shopnames=list(ref.SHOP_DEMANDS);cases=[]
    # Every actual shop name, in every position in the 1/3/8 shop lists.
    for obs in samples:
        for shop in shopnames:
            for count in (1,3,8):
                c=copy.deepcopy(obs);c['town']['unlocked_shops']=[shop]*count;cases.append(('shops',c))
    for i in range(160):
        c=copy.deepcopy(rng.choice(samples));day=rng.choice([0,1,4,14,16,18,21,26,27,28,29]);hour=rng.choice([0,1,12,21,23]);c.update(day=day,hour=hour,step=day*24+hour)
        c['farms'][c['player']]['money']=rng.choice([0,1,2,5,199,200,201,999,1000,1200,3000,50000])
        c['town']['unlocked_shops']=[rng.choice(shopnames) for _ in range(rng.randrange(9))]
        for name in names:c['market']['inventory'][name]=rng.choice([1,9500,9999,10000,10100,11000,13000])
        for sp in ('COW','SHEEP','GOOSE'):c['private']['shed'][sp]=rng.randrange(0,6)
        cases.append(('cash_stock_boundary',c))
    rows=[];culls=0
    for idx,(kind,obs) in enumerate(cases):
        rstate=ref.EvalState();controller=native.EcoBot();sequence=[obs]
        if kind=='cash_stock_boundary' and obs['day']<28:
            # Two negative days expose stateful retention/abandonment; synthetic
            # public states, not asserted to be legal trajectories from this seed.
            base=copy.deepcopy(obs);base['market']['inventory']['FERTILIZER']=20000;base['market']['inventory']['WHEAT']=1
            base['market']['inventory'].update(MILK=20000,WOOL=20000,EGG=20000)
            for add in (0,1,2):
                c=copy.deepcopy(base);c['day']=min(29,base['day']+add);c['step']=c['day']*24+c['hour'];sequence.append(c)
        for seq,ob in enumerate(sequence):
            expected=canon(asdict(ref.evaluate_turn(copy.deepcopy(ob),rstate)));actual=canon(controller.evaluate(ob))
            if expected!=actual:
                (out/'failure.json').write_text(json.dumps(dict(index=idx,sequence=seq,kind=kind,observation=ob,expected=expected,actual=actual),indent=2));raise AssertionError((idx,seq,kind))
            rs=asdict(rstate.cull);rs['downsized']=sorted(rs['downsized']);rs['negative_days']={sp:rs['negative_days'].get(sp,0) for sp in ('COW','SHEEP','GOOSE')}
            cs=controller.debug()['cull'];cs['downsized'].sort();assert canon(rs)==canon(cs),(idx,seq,'cull state')
            culls+=bool(rs['downsized']);rows.append(dict(case=idx,sequence=seq,kind=kind,output_sha256=hashlib.sha256(json.dumps(actual,sort_keys=True).encode()).hexdigest()))
    r=dict(status='PASS_SYNTHETIC_ECONOMY_NOT_GOAL',build=build,source_sha256=sha(source),price_comparisons=prices,base_cases=len(cases),differential_calls=len(rows),downsized_calls=culls,mismatches=0,seconds=time.perf_counter()-started,rows=rows)
    (out/'acceptance.json').write_text(json.dumps(r,indent=2));print(json.dumps({k:v for k,v in r.items() if k not in ('build','rows')}))
if __name__=='__main__':main()
