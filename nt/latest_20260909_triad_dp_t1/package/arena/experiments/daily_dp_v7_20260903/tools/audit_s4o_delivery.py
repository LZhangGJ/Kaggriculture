from pathlib import Path
from collections import Counter,defaultdict
from functools import lru_cache
import gzip,hashlib,json,statistics as st,sys
E=Path(__file__).resolve().parents[1];sys.path.insert(0,str(E/'native/build'))
import _dp7_native as native
root=E/'receipts/s4n_timing_trace_v1';meta=json.loads((root/'acceptance.json').read_text())
assert meta['status']=='PASS_FROZEN_LIVE_TRACE_NOT_STRENGTH'
out=E/'receipts/s4o_delivery_audit_v1';out.mkdir(exist_ok=False)
products=('WHEAT','CARROT','TOMATO','STRAWBERRY','MELON','EGG','MILK','WOOL','FERTILIZER')
sellable=products[1:8];materials=('WHEAT','FERTILIZER','GOOSE','COW','SHEEP');depots=((4,4),(5,4),(4,5),(5,5))
thresholds=(10,20,50,80,100,300,400,500);rows=[];examples=[]
@lru_cache(maxsize=65536)
def quote(item,inv,n):return sum(native.price(item,inv+k) for k in range(n))
for ref in meta['rows']:
    path=root/ref['path'];assert hashlib.sha256(path.read_bytes()).hexdigest()==ref['sha256']
    with gzip.open(path,'rt',encoding='utf8') as f:data=json.load(f)
    seat=data['seat'];counts=Counter();days=defaultdict(set);first={}
    for frame in data['trace']:
        o=frame['before'];d=o['day'];h=o['hour'];farm=o['farms'][seat];priv=o['private'];a=frame['actions'][seat]
        if d>=29 or frame['debug']['phase']!=3:continue
        positions=[farm['farmer'],*farm['hands']];actions=[a['farmer'],*a['hands']];flags=set();room=max(0,100-sum(priv['shed'].values()))
        for u,(pos,act,bag) in enumerate(zip(positions,actions,priv['inventories'])):
            cargo={k:bag.get(k,0) for k in sellable if bag.get(k,0)>0}
            if not cargo:continue
            flags.add('carried_product');mixed=any(bag.get(k,0)>0 for k in materials)
            if mixed:flags.add('mixed_material_product')
            dist=min(abs(pos[0]-x)+abs(pos[1]-y) for x,y in depots)
            depositing=act[0]=='DROP' or (act[0]=='PLACE' and len(act)>1 and act[1] in sellable)
            if dist==0 and not depositing:flags.add('at_depot_not_depositing')
            if act[0]!='PASS' or room<=0:continue
            if dist==0:flags.add('idle_depot_capacity')
            elif h+dist+2<=24:flags.add('idle_geometric_delivery')
            if dist!=0 or a['market'] or h+3>=24:continue
            # Local quote from one owned product, not future harvest collateral.
            for name,n in cargo.items():
                q=min(n,room);value=quote(products.index(name),o['market']['inventory'][name],q)
                crossing=[x for x in thresholds if farm['money']<x<=farm['money']+value]
                if crossing:
                    flags.add('idle_depot_finance_threshold')
                    if mixed:flags.add('mixed_idle_finance_threshold')
                    if len(examples)<80:examples.append(dict(label=ref['label'],opponent=ref['opponent'],seed=ref['seed'],seat=seat,step=o['step'],unit=u,position=pos,cash=farm['money'],bag=bag,room=room,item=name,quantity=q,conditional_revenue=value,official_costs_unlocked=crossing))
        for flag in flags:counts[flag]+=1;days[flag].add(d);first.setdefault(flag,o['step'])
    rows.append(dict(label=ref['label'],opponent=ref['opponent'],seed=ref['seed'],seat=seat,win=ref['win'],counts=dict(counts),days={k:len(v) for k,v in days.items()},first=first))
    if len(rows)%16==0:print(json.dumps(dict(completed=len(rows))),flush=True)
keys=sorted(set().union(*(r['counts'] for r in rows)));groups=[]
for label in sorted({r['label'] for r in rows}):
    for opponent in sorted({r['opponent'] for r in rows}):
        group=[r for r in rows if r['label']==label and r['opponent']==opponent]
        groups.append(dict(label=label,opponent=opponent,games=len(group),metrics={k:dict(games=sum(r['counts'].get(k,0)>0 for r in group),mean_frames=st.fmean(r['counts'].get(k,0) for r in group),mean_days=st.fmean(r['days'].get(k,0) for r in group)) for k in keys}))
receipt=dict(status='COMPLETE_OBSERVED_DELIVERY_AUDIT_NOT_POLICY_VALUE',games=len(rows),groups=groups,rows=rows,examples=examples,trace_meta_sha256=hashlib.sha256((root/'acceptance.json').read_bytes()).hexdigest(),pre_register_sha256=hashlib.sha256((E/'reports/S4O_DELIVERY_FINANCING_AUDIT_PRE_REGISTER_ZH.md').read_bytes()).hexdigest(),caveat='Overlapping observed frames are not independent opportunities or additive money. Finance thresholds are conditional local quotes, not investment profitability, free labour or true future labels.')
(out/'summary.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps(groups,indent=2))
