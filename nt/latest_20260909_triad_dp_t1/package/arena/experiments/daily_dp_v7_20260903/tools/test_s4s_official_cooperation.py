from pathlib import Path
import hashlib,json,sys,copy
E=Path(__file__).resolve().parents[1];root=E.parents[1];host=root/'gpt_review/codex/G001_CPU_FOR_GPT_20260903';sys.path.insert(0,str(host))
from cpu_runtime import LocalGame,pass_agent
out=E/'receipts/s4s_official_cooperation_v2';out.mkdir(exist_ok=False);rows=[]
for seat in (0,1):
    for order in ('plant_water','water_plant'):
        g=LocalGame(203)
        for _ in range(4*24+23):g.advance([pass_agent(g.observation(s)) for s in (0,1)])
        f=g.state[0].observation.farms[seat];p=g.state[seat].observation.private
        f['farmer']=[4,4];f['hands']=[[4,4]];f['tiles'][4][4]=None
        p['inventories']=[{},{}];p['seeds']['WHEAT']=1
        acts=[pass_agent(g.observation(s)) for s in (0,1)]
        pair=[['PLANT','WHEAT'],['WATER']] if order=='plant_water' else [['WATER'],['PLANT','WHEAT']]
        acts[seat]={'farmer':pair[0],'hands':[pair[1]],'market':[]}
        g.advance(acts);tile=g.observation(seat)['farms'][seat]['tiles'][4][4]
        if order=='plant_water':assert tile['kind']=='PLANT' and tile['consecutive_unwatered']==0
        else:assert tile['kind']=='WEED'
        assert g.observation(seat)['private']['seeds']['WHEAT']==0
        rows.append(dict(seat=seat,order=order,tile=tile))
d=dict(status='PASS_OFFICIAL_1327_ORDERED_UNIT_COOPERATION',cases=rows,source_hashes={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (host/'cpu_runtime.py',host/'official/kaggriculture.py',host/'official/kaggriculture.json',Path(__file__))},caveat='Controlled unit-order mechanism only, no win-rate or common-opportunity claim.')
(out/'acceptance.json').write_text(json.dumps(d,indent=2));print(json.dumps(dict(status=d['status'],cases=len(rows))))
