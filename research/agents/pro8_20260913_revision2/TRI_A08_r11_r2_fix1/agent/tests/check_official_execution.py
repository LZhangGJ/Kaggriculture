"""Cross-check single-own-animal streams with frozen official unit/dawn rules.

No games, opponents, stochastic scripts or reconstructed hidden states are used.
A fixed own farmer is at the depot. Movement/delivery constraints are tested
separately and are not silently assumed free or part of the scalar DP theorem.
"""
from __future__ import annotations
import argparse,copy,json,pathlib,sys
ROOT=pathlib.Path(__file__).resolve().parents[1]
EVIDENCE=ROOT/'input' if (ROOT/'input').exists() else ROOT/'evidence'
sys.path.insert(0,str(EVIDENCE/'referee'))
from cpu_runtime import load_engine
p=argparse.ArgumentParser();p.add_argument('--scenarios',type=pathlib.Path,required=True);p.add_argument('--out',type=pathlib.Path,required=True);a=p.parse_args()
engine=load_engine()
checks=0;cases=0;days=0;actions=0;failures=[]
def check(ok,why):
 global checks
 checks+=1
 if not ok: raise AssertionError(why)
def own_state(name='SHEEP',birth=0,yield_units=0,manure=False,hunger=0,bonus=0,fed=False,cared=False):
 t=engine._new_animal(name,birth)
 t.update(yield_units=yield_units,fertilizer_available=bool(manure),consecutive_unfed=hunger,pending_care_bonus=bonus,fed_today=bool(fed),cared_today=bool(cared))
 farm={'tiles':[[None]*10 for _ in range(10)],'farmer':[4,4],'hands':[],'money':0}
 farm['tiles'][4][4]=t
 private={'shed':{},'seeds':{},'inventories':[{'WHEAT':30}]}
 return farm,private
for line in a.scenarios.open():
 s=json.loads(line);cases+=1;name=('GOOSE','COW','SHEEP')[s['kind']-9];product=engine.ANIMALS[name]['product']
 farm,private=own_state(name,s['birth'],s['yield'],s['manure'],s['hunger'],s['bonus'],s['fed'],s['cared'])
 try:
  for d in s['days']:
   days+=1;inv=private['inventories'][0];before=copy.deepcopy(inv);actual_ops=[];tile=farm['tiles'][4][4]
   if isinstance(tile,dict) and 'animal' in tile:
    if tile['yield_units']>0:actual_ops.append('HARVEST')
    if tile['fertilizer_available']:actual_ops.append('COLLECT_FERTILIZER')
   if d['feed']:actual_ops.append('FEED')
   if d['care']:actual_ops.append('CARE')
   for op in actual_ops:
    prior=copy.deepcopy((farm,private))
    engine._apply_unit_action(farm,private,0,[op],10,d['day'],24)
    check((farm,private)!=prior,(s['id'],d['day'],'unexpected no-op',op))
    actions+=1
   delta=lambda item:inv.get(item,0)-before.get(item,0)
   check(delta(product)==d['product'],(s['id'],d['day'],'product'))
   check(delta('FERTILIZER')==d['fertilizer'],(s['id'],d['day'],'manure'))
   check(delta('WHEAT')==d['wheat'],(s['id'],d['day'],'feed consumption'))
   check(len(actual_ops)==d['labor'],(s['id'],d['day'],'collection/service labor',len(actual_ops),d['labor']))
   if d['day']<29:engine._daily_refresh_animals(farm,d['day'])
 except AssertionError as e:failures.append(str(e))
# Explicit actual stock, travel, depot capacity, action and sale-window controls.
market=lambda:{'inventory':{i:10000 for i in engine.PRODUCTS}}
# Three distinct operations are necessary to collect two outputs and deposit.
farm,private=own_state(yield_units=3,manure=True)
private['inventories']=[{}];m=market()
check(not engine._commit_unit('SELL','WOOL',200,farm,private,m),'tile output sold before harvest')
engine._apply_unit_action(farm,private,0,['HARVEST'],10,29,24)
check(not engine._commit_unit('SELL','WOOL',200,farm,private,m),'carried output sold before deposit')
engine._apply_unit_action(farm,private,0,['COLLECT_FERTILIZER'],10,29,24)
check(private['inventories'][0]=={'WOOL':3,'FERTILIZER':1},'two collections need two actions')
engine._apply_unit_action(farm,private,0,['DROP'],10,29,24)
for _ in range(3):check(engine._commit_unit('SELL','WOOL',200,farm,private,m),'deposited wool not saleable')
check(engine._commit_unit('SELL','FERTILIZER',100,farm,private,m),'deposited manure not saleable')
check(farm['money']==700,'quoted-unit physical receipt incorrect')
# This 700 is a synthetic rule check at stated prices, not episode proceeds.
farm,private=own_state(yield_units=3,manure=True);private['inventories']=[{}];m=market()
engine._apply_unit_action(farm,private,0,['HARVEST'],10,29,24)
check(farm['tiles'][4][4]['fertilizer_available'] and not private['shed'],'one action cannot also collect and deliver')
check(not engine._commit_unit('SELL','WOOL',200,farm,private,m),'one-step unfinished route fabricated sale')
# A loaded farmer away from the depot cannot drop merely because output exists.
farm['farmer']=[2,2];before=copy.deepcopy(private)
engine._apply_unit_action(farm,private,0,['DROP'],10,29,24)
check(private==before,'off-depot DROP accepted')
# No wheat in carried inventory means FEED cannot occur, even with shed wheat.
farm,private=own_state();private['inventories']=[{}];private['shed']={'WHEAT':10}
engine._apply_unit_action(farm,private,0,['FEED'],10,28,24)
check(not farm['tiles'][4][4]['fed_today'],'shed wheat treated as carried feed')
# Preserve actual capacity: overflow must not become saleable forecast revenue.
farm,private=own_state();private['inventories']=[{'WOOL':3}];private['shed']={'WHEAT':100}
engine._apply_unit_action(farm,private,0,['DROP'],10,29,24,shed_capacity=100)
check(private['shed'].get('WOOL',0)==0,'capacity overflow credited to shed')
result={'scope':'Official single-own-farm unit/dawn rule checks; not games or global scheduler feasibility','new_games':0,'status':'PASS' if not failures else 'FAIL','lifecycle_cases':cases,'player_days':days,'effective_unit_actions':actions,'checks':checks,'failures':failures}
a.out.parent.mkdir(exist_ok=True,parents=True);a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result));raise SystemExit(bool(failures))
