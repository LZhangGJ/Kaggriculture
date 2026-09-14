#!/usr/bin/env python3
"""Independent frozen-official terminal-route checks. Zero competitive games.
We execute only regenerated own states, no rival orders, zero exogenous supply,
known town demand, and no night transition (last legal action is step 718).
"""
import argparse,copy,ctypes,gzip,hashlib,importlib.util,json,time,collections,sys
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
def load(p,n):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
gate=load(ROOT/'tests/native_completion_gate.py','r4_gate_lib');ITEMS=gate.codec._ITEMS;OPS=gate.codec._OPS
MOVES={1:(0,-1),2:(0,1),3:(1,0),4:(-1,0)}
def atom(a):
 op,i,q,_=a;x=[OPS[op]]
 if i>=0:
  x.append(ITEMS[i])
  if op in (6,7,20,21,22,23):x.append(q)
 return x

def execute(engine,A,obs,plans):
 o=copy.deepcopy(obs);seat=o['player'];f=o['farms'][seat];p=o['private'];cash0=f['money'];idx=[0]*len(plans);sold=collections.Counter();deposited=collections.Counter();work=waits=lost=0;rows=[]
 cfg=A({k:copy.deepcopy(v.get('default') if isinstance(v,dict) else v)for k,v in engine.specification['configuration'].items()});env=A(configuration=cfg)
 while o['step']<719:
  step=o['step'];acts=[atom(pl[idx[u]]) if idx[u]<len(pl) else ['PASS']for u,pl in enumerate(plans)];events=[]
  demand=collections.Counter(x[1]for x in acts if x[0]=='PLANT');blocked={x for x,n in demand.items()if n>p['seeds'].get(x,0)}
  for u,declared in enumerate(acts):
   if idx[u]>=len(plans[u]):continue
   where=f['farmer'] if u==0 else f['hands'][u-1];target=plans[u][idx[u]][3]
   assert target<0 or target==where[1]*10+where[0],('bad route target',step,u,where,target)
   q=sum(p['inventories'][u].values());before=copy.deepcopy(p['inventories'][u]);shed0=copy.deepcopy(p['shed']);act=declared
   if act[0]=='DROP' and q>100-sum(p['shed'].values()):
    waits+=1;work+=1;acts[u]=['PASS'];events.append({'u':u,'wait_for_drop_capacity':q});continue
   if act[0]=='PLANT' and act[1]in blocked:act=['PASS']
   engine._apply_unit_action(f,p,u,act,10,o['day'],24,100);idx[u]+=1;work+=declared[0]!='PASS'
   if act[0]=='DROP':
    delta={x:p['shed'].get(x,0)-shed0.get(x,0)for x in set(p['shed'])|set(shed0)};deposited.update(delta);lost+=q-sum(delta.values());events.append({'u':u,'drop_before':before,'deposited':delta})
  prices=o['market']['prices'];orders=[['SELL',i,p['shed'][i]]for i in sorted(ITEMS[:9],key=lambda i:-prices[i])if p['shed'].get(i,0)>0];shed0=copy.deepcopy(p['shed']);money0=f['money']
  public=A(step=step,day=o['day'],hour=o['hour'],farms=o['farms'],market=o['market'],town=o['town']);states=[A(observation=A(public),action={'market':[]})for _ in range(2)];states[seat].observation.private=p;states[1-seat].observation.private={'shed':{},'seeds':{},'inventories':[{}]};states[seat].action={'market':orders}
  engine._process_market(states,env)
  sold.update({i:shed0.get(i,0)-p['shed'].get(i,0)for i in ITEMS[:9]})
  rows.append({'step':step,'units':acts,'orders':orders,'cash_delta':f['money']-money0,'unit_events':events})
  engine._town_consume(env,states,step);engine._decay_plants(f,step)
  o['step']+=1;o['day']=o['step']//24;o['hour']=o['step']%24
 stranded=collections.Counter(p['shed'])
 for b in p['inventories']:stranded.update(b)
 return {'cash':f['money']-cash0,'sold':[sold[i]for i in ITEMS],'stranded':[stranded[i]for i in ITEMS],'deposited':[deposited[i]for i in ITEMS],'work':work,'waits':waits,'lost':lost,'completed':all(i==len(pl)for i,pl in zip(idx,plans)),'rows':rows,'final_observation':o}

def reflect(o,plans,position,m):
 o=copy.deepcopy(o)
 def c(pos):
  x,y=pos%10,pos//10
  if m&1:x=9-x
  if m&2:y=9-y
  return y*10+x
 for f in o['farms']:
  tiles=[[None for _ in range(10)]for _ in range(10)]
  for y,row in enumerate(f['tiles']):
   for x,t in enumerate(row):q=c(y*10+x);tiles[q//10][q%10]=t
  f['tiles']=tiles
  for pos in [f['farmer'],*f['hands']]:q=c(pos[1]*10+pos[0]);pos[:]=[q%10,q//10]
 out=[]
 for pl in plans:
  r=[]
  for op,i,q,pos in pl:
   if m&1:op={3:4,4:3}.get(op,op)
   if m&2:op={1:2,2:1}.get(op,op)
   r.append([op,i,q,c(pos)if pos>=0 else -1])
  out.append(r)
 return o,out,c(position)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--library',type=Path,required=True);ap.add_argument('--referee',type=Path,default=ROOT/'evidence/round4_feedback/referee');ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
 rt=load(args.referee/'cpu_runtime.py','r4_official_runtime');eng=rt.load_engine();n=gate.Native(args.library);start=time.perf_counter();checks=0;records=[];inputs=[];positives=0;negatives=0;raw=[]
 def check(c,why):
  nonlocal checks
  checks+=1
  if not c:
   args.out.parent.mkdir(parents=True,exist_ok=True);args.out.with_suffix('.FAILURE.json').write_text(json.dumps({'failure':why,'checks':checks,'records':records},indent=2));raise AssertionError(why)
 def test(label,o,ps,pos=46,whole=((10,-1,1),),missing=None,shadow=2,mode=0,expect=None):
  nonlocal positives,negatives
  eng._refresh_prices(o['market']);r=n.query(o,ps,pos,whole,missing,shadow,mode)
  if expect is not None:check(r['selected']==expect,[label,'selection',r])
  positives+=r['selected'];negatives+=not r['selected']
  a=execute(eng,rt.AttrDict,o,r['base']);b=execute(eng,rt.AttrDict,o,r['best'])
  for name,actual in [('before',a),('after',b)]:
   for key in ['cash','sold','stranded','deposited','work','waits']:check(r[name][key]==actual[key],[label,name,key,r[name][key],actual[key]])
  if r['selected']:
   check(b['lost']==0 and b['completed'],[label,'execution',b])
   check(b['cash']-a['cash']>shadow*max(0,b['work']-a['work'])+1e-6,[label,'net cash',a['cash'],b['cash']])
   for i in range(9):check(b['stranded'][i]<=a['stranded'][i] and b['sold'][i]>=a['sold'][i],[label,'inventory preservation',i])
   # Complete original semantic route remains a subsequence of the new one.
   for old,new in zip(ps,r['best']):
    old=[x for x in old if x[0]not in MOVES];new=[x for x in new if x[0]not in MOVES];i=0
    for x in new:
     if i<len(old)and x==old[i]:i+=1
    check(i==len(old),[label,'dropped old commitment',old,new])
  rec={'name':label,'selected':r['selected'],'native':r,'official_before':{k:v for k,v in a.items()if k not in ('rows','final_observation')},'official_after':{k:v for k,v in b.items()if k not in ('rows','final_observation')}};records.append(rec);inputs.append({'name':label,'observation':o,'plans':ps,'pos':pos,'whole':whole,'missing':missing,'shadow':shadow,'mode':mode});raw.append({'name':label,'before':a['rows'],'after':b['rows']})
 # General geometry, remaining horizon, output size, price pressure, labor.
 for mirror in range(4):
  for step in (698,706,710,713,714,715,716,717,718):
   for distance in (1,2,3):
    for qty in (1,4):
     for inv in (9990,10070):
      o=gate.synthetic(step,qty);tile=o['farms'][0]['tiles'][4][6];o['farms'][0]['tiles'][4][6]=None;o['farms'][0]['tiles'][4][5+distance]=tile;o['market']['inventory']['TOMATO']=inv
      o,ps,pos=reflect(o,[[[5,-1,1,45]]],45+distance,mirror)
      test(f'm{mirror}_t{step}_d{distance}_q{qty}_p{inv}',o,ps,pos,shadow=2)
 # Admission must not deplete carried inputs promised to an old service.
 o=gate.synthetic(708,cargo={'WHEAT':1,'EGG':1});o['farms'][0]['tiles'][4][7]=eng._new_animal('COW',10);o['farms'][0]['tiles'][4][7]['yield_units']=0
 ps=[[[3,-1,1,-1],[3,-1,1,-1],[15,-1,1,47],[17,-1,1,47],[4,-1,1,-1],[4,-1,1,-1],[5,-1,1,45]]]
 test('preserve_feed_care_and_carried_wheat',o,ps,expect=True)
 o['private']['inventories'][0].pop('WHEAT');test('unfunded_old_feed_rejected',o,ps,expect=False)
 # Two deposits really compete for space; market frees it only after units.
 o=gate.synthetic(715,cargo={'EGG':60});o['farms'][0]['hands']=[[5,4]];o['private']['inventories'].append({'MILK':60})
 test('simultaneous_drop_capacity_wait',o,[[[5,-1,1,45]],[[5,-1,1,45]]],expect=True)
 # Need a SECOND return, not an oversized one-shot cargo drop.
 o=gate.synthetic(714,qty=2,cargo={'EGG':99});test('two_deposit_waves_feasible',o,[[[5,-1,1,45]]],expect=True)
 o=gate.synthetic(715,qty=2,cargo={'EGG':99});test('two_deposit_waves_one_tick_short',o,[[[5,-1,1,45]]],expect=False)
 o=gate.synthetic(714);o['private']['shed']={'SHEEP':99};test('unsellable_warehouse_block',o,[[[5,-1,1,45]]],expect=False)
 o=gate.synthetic(715);o['market']['inventory']['TOMATO']=100000;test('price_floor_not_free_labor',o,[[[5,-1,1,45]]],expect=False)
 test('large_labor_cost_reject',gate.synthetic(715),[[[5,-1,1,45]]],shadow=10000,expect=False)
 test('positive_return_close_old_unclosed_route',gate.synthetic(714),[[[5,-1,1,45],[3,-1,1,-1],[10,-1,1,46]]],mode=1,expect=True)
 # One unreachable worker must not veto a separately reachable last-tick DROP.
 o=gate.synthetic(718,cargo={'EGG':1});o['farms'][0]['hands']=[[0,0]];o['private']['inventories'].append({'MILK':1})
 test('reachable_individual_tail_despite_unreachable_peer',o,[[],[]],mode=1,shadow=4,expect=True)
 test('individual_tail_negative_after_work_cost',o,[[],[]],mode=1,shadow=10000,expect=False)
 # Stored predecessor may be on a different worker: water MUST precede harvest.
 o=gate.synthetic(715,qty=1,cargo={});o['farms'][0]['tiles'][4][6]=eng._new_plant('WHEAT',26,24);o['farms'][0]['tiles'][4][6]['yield_units']=1;o['farms'][0]['tiles'][4][6]['fertilized_until_day']=-1;o['farms'][0]['farmer']=[6,4];o['farms'][0]['hands']=[[6,4]];o['private']['inventories'].append({})
 ps=[[[10,-1,1,46],[4,-1,1,-1],[5,-1,1,45]],[]]
 test('cross_worker_water_before_existing_harvest',o,ps,whole=((9,-1,1),(10,-1,1)),missing=((9,-1,1),),shadow=0,expect=True)
 # Existing future FERTILIZE/PLANT material sequences keep their resources.
 o=gate.synthetic(710);o['private']['seeds']={'WHEAT':1};ps=[[[8,0,1,45],[9,-1,1,45],[5,-1,1,45]]]
 test('funded_old_plant_water_kept',o,ps,expect=True)
 o['private']['seeds']={};test('seed_unfunded_not_rescued_by_sale',o,ps,expect=False)
 # Crop decay is applied by the official kernel at each generated step.
 for expiry in (712,713,714,715):
  o=gate.synthetic(713,qty=4);o['farms'][0]['tiles'][4][6]['max_lifespan_step']=expiry;test('expiry_'+str(expiry),o,[[[5,-1,1,45]]],shadow=0)
 args.out.parent.mkdir(parents=True,exist_ok=True)
 result={'status':'PASS','scope':'conditional terminal-only official execution; no rival market orders; zero exogenous supply; no new games','native_sha256':hashlib.sha256(args.library.read_bytes()).hexdigest(),'cases':len(records),'checks':checks,'selected':positives,'not_selected':negatives,'wall_seconds':time.perf_counter()-start,'new_games':0,'records':records}
 args.out.write_text(json.dumps(result,indent=2)+'\n');args.out.with_suffix('.inputs.json.gz').write_bytes(gzip.compress(json.dumps(inputs).encode()));args.out.with_suffix('.official_steps.jsonl.gz').write_bytes(gzip.compress(('\n'.join(json.dumps(x)for x in raw)+'\n').encode()));print(json.dumps({k:v for k,v in result.items()if k!='records'}))
if __name__=='__main__':main()
