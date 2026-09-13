"""Bounded conditional continuations using the supplied unmodified official interpreter.
Not matches: rival inventory starts empty, public forecast receipts are injected;
rival visible service is stipulated, weeds disabled and future shop identities
are discarded. Expected unknown-shop consumption follows the parent's scenario.
No recorded suffix, private opponent state, or actual environment seed is read.
"""
from pathlib import Path
import argparse,collections,copy,ctypes,gzip,hashlib,importlib.util,json,math,resource,sys,time
ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--kit',type=Path,required=True);ap.add_argument('--feedback',type=Path,required=True);ap.add_argument('--history',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--ticks',type=int,default=312);a=ap.parse_args()
assert 1<=a.ticks<=336
R=Path(__file__).resolve().parents[1];D=a.out;D.mkdir(parents=True,exist_ok=False)
H=a.kit/'Kaggriculture/nt/latest_20260910_r2p16/referee';sys.path.insert(0,str(H));from cpu_runtime import AttrDict,load_engine
s=importlib.util.spec_from_file_location('r12codec',R/'policy/agent.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);cfg=json.loads((R/'policy/config.json').read_text())
def save(p,x):
 b=json.dumps(x,ensure_ascii=False,indent=2).encode();p.write_bytes(gzip.compress(b,mtime=0) if p.suffix=='.gz' else b+b'\n')
def funcs(z):
 for fn,ret,args in [('td_clone',ctypes.c_void_p,[ctypes.c_void_p]),('td_r11_enable',ctypes.c_int,[ctypes.c_void_p,ctypes.c_int]),('td_r12_enable',ctypes.c_int,[ctypes.c_void_p,ctypes.c_int]),('td_contract_json',ctypes.c_char_p,[ctypes.c_void_p]),('td_crop_clock_json',ctypes.c_char_p,[ctypes.c_void_p])]:
  f=getattr(z.lib,fn);f.restype=ret;f.argtypes=args

def clone(z,on):
 q=m.Agent(cfg,R/'policy/a06.so');q.close();q.handle=z.lib.td_clone(z.handle);q.last=z.last;q.seat=z.seat;funcs(q);q.lib.td_r11_enable(q.handle,1);q.lib.td_r12_enable(q.handle,on);return q
class Conditional:
 def __init__(self,obs,flow):
  self.engine=load_engine();self.configuration=AttrDict({k:copy.deepcopy(v.get('default') if isinstance(v,dict) else v) for k,v in self.engine.specification['configuration'].items()})
  self.configuration.weedSpawnChance=0.0
  # This dummy internal value is never used by the agent. Weed generation is
  # disabled and synthetic new shop identities are immediately discarded.
  self.info={'seed':0};self.done=False;self.t=obs['step'];self.seat=obs['player'];self.flow=flow;self.fraction=[0.0]*9;self.known=copy.deepcopy(obs['town']['unlocked_shops'])
  common={k:copy.deepcopy(obs[k]) for k in ('farms','market','town','day','hour')}
  self.state=[]
  for seat in (0,1):
   o=AttrDict(common.copy());o.step=self.t;o.player=seat;o.remainingOverageTime=60
   o.private=copy.deepcopy(obs['private']) if seat==self.seat else self.engine._new_private()
   if seat!=self.seat:o.private['inventories']=[{} for _ in range(1+len(common['farms'][seat]['hands']))]
   self.state.append(AttrDict(observation=o,action={},reward=0,status='ACTIVE'))
  self.events=[];self.totals=[collections.defaultdict(float),collections.defaultdict(float)];self.quantities=[collections.defaultdict(int),collections.defaultdict(int)]
  commit=self.engine._commit_unit
  def wrap(op,item,price,farm,private,market,shed_capacity=100):
   seat=next(i for i,f in enumerate(self.state[0].observation.farms) if f is farm);before=farm['money'];ok=commit(op,item,price,farm,private,market,shed_capacity)
   if ok:
    delta=farm['money']-before;key=op+':'+item;self.events.append({'step':self.t,'seat':seat,'op':op,'item':item,'cash':delta,'quantity':1});self.totals[seat][key]+=delta;self.quantities[seat][key]+=1
   return ok
  self.engine._commit_unit=wrap
  for fn in ('_do_hire','_do_buy_land'):
   original=getattr(self.engine,fn)
   def helper(farm,*args,_fn=fn,_original=original,**kw):
    seat=next(i for i,f in enumerate(self.state[0].observation.farms) if f is farm);before=farm['money'];out=_original(farm,*args,**kw);delta=farm['money']-before
    if delta:self.events.append({'step':self.t,'seat':seat,'op':_fn,'cash':delta,'quantity':1});self.totals[seat][_fn]+=delta;self.quantities[seat][_fn]+=1
    return out
   setattr(self.engine,fn,helper)
  self.startcash=[f['money'] for f in common['farms']]
 def observation(self):
  return copy.deepcopy(self.state[self.seat].observation)
 def advance(self,act):
  step=self.t;day=step//24;hour=step%24;other=1-self.seat;obs0=self.state[0].observation
  rival={'farmer':['PASS'],'hands':[['PASS'] for _ in obs0.farms[other]['hands']],'market':[]}
  # Same public-flow condition for both branches, not either real opponent's
  # historical future decisions. Quantities are assumptions, fills are official.
  if hour in (1,17):
   for i in range(9):
    x=self.flow[day][i]*cfg['supply'];q=math.floor(x/2) if hour==1 else int(math.copysign(math.floor(abs(x)+0.5),x))-math.floor(x/2)
    item=m._ITEMS[i]
    if q>0:
     pr=self.state[other].observation.private;pr['shed'][item]=pr['shed'].get(item,0)+q;rival['market'].append(['SELL',item,q])
    elif q<0 and i in (0,8):rival['market'].append(['BUY_PRODUCT',item,-q])
  for row in obs0.farms[other]['tiles']:
   for tile in row:
    if not isinstance(tile,dict):continue
    if tile.get('animal'):
     tile.update(fed_today=True,cared_today=True,yield_units=0,fertilizer_available=False)
    elif tile.get('kind')=='PLANT':tile.update(watered_today=True,yield_units=0,max_lifespan_step=-1)
  self.state[self.seat].action=copy.deepcopy(act);self.state[other].action=rival
  for s in self.state:s.observation.step=step
  self.events=[];self.engine.interpreter(self.state,self)
  if step%4==0:
   unknown=max(0,min(8,day//3)-len(self.known));avg=[5/8,3/8,2/8,4/8,0,2/8,3/8,2/8,0]
   for i in range(9):
    self.fraction[i]+=unknown*avg[i];n=math.floor(self.fraction[i]+1e-9);obs0.market['inventory'][m._ITEMS[i]]-=n;self.fraction[i]-=n
   self.engine._refresh_prices(obs0.market)
  obs0.town['unlocked_shops']=copy.deepcopy(self.known)
  self.t+=1
  for s in self.state:s.observation.step=self.t;s.observation.town=obs0.town
  self.done=all(s.status=='DONE' for s in self.state)
  for i,f in enumerate(obs0.farms):assert abs(self.startcash[i]+sum(self.totals[i].values())-f['money'])<1e-7
  return copy.deepcopy(self.events),rival

summary={'scope':__doc__,'official_engine_sha256':hashlib.sha256((H/'official/kaggriculture.py').read_bytes()).hexdigest(),'native_sha256':hashlib.sha256((R/'policy/a06.so').read_bytes()).hexdigest(),'new_games':0,'new_seeds':0,'cases':[]}
cases=['thomas_955_v2_3293516959_seat0','aurax_shop_v2_3293516958_seat1','thomas_955_v2_3293516972_seat0','submission_56149565_3293516949_seat0']
for name in cases:
 replay=json.loads(gzip.decompress((a.feedback/'replays'/(name+'.replay.json.gz')).read_bytes()));seat=replay['result']['seat']
 exact=a.history/(name+'_first_r12.json.gz')
 if exact.exists():start=json.loads(gzip.decompress(exact.read_bytes()))['step'];reason='first r12 difference from parent while historical action prefix is identical'
 else:
  probes=json.loads(gzip.decompress((a.history/(name+'_probes.json.gz')).read_bytes()));changes=[p for p in probes if p['different']];start=changes[0]['step'] if changes else probes[0]['step'];reason='matched one-call mechanism probes from exact r6 history; not claimed full parent prefix'
 track=m.Agent(cfg|{'a06_r11_recovery':(1 if exact.exists() else 0),'a06_r12_calendar':0},R/'policy/a06.so');funcs(track)
 for t in range(start):assert track(replay['steps'][t][seat]['observation'])==replay['actions'][t][seat]
 o=replay['steps'][start][seat]['observation'];branches=[clone(track,on) for on in (0,1)];track.close()
 actions=[b(o) for b in branches];flow=json.loads(branches[0].lib.td_crop_clock_json(branches[0].handle))['live_forecast']
 both=[]
 for on,b in enumerate(branches):
  world=Conditional(o,flow);records=[];daily=[];tick0=time.perf_counter();first_contract=json.loads(b.lib.td_contract_json(b.handle));first_debug=b.debug()
  n=min(a.ticks,718-start) # never finish a new 719-step game
  for t in range(n):
   obs=world.observation();action=actions[on] if t==0 else b(obs)
   events,rival=world.advance(action);post=world.observation()
   records.append({'step':obs['step'],'observation':obs,'action':action,'conditional_rival_action':rival,'successful_official_market_events':events,'cash_after':[f['money'] for f in post['farms']],'own_private_after':post['private'],'own_tiles_after':post['farms'][seat]['tiles']})
   if obs['hour']==0 or t==0:
    dbg=b.debug();daily.append({'step':obs['step'],'cash_observed':[f['money'] for f in obs['farms']],'r12_last_change':dbg.get('r12_last_change'),'last_search':dbg.get('last_search'),'contracts':json.loads(b.lib.td_contract_json(b.handle))})
   if t%72==0:print(name,on,t,[f['money'] for f in post['farms']],flush=True)
  endpoint=world.observation();row={'enabled':on,'start_step':start,'ticks':n,'endpoint_cash':[f['money'] for f in endpoint['farms']],'own_cash':endpoint['farms'][seat]['money'],'opponent_cash':endpoint['farms'][1-seat]['money'],'margin':endpoint['farms'][seat]['money']-endpoint['farms'][1-seat]['money'],'wall_seconds':time.perf_counter()-tick0,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'actual_conditional_flow':world.totals,'actual_conditional_units':world.quantities,'endpoint_own_private':endpoint['private'],'endpoint_own_tiles':endpoint['farms'][seat]['tiles']}
  save(D/(name+f'_on{on}.json.gz'),{'summary':row,'first_observation':o,'first_debug':first_debug,'first_contract':first_contract,'flow_condition':flow,'daily':daily,'steps':records});both.append(row);b.close()
 case={'id':name,'start_step':start,'selection_reason':reason,'control':both[0],'candidate':both[1],'own_delta':both[1]['own_cash']-both[0]['own_cash'],'opponent_delta':both[1]['opponent_cash']-both[0]['opponent_cash'],'margin_delta':both[1]['margin']-both[0]['margin'],'not_real_game':True};summary['cases'].append(case);save(D/'SUMMARY.json',summary);print('DONE',name,case['own_delta'],case['margin_delta'],flush=True)
