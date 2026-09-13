"""No matches: recorded public/self observation prefixes only, stop diverged policies.
One-step probes clone an exactly reconstructed r6-equivalent prefix before enabling
r11. No historical suffix is fed to a deviated context. No seed is sent to Agent.
"""
from pathlib import Path
import gzip,json,hashlib,ctypes,importlib.util,time,sys,statistics
import argparse
ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
C=Path(__file__).resolve().parents[1];H=C/'validation/history';D=args.out.resolve();D.mkdir(parents=True,exist_ok=False)
def load(p,n):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
m=load(C/'policy/agent.py','newcodec');cfg=json.loads((C/'policy/config.json').read_text());off=cfg|{'a06_r11_recovery':0}
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def funcs(a):
 for fn,rest,args in [('td_clone',ctypes.c_void_p,[ctypes.c_void_p]),('td_r11_enable',ctypes.c_int,[ctypes.c_void_p,ctypes.c_int]),('td_contract_json',ctypes.c_char_p,[ctypes.c_void_p])]:
  f=getattr(a.lib,fn);f.restype=rest;f.argtypes=args

def clone(a):
 z=m.Agent(cfg,C/'policy/a06.so');z.close();z.handle=a.lib.td_clone(a.handle);assert z.handle;z.last=a.last;z.seat=a.seat;funcs(z);assert z.lib.td_r11_enable(z.handle,1)==0;return z

def validate(a,o):
 assert set(a)=={'farmer','hands','market'};assert len(a['hands'])==len(o['farms'][o['player']]['hands']);assert len(a['market'])<=10
 units={'PASS','NORTH','SOUTH','EAST','WEST','DROP','PICKUP','PLACE','PLANT','WATER','HARVEST','FERTILIZE','DIG','BUILD_COOP','BUILD_PASTURE','FEED','COLLECT_FERTILIZER','CARE'}
 for x in [a['farmer'],*a['hands']]:assert isinstance(x,list) and x[0] in units
 for x in a['market']:assert isinstance(x,list) and x[0] in {'HIRE','BUY_LAND','BUY_SEED','BUY_PRODUCT','BUY_ANIMAL','SELL'}
 for x in [a['farmer'],*a['hands'],*a['market']]:
  if len(x)==3:assert type(x[2])==int and x[2]>0
 return True
summary={'scope':'recorded legal observations only; zero new games, zero seeds drawn','native_sha256':sha(C/'policy/a06.so'),'cases':[],'errors':[]}
for fn in ['author_A05_r4_2610020000_seat0','submission_56149565_2610020001_seat0']:
 p=H/(fn+'.replay.json.gz');r=json.loads(gzip.decompress(p.read_bytes()));seat=r['result']['seat'];assert len(r['actions'])==719
 track=m.Agent(off,C/'policy/a06.so');funcs(track);enabled=m.Agent(cfg,C/'policy/a06.so');funcs(enabled);live=True;prefix=0;first=None;probes=[];records=[];times=[];no_errors=True
 # Replay location selects an audit fixture, never passes the fixture identity to the strategy.
 chosen={166,167,168,169,170,171,216,217,218,219,220,221,222,223,224,225,226,227,228,229,230,231,232,233,234,235,236,237,238,239,240,241,246,290,360,384,552,696,710,718}
 for t in range(719):
  o=r['steps'][t][seat]['observation'];assert o['step']==t
  # The codec must ignore metadata that is not part of its allowed View.
  if t in {0,217,384}:
   extra=dict(o,seed=987654321,opponent_id='FORBIDDEN_TEST_VALUE',future_shops=['FORBIDDEN'],opponent_private={'shed':{'WOOL':999999}})
   assert list(m._pack(o))==list(m._pack(extra))
  probe=clone(track) if t in chosen else None
  before=json.loads(track.lib.td_contract_json(track.handle)) if t in chosen else None
  ts=time.perf_counter();a=track(o);times.append(time.perf_counter()-ts);validate(a,o)
  assert a==r['actions'][t][seat],f'disabled mismatch {fn} {t}'
  if live:
   ts=time.perf_counter();b=enabled(o);dt=time.perf_counter()-ts;validate(b,o)
   if b!=a:
    first={'step':t,'same_prior_action_count':prefix,'observation':o,'r6_action':a,'r11_action':b,'r11_debug':enabled.debug(),'r11_contract':json.loads(enabled.lib.td_contract_json(enabled.handle)),'seconds':dt}
    save(D/(fn+'_first_divergence.json'),first);live=False;enabled.close()
   else:prefix+=1
  if probe:
   ts=time.perf_counter();b=probe(o);dt=time.perf_counter()-ts;validate(b,o);dbg=probe.debug()
   item={'step':t,'historical_prefix_matches':t,'observation':o,'before_contract':before,'r6_action':a,'probe_action':b,'changed':a!=b,'seconds':dt,'debug':dbg,'after_contract':json.loads(probe.lib.td_contract_json(probe.handle))}
   probes.append(item);probe.close()
  records.append({'step':t,'action':a,'seconds':times[-1],'actual_cash_observed':o['farms'][seat]['money'],'actual_workers_observed':1+len(o['farms'][seat]['hands'])})
  if t%72==0:print(fn,t,'prefix',prefix,'probes',len(probes),flush=True)
 track.close();enabled.close()
 (D/(fn+'_cloned_probes.json.gz')).write_bytes(gzip.compress(json.dumps(probes).encode(),mtime=0))
 (D/(fn+'_disabled_prefix.json.gz')).write_bytes(gzip.compress(json.dumps(records).encode(),mtime=0))
 entry={'id':fn,'source_replay_sha256':sha(p),'recorded_control_cash_not_new':r['result']['cash'],'disabled_identical_actions':719,'root_enabled_matching_prefix':prefix,'enabled_first_divergence':first['step'] if first else None,'single_step_clones':len(probes),'clone_action_changes':sum(x['changed'] for x in probes),'clone_r11_proposals':sum(x['debug'].get('r11_offered',0)>0 for x in probes),'new_full_games':0,'disabled_call_seconds_sum':sum(times),'disabled_max_seconds':max(times),'max_probe_seconds':max(x['seconds'] for x in probes)}
 summary['cases'].append(entry);save(D/'SUMMARY.json',summary)
 print(json.dumps(entry),flush=True)
