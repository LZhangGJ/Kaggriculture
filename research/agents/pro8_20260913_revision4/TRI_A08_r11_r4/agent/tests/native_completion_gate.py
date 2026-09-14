#!/usr/bin/env python3
"""Offline gate for the actual linked rolling-completion implementation.
No official-engine dependency; larger independent official tests are separate.
"""
from pathlib import Path
import argparse,ctypes,json,hashlib,importlib.util,copy,time
ROOT=Path(__file__).resolve().parents[1]
def load_codec():
 s=importlib.util.spec_from_file_location('completion_codec',ROOT/'policy/agent.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
codec=load_codec()
def synthetic(step=715,qty=2,cargo=None,shade=0):
 tiles=[[None for x in range(10)]for y in range(10)]
 tiles[4][6]={'kind':'PLANT','crop':'TOMATO','planted_day':10,'yield_units':qty,'watered_today':True,'consecutive_unwatered':0,'fertilized_until_day':30,'max_lifespan_step':-1}
 f={'money':50000,'tiles':tiles,'farmer':[5,4],'hands':[],'unlocked_quadrants':['NW','NE','SW','SE'],'hires_today':0}
 return {'step':step,'day':step//24,'hour':step%24,'player':0,'farms':[f,copy.deepcopy(f)],'private':{'shed':{},'seeds':{},'inventories':[cargo if cargo is not None else {'EGG':1}]},'market':{'inventory':{i:10000 for i in codec._ITEMS[:9]},'prices':dict(zip(codec._ITEMS[:9],[25,35,60,120,250,50,160,200,100]))},'town':{'unlocked_shops':[]}}
class Native:
 def __init__(self,path):
  self.lib=ctypes.CDLL(str(Path(path).resolve()));self.f=self.lib.td_completion_snapshot
  self.f.argtypes=[ctypes.POINTER(ctypes.c_double),ctypes.c_size_t,ctypes.POINTER(ctypes.c_int32),ctypes.c_size_t,ctypes.POINTER(ctypes.c_int32),ctypes.c_size_t,ctypes.c_double,ctypes.c_int];self.f.restype=ctypes.c_char_p
 def query(self,o,plans,pos=46,whole=((10,-1,1),),missing=None,shadow=2,mode=0):
  if missing is None:missing=whole
  pack=codec._pack(o);raw=[len(plans)]
  for p in plans:raw += [len(p)]+[v for x in p for v in x]
  jobs=[pos,len(whole)]+[v for x in whole for v in x]+[len(missing)]+[v for x in missing for v in x]
  rr=(ctypes.c_int32*len(raw))(*raw);jj=(ctypes.c_int32*len(jobs))(*jobs)
  r=json.loads(self.f(pack,len(pack),rr,len(rr),jj,len(jj),shadow,mode))
  if 'error' in r:raise AssertionError(r)
  return r

def run(lib,enabled=1):
 n=Native(lib);rows=[];checks=0
 def check(c,msg):
  nonlocal checks
  checks+=1
  if not c:raise AssertionError(msg)
 o=synthetic();old=[[[5,-1,1,45]]]
 r=n.query(o,old);rows.append({'name':'before_existing_final_DROP','result':r})
 if enabled:
  check(r['selected'],r);check([x[0] for x in r['best'][0]]==[3,10,4,5],r)
  check(r['after']['executed'] and not any(r['after']['stranded'][:9]),r)
  check(r['after']['sold'][2]==2 and r['after']['sold'][5]==1,r)
  check(r['after']['cash']>r['before']['cash']+2*(r['after']['work']-r['before']['work']),r)
 else:check(not r['selected'],r)
 for label,obs,work in [('one_tick_short',synthetic(716),2),('negative_net',synthetic(),10000)]:
  x=n.query(obs,old,shadow=work);rows.append({'name':label,'result':x});check(not x['selected'],x)
 # Close a genuinely open pre-existing route, not merely an empty old DROP.
 broken=[[[5,-1,1,45],[3,-1,1,-1],[10,-1,1,46]]]
 x=n.query(synthetic(714),broken,mode=1);rows.append({'name':'repair_existing_unclosed_tail','result':x})
 check(x['selected']==bool(enabled),x)
 if enabled:check(x['after']['sold'][2]==2 and not any(x['after']['stranded']),x)
 # The full actual recoordinate branch, not only the isolated helper.
 x=n.query(o,old,mode=2);rows.append({'name':'actual_recoordinate','result':x})
 if enabled:
  check(x['after']['sold'][2]==2 and not any(x['after']['stranded']),x)
 else:
  check(x['after']['stranded'][2]==2 and x['after']['sold'][2]==0,x)
 return {'status':'PASS','enabled':enabled,'checks':checks,'native_sha256':hashlib.sha256(Path(lib).read_bytes()).hexdigest(),'new_games':0,'cases':rows}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--library',required=True,type=Path);p.add_argument('--enabled',type=int,choices=[0,1],default=1);p.add_argument('--out',required=True,type=Path);a=p.parse_args();t=time.perf_counter();r=run(a.library,a.enabled);r['seconds']=time.perf_counter()-t;a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k!='cases'}))
