"""Focused funding / labor / whole-land-plan tests; ZERO new matches.
Saved states are independent test inputs. Conditional execution stops before
midnight, has no other market orders and uses only current known shops. It never
splices recorded future observations after an action changes.
"""
from pathlib import Path
import argparse,copy,ctypes,gzip,hashlib,importlib.util,json,resource,shutil,subprocess,sys,time
R=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--cxx',default=shutil.which('g++'));p.add_argument('--out',type=Path,default=R/'build/r3_units');p.add_argument('--feedback',type=Path,default=R/'evidence/r3_feedback');p.add_argument('--reuse-compiled',type=Path);p.add_argument('--quick',action='store_true');a=p.parse_args();a.out=a.out.resolve();a.out.mkdir(parents=True,exist_ok=True);tick=time.perf_counter()
def save(path,x):
 path=Path(path);text=json.dumps(x,ensure_ascii=False,separators=(',',':'))+'\n'
 if path.suffix=='.gz':
  with gzip.open(path,'wt') as f:f.write(text)
 else:path.write_text(text)
lib=a.out/'r3_checks.so'
if a.reuse_compiled:shutil.copy2(a.reuse_compiled,lib)
else:
 flags=json.loads((R/'COMPILER_FLAGS.json').read_text());cmd=[a.cxx,*flags,'-I'+str(R/'policy'),str(R/'tests/r3/r3_checks.cpp'),str(R/'policy/executor/vendor/simulator.cpp'),'-o',str(lib)];save(a.out/'compile.command.json',cmd);tt=time.perf_counter()
 q=subprocess.run(['/usr/bin/time','-v','-o',str(a.out/'compile.time'),'timeout','-k','5s','150s',*cmd],stdout=open(a.out/'compile.stdout','w'),stderr=open(a.out/'compile.stderr','w'));save(a.out/'compile.receipt.json',{'exit':q.returncode,'seconds':time.perf_counter()-tt,'compiler':subprocess.check_output([a.cxx,'--version'],text=True)})
 if q.returncode:raise RuntimeError((a.out/'compile.stderr').read_text())
sp=importlib.util.spec_from_file_location('r3_unit_main',R/'main.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
from official_prefix import load_referee,decode_action,own_signature
runtime,engine=load_referee(a.feedback)
ag=m.create_agent(binary_path=lib)
for name in ['td_r3_state','td_r3_synthetic','td_r3_branch']:
 fn=getattr(ag.lib,name);fn.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_double),ctypes.c_size_t]+([ctypes.c_int,ctypes.c_int] if name=='td_r3_branch' else []);fn.restype=ctypes.c_char_p

fixtures=json.loads((R/'evidence/parent_service_fixtures.json').read_text())
load_fixture=ag.lib.td_r3_loadfixture;load_fixture.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_double),ctypes.c_size_t];load_fixture.restype=ctypes.c_int

def install_parent_fixture(f):
 z=list(f['settings'])+[f[k] for k in ('day','phase','step','planned_land','feed_stock_target','crop_service_day','animal_service_day')]
 for k in ('daily_need','prepared_seed_need','crop_birth','crop_kind','crop_water','crop_fertilize','triad_crop_age','plant_not_before','service_feed','service_care'):z+=f[k]
 z+=[len(f['target'])]
 for x in f['target']:z+=x
 z+=[len(f['remaining_plans'])]
 for plan in f['remaining_plans']:
  z+=[len(plan)]
  for atom in plan:z+=atom
 packed=(ctypes.c_double*len(z))(*z);assert load_fixture(ag.handle,packed,len(z))==0,ag.debug()

def call(name,ob,*args):
 z=m.codec._pack(ob);raw=getattr(ag.lib,name)(ag.handle,z,len(z),*args).decode()
 if raw.startswith('ERROR'):
  save(a.out/'failure.json',{'api':name,'args':args,'step':ob['step'],'error':raw});raise RuntimeError(raw)
 return json.loads(raw)

def official_sequence(trace,initial,branch,root_compare):
 ob=copy.deepcopy(initial);seat=ob['player'];cfg=runtime.AttrDict(trace['configuration']);env=runtime.AttrDict(configuration=cfg);results=[]
 real=m.create_agent() if root_compare else None;cloned=None
 try:
  for row in branch['ticks']:
   assert ob['step']==row['step'];act=decode_action(row['action'],m.codec)
   if real is not None:
    actual=real(ob,trace['configuration'])
    if actual!=act:raise AssertionError(('test native / production root mismatch',ob['step'],actual,act))
    if cloned is not None:
     other=cloned(ob,trace['configuration']);assert other==actual,('clone callback mismatch',ob['step'])
    elif len(results)==0:
     cloned=m.create_agent();cloned.close();real.lib.td_clone.argtypes=[ctypes.c_void_p];real.lib.td_clone.restype=ctypes.c_void_p;cloned.handle=real.lib.td_clone(real.handle);assert cloned.handle;cloned.last=real.last;cloned.seat=real.seat
   farm,priv=ob['farms'][seat],ob['private'];units=[act['farmer'],*act['hands']];demand={}
   for x in units:
    if x[0]=='PLANT':demand[x[1]]=demand.get(x[1],0)+1
   blocked={k for k,v in demand.items() if v>priv['seeds'].get(k,0)}
   for u,x in enumerate(units):
    if x[0]=='PLANT' and x[1] in blocked:continue
    engine._apply_unit_action(farm,priv,u,x,10,ob['day'],24,int(cfg.get('shedCapacity',100)))
   state=[runtime.AttrDict(observation=runtime.AttrDict(farms=ob['farms'],market=ob['market'],town=ob['town'],private=priv if k==seat else {}),action=act if k==seat else {}) for k in range(2)]
   engine._process_market(state,env)
   official=own_signature(ob,m.codec);expected=row['prefix_signature'];diff=[(i,x,y) for i,(x,y) in enumerate(zip(official,expected)) if x!=y]
   if len(official)!=len(expected) or diff:
    save(a.out/'official_failure.json',{'game_id':trace['game_id'],'step':ob['step'],'mode':branch['mode'],'differences':diff,'official':official,'native':expected,'action':act});raise AssertionError(('official prefix mismatch',ob['step'],diff[:5]))
   results.append({'step':ob['step'],'values_checked':len(official),'cash_after_prefix':farm['money'],'hands_after_prefix':len(farm['hands']),'pass':True})
   engine._town_consume(env,state,ob['step'])
   for f in ob['farms']:engine._decay_plants(f,ob['step'])
   ob['step']+=1;ob['day']=ob['step']//24;ob['hour']=ob['step']%24
   assert ob['hour']!=0,'test crossed midnight'
  assert branch['final_risk']==sum(isinstance(t,dict) and t.get('kind')=='PLANT' and t.get('consecutive_unwatered',0)>=1 and not t.get('watered_today',False) for row in ob['farms'][seat]['tiles'] for t in row)
  return results
 finally:
  if real:real.close()
  if cloned:cloned.close()

paths=sorted((a.feedback/'own_traces').glob('*.gz'))+sorted((R/'evidence/own_visible/own_traces').glob('*.gz'))
assert paths,'missing fixtures'
state_rows=[];branch_summaries=[];synthetic=None;prefixes=0;root_calls=0;clone_calls=0;all_case_ids=[]
for path in paths:
 t=json.load(gzip.open(path,'rt'));all_case_ids.append(t['game_id']);states=[];branches=[]
 if synthetic is None:synthetic=call('td_r3_synthetic',t['observations'][216]);save(a.out/'synthetic.json',synthetic)
 indices=[0,216,312,696] if a.quick else list(range(0,719,24))
 for index in indices:
  q=call('td_r3_state',t['observations'][index]);q['game_id']=t['game_id'];states.append(q);state_rows.append(q)
 # Actual production root and copy/callback equivalence on bounded same-day
 # execution at opening and explicitly selected diagnostic maintenance days.
 branch_indices=[0,216,312] if path.parent.parent==a.feedback.resolve() else [0,216]
 if a.quick:branch_indices=[216]
 for index in branch_indices:
  ob=t['observations'][index];q=call('td_r3_branch',ob,0,0);q['scope']='Conditional own-only same-day test, no opponent orders. Cold real controller from saved legal state, not continuation of source match.';q['official_checks']=official_sequence(t,ob,q,True);branches.append(q);prefixes+=len(q['ticks']);root_calls+=len(q['ticks']);clone_calls+=max(0,len(q['ticks'])-1)
 if t['game_id']=='submission_56149565_389573676_seat0':
  for index,desired in [(219,7),(315,8)]:
   for mode in [1,2]:
    ob=t['observations'][index];q=call('td_r3_branch',ob,mode,desired);q['scope']='Isolated service fixture from observed farm/resources/current crew; targets restricted to incumbents; fresh current-crew compile; no opponent orders, no new projects, no saved future.';q['official_checks']=official_sequence(t,ob,q,False);branches.append(q);prefixes+=len(q['ticks'])

 if t['game_id']=='submission_56149565_389573676_seat0':
  for f in fixtures:
   for mode in (3,4):
    ag.reset();install_parent_fixture(f);ob=t['observations'][f['next_observation_index']]
    q=call('td_r3_branch',ob,mode,f['desired_hands']);q['scope']='Parent-exported remaining routes, targets, selected services and reservations at the actual later-cash observation. Only labor recovery differs between mode3/4; fresh project admission disabled in both; own-only conditional continuation, not original game.'
    q['official_checks']=official_sequence(t,ob,q,False);branches.append(q);prefixes+=len(q['ticks'])
  ag.reset()
 save(a.out/(t['game_id']+'.json.gz'),{'fixture':str(path.relative_to(R)) if path.is_relative_to(R) else str(path),'states':states,'branches':branches})
 for q in branches:branch_summaries.append({'game_id':t['game_id'],**{k:v for k,v in q.items() if k not in ('ticks','official_checks')}})
 print(t['game_id'],'states',len(states),'branches',len(branches),flush=True)
ag.close()
summary={'scope':'Independent saved-state invariants; conditional same-day execution; official own prefix and packaged root equivalence. No new games, no win rate, no counterfactual final cash.','new_games':0,'case_ids':all_case_ids,'independent_states':len(state_rows),'whole_plan_expansion_comparisons':sum(x['compare'] for x in state_rows),'positive_expansions_kept':sum(x['retained_expansion'] for x in state_rows),'nonpositive_expansions_deferred':sum(x['deferred_expansion'] for x in state_rows),'funding_reordered_states':sum(x['funding_reordered'] for x in state_rows),'queue_orders_checked':sum(x['queue_size'] for x in state_rows),'synthetic_positive_negative_cases':len(synthetic['cases']),'price_floor_regressions':synthetic['floor_crossing_checks']+1,'conditional_branches':len(branch_summaries),'official_prefix_passes':prefixes,'real_root_calls_matched':root_calls,'cloned_controller_calls_matched':clone_calls,'test_native_sha256':hashlib.sha256(lib.read_bytes()).hexdigest(),'production_native_sha256':hashlib.sha256((R/'policy/a06.so').read_bytes()).hexdigest(),'seconds':time.perf_counter()-tick,'rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'branches':branch_summaries}
assert summary['positive_expansions_kept']>0 and summary['nonpositive_expansions_deferred']>0 and summary['funding_reordered_states']>0
save(a.out/'STATES.json',state_rows);save(a.out/'SUMMARY.json',summary);print(json.dumps({k:v for k,v in summary.items() if k!='branches'}),flush=True)

# Source transport: inherited tests above are unchanged.
subprocess.run([sys.executable,str(R/"tests/r4/run_units.py"),"--cxx",a.cxx],cwd=R,check=True)
