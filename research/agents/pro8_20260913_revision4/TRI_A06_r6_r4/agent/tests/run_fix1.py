"""Focused fix1 checks. Saved prefixes and same-day own-only branches are NOT games."""
from pathlib import Path
import argparse,copy,ctypes,gzip,hashlib,importlib.util,json,sys,time,subprocess
R=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--feedback',type=Path,default=R/'evidence/closed_loop_feedback');p.add_argument('--checks',type=Path,default=R/'validation_fix1/checks.so');p.add_argument('--out',type=Path,default=R/'validation_fix1/focused');a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
s=importlib.util.spec_from_file_location('fix1_root',R/'main.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
from official_prefix import load_referee,own_signature,decode_action
runtime,engine=load_referee(a.feedback)
check=m.create_agent(binary_path=a.checks)
for name,extra in [('td_fix1_tests',[]),('td_r3_branch',[ctypes.c_int,ctypes.c_int])]:
 f=getattr(check.lib,name);f.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_double),ctypes.c_size_t,*extra];f.restype=ctypes.c_char_p
def invoke(name,ob,*xs):
 z=m.codec._pack(ob);raw=getattr(check.lib,name)(check.handle,z,len(z),*xs).decode();assert not raw.startswith('ERROR'),raw;return json.loads(raw)
def save(n,j):
 (a.out/n).write_text(json.dumps(j,indent=2))
traces=sorted((a.feedback/'own_traces/candidate_r3').glob('*.gz'));assert len(traces)==9
principal=json.load(gzip.open(next(t for t in traces if '201542742' in t.name),'rt'))
synth=invoke('td_fix1_tests',principal['observations'][600]);save('synthetic.json',synth)
# Stop immediately at the first policy divergence. Never feed its stored future.
prefixes=[]
for path in traces:
 t=json.load(gzip.open(path,'rt'));m.reset();eq=0;rec=[]
 for i,ob in enumerate(t['observations'][:-1]):
  actual=m.agent(copy.deepcopy(ob),t['configuration']);same=actual==t['own_actions'][i];rec.append({'step':i,'same':same,'actual':actual,'source':t['own_actions'][i] if not same else None})
  if not same:break
  eq+=1
 m.reset();prefixes.append({'case':t['game_id'],'calls':len(rec),'same':eq,'first_difference':None if eq==719 else i});
 with gzip.open(a.out/(path.name),'wt') as f:json.dump({'scope':'Saved own prefix; stop at first differing action. No candidate future or game.','rows':rec},f)
 print(prefixes[-1],flush=True)
save('prefixes.json',prefixes)
# Official own-only continuous same-day check, starting independently at a real saved state.
branches=[]
for name,index in [('submission_56149565_201542742_seat0',600),('submission_56149565_389573676_seat0',216)]:
 t=json.load(gzip.open(next(x for x in traces if x.stem==name+'.json'),'rt'));initial=copy.deepcopy(t['observations'][index]);check.reset();q=invoke('td_r3_branch',initial,0,0)
 ob=copy.deepcopy(initial);seat=ob['player'];env=runtime.AttrDict(configuration=runtime.AttrDict(t['configuration']));m.reset();audit=[];fert=0;hire=0
 for row in q['ticks']:
  act=decode_action(row['action'],m.codec);actual=m.agent(copy.deepcopy(ob),t['configuration']);assert actual==act,('root/check mismatch',ob['step'])
  farm,pr=ob['farms'][seat],ob['private'];beforefert=[t.get('fertilized_until_day',-1) if isinstance(t,dict) else -1 for rr in farm['tiles'] for t in rr];beforehands=len(farm['hands'])
  demand={};units=[act['farmer'],*act['hands']]
  for x in units:
   if x[0]=='PLANT':demand[x[1]]=demand.get(x[1],0)+1
  blocked={k for k,v in demand.items() if v>pr['seeds'].get(k,0)}
  for u,x in enumerate(units):
   if x[0]=='PLANT' and x[1] in blocked:continue
   engine._apply_unit_action(farm,pr,u,x,10,ob['day'],24,int(env.configuration.get('shedCapacity',100)))
  afterfert=[t.get('fertilized_until_day',-1) if isinstance(t,dict) else -1 for rr in farm['tiles'] for t in rr];fert+=sum(y>x for x,y in zip(beforefert,afterfert))
  states=[runtime.AttrDict(observation=runtime.AttrDict(farms=ob['farms'],market=ob['market'],town=ob['town'],private=pr if p==seat else {}),action=act if p==seat else {}) for p in range(2)]
  engine._process_market(states,env);hire+=len(farm['hands'])-beforehands;got=own_signature(ob,m.codec);assert got==row['prefix_signature'],('official prefix',ob['step'])
  audit.append({'step':ob['step'],'actual_root':act,'cash':farm['money'],'hands':len(farm['hands']),'fert_stock':pr['shed']['FERTILIZER'],'pass':True})
  engine._town_consume(env,states,ob['step'])
  for ff in ob['farms']:engine._decay_plants(ff,ob['step'])
  ob['step']+=1;ob['day']=ob['step']//24;ob['hour']=ob['step']%24;assert ob['hour']!=0
 m.reset();q['actual_official_fertilized']=fert;q['actual_official_hired']=hire;q['root_official_checks']=audit;q['scope']='Independent cold start at saved real state, only our orders, current shops, stops before midnight; no game or causal final money.'
 save(name+'_branch_'+str(index)+'.json',q);branches.append({'case':name,'start':index,'steps':len(audit),'fertilized':fert,'hired':hire,'root_matches':len(audit),'official_matches':len(audit),'market':audit[0]['actual_root']['market']});print(branches[-1],flush=True)
check.close();save('SUMMARY.json',{'new_games':0,'synthetic_cases':synth['case_count'],'prefixes':prefixes,'branches':branches,'native_sha256':hashlib.sha256((R/'policy/a06.so').read_bytes()).hexdigest()})
