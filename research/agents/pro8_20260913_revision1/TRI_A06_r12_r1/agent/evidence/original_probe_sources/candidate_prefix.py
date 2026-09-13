"""Stop each candidate historical prefix at its FIRST action divergence.
Saved suffixes are never used to estimate candidate outcomes.
"""
from pathlib import Path
import importlib.util,json,gzip,time,resource
R=Path(__file__).resolve().parents[1]
def load(label,path):
 s=importlib.util.spec_from_file_location(label,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
cmod=load('candidate_main',R/'candidate/main.py');pmod=load('parent_main',R/'input/agent/main.py')
summary={'scope':'legal historical common prefixes only; zero candidate game outcomes','cases':[]};start=time.perf_counter()
for cp in json.load(open(R/'input/SELECTED_CASES.json')):
 r=json.loads(gzip.decompress((R/'input/replays'/(cp['id']+'.json.gz')).read_bytes()));seat=r['result']['seat'];p=pmod.create_agent();c=cmod.create_agent();case={'id':cp['id'],'historical_margin':cp.get('margin'),'compared':0,'matched':0,'first_difference':None};t0=time.perf_counter()
 for t in range(719):
  o=r['steps'][t][seat]['observation'];pa=p(o);assert pa==r['actions'][t][seat],(cp['id'],t,'parent mismatch');ca=c(o);case['compared']+=1
  if pa==ca:case['matched']+=1;continue
  case['first_difference']=t;case['day']=t//24;case['hour']=t%24
  data={'id':cp['id'],'step':t,'scope':'first divergence; NO suffix evaluated','legal_observation':o,'parent_action':pa,'candidate_action':ca,'parent_debug':p.debug(),'candidate_debug':c.debug()}
  (R/'logs'/(cp['id']+'_first_difference.json.gz')).write_bytes(gzip.compress(json.dumps(data).encode(),mtime=0));break
 case.update(seconds=time.perf_counter()-t0,all_719_actions_identical=case['matched']==719);summary['cases'].append(case);p.close();c.close();print(json.dumps(case),flush=True);del r
summary.update(seconds=time.perf_counter()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
(R/'logs/candidate_prefix.json').write_text(json.dumps(summary,indent=2))
