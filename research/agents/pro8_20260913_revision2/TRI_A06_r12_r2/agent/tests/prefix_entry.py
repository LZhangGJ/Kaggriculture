"""Compare real entry points only on their common recorded history.

Each candidate stops at its first differing action. No suffix of a saved replay
is interpreted as the candidate's trajectory; this script runs zero games.
"""
from pathlib import Path
import argparse,copy,ctypes,gzip,hashlib,importlib.util,json,sys,time
p=argparse.ArgumentParser();p.add_argument('--parent',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True);p.add_argument('--feedback',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
def load(path,name):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
pm=load(a.parent,'prefix_parent');cm=load(a.candidate,'prefix_candidate');rows=[]
def shape(action,obs):
 assert isinstance(action,dict) and set(action)=={'farmer','hands','market'}
 assert isinstance(action['farmer'],list) and action['farmer']
 assert len(action['hands'])==len(obs['farms'][obs['player']]['hands'])
 assert len(action['market'])<=10
 for order in [action['farmer'],*action['hands'],*action['market']]:assert isinstance(order,list) and isinstance(order[0],str)
for case in json.loads((a.feedback/'SELECTED_CASES.json').read_text()):
 tr=json.load(gzip.open(a.feedback/case['trace'],'rt'));parent=pm.create_agent();candidate=cm.create_agent();matched=0;first=None;started=time.monotonic();calls=0;acts=hashlib.sha256()
 for i,obs in enumerate(tr['observations'][:719]):
  old=parent(copy.deepcopy(obs),tr['configuration']);assert old==tr['own_actions'][i],(case['id'],i,'parent identity failure')
  new=candidate(copy.deepcopy(obs),tr['configuration']);shape(new,obs);calls+=1
  acts.update(json.dumps(new,sort_keys=True).encode())
  if new!=old:
   def contract(obj):
    obj.lib.td_contract_json.argtypes=[ctypes.c_void_p];obj.lib.td_contract_json.restype=ctypes.c_char_p
    return json.loads(obj.lib.td_contract_json(obj.handle))
   first={'step':i,'day':obs['day'],'hour':obs['hour'],'parent':old,'candidate':new,'parent_debug':parent.debug(),'candidate_debug':candidate.debug(),'parent_contract':contract(parent),'candidate_contract':contract(candidate)};break
  matched+=1
 row={'case':case['id'],'historical_margin':case['row']['margin'],'calls':calls,'common_actions':matched,'first_divergence':first,'all_719_same':matched==719,'seconds':time.monotonic()-started,'prefix_action_sha256':acts.hexdigest(),'new_games':0}
 rows.append(row);parent.close();candidate.close();print(json.dumps({k:v for k,v in row.items() if k!='first_divergence'},ensure_ascii=False),flush=True)
 a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps({'kind':'common historical prefix only; zero new matches','parent_native_sha256':hashlib.sha256((a.parent.parent/'policy/a06.so').read_bytes()).hexdigest(),'candidate_native_sha256':hashlib.sha256((a.candidate.parent/'policy/a06.so').read_bytes()).hexdigest(),'cases':rows},indent=2))
