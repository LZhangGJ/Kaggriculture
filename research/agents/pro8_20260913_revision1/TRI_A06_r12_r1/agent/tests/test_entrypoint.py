"""Direct main.agent vs independent contexts; two seats and automatic new-game reset."""
import argparse,hashlib,importlib.util,json,time,resource
from pathlib import Path
R=Path(__file__).resolve().parents[1]
def module(label,path):
 s=importlib.util.spec_from_file_location(label,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def main():
 p=argparse.ArgumentParser();p.add_argument('--referee',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();host=module('entry_host',a.referee/'cpu_runtime.py');m=module('entry_under_test',R/'main.py');start=time.perf_counter();checks=0
 for seed in [2613091201,2613091202]:
  game=host.LocalGame(seed);independent=[m.create_agent(),m.create_agent()]
  for t in range(20):
   actions=[]
   for seat in [0,1]:
    obs=game.observation(seat);actual=m.agent(obs,game.configuration);expected=independent[seat](obs,game.configuration);assert actual==expected,(seed,t,seat);checks+=1;actions.append(actual)
   game.advance(actions)
  for x in independent:x.close()
 m.reset();assert not m._seats
 data={'scope':'80 legal direct-entry/interface comparisons over two 20-tick real-policy prefixes; not completed games','checks':checks,'automatic_step_zero_reset':True,'per_seat_isolation':True,'explicit_reset':True,'exceptions':0,'seconds':time.perf_counter()-start,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'main_sha256':hashlib.sha256((R/'main.py').read_bytes()).hexdigest(),'native_sha256':hashlib.sha256((R/'policy/a06.so').read_bytes()).hexdigest()}
 a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(data,indent=2));print(json.dumps(data))
if __name__=='__main__':main()
