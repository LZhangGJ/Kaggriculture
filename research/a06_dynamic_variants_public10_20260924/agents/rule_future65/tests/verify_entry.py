"""Frozen-library/root-entry regression. Saved prefixes stop before any divergence.
Seat relabeling is an entry isolation test, not a new match or legal future.
"""
from pathlib import Path
import argparse, copy, gzip, hashlib, importlib.util, json, resource, time
p=argparse.ArgumentParser()
p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
p.add_argument('--evidence',type=Path)
p.add_argument('--out',type=Path,required=True)
p.add_argument('--expected-sha',default='ed0d275ca9caf13fdaaf2c833e164c5b8c1ccf313dbf96296fb9f7896254ab90')
a=p.parse_args(); r=a.root.resolve(); evidence=(a.evidence or r/'feedback').resolve(); t=time.monotonic()
trace_dir=evidence/'own_traces'
if (trace_dir/'candidate_r3').is_dir():trace_dir=trace_dir/'candidate_r3'
def sha(f):return hashlib.sha256(f.read_bytes()).hexdigest()
assert sha(r/'policy/a06.so')==a.expected_sha
build=json.loads((r/'policy/a06.BUILD.json').read_text())
for name,h in build['all_build_inputs'].items():assert sha(r/name)==h, name
s=importlib.util.spec_from_file_location('delivery_root',r/'main.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
calls=0;cases=[];max_call=0.0;resets=0
for path in sorted(trace_dir.glob('*.json.gz')):
 tr=json.loads(gzip.decompress(path.read_bytes())); m.reset(); n=0
 for step,obs in enumerate(tr['observations'][:48]):
  for mirror in (False,True):
   q=copy.deepcopy(obs)
   if mirror:q['farms'].reverse();q['player']=1-q['player']
   tt=time.monotonic();out=m.agent(q,tr['configuration']);max_call=max(max_call,time.monotonic()-tt)
   assert out==tr['own_actions'][step],(path.name,step,mirror)
   n+=1;calls+=1
 # Step zero resets an existing per-seat context, without explicit reset first.
 for mirror in (False,True):
  q=copy.deepcopy(tr['observations'][0])
  if mirror:q['farms'].reverse();q['player']=1-q['player']
  assert m.agent(q,tr['configuration'])==tr['own_actions'][0];resets+=1
 m.reset();assert not m._seats
 cases.append({'case':tr['game_id'],'prefix_and_relabel_calls':n})
assert len(cases)==4 and calls==384 and resets==8
negatives=0
try:m.agent({'player':2})
except ValueError:negatives+=1
else:raise AssertionError('invalid seat was silently accepted')
try:m.create_agent(r/'does_not_exist.so')
except FileNotFoundError:negatives+=1
else:raise AssertionError('missing native was silently accepted')
obj=m.create_agent();tr=json.loads(gzip.decompress(next(trace_dir.glob('*.json.gz')).read_bytes()))
obj(copy.deepcopy(tr['observations'][0]),tr['configuration']); obj(copy.deepcopy(tr['observations'][1]),tr['configuration'])
try:obj(copy.deepcopy(tr['observations'][1]),tr['configuration'])
except ValueError:negatives+=1
else:raise AssertionError('nonmonotonic state was accepted')
obj.close();assert not obj.handle
m.reset()
report={'scope':'Entry and matching-build checks; zero games; 48-step saved prefixes before divergence, and synthetic seat relabeling only',
 'native_sha256':a.expected_sha,'build_inputs_checked':len(build['all_build_inputs']),
 'prefix_and_relabel_calls':calls,'step_zero_resets':resets,'negative_checks':negatives,
 'cases':cases,'seconds':time.monotonic()-t,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
 'max_call_seconds':max_call,'all_passed':True}
a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,indent=2));print(json.dumps(report))
