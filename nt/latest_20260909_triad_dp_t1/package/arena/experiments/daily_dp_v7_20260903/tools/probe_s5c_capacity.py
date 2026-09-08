"""Paired current-observation plan alternatives; live policy unchanged."""
from pathlib import Path
import argparse,hashlib,json,shlex,subprocess,sys,sysconfig,time,zlib
E=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--compile-only',action='store_true');p.add_argument('--version',default='v1');a=p.parse_args()
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
build=json.loads((E/'native/build/build_receipt.json').read_text())
for rel,h in build['source_hashes'].items():assert sha(E/rel)==h
dst=E/('native/capacity_probe_build_'+a.version);src=E/'native/compile_capacity_probe.cpp'
if a.compile_only:
 dst.mkdir(exist_ok=False);inc=shlex.split(subprocess.check_output([sys.executable,'-m','pybind11','--includes'],text=True))
 binary=dst/('_dp7_capacity_probe'+sysconfig.get_config_var('EXT_SUFFIX'));cmd=['g++','-std=c++20','-O2','-fPIC','-shared',*inc,'-I'+str(E/'native'),str(src),str(E/'native/build/simulator.o'),'-fopenmp','-o',str(binary)]
 tic=time.perf_counter();r=subprocess.run(cmd,capture_output=True,text=True);(dst/'compile.log').write_text(r.stdout+r.stderr)
 if r.returncode:print(r.stderr[-6000:]);raise SystemExit(r.returncode)
 (dst/'source.cpp').write_bytes(src.read_bytes())
 (dst/'build_receipt.json').write_text(json.dumps(dict(production=build,source_sha256=sha(src),binary_sha256=sha(binary),command=cmd,seconds=time.perf_counter()-tic),indent=2));print('COMPILE_PASS');raise SystemExit(0)
br=json.loads((dst/'build_receipt.json').read_text());assert br['production']==build and br['source_sha256']==sha(src)
sys.path[:0]=[str(E/'native/build'),str(dst)];import _dp7_native as n;import _dp7_capacity_probe as probe
panel=json.loads((E/'receipts/s5b_eightway_N10_v1/results.json').read_text());assert panel['build']==build
cases=json.loads((E/'receipts/s5b_escape_triage_v1/acceptance.json').read_text())['rows']
out=E/('receipts/s5c_capacity_'+a.version);out.mkdir(exist_ok=False);rows=[];games=[];started=time.perf_counter()
for case in cases:
 label,seed,seat,opponent=[case[k] for k in ('label','seed','seat','opponent')]
 if opponent=='kaito_v58':
  asset=E/panel['identities'][opponent]['asset'];assert sha(asset)==panel['identities'][opponent]['asset_sha256'];rival=n.KaitoV58(json.loads(zlib.decompress(asset.read_bytes())));state=n.KaitoState()
  def other(env):return rival.act(env,1-seat,state)
 else:
  assert opponent=='yhay81_three_day';rival=n.ThreeDay()
  def other(env):return rival.act(env,1-seat)
 env=n.Env(seed);c=n.Controller(panel['configurations'][label]);sampled=set()
 while not env.done:
  day=env.step_count//24
  if max(case['escape_days'])-1<=day<=max(case['escape_days']) and c.debug()['phase']==2 and day not in sampled:
   # Do not inspect the previous day's leftover phase before new_day.
   if env.step_count%24:
    before=c.debug();r=probe.inspect(c,env,seat,True);assert before==c.debug();r.update(case=case);rows.append(r);sampled.add(day)
    (out/'progress.json').write_text(json.dumps(rows,indent=2));print(json.dumps(dict(label=label,seed=seed,seat=seat,step=r['step'],seconds=r['seconds'],trials=[{k:v for k,v in t.items() if k!='plans'} for t in r['trials']])),flush=True)
  own=c.act(env,seat);op=other(env);env.step([own,op] if seat==0 else [op,own])
 money=[f['money'] for f in env.observation(seat)['farms']];ref=next(r for r in panel['rows'] if (r['variant'],r['opponent'],r['seed'],r['seat'])==(label,opponent,seed,seat))
 assert money[seat]==ref['cash'] and money[1-seat]==ref['opponent_cash'];games.append(dict(case=case,money=money,control_equal=True))
(out/'acceptance.json').write_text(json.dumps(dict(status='PASS_OBSERVATION_ONLY_ALTERNATIVES',build=br,rows=rows,games=games,seconds=time.perf_counter()-started,caveat='Conditional day and residual predictions, not actual full-game improvement or promotion.'),indent=2));print('PASS_OBSERVATION_ONLY_ALTERNATIVES',flush=True)
