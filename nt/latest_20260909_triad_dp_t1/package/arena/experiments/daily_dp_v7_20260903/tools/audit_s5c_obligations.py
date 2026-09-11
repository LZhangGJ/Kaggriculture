"""Unchanged failure replays, with read-only task/plan snapshots; not Oracle."""
from pathlib import Path
import argparse,gzip,hashlib,json,shlex,subprocess,sys,sysconfig,time,zlib
E=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--compile-only',action='store_true');p.add_argument('--version',default='v1');a=p.parse_args()
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
build=json.loads((E/'native/build/build_receipt.json').read_text())
for rel,h in build['source_hashes'].items():assert sha(E/rel)==h
dst=E/('native/obligation_probe_build_'+a.version);src=E/'native/execution_obligation_probe.cpp'
if a.compile_only:
 dst.mkdir(exist_ok=False);inc=shlex.split(subprocess.check_output([sys.executable,'-m','pybind11','--includes'],text=True))
 binary=dst/('_dp7_obligation_probe'+sysconfig.get_config_var('EXT_SUFFIX'))
 cmd=['g++','-std=c++20','-O2','-fPIC','-shared',*inc,'-I'+str(E/'native'),str(src),str(E/'native/build/simulator.o'),'-fopenmp','-o',str(binary)]
 t=time.perf_counter();r=subprocess.run(cmd,capture_output=True,text=True);(dst/'compile.log').write_text(r.stdout+r.stderr)
 if r.returncode:print(r.stderr[-6000:]);raise SystemExit(r.returncode)
 (dst/'build_receipt.json').write_text(json.dumps(dict(production=build,probe_sha256=sha(src),binary_sha256=sha(binary),command=cmd,seconds=time.perf_counter()-t),indent=2));print('COMPILE_PASS');raise SystemExit(0)
receipt=json.loads((dst/'build_receipt.json').read_text());assert receipt['production']==build and receipt['probe_sha256']==sha(src)
sys.path[:0]=[str(E/'native/build'),str(dst)];import _dp7_native as n;import _dp7_obligation_probe as probe
panel=json.loads((E/'receipts/s5b_eightway_N10_v1/results.json').read_text());assert panel['build']==build
cases=json.loads((E/'receipts/s5b_escape_triage_v1/acceptance.json').read_text())['rows']
out=E/('receipts/s5c_obligations_'+a.version);out.mkdir(exist_ok=False);rows=[];started=time.perf_counter()
for case in cases:
 label=case['label'];seed=case['seed'];seat=case['seat'];opponent=case['opponent']
 if opponent=='kaito_v58':
  asset=E/panel['identities'][opponent]['asset'];assert sha(asset)==panel['identities'][opponent]['asset_sha256']
  rival=n.KaitoV58(json.loads(zlib.decompress(asset.read_bytes())));state=n.KaitoState()
  def act(env):return rival.act(env,1-seat,state)
 else:
  assert opponent=='yhay81_three_day';rival=n.ThreeDay()
  def act(env):return rival.act(env,1-seat)
 env=n.Env(seed);ctl=n.Controller(panel['configurations'][label]);trace=[];first=max(0,min(case['escape_days'])-3)*24;last=(max(case['escape_days'])+1)*24
 while not env.done:
  step=env.step_count;selected=first<=step<last
  if selected:before=env.observation(seat);debug=ctl.debug();pre=probe.inspect(ctl,env,seat,True);assert debug==ctl.debug()
  own=ctl.act(env,seat)
  if selected:post=probe.inspect(ctl,env,seat,False)
  other=act(env);env.step([own,other] if seat==0 else [other,own])
  if selected:trace.append(dict(step=step,before=before,pre=pre,post=post,own=own,other=other,after=env.observation(seat)))
 money=[f['money'] for f in env.observation(seat)['farms']];ref=next(r for r in panel['rows'] if (r['variant'],r['opponent'],r['seed'],r['seat'])==(label,opponent,seed,seat))
 assert money[seat]==ref['cash'] and money[1-seat]==ref['opponent_cash']
 path=out/f'{label}_{opponent}_{seed}_{seat}.json.gz'
 with gzip.open(path,'wt') as f:json.dump(dict(case=case,trace=trace),f)
 row=dict(case=case,trace=path.name,sha256=sha(path),steps=len(trace),control_equal=True);rows.append(row)
 (out/'progress.json').write_text(json.dumps(rows,indent=2));print(json.dumps(dict(label=label,seed=seed,seat=seat,cash=money[seat],control_equal=True)),flush=True)
(out/'acceptance.json').write_text(json.dumps(dict(status='PASS_UNCHANGED_FAILURE_TRACE',build=receipt,rows=rows,seconds=time.perf_counter()-started),indent=2));print('PASS_UNCHANGED_FAILURE_TRACE',flush=True)
