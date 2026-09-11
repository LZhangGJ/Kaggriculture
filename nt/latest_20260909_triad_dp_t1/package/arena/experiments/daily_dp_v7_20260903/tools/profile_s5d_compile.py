from pathlib import Path
import argparse, hashlib, json, shlex, subprocess, sys, sysconfig, time, zlib
E=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--version',default='v1');p.add_argument('--runtime',default='v1');p.add_argument('--configs',default=str(E/'profiles/s5d/configs.json'));p.add_argument('--reuse-probe-version');a=p.parse_args()
out=E/('receipts/s5d_cost_'+a.version);out.mkdir(exist_ok=False)
build=json.loads((E/'native/build/build_receipt.json').read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for rel,h in build['source_hashes'].items():assert sha(E/rel)==h
src=E/'native/compile_cost_probe.cpp';(out/'source.cpp').write_bytes(src.read_bytes())
inc=shlex.split(subprocess.check_output([sys.executable,'-m','pybind11','--includes'],text=True))
binary=out/('_dp7_compile_cost'+sysconfig.get_config_var('EXT_SUFFIX'))
cmd=['g++','-std=c++20','-O3','-march=native','-fPIC','-shared',*inc,'-I'+str(E/'native'),str(src),str(E/'native/build/simulator.o'),'-fopenmp','-o',str(binary)]
tic=time.perf_counter()
if a.reuse_probe_version:
 previous=E/('receipts/s5d_cost_'+a.reuse_probe_version);br=json.loads((previous/'build.json').read_text());oldbin=previous/binary.name
 assert br['production']==build and br['source_hash']==sha(src) and br['binary_hash']==sha(oldbin)
 binary.write_bytes(oldbin.read_bytes());br={**br,'reused_from':str(previous)}
else:
 r=subprocess.run(cmd,capture_output=True,text=True);(out/'compile.log').write_text(r.stdout+r.stderr)
 if r.returncode:print(r.stderr);raise SystemExit(r.returncode)
 br=dict(production=build,source_hash=sha(src),binary_hash=sha(binary),seconds=time.perf_counter()-tic,command=cmd)
(out/'build.json').write_text(json.dumps(br,indent=2))
sys.path[:0]=[str(E/'native/build'),str(out)];import _dp7_native as n;import _dp7_compile_cost as probe
failure=json.loads((E/f'receipts/s5d_runtime_{a.runtime}/acceptance.json').read_text());label=failure['label'];seat=failure['seat'];seed=failure['seed'];step=failure['step']
cfg=json.loads(Path(a.configs).read_text());env=n.Env(seed);c=n.Controller(cfg[label]);op=n.G001(json.loads(zlib.decompress((E/'native/g003_frozen.json.zlib').read_bytes())));state=n.G001State()
while env.step_count<step:
 own=c.act(env,seat);other=op.act(env,1-seat,state);env.step([own,other] if seat==0 else [other,own])
assert json.loads(json.dumps(env.observation(seat)))==failure['observation'], 'must reproduce the exact stopped state'
before=c.debug();result=probe.newday(c,env,seat) if step%24==0 else [probe.inspect(c,env,seat,bounded) for bounded in (False,True)];assert c.debug()==before
(out/'acceptance.json').write_text(json.dumps(dict(status='PASS_READ_ONLY_COST',label=label,seed=seed,seat=seat,step=step,build=br,result=result),indent=2));print(json.dumps(result,indent=2),flush=True)
