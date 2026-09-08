"""Build/run isolated S4W prototype; never rewrites the production module."""
from pathlib import Path
import argparse,gzip,hashlib,json,shlex,subprocess,sys,sysconfig,time,zlib
E=Path(__file__).resolve().parents[1]
cli=argparse.ArgumentParser();cli.add_argument('--compile-only',action='store_true');cli.add_argument('--build',default='native/service_replacement_probe_build_v1');cli.add_argument('--out',default='receipts/s4w_replacement_probe_v1');a=cli.parse_args()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
build=json.loads((E/'native/build/build_receipt.json').read_text())
for rel,h in build['source_hashes'].items():assert sha(E/rel)==h
dst=E/a.build;source=[E/'native/service_replacement_prototype.hpp',E/'native/service_replacement_probe.cpp',E/'native/test_service_replacement.cpp']
if a.compile_only:
    dst.mkdir(exist_ok=False)
    inc=shlex.split(subprocess.check_output([sys.executable,'-m','pybind11','--includes'],text=True));binary=dst/('_dp7_replacement_probe'+sysconfig.get_config_var('EXT_SUFFIX'))
    commands=[['g++','-std=c++20','-O2','-I'+str(E/'native'),str(source[2]),str(E/'native/vendor/simulator.cpp'),'-o',str(dst/'test')],['g++','-std=c++20','-O2','-fPIC','-shared',*inc,'-I'+str(E/'native'),str(source[1]),str(E/'native/vendor/simulator.cpp'),'-o',str(binary)]]
    for i,cmd in enumerate(commands):
        r=subprocess.run(cmd,capture_output=True,text=True);(dst/f'compile_{i}.log').write_text(r.stdout+r.stderr)
        if r.returncode:print(r.stderr[-4000:]);raise SystemExit(r.returncode)
    (dst/'build_receipt.json').write_text(json.dumps(dict(build=build,source_hashes={str(p.relative_to(E)):sha(p) for p in source},binary_sha256=sha(binary),test_sha256=sha(dst/'test'),commands=commands),indent=2));print('COMPILE_ONLY_PASS');raise SystemExit(0)
frozen=json.loads((dst/'build_receipt.json').read_text());assert frozen['build']==build
for rel,h in frozen['source_hashes'].items():assert sha(E/rel)==h
out=E/a.out;out.mkdir(exist_ok=False);tic=time.perf_counter()
r=subprocess.run([str(dst/'test')],capture_output=True,text=True);(out/'mechanism.log').write_text(r.stdout+r.stderr)
if r.returncode:
    (out/'failure.json').write_text(json.dumps(dict(stage='mechanisms',stderr=r.stderr,build=frozen),indent=2));print(r.stderr);raise SystemExit(r.returncode)
mechanisms=json.loads(r.stdout);print(json.dumps(mechanisms),flush=True)
sys.path.insert(0,str(E/'native/build'));sys.path.insert(0,str(dst));import _dp7_native as n;import _dp7_replacement_probe as probe
panel=json.loads((E/'receipts/s4u_fiveway_N50_v1/results.json').read_text());identity=panel['identities']['g001'];asset=E/identity['asset'];assert sha(asset)==identity['asset_sha256']
rival=n.G001(json.loads(zlib.decompress(asset.read_bytes())));rows=[];games=[]
refs={ (r['seed'],r['seat']):r for r in json.load(gzip.open(E/'receipts/s4u_pool_audit_N50_v1/full_chain_autonomous_g001.json.gz','rt'))['rows'] }
for seed in (20262703,20262736):
    for seat in (0,1):
        env=n.Env(seed);c=n.Controller(panel['configurations']['full_chain_autonomous']);state=n.G001State();valued=set()
        for step in range(719):
            if 384<=step<432:
                before=c.debug();row=probe.inspect(c,env,seat,False);assert before==c.debug()
                if row['eligible']:
                    if row['candidates'] and step//24 not in valued:
                        row=probe.inspect(c,env,seat,True);assert before==c.debug();valued.add(step//24)
                    row.update(seed=seed,seat=seat);rows.append(row);print(json.dumps({k:v for k,v in row.items() if k!='candidates'}),flush=True)
            actions=[None,None];actions[seat]=c.act(env,seat);actions[1-seat]=rival.act(env,1-seat,state);env.step(actions)
        # Completion receipt verifies diagnostics did not alter the actual path.
        raw=env.observation(seat);money=[f['money'] for f in raw['farms']]
        assert money==refs[seed,seat]['money'],(seed,seat,money,refs[seed,seat]['money'])
        games.append(dict(seed=seed,seat=seat,debug=c.debug(),money=money,control_equal=True))
(out/'acceptance.json').write_text(json.dumps(dict(status='COMPLETE_READ_ONLY_PROTOTYPE_AUDIT',build=frozen,mechanisms=mechanisms,rows=rows,games=games,seconds=time.perf_counter()-tic,script_sha256=sha(Path(__file__))),indent=2));print('COMPLETE_READ_ONLY_PROTOTYPE_AUDIT',flush=True)
