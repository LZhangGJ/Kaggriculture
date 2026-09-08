from pathlib import Path
import hashlib,json,shlex,subprocess,sys,sysconfig,zlib
E=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
build=json.loads((E/'native/build/build_receipt.json').read_text())
for rel,h in build['source_hashes'].items():assert sha(E/rel)==h
dst=E/'native/feed_insertion_probe_build';dst.mkdir(exist_ok=False)
src=E/'native/feed_insertion_probe.cpp';binary=dst/('_dp7_feed_probe'+sysconfig.get_config_var('EXT_SUFFIX'))
inc=shlex.split(subprocess.check_output([sys.executable,'-m','pybind11','--includes'],text=True))
cmd=['g++','-std=c++20','-O2','-fPIC','-shared',*inc,'-I'+str(E/'native'),str(src),'-o',str(binary)];subprocess.run(cmd,check=True)
(dst/'build_receipt.json').write_text(json.dumps(dict(build=build,source_sha256=sha(src),binary_sha256=sha(binary),command=cmd),indent=2))
sys.path.insert(0,str(E/'native/build'));sys.path.insert(0,str(dst));import _dp7_native as n;import _dp7_feed_probe as probe
panel=json.loads((E/'receipts/s4u_fiveway_N50_v1/results.json').read_text());identity=panel['identities']['g001'];asset=E/identity['asset'];assert sha(asset)==identity['asset_sha256']
rival=n.G001(json.loads(zlib.decompress(asset.read_bytes())));rows=[]
for seed in (20262703,20262736):
    env=n.Env(seed);c=n.Controller(panel['configurations']['full_chain_autonomous']);state=n.G001State()
    for step in range(432):
        # Probe after prior observation, before current decision, to preserve
        # outstanding current actions and exact remaining-time convention.
        if step in (388,400,412,420,425,429,430,431):
            before=c.debug();r=probe.inspect(c,env,0);assert before==c.debug();r['seed']=seed;rows.append(r);print(json.dumps(r),flush=True)
        a=c.act(env,0);b=rival.act(env,1,state);env.step([a,b])
out=E/'receipts/s4v_feed_exclusions_v1';out.mkdir(exist_ok=False)
(out/'acceptance.json').write_text(json.dumps(dict(status='COMPLETE_READ_ONLY_EXCLUSION_AUDIT',build=build,rows=rows,script_sha256=sha(Path(__file__))),indent=2))
