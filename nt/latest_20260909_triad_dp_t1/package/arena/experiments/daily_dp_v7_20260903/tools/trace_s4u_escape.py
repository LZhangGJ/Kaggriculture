"""Replay identified real failures unchanged; never a policy feature."""
from pathlib import Path
import argparse,gzip,hashlib,json,shlex,subprocess,sys,sysconfig,zlib
E=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
sys.path.insert(0,str(E/'native/build'))
import _dp7_native as n

def main():
    cli=argparse.ArgumentParser();cli.add_argument('--plans',action='store_true');args=cli.parse_args()
    build=json.loads((E/'native/build/build_receipt.json').read_text())
    for rel,h in build['source_hashes'].items():assert sha(E/rel)==h
    if args.plans:
        dst=E/'native/plan_execution_probe_build';dst.mkdir(exist_ok=True)
        binary=dst/('_dp7_plan_probe'+sysconfig.get_config_var('EXT_SUFFIX'));assert not binary.exists()
        src=E/'native/plan_execution_probe.cpp'
        include=shlex.split(subprocess.check_output([sys.executable,'-m','pybind11','--includes'],text=True))
        cmd=['g++','-std=c++20','-O2','-fPIC','-shared',*include,'-I'+str(E/'native'),str(src),'-o',str(binary)]
        subprocess.run(cmd,check=True)
        (dst/'build_receipt.json').write_text(json.dumps(dict(build=build,probe_source_sha256=sha(src),binary_sha256=sha(binary),command=cmd),indent=2))
        sys.path.insert(0,str(dst));import _dp7_plan_probe as probe
    panel_path=E/'receipts/s4u_fiveway_N50_v1/results.json'
    panel=json.loads(panel_path.read_text());label='full_chain_autonomous'
    asset=E/panel['identities']['g001']['asset']
    rival=n.G001(json.loads(zlib.decompress(asset.read_bytes())))
    records=json.load(gzip.open(E/'receipts/s4u_pool_audit_N50_v1/full_chain_autonomous_g001.json.gz','rt'))['rows']
    failed=[r for r in records if any(sum(d[r['seat']]['escaped']) for d in r['production'])]
    out=E/('receipts/s4u_escape_plan_trace_v1' if args.plans else 'receipts/s4u_escape_trace_v1');out.mkdir(exist_ok=False);rows=[]
    for r in failed:
        seed,seat=r['seed'],r['seat'];env=n.Env(seed);c=n.Controller(panel['configurations'][label]);state=n.G001State();trace=[];events=[]
        for step in range(719):
            obs=env.observation(seat);farm=obs['farms'][seat]
            own=c.act(env,seat);other=rival.act(env,1-seat,state)
            env.step([own,other] if seat==0 else [other,own])
            after=env.observation(seat);af=after['farms'][seat]
            escaped=[]
            for y,row in enumerate(farm['tiles']):
                for x,t in enumerate(row):
                    at=af['tiles'][y][x]
                    if isinstance(t,dict) and t.get('animal') and not (isinstance(at,dict) and at.get('animal')):
                        escaped.append(dict(x=x,y=y,before=t,after=at))
            if escaped:events.append(dict(step=step,animals=escaped))
            if 15*24<=step<19*24:
                trace.append(dict(step=step,observation=obs,own_action=own,opponent_action=other,debug=c.debug(),service=n.service_stats(c),escaped=escaped,
                    plans=probe.inspect(c) if args.plans else None))
        assert [f['money'] for f in env.observation(seat)['farms']]==r['money']
        name=f'{seed}_{seat}.json.gz';(out/name).write_bytes(gzip.compress(json.dumps(trace).encode()))
        rows.append(dict(seed=seed,seat=seat,events=events,cash=r['money'],trace=name))
        print(json.dumps(rows[-1]),flush=True)
    (out/'acceptance.json').write_text(json.dumps(dict(status='PASS_UNCHANGED_FAILURE_REPRODUCTION_NOT_STRATEGY_ACCEPTANCE',build=build,rows=rows,
        panel_sha256=sha(panel_path),script_sha256=sha(Path(__file__)),classification='UNPLANNED_UNTIL_SUPPORTED_BY_EXPLICIT_RELEASE_INTENT'),indent=2))
if __name__=='__main__':main()
