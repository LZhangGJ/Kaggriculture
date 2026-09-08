"""Diagnose current proposal score/budget/realization using unchanged games."""
from pathlib import Path
import argparse,gzip,hashlib,json,shlex,subprocess,sys,sysconfig,time,zlib
E=Path(__file__).resolve().parents[1]
cli=argparse.ArgumentParser();cli.add_argument('--build-only',action='store_true');a=cli.parse_args()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
build=json.loads((E/'native/build/build_receipt.json').read_text())
for rel,h in build['source_hashes'].items():assert sha(E/rel)==h
dst=E/'native/investment_choice_probe_build_v2';src=E/'native/investment_choice_probe.cpp';stamp=dst/'build_receipt.json'
if a.build_only:
 dst.mkdir(exist_ok=False);binary=dst/('_dp7_investment_choice_probe'+sysconfig.get_config_var('EXT_SUFFIX'))
 inc=shlex.split(subprocess.check_output([sys.executable,'-m','pybind11','--includes'],text=True));cmd=['g++','-std=c++20','-O2','-ffp-contract=off','-fPIC','-shared',*inc,'-I'+str(E/'native'),str(src),str(E/'native/vendor/simulator.cpp'),'-o',str(binary)]
 r=subprocess.run(cmd,capture_output=True,text=True);(dst/'compile.log').write_text(r.stdout+r.stderr)
 if r.returncode:print(r.stderr[-3500:]);raise SystemExit(r.returncode)
 stamp.write_text(json.dumps(dict(build=build,source_sha256=sha(src),binary_sha256=sha(binary),command=cmd),indent=2));print('COMPILE_ONLY_PASS');raise SystemExit(0)
frozen=json.loads(stamp.read_text());assert frozen['build']==build and frozen['source_sha256']==sha(src)
sys.path.insert(0,str(E/'native/build'));sys.path.insert(0,str(dst));import _dp7_native as n;import _dp7_investment_choice_probe as probe
panel_path=E/'receipts/s4v_twelveway_N50_v1/results.json';panel=json.loads(panel_path.read_text());refs={(r['variant'],r['opponent'],r['seed'],r['seat']):r for r in panel['rows']}
classes={'g001':(n.G001,n.G001State),'g003':(n.G001,n.G001State),'boatlee_v29':(n.BoatleeV29,n.BoatleeState),'kaito_v58':(n.KaitoV58,n.KaitoState),'lynn_v5':(n.LynnV5,n.LynnState)}
direct={'yhay81_six_day':n.Fieldbook,'yhay81_three_day':n.ThreeDay,'ecobot_v7':n.EcoBotV7}
out=E/'receipts/s4y_investment_choice_audit_v1';out.mkdir(exist_ok=False);games=[];tic=time.perf_counter()
for label in ('all_intraday_insert','full_chain_autonomous'):
 for opp,entry in panel['identities'].items():
  if opp=='pass':continue
  rival=None
  if opp in classes:
   asset=E/entry['asset'];assert sha(asset)==entry['asset_sha256'];rival=classes[opp][0](json.loads(zlib.decompress(asset.read_bytes())))
  for seed in (20262701,20262702):
   for seat in (0,1):
    env=n.Env(seed);c=n.Controller(panel['configurations'][label]);state=classes[opp][1]() if rival else direct[opp]();rows=[]
    for step in range(719):
     own=c.act(env,seat)
     if step%24==0:
      before=c.debug();r=probe.inspect(c,env,seat);assert before==c.debug();rows.append(r)
     other=rival.act(env,1-seat,state) if rival else state.act(env,1-seat)
     env.step([own,other] if seat==0 else [other,own])
    ref=refs[label,opp,seed,seat];farms=env.observation(seat)['farms'];assert farms[seat]['money']==ref['cash'] and farms[1-seat]['money']==ref['opponent_cash']
    summary=dict(label=label,opponent=opp,seed=seed,seat=seat,cash=ref['cash'],opponent_cash=ref['opponent_cash'],control_equal=True)
    (out/f'{label}_{opp}_{seed}_{seat}.json.gz').write_bytes(gzip.compress(json.dumps(dict(summary=summary,rows=rows)).encode()));games.append(summary);(out/'progress.json').write_text(json.dumps(games,indent=2));print(json.dumps(summary),flush=True)
(out/'acceptance.json').write_text(json.dumps(dict(status='PASS_READ_ONLY_INVESTMENT_AUDIT',build=frozen,rows=games,seconds=time.perf_counter()-tic,panel_sha256=sha(panel_path),script_sha256=sha(Path(__file__)),final_goal_acceptance=False,
 caveat='Greedy trace fixes final land capacity; actual portfolio/rotation may differ. Preparation is conditional, not promised opponent fills. No policy mutation or hindsight action selection.'),indent=2));print('PASS_READ_ONLY_INVESTMENT_AUDIT',flush=True)
