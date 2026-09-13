"""Offline bounded regression checks. Never starts a game or draws a seed."""
from pathlib import Path
import argparse,json,os,subprocess,sys,time
os.environ['PYTHONDONTWRITEBYTECODE']='1';sys.dont_write_bytecode=True
R=Path(__file__).resolve().parents[1];p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=R/'build/validation');a=p.parse_args();D=a.out.resolve();D.mkdir(parents=True,exist_ok=True)
F=R/'evidence/input';P=R/'provenance/parent_agent';unit=D/'units.stdout';audit=D/'fixed_core_audit.so'
commands=[('units',[sys.executable,str(R/'tests/run_units.py')]),('entry',[sys.executable,str(R/'tests/entry_smoke.py'),'--agent',str(R/'main.py'),'--feedback',str(F),'--out',str(D/'entry.json')]),('prefix',[sys.executable,str(R/'tests/prefix_entry.py'),'--parent',str(P/'main.py'),'--candidate',str(R/'main.py'),'--feedback',str(F),'--out',str(D/'prefix.json')]),('atomic_market',[sys.executable,str(R/'tests/market_reservation.py'),'--feedback',str(F),'--unit-json',str(unit),'--out',str(D/'atomic_market.json')]),('audit_build',[sys.executable,str(R/'tests/build_audit.py'),'--parent',str(P),'--out',str(audit)]),('fixed_routes',[sys.executable,str(R/'tests/fixed_execution.py'),'--parent',str(P/'main.py'),'--audit-lib',str(audit),'--feedback',str(F),'--unit-json',str(unit),'--out',str(D/'fixed_routes.json')])]
rows=[]
for name,cmd in commands:
 t=time.monotonic()
 with open(D/f'{name}.stdout','w') as stdout,open(D/f'{name}.stderr','w') as stderr:r=subprocess.run(['timeout','--kill-after=5s','120s',*cmd],stdout=stdout,stderr=stderr,timeout=130)
 row={'name':name,'command':cmd,'returncode':r.returncode,'seconds':time.monotonic()-t};rows.append(row);(D/'suite.json').write_text(json.dumps({'new_games':0,'tests':rows},indent=2));print(json.dumps(row),flush=True)
 if r.returncode:raise SystemExit(r.returncode)
