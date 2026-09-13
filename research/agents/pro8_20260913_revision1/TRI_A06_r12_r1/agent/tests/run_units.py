"""Compile/run the retained six units and terminal-calendar regression, offline."""
from pathlib import Path
import subprocess,sys,argparse,json,hashlib,time
from datetime import datetime,timezone
R=Path(__file__).resolve().parents[1]
names=['test_a06_fleet','test_r3_pending','test_r5_contract','test_r6_contract','test_r11_recovery','test_r12_calendar','test_terminal_calendar']
p=argparse.ArgumentParser();p.add_argument('--cxx',default='g++');p.add_argument('--only',nargs='+',choices=names);a=p.parse_args();(R/'build').mkdir(exist_ok=True)
flags=['-std=c++20','-O2','-ffp-contract=off','-DA06_EXEC_MODE=2','-DR2_LOCAL_SALE_TIMING=1','-DR2_FINITE_FERTILIZER=1','-DR2_CROP_CLOCK_MODE=1','-DR2_OBSERVE_PUBLIC_TRADES=1','-DR2_SALE_CLOCK_MODE=0']
result_path=R/'tests/UNIT_RESULTS.json'
rows=json.loads(result_path.read_text()) if a.only and result_path.exists() else []
for name in a.only or names:
 source=R/'tests'/(name+'.cpp');out=R/'build'/name;cmd=[a.cxx,*flags,'-I',str(R),str(source),str(R/'policy/executor/vendor/simulator.cpp'),'-o',str(out)];start=time.perf_counter()
 built=subprocess.run(cmd,capture_output=True,text=True,timeout=120);result=subprocess.run([str(out)],capture_output=True,text=True,timeout=45) if built.returncode==0 else built
 row={'test':name,'tested_at_utc':datetime.now(timezone.utc).isoformat(),'command':cmd,'compile_exit':built.returncode,'exit':result.returncode,'stdout':result.stdout,'stderr':built.stderr+result.stderr,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'production_source_sha256':{str(x.relative_to(R)):hashlib.sha256(x.read_bytes()).hexdigest() for x in sorted((R/'policy').rglob('*')) if x.is_file() and x.suffix in ('.cpp','.hpp','.inc')},'seconds':time.perf_counter()-start};rows=[x for x in rows if x['test']!=name];rows.append(row)
 result_path.write_text(json.dumps(rows,indent=2));print(name,result.returncode,result.stdout,flush=True)
 if result.returncode:raise SystemExit(result.returncode)
