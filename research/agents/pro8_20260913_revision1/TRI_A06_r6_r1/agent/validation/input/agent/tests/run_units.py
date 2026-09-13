from pathlib import Path
import subprocess,sys,argparse,json,hashlib,time
R=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--cxx',default='g++');a=p.parse_args();(R/'build').mkdir(exist_ok=True)
flags=['-std=c++20','-O2','-ffp-contract=off','-DA06_EXEC_MODE=2','-DR2_LOCAL_SALE_TIMING=1','-DR2_FINITE_FERTILIZER=1','-DR2_CROP_CLOCK_MODE=1','-DR2_OBSERVE_PUBLIC_TRADES=1','-DR2_SALE_CLOCK_MODE=0']
rows=[]
for name in ['test_a06_fleet','test_r3_pending','test_r5_contract','test_r6_contract']:
 source=R/'tests'/(name+'.cpp');out=R/'build'/name;cmd=[a.cxx,*flags,'-I',str(R),str(source),str(R/'policy/executor/vendor/simulator.cpp'),'-o',str(out)];start=time.perf_counter()
 built=subprocess.run(cmd,capture_output=True,text=True);result=subprocess.run([str(out)],capture_output=True,text=True) if built.returncode==0 else built
 row={'test':name,'command':cmd,'compile_exit':built.returncode,'exit':result.returncode,'stdout':result.stdout,'stderr':built.stderr+result.stderr,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'seconds':time.perf_counter()-start};rows.append(row)
 (R/'tests/UNIT_RESULTS.json').write_text(json.dumps(rows,indent=2));print(name,result.returncode,result.stdout,flush=True)
 if result.returncode:raise SystemExit(result.returncode)
