from pathlib import Path
import argparse,hashlib,json,shutil,subprocess,time
E=Path(__file__).resolve().parents[1];p=argparse.ArgumentParser();p.add_argument('--version',default='v1');p.add_argument('--insertion',action='store_true');a=p.parse_args();out=E/('receipts/s5d_'+('insertion_' if a.insertion else 'mechanisms_')+a.version);out.mkdir(exist_ok=False)
test='test_insertion_incremental.cpp' if a.insertion else 'test_compile_choices.cpp'
sources=['policy.hpp','compile_choices.hpp',test];hashes={}
for s in sources:shutil.copy2(E/'native'/s,out/s);hashes[s]=hashlib.sha256((out/s).read_bytes()).hexdigest()
cmd=['g++','-std=c++20','-O2','-I'+str(E/'native'),str(E/'native'/test),str(E/'native/vendor/simulator.cpp'),'-o',str(out/'test')]
start=time.perf_counter();r=subprocess.run(cmd,capture_output=True,text=True);(out/'compile.log').write_text(r.stdout+r.stderr)
if not r.returncode:r=subprocess.run([str(out/'test')],capture_output=True,text=True);(out/'test.log').write_text(r.stdout+r.stderr)
receipt=dict(status='PASS' if r.returncode==0 else 'FAIL_PRESERVED',returncode=r.returncode,seconds=time.perf_counter()-start,command=cmd,source_hashes=hashes,output=r.stdout,error=r.stderr)
(out/'acceptance.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt,indent=2));raise SystemExit(r.returncode)
