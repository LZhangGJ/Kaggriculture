"""Portable, benchmark-free R5 execution-candidate and phase-contract tests."""
from pathlib import Path
import argparse,shutil,subprocess,json,hashlib
r=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--cxx',default=shutil.which('g++'));a=p.parse_args()
if not a.cxx:raise SystemExit('C++20 compiler required')
(r/'build').mkdir(exist_ok=True)
flags=['-std=c++20','-O2','-ffp-contract=off','-DA06_EXEC_MODE=2','-DR2_LOCAL_SALE_TIMING=1','-DR2_FINITE_FERTILIZER=1','-DR2_CROP_CLOCK_MODE=1','-DR2_OBSERVE_PUBLIC_TRADES=1','-DR2_SALE_CLOCK_MODE=0']
cmd=[a.cxx,*flags,'-I',str(r),str(r/'tests/test_r5_contract.cpp'),str(r/'policy/executor/vendor/simulator.cpp'),'-o',str(r/'build/test_r5_contract')]
subprocess.run(cmd,check=True);res=subprocess.run([str(r/'build/test_r5_contract')],capture_output=True,text=True)
(r/'tests/r5_unit_results.txt').write_text(res.stdout);(r/'tests/r5_unit_stderr.txt').write_text(res.stderr)
(r/'tests/r5_unit_receipt.json').write_text(json.dumps({'command':cmd,'exit':res.returncode,'test_source_sha256':hashlib.sha256((r/'tests/test_r5_contract.cpp').read_bytes()).hexdigest(),'native_source_receipt':'build/joint_v2.BUILD.json'},indent=2))
res.check_returncode();print(res.stdout)
