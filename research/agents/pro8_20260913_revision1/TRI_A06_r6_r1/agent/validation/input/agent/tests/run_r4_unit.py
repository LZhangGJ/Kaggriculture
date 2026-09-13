"""Portable R4 origin and feed-cover unit tests; benchmark-free."""
from pathlib import Path
import argparse, shutil, subprocess
p=Path(__file__).resolve().parents[1]
a=argparse.ArgumentParser();a.add_argument('--cxx',default=shutil.which('g++'));v=a.parse_args()
if not v.cxx: raise SystemExit('C++20 compiler required')
flags=['-std=c++20','-O2','-ffp-contract=off','-DR2_LOCAL_SALE_TIMING=1','-DR2_FINITE_FERTILIZER=1','-DR2_CROP_CLOCK_MODE=1','-DR2_OBSERVE_PUBLIC_TRADES=1','-DR2_SALE_CLOCK_MODE=0']
cmd=[v.cxx,*flags,'-I',str(p),str(p/'tests/test_r4_scope.cpp'),str(p/'policy/executor/vendor/simulator.cpp'),'-o',str(p/'build/test_r4_scope')]
subprocess.run(cmd,check=True)
r=subprocess.run([str(p/'build/test_r4_scope')],capture_output=True,text=True)
(p/'tests/r4_unit_stderr.txt').write_text(r.stderr)
r.check_returncode()
(p/'tests/r4_unit_results.txt').write_text(r.stdout)
print(r.stdout)
