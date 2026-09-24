"""Compile and run deterministic forecast invariants; no game scores are asserted."""
from pathlib import Path
import argparse,shutil,subprocess
P=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cxx',default=shutil.which('g++'));a=p.parse_args()
    if not a.cxx:p.error('C++20 compiler not found')
    out=P/'build/forecast_tests';out.parent.mkdir(exist_ok=True)
    command=[a.cxx,'-std=c++20','-O0','-DR2_SALE_CLOCK_MODE=2','-I'+str(P),str(P/'tests/forecast_tests.cpp'),str(P/'policy/executor/vendor/simulator.cpp'),'-o',str(out)]
    subprocess.run(command,check=True);subprocess.run([str(out)],check=True)
if __name__=='__main__':main()
