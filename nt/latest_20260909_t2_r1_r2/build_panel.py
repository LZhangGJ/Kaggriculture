"""Rebuild only the arena for another Linux Python ABI. Does not rebuild policies."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import argparse
import hashlib
import json
import subprocess
import sysconfig
import time
import pybind11

R=Path(__file__).resolve().parent
N=R.parent/'latest_20260909_triad_dp_t1/package/arena/experiments/daily_dp_v7_20260903/native'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--jobs',type=int,default=4);ap.add_argument('--cxx',default='g++');a=ap.parse_args();assert 1<=a.jobs<=16
    provenance=json.loads((R/'SOURCE_PROVENANCE.json').read_text())
    for rel,h in provenance['shared_arena_hashes'].items():assert (N/rel).is_file() and sha(N/rel)==h,rel
    B=R/'build';B.mkdir(exist_ok=True);started=time.perf_counter()
    names=['simulator','fieldbook_adapter','boatlee_v29','kaito_v58','lynn_v5','three_day_adapter','ecobot_v7_core','ecobot_v7']
    def compile(name):
        src=N/('vendor/simulator.cpp' if name=='simulator' else name+'.cpp');out=B/(name+'.o')
        cmd=[a.cxx,'-std=c++20','-O3','-DNDEBUG','-march=x86-64','-fopenmp','-fPIC','-I'+pybind11.get_include(),'-I'+sysconfig.get_paths()['include'],'-I'+str(N)]
        if name in ('boatlee_v29','kaito_v58','lynn_v5','ecobot_v7','ecobot_v7_core'):cmd+=['-ffp-contract=off']
        if name=='three_day_adapter':cmd+=['-I'+str(N.parent/'opponents/yhay81_three_day/output/source/include')]
        cmd+=['-c',str(src),'-o',str(out)]
        with (B/(name+'.log')).open('w') as log:subprocess.run(cmd,check=True,stdout=log,stderr=subprocess.STDOUT)
        return out
    with ThreadPoolExecutor(max_workers=a.jobs) as pool:objects=list(pool.map(compile,names))
    output=R/'arena'/('_triad_panel'+sysconfig.get_config_var('EXT_SUFFIX'))
    cmd=[a.cxx,'-std=c++20','-O2','-DNDEBUG','-fPIC','-fopenmp','-shared','-Wl,-Bsymbolic','-I'+pybind11.get_include(),'-I'+sysconfig.get_paths()['include'],'-I'+str(N),str(R/'arena/panel.cpp'),*map(str,objects),'-ldl','-o',str(output)]
    subprocess.run(cmd,check=True)
    (B/'BUILD.json').write_text(json.dumps(dict(command=cmd,sha256=sha(output),seconds=time.perf_counter()-started),indent=2))
    print('Built arena. Run both --reference tests before trusting it. Policy binaries were not modified.')

if __name__=='__main__':main()
