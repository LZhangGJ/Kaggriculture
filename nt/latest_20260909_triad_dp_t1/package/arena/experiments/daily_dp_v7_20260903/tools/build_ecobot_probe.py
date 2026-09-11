"""Build independent test module; do not invalidate the accepted arena binary."""
from pathlib import Path
import hashlib
import json
import shlex
import subprocess
import sys
import sysconfig
import time

EXP=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    build=EXP/'native/eco7_probe_build';build.mkdir(exist_ok=True)
    sources=['native/ecobot_v7_core.cpp','native/ecobot_v7_core.hpp','native/ecobot_v7_probe.cpp','native/vendor/simulator.hpp','native/ecobot_v7.cpp','native/ecobot_v7.hpp']
    original=EXP/'opponents/ecobot_v7/output/main.py'
    assert sha(original)=='0dc02e03c94ef60c06b5093efc2e2fd0530aa6eea20df507a90b90d6651bd067'
    includes=shlex.split(subprocess.check_output([sys.executable,'-m','pybind11','--includes'],text=True))
    binary=build/('_eco7_probe'+sysconfig.get_config_var('EXT_SUFFIX'))
    if binary.exists():
        history=build/'history';history.mkdir(exist_ok=True)
        old=sha(binary);(history/(old+binary.suffix)).write_bytes(binary.read_bytes())
        if (build/'build_receipt.json').exists():(history/(old+'.json')).write_bytes((build/'build_receipt.json').read_bytes())
    cmd=['g++','-O2','-std=c++20','-fPIC','-shared','-ffp-contract=off',*includes,'-I'+str(EXP/'native'),str(EXP/sources[0]),str(EXP/sources[2]),str(EXP/'native/ecobot_v7.cpp'),'-o',str(binary)]
    start=time.perf_counter();subprocess.run(cmd,check=True)
    r=dict(status='BUILT_PARTIAL_CORE_NOT_AGENT',source_hashes={s:sha(EXP/s) for s in sources},source_sha256=sha(original),binary_sha256=sha(binary),command=cmd,seconds=time.perf_counter()-start)
    (build/'build_receipt.json').write_text(json.dumps(r,indent=2));print(json.dumps({k:v for k,v in r.items() if k not in ('command','source_hashes')}))
if __name__=='__main__':main()
