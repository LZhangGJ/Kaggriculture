"""Rebuild into build/, never replace the tested P16 binary."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import time

ROOT = Path(__file__).resolve().parent
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--cxx',default='g++-13')
    parser.add_argument('--out',type=Path,default=ROOT/'build/p16_rebuilt.so')
    parser.add_argument('--unit',action='store_true')
    args=parser.parse_args()
    target=args.out.resolve()
    assert not target.exists(), f'Refusing overwrite: {target}'
    assert target != ROOT/'policy/startupsupply2.so'
    target.parent.mkdir(parents=True,exist_ok=True)
    receipt=json.loads((ROOT/'development/candidates/candidate_r2p16/startupsupply2.BUILD.json').read_text())
    for name, expected in receipt['sources'].items():
        assert hashlib.sha256((ROOT/'policy'/name).read_bytes()).hexdigest()==expected, name
    flags=[]
    for arg in receipt['command'][1:]:
        if arg.endswith('.cpp'):
            break
        flags.append(arg)
    cmd=[args.cxx,*flags,str(ROOT/'policy/bridge.cpp'),str(ROOT/'policy/executor/vendor/simulator.cpp'),'-o',str(target)]
    tick=time.perf_counter()
    subprocess.run(cmd,check=True)
    out=dict(command=cmd,seconds=time.perf_counter()-tick,sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
             status='COMPILED_NOT_BEHAVIOR_VALIDATED')
    if args.unit:
        unit=target.parent/(target.name+'.unit')
        assert not unit.exists()
        subprocess.run([args.cxx,'-std=c++20','-O2','-ffp-contract=off','-I',str(ROOT),
                        str(ROOT/'tests/test_startup_supply.cpp'),str(ROOT/'policy/executor/vendor/simulator.cpp'),'-o',str(unit)],check=True)
        out['unit']=subprocess.run([str(unit)],capture_output=True,text=True,check=True).stdout.strip()
    target.with_suffix('.BUILD.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))
if __name__=='__main__':
    main()
