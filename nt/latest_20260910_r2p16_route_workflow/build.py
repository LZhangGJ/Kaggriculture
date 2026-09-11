"""Compile the workflow candidate with the frozen P16 policy flags."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import time

ROOT = Path(__file__).resolve().parent
FLAGS = ['-std=c++20', '-O3', '-DNDEBUG', '-march=x86-64', '-ffp-contract=off',
         '-DR2_STARTUP_SUPPLY_MODE=2', '-DR2_LOCAL_SALE_TIMING=1',
         '-DR2_FINITE_FERTILIZER=1', '-DR2_CROP_CLOCK_MODE=1',
         '-DR2_OBSERVE_PUBLIC_TRADES=1', '-DR2_MARKET_INTEGRAL=0', '-DR2_SALE_CLOCK_MODE=0']

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', type=int, choices=range(4), default=3)
    p.add_argument('--cxx', default='g++-13')
    p.add_argument('--out', type=Path)
    p.add_argument('--unit', action='store_true')
    a = p.parse_args()
    out = (a.out or ROOT / 'build/revision2' / f'route{a.mode}.so').resolve()
    assert not out.exists(), f'Refusing overwrite: {out}'
    out.parent.mkdir(parents=True, exist_ok=True)
    flags = [*FLAGS, f'-DR2_ROUTE_ECONOMICS={a.mode}']
    vendor = ROOT / 'policy/executor/vendor/simulator.cpp'
    command = [a.cxx, *flags, '-fPIC', '-shared', '-Wl,-Bsymbolic',
               str(ROOT / 'policy/bridge.cpp'), str(vendor), '-o', str(out)]
    def sources():
        return {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest()
                for f in sorted((ROOT / 'policy').rglob('*')) if f.suffix in {'.hpp', '.cpp', '.inc', '.json', '.py'}}
    before = sources()
    start = time.perf_counter()
    subprocess.run(command, check=True)
    result = dict(mode=a.mode, command=command, seconds=time.perf_counter()-start,
                  sha256=hashlib.sha256(out.read_bytes()).hexdigest(),
                  sources=before)
    if a.unit:
        test = out.with_suffix('.unit')
        subprocess.run([a.cxx, *flags, '-UNDEBUG', '-I', str(ROOT),
                        str(ROOT / 'tests/test_route_workflow.cpp'), str(vendor), '-o', str(test)], check=True)
        checked = subprocess.run([str(test)], capture_output=True, text=True)
        if checked.returncode:
            print(checked.stdout + checked.stderr, flush=True)
            checked.check_returncode()
        result['unit'] = checked.stdout.strip()
    assert before == sources(), 'Source changed during compilation'
    out.with_suffix('.BUILD.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'sources'}, indent=2))

if __name__ == '__main__':
    main()
