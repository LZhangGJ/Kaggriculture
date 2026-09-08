"""Portable CPU build. Run from an extracted bundle, not the original checkout."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import argparse, hashlib, json, os, subprocess, sysconfig, time
try:
    import pybind11
except ImportError:
    from types import SimpleNamespace
    candidates=[Path(sysconfig.get_paths()["purelib"])/"torch/include"]
    inc=next((x for x in candidates if (x/"pybind11/pybind11.h").exists()),None)
    if inc is None: raise RuntimeError("Install pybind11 or provide its headers")
    pybind11=SimpleNamespace(get_include=lambda: str(inc))

ROOT = Path(__file__).resolve().parent
EXP = ROOT / 'experiments/daily_dp_v7_20260903'
N = EXP / 'native'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--jobs', type=int, default=2)
    args = parser.parse_args()
    dest = N / 'build'
    dest.mkdir(exist_ok=True)
    names = ['vendor/simulator.cpp','module.cpp','fieldbook_adapter.cpp','boatlee_v29.cpp',
             'kaito_v58.cpp','lynn_v5.cpp','three_day_adapter.cpp','ecobot_v7_core.cpp','ecobot_v7.cpp']
    inc = ['-I'+pybind11.get_include(), '-I'+sysconfig.get_paths()['include'], '-I'+str(N)]
    commands = []
    started = time.perf_counter()
    def compile_one(name):
        source = N/name
        obj = dest/(source.stem+'.o')
        cmd = ['g++','-std=c++20','-O3','-DNDEBUG','-march=x86-64','-fopenmp','-fPIC',*inc]
        if source.stem in ('boatlee_v29','kaito_v58','lynn_v5','ecobot_v7','ecobot_v7_core'):
            cmd += ['-ffp-contract=off']
        if source.stem == 'three_day_adapter':
            cmd += ['-I'+str(EXP/'opponents/yhay81_three_day/output/source/include')]
        cmd += ['-c',str(source),'-o',str(obj)]
        commands.append(cmd)
        subprocess.run(cmd,check=True)
        print('Compiled',name,flush=True)
        return str(obj)
    with ThreadPoolExecutor(max_workers=max(1,min(args.jobs,4))) as pool:
        objects = list(pool.map(compile_one,names))
    binary = dest/('_dp7_native'+sysconfig.get_config_var('EXT_SUFFIX'))
    link = ['g++','-shared','-fopenmp',*objects,'-o',str(binary)]
    commands.append(link)
    subprocess.run(link,check=True)
    winner = EXP/'strategy_switch7_20260905/final_local_best_v1'
    cmd = ['g++','-std=c++20','-O3','-DNDEBUG','-march=x86-64','-fPIC','-shared','-Wl,-Bsymbolic',
           '-I'+str(N),'-I'+str(winner),str(EXP/'strategy_switch7_20260905/online_bridge.cpp'),
           str(N/'vendor/simulator.cpp'),'-o',str(winner/'agent.so')]
    commands.append(cmd)
    subprocess.run(cmd,check=True)
    files = [p for p in N.rglob('*') if p.is_file() and p.suffix in ('.cpp','.hpp','.inc')]
    receipt = dict(status='BUILT_NOT_YET_VALIDATED',seconds=time.perf_counter()-started,
                   architecture='generic x86-64, no -march=native',commands=commands,
                   source_hashes={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                   binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                   j7_binary_sha256=hashlib.sha256((winner/'agent.so').read_bytes()).hexdigest())
    (dest/'portable_build_receipt.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps({'status':receipt['status'],'seconds':receipt['seconds']}),flush=True)

if __name__ == '__main__': main()
