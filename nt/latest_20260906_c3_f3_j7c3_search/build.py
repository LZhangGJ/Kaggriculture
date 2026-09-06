"""Build only from this package; no outside objects, absolute paths or GPU."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import argparse, hashlib, json, subprocess, sys, sysconfig, time

P = Path(__file__).resolve().parent
S = P / 'src'
B = P / 'build'
ARENA = S / 'stage/arena'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jobs', type=int, default=3)
    args = ap.parse_args()
    if not 1 <= args.jobs <= 16:
        ap.error('--jobs must be 1..16')
    if sys.platform != 'linux':
        raise SystemExit('Use Linux/WSL2. This package uses dlopen and OpenMP.')
    B.mkdir(exist_ok=True)
    records = []
    common = ['g++','-std=c++20','-O3','-DNDEBUG','-fPIC','-ffp-contract=off','-pthread']
    includes = subprocess.check_output([sys.executable,'-m','pybind11','--includes'],text=True).split()
    def run(job):
        name, command = job
        t = time.perf_counter()
        r = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (B / (name+'.log')).write_text(r.stdout,encoding='utf-8')
        record = dict(name=name,command=command,seconds=time.perf_counter()-t,exit_code=r.returncode)
        print(name, r.returncode, round(record['seconds'],2), r.stdout[-1200:],flush=True)
        return record
    object_sources = [ARENA/'vendor/simulator.cpp'] + [ARENA/(n+'.cpp') for n in (
        'boatlee_v29','kaito_v58','lynn_v5','fieldbook_adapter','three_day_adapter',
        'ecobot_v7_core','ecobot_v7')]
    public_include = S/'stage/opponents/yhay81_three_day/output/source/include'
    jobs = [(f.stem, common+['-I'+str(public_include),'-fopenmp','-c',str(f),'-o',str(B/(f.stem+'.o'))]) for f in object_sources]
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        records.extend(pool.map(run,jobs))
    if any(r['exit_code'] for r in records):
        raise SystemExit('Object build failed; inspect build/*.log')
    shared = common + ['-shared','-Wl,-Bsymbolic']
    jobs = []
    for arm in ('c3auto','c3j7','f3'):
        flags = ['-DRL_C3_AUTO'] if arm=='c3auto' else []
        dependencies = [] if arm=='f3' else [str(S/f'stage/{arm}/executor/vendor/simulator.cpp')]
        if arm=='f3':
            flags += ['-I'+str(P/'vendor')]
        jobs.append((arm,shared+flags+[str(S/('f3_policy.cpp' if arm=='f3' else 'c3_policy.cpp'))]+dependencies+['-o',str(B/(arm+'.so'))]))
        jobs.append((arm+'_plan',shared+flags+(['-DHANDOFF_F3'] if arm=='f3' else [])+[str(S/'day_plan_policy.cpp')]+dependencies+['-o',str(B/(arm+'_plan.so'))]))
    objects = [str(B/(f.stem+'.o')) for f in object_sources]
    jobs.append(('runner',shared+['-fopenmp',*includes,'-I'+str(ARENA),'-I'+str(S),str(S/'runner_module.cpp'),*objects,'-ldl','-o',str(B/('_economic_rl_native'+sysconfig.get_config_var('EXT_SUFFIX')))]))
    # Unit test intentionally keeps assertions enabled, as in the accepted fix.
    unitflags = [flag for flag in common if flag!='-DNDEBUG']
    jobs.append(('test_ledger',unitflags+['-I'+str(P/'vendor'),str(S/'test_ledger.cpp'),str(ARENA/'vendor/simulator.cpp'),'-o',str(B/'test_ledger')]))
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        records.extend(pool.map(run,jobs))
    receipt = dict(status='PASS' if all(r['exit_code']==0 for r in records) else 'FAIL',
        python=sys.version,compiler=subprocess.check_output(['g++','--version'],text=True),builds=records,
        source_hashes={str(f.relative_to(P)):sha(f) for root in (S,P/'vendor') for f in root.rglob('*') if f.is_file()},
        artifacts={str(f.relative_to(P)):sha(f) for f in B.glob('*.so')})
    (B/'BUILD_RECEIPT.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    if receipt['status']!='PASS':
        raise SystemExit('Build failed')
    subprocess.run([str(B/'test_ledger')],check=True)

if __name__=='__main__':
    main()
