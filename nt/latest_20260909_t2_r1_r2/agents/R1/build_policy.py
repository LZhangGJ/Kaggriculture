"""Build T2-R1 service-aligned in isolation; publish only after full frozen-trajectory conformance.
Usage: python3 build_policy.py --cxx g++-13
A failed compiler/build/check never replaces the existing policy/agent.so.
Baselines remain supported, but are explicitly NOT behavior-validated here.
"""
from __future__ import annotations
import argparse, fcntl, hashlib, json, os, shlex, shutil, subprocess, sys, tempfile, time, zipfile
from pathlib import Path
ROOT = Path(__file__).resolve().parent

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def stable_hash(mapping: dict) -> str:
    return hashlib.sha256(json.dumps(mapping, sort_keys=True).encode()).hexdigest()

def capture_dependencies(cxx: str, source: Path, defines: list[str]) -> list[Path]:
    text = subprocess.check_output([cxx, '-std=c++20', *defines, '-M', '-MT', 't2_r1_dependencies', str(source)], text=True)
    words = shlex.split(text.replace('\\\n', ' ').split(':', 1)[1])
    return [Path(word).resolve(strict=True) for word in words]

def validate(candidate: Path, report: Path, workers: int = 2) -> dict:
    proc = subprocess.run([sys.executable, str(ROOT/'rebuild_checks/check_binary.py'),
                           '--binary', str(candidate), '--report', str(report),
                           '--workers', str(workers)], check=False)
    if not report.exists():
        raise RuntimeError(f'Conformance checker failed without a report (exit {proc.returncode})')
    result = json.loads(report.read_text())
    if proc.returncode or result.get('status') != 'PASS':
        raise RuntimeError(f'BUILD REJECTED: behavior differs from T2-R1 service-aligned; see {report}')
    if result['binary_sha256'] != sha(candidate):
        raise RuntimeError('Candidate changed during conformance check')
    return result

def validate_and_publish(candidate: Path, output: Path, report: Path, workers: int = 2) -> dict:
    """This same gate is used by normal builds and the negative-control test."""
    result = validate(candidate, report, workers)
    os.replace(candidate, output)
    return result

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cxx', default=os.environ.get('CXX', 'g++'), help='Compiler executable; defaults to CXX or g++')
    ap.add_argument('--out', type=Path, help='Optional output path; default policy/agent.so')
    ap.add_argument('--workers', type=int, default=2, help='Validation processes, 1..4')
    # Only the recovered candidate is built by this gated script.
    args = ap.parse_args(); args.baseline = None
    found = shutil.which(args.cxx)
    if not found:
        raise FileNotFoundError(f'Compiler executable not found: {args.cxx}')
    # Preserve argv[0]: clang++ and clang may be symlinks to the same executable,
    # but they select different default C++ linker behavior.
    cxx = str(Path(found).absolute())
    source = ROOT / ('src/baseline_bridge.cpp' if args.baseline else 'policy/bridge.cpp')
    simulator = ROOT / 'policy/executor/vendor/simulator.cpp'
    out = (args.out or ROOT / (f'baseline_{args.baseline}.so' if args.baseline else 'policy/agent.so')).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    lock = out.with_name(out.name + '.build.lock')
    with lock.open('a') as lockfile:
        fcntl.flock(lockfile.fileno(), fcntl.LOCK_EX)
        return locked_build(cxx, source, simulator, out, args)

def locked_build(cxx: str, source: Path, simulator: Path, out: Path, args) -> int:
    start = time.perf_counter()
    stage = Path(tempfile.mkdtemp(prefix='.t2-build-', dir=out.parent))
    candidate = stage/'agent.so'
    report = stage/'conformance.json'
    defines = ['-DBASE_AUTO'] if args.baseline == 'auto' else []
    flags = ['-std=c++20','-O3','-DNDEBUG','-march=x86-64','-ffp-contract=off','-fPIC','-shared','-Wl,-Bsymbolic',*defines]
    root = ROOT / ('agent' if args.baseline else 'policy')
    sources = {p.resolve() for p in root.rglob('*') if p.is_file() and
               p.suffix in ('.hpp','.cpp','.h','.inc','.py','.json') and
               not p.name.endswith('.build.json') and not any(x.startswith('.t2-build-') for x in p.parts)}
    sources.update([source.resolve(),simulator.resolve(),Path(__file__).resolve()])
    if not args.baseline:
        sources.update(p.resolve() for p in (ROOT/'rebuild_checks').rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    dependencies = set(capture_dependencies(cxx, source, defines) + capture_dependencies(cxx, simulator, []))
    tracked = sources | dependencies
    before = {str(p):sha(p) for p in sorted(tracked)}
    relative = {str(p.relative_to(ROOT)):before[str(p)] for p in sorted(sources|{x for x in dependencies if x.is_relative_to(ROOT)})}
    cmd = [cxx,*flags,str(source),str(simulator),'-o',str(candidate)]
    compiler = dict(executable=cxx,resolved_executable=str(Path(cxx).resolve()),sha256=sha(Path(cxx)),
                    version=subprocess.check_output([cxx,'--version'],text=True).strip(),
                    target=subprocess.check_output([cxx,'-dumpmachine'],text=True).strip())
    frontend = subprocess.check_output([cxx,'-print-prog-name=cc1plus'],text=True).strip()
    if Path(frontend).is_file():
        compiler['frontend']={'path':str(Path(frontend).resolve()),'sha256':sha(Path(frontend))}
    receipt = dict(status='STARTED',command=cmd,compiler=compiler,source_files=relative,
                   source_hash=stable_hash(relative),dependency_files=before,
                   dependency_hash=stable_hash(before),build_gate='five full focus-trajectory check before replacing output')
    try:
        with (stage/'compile.log').open('w') as log:
            subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
        if {str(p):sha(p) for p in sorted(tracked)} != before:
            raise RuntimeError('A source or included dependency changed during compilation')
        receipt['binary_hash'] = sha(candidate)
        if args.baseline:
            receipt['status']='BUILT_NOT_BEHAVIOR_VALIDATED_BASELINE'
        else:
            receipt['validation']=validate(candidate,report,args.workers)
            receipt['status']='PASS'
        if {str(p):sha(p) for p in sorted(tracked)} != before:
            raise RuntimeError('A source or included dependency changed during validation')
        store=ROOT/'snapshots';store.mkdir(exist_ok=True)
        # Preserve the incumbent by content hash before publishing a passing candidate.
        if out.is_file():
            incumbent=store/(sha(out)+'.so')
            if not incumbent.exists():shutil.copy2(out,incumbent)
        frozen=store/(receipt['binary_hash']+'.so')
        if not frozen.exists():shutil.copy2(candidate,frozen)
        if sha(frozen)!=receipt['binary_hash']:raise RuntimeError('Immutable snapshot hash mismatch')
        with zipfile.ZipFile(store/(receipt['source_hash']+'.zip'),'w',zipfile.ZIP_DEFLATED) as z:
            for rel in relative:z.write(ROOT/rel,rel)
        receipt.update(source_stability_check='PASS',seconds=time.perf_counter()-start)
        receipt_tmp=stage/'receipt.json';receipt_tmp.write_text(json.dumps(receipt,indent=2))
        # Check gate completed above. No failed candidate can reach publication.
        os.replace(candidate,out);os.replace(receipt_tmp,out.with_suffix('.build.json'))
        archive=ROOT/'rebuild_evidence/builds'/receipt['binary_hash'];archive.mkdir(parents=True,exist_ok=True)
        for file in stage.iterdir():
            if file.is_file():shutil.copy2(file,archive/file.name)
        shutil.rmtree(stage)
        print(json.dumps(dict(status=receipt['status'],output=str(out),binary_hash=receipt['binary_hash'],
                              source_hash=receipt['source_hash'],seconds=receipt['seconds'])),flush=True)
        return 0
    except Exception as exc:
        receipt.update(status='FAIL_NOT_PUBLISHED',error=f'{type(exc).__name__}: {exc}',seconds=time.perf_counter()-start)
        (stage/'failure.json').write_text(json.dumps(receipt,indent=2))
        print(f'BUILD FAILED; previous binary left untouched. Evidence: {stage}\n{exc}',file=sys.stderr)
        return 1

if __name__ == '__main__':
    raise SystemExit(main())
