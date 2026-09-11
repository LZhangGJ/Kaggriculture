"""Build into a new output, validate contracts and frozen actions, then publish.
Never changes the shipped policy/joint.so or existing baselines.
"""
from __future__ import annotations
import argparse,ctypes,hashlib,json,os,shutil,subprocess,sys,tempfile,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def source_hashes():
    return {str(p.relative_to(ROOT/'policy')):sha(p)for p in sorted((ROOT/'policy').rglob('*'))if p.is_file()and p.suffix not in ('.so','.pyc')and '__pycache__'not in p.parts}
def run(cmd,log):
    with log.open('w')as f:subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--cxx',default='g++');parser.add_argument('--out',type=Path,default=ROOT/'build/local.so');a=parser.parse_args()
    compiler=shutil.which(a.cxx)
    if not compiler:parser.error('Compiler not found: '+a.cxx)
    target=a.out.resolve()
    if target.exists()or any(target.is_relative_to(ROOT/x)for x in ['policy','baselines','references']):parser.error('Choose a new build output. Frozen files cannot be replaced.')
    flags=json.loads((ROOT/'references/BUILD_FLAGS.json').read_text());release=json.loads((ROOT/'RELEASE_BUILD.json').read_text());sources=source_hashes()
    if sources!=release['sources']or flags!=release['flags']:raise SystemExit('Frozen source or flags changed. Use a separate experiment; do not rewrite reference hashes.')
    target.parent.mkdir(parents=True,exist_ok=True);folder=Path(tempfile.mkdtemp(prefix=target.stem+'_',dir=target.parent));pending=folder/'candidate.pending.so';start=time.monotonic()
    r={'status':'BUILDING','compiler':subprocess.check_output([compiler,'--version'],text=True),'sources':sources,'flags':flags,'output':str(target),'contracts':[]}
    try:
        cmd=[compiler,*flags,str(ROOT/'policy/bridge.cpp'),str(ROOT/'policy/executor/vendor/simulator.cpp'),'-o',str(pending)];r['command']=cmd;run(cmd,folder/'compile.log')
        r['binary_sha256']=sha(pending);lib=ctypes.CDLL(str(pending));lib.td_settings_count.restype=ctypes.c_size_t
        if lib.td_settings_count()!=38:raise RuntimeError('ABI mismatch')
        macros=[x for x in flags if x.startswith('-D')and x!='-DNDEBUG']
        for name in ['execution_contracts','takeover_contracts','test_startup_supply','integration_contracts','joint_contracts']:
            exe=folder/name;cmd=[compiler,'-std=c++20','-O2','-ffp-contract=off',*macros,'-I',str(ROOT),str(ROOT/'tests'/f'{name}.cpp'),str(ROOT/'policy/executor/vendor/simulator.cpp'),'-o',str(exe)]
            run(cmd,folder/(name+'.compile.log'));args=[str(exe)]
            if name in ['takeover_contracts','integration_contracts']:args+=[str(ROOT/'tests/handoff24.txt'),str(ROOT/'tests/settings38.txt')]
            if name=='joint_contracts':args+=[str(ROOT/'tests/settings38.txt')]
            tested=subprocess.run(args,cwd=ROOT,capture_output=True,text=True,check=True);r['contracts'].append({'name':name,'output':tested.stdout.strip()})
        run([sys.executable,str(ROOT/'check_binary.py'),'--binary',str(pending),'--out',str(folder/'BEHAVIOR.json')],folder/'behavior.log')
        r['behavior']=json.loads((folder/'BEHAVIOR.json').read_text())
        if r['behavior']['status']!='PASS'or source_hashes()!=sources:raise RuntimeError('Behavior check failed or sources changed')
        os.replace(pending,target);r['status']='PASS_CONTRACTS_AND_FROZEN_ACTIONS'
    except Exception as e:
        r['status']='REJECTED_NOT_PUBLISHED';r['error']=repr(e);raise
    finally:
        r['seconds']=time.monotonic()-start;(folder/'BUILD.json').write_text(json.dumps(r,indent=2)+'\n')
        if r['status']=='PASS_CONTRACTS_AND_FROZEN_ACTIONS':target.with_suffix('.BUILD.json').write_text(json.dumps(r,indent=2)+'\n')
        print(json.dumps({k:v for k,v in r.items()if k not in ['sources','compiler']},indent=2))
if __name__=='__main__':main()
