#!/usr/bin/env python3
"""Reproducible offline r4 checks. Build first using python build.py.
Uses the exact production native, frozen official unit functions and a C++
transport property test. --sanitize adds ASan/UBSan; does not replace -O3.
"""
from pathlib import Path
import argparse,datetime,hashlib,json,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--sanitize',action='store_true');p.add_argument('--cxx',default=os.environ.get('CXX','g++'));p.add_argument('--library',type=Path,default=ROOT/'policy/tri_a08_r11_r4.so');p.add_argument('--referee',type=Path,default=ROOT/'evidence/round4_feedback/referee');p.add_argument('--out-dir',type=Path);a=p.parse_args()
 dest=a.out_dir or ROOT/'test_runs'/datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');dest.mkdir(parents=True,exist_ok=True)
 receipt_path=a.library.with_suffix('.BUILD.json')
 if not receipt_path.is_file():raise SystemExit('Missing actual build receipt: '+str(receipt_path)+'; run build.py first')
 receipt=json.loads(receipt_path.read_text());flags=receipt['flags'];sha=hashlib.sha256(a.library.read_bytes()).hexdigest()
 if receipt['binary_sha256']!=sha:raise SystemExit('Native/build identity mismatch')
 for name,expected in receipt['sources'].items():
  if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=expected:raise SystemExit('Source differs from tested native: '+name)
 for name,expected in receipt.get('validation_sources',{}).items():
  if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=expected:raise SystemExit('Build gate source changed: '+name)
 record={'native_sha256':sha,'build_flags':flags,'compiler':subprocess.check_output([a.cxx,'--version'],text=True).splitlines()[0],'new_games':0,'checks':[]}
 def run(name,cmd,timeout=120,env=None):
  t=time.perf_counter();(dest/(name+'.command.json')).write_text(json.dumps(cmd,indent=2))
  with (dest/(name+'.stdout')).open('w') as out,(dest/(name+'.stderr')).open('w') as err:
   r=subprocess.run(cmd,stdout=out,stderr=err,timeout=timeout,env=env)
  x={'name':name,'command':cmd,'returncode':r.returncode,'wall_seconds':time.perf_counter()-t};record['checks'].append(x);(dest/'RESULTS.json').write_text(json.dumps(record,indent=2));print(json.dumps(x),flush=True)
  if r.returncode:raise SystemExit('FAILED '+name+'; raw logs retained in '+str(dest))
 run('native_choice_gate',[sys.executable,str(ROOT/'tests/native_choice_gate.py'),'--library',str(a.library.resolve()),'--enabled','1','--out',str(dest/'native_choice_gate.json')])
 run('native_completion_gate',[sys.executable,str(ROOT/'tests/native_completion_gate.py'),'--library',str(a.library.resolve()),'--enabled','1','--out',str(dest/'native_completion_gate.json')])
 run('completion_official',[sys.executable,str(ROOT/'tests/completion_official.py'),'--library',str(a.library.resolve()),'--referee',str(a.referee.resolve()),'--out',str(dest/'completion_official.json')])
 run('delivery_official',[sys.executable,str(ROOT/'tests/delivery_checks.py'),'--library',str(a.library.resolve()),'--referee',str(a.referee.resolve()),'--out',str(dest/'delivery_official.json')])
 unit_flags=[f for f in flags if f not in ('-fPIC','-shared','-Wl,-Bsymbolic')]
 target=dest/'delivery_unit_O3';run('compile_O3',[a.cxx,*unit_flags,str(ROOT/'tests/delivery_unit.cpp'),'-o',str(target)])
 run('unit_O3',[str(target)])
 target=dest/'completion_unit_O3';run('compile_completion_O3',[a.cxx,*unit_flags,str(ROOT/'tests/completion_unit.cpp'),str(ROOT/'policy/executor/vendor/simulator.cpp'),'-o',str(target)])
 run('completion_unit_O3',[str(target)])
 if a.sanitize:
  sf=[f for f in unit_flags if f not in ('-O3','-DNDEBUG')]+['-O1','-g','-fsanitize=address,undefined','-fno-omit-frame-pointer']
  target=dest/'delivery_unit_ASan_UBSan';run('compile_sanitize',[a.cxx,*sf,str(ROOT/'tests/delivery_unit.cpp'),'-o',str(target)])
  env=os.environ.copy();env['ASAN_OPTIONS']='detect_leaks=0:halt_on_error=1';env['UBSAN_OPTIONS']='halt_on_error=1:print_stacktrace=1';run('unit_sanitize',[str(target)],env=env)
  target=dest/'completion_unit_ASan_UBSan';run('compile_completion_sanitize',[a.cxx,*sf,str(ROOT/'tests/completion_unit.cpp'),str(ROOT/'policy/executor/vendor/simulator.cpp'),'-o',str(target)])
  run('completion_unit_sanitize',[str(target)],env=env)
 record['status']='PASS';(dest/'RESULTS.json').write_text(json.dumps(record,indent=2)+'\n');print('PASS '+str(dest))
if __name__=='__main__':main()
