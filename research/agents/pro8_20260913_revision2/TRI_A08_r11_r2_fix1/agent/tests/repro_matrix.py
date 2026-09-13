#!/usr/bin/env python3
"""Offline reduction/optimization matrix. Original failures are diagnostic.

The original standalone is a reduction candidate, not a claim that GCC13 was
locally available or that the reduction reproduced the central failure here.
The unmodified original full unit is also run to preserve the known trigger.
"""
from __future__ import annotations
import argparse,datetime,json,os,pathlib,shutil,subprocess,time
ROOT=pathlib.Path(__file__).resolve().parents[1]

def main():
 p=argparse.ArgumentParser();p.add_argument('--cxx',default='g++');p.add_argument('--out',type=pathlib.Path,required=True);p.add_argument('--reduced-only',action='store_true');a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
 if shutil.which(a.cxx) is None:
  result={'status':'COMPILER_UNAVAILABLE','compiler':a.cxx,'new_games':0};(a.out/'RESULT.json').write_text(json.dumps(result,indent=2));print(json.dumps(result));raise SystemExit(2)
 rec=json.loads((ROOT/'policy/tri_a08_r11_r2_fix1.BUILD.json').read_text());base=[f for f in rec.get('flags',rec['command'][1:-5]) if f not in ('-shared','-fPIC','-Wl,-Bsymbolic')]
 repro=ROOT/'fix1/repro' if (ROOT/'fix1/repro').exists() else ROOT.parent/'repro'
 cases=[('original_standalone',repro/'original_standalone.cpp',None),('boundary_only',repro/'original_standalone_boundary.cpp',None),('argmax_only',repro/'fixed_standalone_no_boundary.cpp',None),('fixed_standalone',repro/'fixed_standalone.cpp',None)]
 if not a.reduced_only:cases.append(('original_full_unit',ROOT/'history/r2/animal_collection_test.cpp',ROOT/'reference/r2'))
 rows=[];start=time.perf_counter();env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
 for name,source,include in cases:
  for mode in ('O3','O0','O1_asan_ubsan'):
   flags=base.copy()
   if mode=='O0':flags=[f for f in flags if f!='-O3']+['-O0']
   if mode=='O1_asan_ubsan':flags=[f for f in flags if f not in ('-O3','-DNDEBUG')]+['-O1','-g','-fsanitize=address,undefined','-fno-sanitize-recover=all','-fno-omit-frame-pointer']
   target=a.out/(name+'_'+mode);extra=['-I',str(include),str(include/'policy/executor/vendor/simulator.cpp')] if include else ['-Wall','-Wextra','-Wpedantic']
   cmd=[a.cxx,*flags,*extra,str(source),'-o',str(target)];(a.out/(target.name+'.command.json')).write_text(json.dumps(cmd,indent=2)+'\n')
   outputs={}
   for phase,command,limit in (('build',cmd,50),('run',[str(target)],15)):
    if phase=='run' and outputs['build']!=0:outputs['run']=None;break
    with (a.out/(target.name+'.'+phase+'.stdout')).open('w') as o,(a.out/(target.name+'.'+phase+'.stderr')).open('w') as e:
     done=subprocess.run(['/usr/bin/time','-v','-o',str(a.out/(target.name+'.'+phase+'.time')),'timeout','-k','3s',str(limit)+'s',*command],stdout=o,stderr=e,env=env)
    outputs[phase]=done.returncode
   row={'case':name,'mode':mode,'source':str(source),'command':cmd,'results':outputs};rows.append(row);print(json.dumps(row),flush=True)
 fixed_ok=all(row['results']=={'build':0,'run':0} for row in rows if row['case']=='fixed_standalone')
 result={'status':'FIXED_CASES_PASS' if fixed_ok else 'FIXED_CASE_FAILURE','compiler':subprocess.check_output([a.cxx,'--version'],text=True).splitlines()[0],'rows':rows,'seconds':time.perf_counter()-start,'new_games':0,'limitation':'Original reductions are diagnostic. A compiler that does not fail on original source cannot establish root cause of central GCC13 failure.'}
 (a.out/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True);raise SystemExit(not fixed_ok)
if __name__=='__main__':main()
