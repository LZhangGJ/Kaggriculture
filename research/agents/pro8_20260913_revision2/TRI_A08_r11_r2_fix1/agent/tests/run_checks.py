#!/usr/bin/env python3
"""Offline focused checks. No matches, opponents, random game seeds or networking.

Production must already have been built with the default build.py options.
Outputs go to a new rerun/<UTC>/ directory; delivered evidence is not overwritten.
"""
from __future__ import annotations
import argparse,datetime,json,os,pathlib,subprocess,sys
ROOT=pathlib.Path(__file__).resolve().parents[1]
PROD=ROOT/'candidate' if (ROOT/'candidate').exists() else ROOT
EVIDENCE=ROOT/'input' if (ROOT/'input').exists() else ROOT/'evidence'
p=argparse.ArgumentParser();p.add_argument('--sanitize',action='store_true');p.add_argument('--cxx',default=None);p.add_argument('--library',type=pathlib.Path);p.add_argument('--saved-limit',type=int,choices=[48,719]);a=p.parse_args()
stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');OUT=ROOT/'rerun'/stamp;OUT.mkdir(parents=True)
ENV=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
receipt=json.loads((PROD/'policy/tri_a08_r11_r2_fix1.BUILD.json').read_text())
base=[f for f in receipt.get('flags',receipt['command'][1:-5]) if f not in ('-fPIC','-shared','-Wl,-Bsymbolic') and not f.startswith('-DA08_ANIMAL_COLLECTION_LABOR=')]
results=[]
def run(name,command,seconds=40,stdout_path=None):
 (OUT/(name+'.command.json')).write_text(json.dumps(command,indent=2)+'\n')
 with (stdout_path or OUT/(name+'.stdout')).open('w') as o,(OUT/(name+'.stderr')).open('w') as e:
  r=subprocess.run(['/usr/bin/time','-v','-o',str(OUT/(name+'.time')),'timeout','-k','3s',str(seconds)+'s',*command],stdout=o,stderr=e,env=ENV)
 results.append({'name':name,'exit_code':r.returncode});(OUT/(name+'.exit')).write_text(str(r.returncode)+'\n')
 if r.returncode:raise RuntimeError(f'{name}: exit {r.returncode}; see {OUT}')
def compile_source(name,source,enabled=1,sanitize=False):
 flags=[*base,f'-DA08_ANIMAL_COLLECTION_LABOR={enabled}','-I',str(PROD)]
 if sanitize:flags=[x for x in flags if x not in ('-O3','-DNDEBUG')]+['-O1','-g','-fsanitize=address,undefined','-fno-sanitize-recover=all','-fno-omit-frame-pointer']
 out=OUT/name
 run(name+'_build',[a.cxx or receipt['command'][0],*flags,str(ROOT/'tests'/source),str(PROD/'policy/executor/vendor/simulator.cpp'),'-o',str(out)])
 return out
try:
 for enabled in (1,0):
  name=f'unit_{enabled}';binary=compile_source(name,'animal_collection_test.cpp',enabled=enabled)
  run(name,[str(binary)],15)
 if a.sanitize:
  binary=compile_source('unit_sanitize','animal_collection_test.cpp',sanitize=True);run('unit_sanitize',[str(binary)],20)
 binary=compile_source('emit_animal_paths','emit_animal_paths.cpp')
 scenarios=OUT/'official_scenarios.jsonl';run('emit_scenarios',[str(binary)],10,stdout_path=scenarios)
 run('official_execution',[sys.executable,str(ROOT/'tests/check_official_execution.py'),'--scenarios',str(scenarios),'--out',str(OUT/'official_execution.json')],20)
 run('native_choice_gate',[sys.executable,str(ROOT/'tests/native_choice_gate.py'),'--library',str(a.library or PROD/'policy/tri_a08_r11_r2_fix1.so'),'--out',str(OUT/'native_choice_gate.json')],20)
 run('entry_contract',[sys.executable,str(ROOT/'tests/test_entry_contract.py')],20)
 if a.saved_limit:
  for trace in sorted((EVIDENCE/'own_traces').glob('*.gz')):
   name='saved_'+trace.name.removesuffix('.json.gz')
   run(name,[sys.executable,str(EVIDENCE/'probe_observations.py'),'--agent',str(PROD/'main.py'),'--trace',str(trace),'--limit',str(a.saved_limit),'--out',str(OUT/(name+'.json'))],20)
 result={'status':'PASS','new_games':0,'output':str(OUT),'commands':results}
except Exception as e:
 result={'status':'FAIL','new_games':0,'output':str(OUT),'commands':results,'error':str(e)}
finally:
 (OUT/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
raise SystemExit(result['status']!='PASS')
