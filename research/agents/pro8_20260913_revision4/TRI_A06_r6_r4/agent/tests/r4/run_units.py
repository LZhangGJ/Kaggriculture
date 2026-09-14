"""Offline delivery regression suite. Synthetic tests and declared own-only
continuations only; this script does not create a competitive match.
"""
from pathlib import Path
import argparse,hashlib,json,subprocess,sys,time,shutil,resource
R=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser();p.add_argument('--cxx',default=shutil.which('g++'));p.add_argument('--out',type=Path,default=R/'build/r4_units');a=p.parse_args();a.out=a.out.resolve();a.out.mkdir(parents=True,exist_ok=True)
flags=json.loads((R/'COMPILER_FLAGS.json').read_text());start=time.perf_counter();records=[]
def run(name,cmd,timeout=180):
 t=time.perf_counter();(a.out/(name+'.command.json')).write_text(json.dumps(cmd,indent=2));proc=subprocess.run(['/usr/bin/time','-v','-o',str(a.out/(name+'.time')),'timeout','-k','5',str(timeout),*map(str,cmd)],cwd=R,stdout=open(a.out/(name+'.stdout'),'w'),stderr=open(a.out/(name+'.stderr'),'w'));row={'name':name,'exit':proc.returncode,'seconds':time.perf_counter()-t};records.append(row);print(row,flush=True)
 if proc.returncode:raise RuntimeError(f'{name} failed: '+(a.out/(name+'.stderr')).read_text()[-4000:])
unit=a.out/'unit';lib=a.out/'audit.so';outflags=[f for f in flags if f not in ('-fPIC','-shared','-Wl,-Bsymbolic')]
run('compile_unit',[a.cxx,*outflags,'-I'+str(R/'policy'),str(R/'tests/r4/unit.cpp'),str(R/'policy/executor/vendor/simulator.cpp'),'-o',str(unit)])
run('synthetic',[str(unit)],90)
run('compile_audit',[a.cxx,*flags,'-DR4_TESTING=1','-I'+str(R/'policy'),str(R/'tests/r4/audit.cpp'),str(R/'policy/executor/vendor/simulator.cpp'),'-o',str(lib)])
# The text transport intentionally excludes the four multi-megabyte r4 replays.
# Reuse the exact parent's byte-identical official rules for artificial terminal
# fixtures; never rename an inherited replay as a round4 game.
feedback=R/'evidence/r4_feedback';omitted=[]
if feedback.exists():
 required={'aurax_reactive_v1_395620922_seat0.json.gz','submission_56149565_1075824552_seat0.json.gz','submission_56149565_563140739_seat1.json.gz','thomas_955_v2_20452605_seat1.json.gz'}
 assert required=={x.name for x in (feedback/'own_traces').glob('*.gz')},'incomplete round4 fixtures'
else:
 feedback=R/'evidence/r3_feedback'
 omitted=['warm_root_branches','root_prefixes']
 assert (feedback/'referee/official/kaggriculture.py').is_file()
 assert list((feedback/'own_traces').glob('*.gz')),'missing inherited fixtures'
common=['--root',str(R),'--feedback',str(feedback)]
run('terminal_paths',[sys.executable,str(R/'tests/r4/terminal_paths.py'),*common,'--lib',str(lib),'--out',str(a.out/'terminal_paths')],90)
if not omitted:
 run('warm_root_branches',[sys.executable,str(R/'tests/r4/check_live.py'),*common,'--lib',str(lib),'--out',str(a.out/'warm_root_branches')],180)
 run('root_prefixes',[sys.executable,str(R/'tests/r4/root_prefixes.py'),*common,'--out',str(a.out/'root_prefixes')],180)
else:
 print({'not_rerun':omitted,'reason':'r4 source-match replays not transmitted; no substituted match result'},flush=True)
summary={'pass':True,'not_rerun':omitted,'fixture_source':str(feedback.relative_to(R)),'new_competitive_games':0,'records':records,'production_native_sha256':hashlib.sha256((R/'policy/a06.so').read_bytes()).hexdigest(),'test_native_sha256':hashlib.sha256(lib.read_bytes()).hexdigest(),'seconds':time.perf_counter()-start,'rss_kib':resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss}
(a.out/'SUMMARY.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
