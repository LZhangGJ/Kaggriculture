"""Offline focused tests. All 24-tick exercises are forecasts, not new matches."""
from pathlib import Path
import argparse,ctypes,gzip,hashlib,importlib.util,json,os,resource,shutil,subprocess,sys,time
R=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--cxx',default=shutil.which('g++'));p.add_argument('--fixtures',type=Path,default=R/'evidence/own_visible');p.add_argument('--out',type=Path,default=R/'build/units');p.add_argument('--reuse-compiled',action='store_true');a=p.parse_args()
a.out=a.out.resolve();a.out.mkdir(parents=True,exist_ok=True);a.fixtures=a.fixtures.resolve()
if not a.cxx:raise SystemExit('C++20 compiler required')
flags=json.loads((R/'COMPILER_FLAGS.json').read_text());lib=a.out/'land_choice_checks.so'
cmd=[a.cxx,*flags,'-I'+str(R/'policy'),str(R/'tests/land_choice_checks.cpp'),str(R/'policy/executor/vendor/simulator.cpp'),'-o',str(lib)]
tick=time.perf_counter()
if not a.reuse_compiled:
 (a.out/'compile.command.json').write_text(json.dumps(cmd,indent=2)+'\n')
 r=subprocess.run(['timeout','-k','5s','150s',*cmd],capture_output=True,text=True);(a.out/'compile.stdout').write_text(r.stdout);(a.out/'compile.stderr').write_text(r.stderr);(a.out/'compile.exit').write_text(str(r.returncode)+'\n')
 if r.returncode:print(r.stderr);raise SystemExit(r.returncode)
 (a.out/'compile.receipt.json').write_text(json.dumps({'command':cmd,'seconds':time.perf_counter()-tick,'compiler':subprocess.check_output([a.cxx,'--version'],text=True),'test_native_sha256':hashlib.sha256(lib.read_bytes()).hexdigest()},indent=2)+'\n');print('test native compiled',flush=True)
spec=importlib.util.spec_from_file_location('unit_production_main',R/'main.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
from official_prefix import load_referee,check_prefix
runtime,engine=load_referee(a.fixtures);prefix_checks=[]
results=[];scope='Own-visible invariant checks and 24-tick declared public-flow forecast exercises. No new games, opponents, win rate, or recovered cash.'
for f in sorted((a.fixtures/'own_traces').glob('*.gz')):
 t=json.load(gzip.open(f,'rt'));agent=m.create_agent(binary_path=lib);fn=agent.lib.td_r2_choice_checks;fn.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_double),ctypes.c_size_t,ctypes.c_int];fn.restype=ctypes.c_char_p
 buys={i for i,x in enumerate(t['own_actions']) if any(y[0]=='BUY_LAND' for y in x['market'])};rows=[]
 for i,ob in enumerate(t['observations'][:-1]):
  if ob['hour']==0 and (i in buys or i in (0,696)):
   packed=m.codec._pack(ob);raw=fn(agent.handle,packed,len(packed),int(i in buys)).decode()
   if raw.startswith('ERROR'):
    (a.out/(t['game_id']+'.failure.json')).write_text(json.dumps({'id':t['game_id'],'step':i,'error':raw},indent=2));raise RuntimeError(raw+f' at {t["game_id"]} {i}')
   rows.append(json.loads(raw))
   for branch in rows[-1]['branch_exercises']:
    checked=check_prefix(ob,t['configuration'],branch,m.codec,runtime,engine);checked['game_id']=t['game_id'];prefix_checks.append(checked)
    (a.out/'OFFICIAL_PREFIX_CHECKS.json').write_text(json.dumps(prefix_checks,indent=2)+'\n')
    if not checked['pass']:raise RuntimeError('official prefix mismatch '+str(checked))
  agent(ob,t['configuration'])
 agent.close();(a.out/(t['game_id']+'.json')).write_text(json.dumps({'scope':scope,'new_games':0,'checks':rows},separators=(',',':'))+'\n');results.extend(rows);print(t['game_id'],'checks',len(rows),flush=True)
summary={'scope':scope,'new_games':0,'official_prefix_checks':len(prefix_checks),'official_prefix_passes':sum(x['pass'] for x in prefix_checks),'fixture_states':len(results),'original_candidates_checked':sum(x['original'] for x in results),'deferred_candidates_checked':sum(x['deferred'] for x in results),'funded_commitments_checked':sum(x['funded_commitments_checked'] for x in results),'conditional_branch_exercises':sum(len(x['branch_exercises']) for x in results),'conditional_steps':sum(y['ticks'] for x in results for y in x['branch_exercises']),'rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'seconds':time.perf_counter()-tick}
assert summary['deferred_candidates_checked']>0
(a.out/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)
