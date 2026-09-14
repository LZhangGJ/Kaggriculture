"""Rebuild a read-only parent inspection shim and reproduce the own route fixtures.
No matches, hidden observations or future tape are used by the candidate.
"""
from pathlib import Path
import argparse,ctypes,gzip,hashlib,importlib.util,json,subprocess,time,shutil
R=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--parent',type=Path,default=R/'parent');p.add_argument('--out',type=Path,default=R/'build/parent_fixture_export');p.add_argument('--cxx',default=shutil.which('g++'));a=p.parse_args();a.parent=a.parent.resolve();a.out=a.out.resolve();a.out.mkdir(parents=True,exist_ok=True)
lib=a.out/'parent_fixture_export.so';flags=json.loads((a.parent/'COMPILER_FLAGS.json').read_text());cmd=[a.cxx,*flags,'-I'+str(a.parent/'policy'),str(R/'tests/parent_fixture_export.cpp'),str(a.parent/'policy/executor/vendor/simulator.cpp'),'-o',str(lib)];(a.out/'compile.command.json').write_text(json.dumps(cmd,indent=2)+'\n');tick=time.perf_counter();q=subprocess.run(['timeout','-k','5s','150s',*cmd],stdout=open(a.out/'compile.stdout','w'),stderr=open(a.out/'compile.stderr','w'));assert q.returncode==0
s=importlib.util.spec_from_file_location('frozen_parent_entry',a.parent/'main.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);agent=m.create_agent(binary_path=lib);fn=agent.lib.td_fixture;fn.argtypes=[ctypes.c_void_p];fn.restype=ctypes.c_char_p
t=json.load(gzip.open(R/'evidence/r3_feedback/own_traces/submission_56149565_389573676_seat0.json.gz','rt'));rows=[]
for i,ob in enumerate(t['observations'][:-1]):
 act=agent(ob,t['configuration']);assert act==t['own_actions'][i],i
 if i in (218,314):
  f=json.loads(fn(agent.handle));f.update(next_observation_index=i+1,desired_hands=7 if i==218 else 8,game_id=t['game_id'],parent_native_sha256=hashlib.sha256((a.parent/'policy/a06.so').read_bytes()).hexdigest(),scope='Own controller service/route state exported after reproducing all parent actions up through this step.');rows.append(f)
 if i>=314:break
agent.close();assert rows==json.loads((R/'evidence/parent_service_fixtures.json').read_text())
(a.out/'parent_service_fixtures.json').write_text(json.dumps(rows,indent=2)+'\n');(a.out/'receipt.json').write_text(json.dumps({'parent_actions_reproduced':315,'fixtures':2,'fixture_json_exact':True,'seconds':time.perf_counter()-tick,'new_games':0},indent=2)+'\n');print('315 actions exact, 2 service fixtures exact; no games')
