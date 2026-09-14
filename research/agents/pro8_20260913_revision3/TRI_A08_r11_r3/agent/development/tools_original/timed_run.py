#!/usr/bin/env python3
import sys,os,time,json,datetime,subprocess
from pathlib import Path
name,seconds,*cmd=sys.argv[1:]; dest=Path('/mnt/data/r3_work/logs');dest.mkdir(exist_ok=True)
base=dest/name;start=time.time();record={'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'timeout_seconds':int(seconds),'command':cmd,'cwd':os.getcwd()}
record['memory_current_start']=Path('/sys/fs/cgroup/memory.current').read_text().strip()
with base.with_suffix('.stdout').open('w') as out,base.with_suffix('.stderr').open('w') as err:
 p=subprocess.run(['/usr/bin/time','-v','-o',str(base.with_suffix('.time')),'timeout','--signal=TERM','--kill-after=5',seconds,*cmd],stdout=out,stderr=err)
record.update(returncode=p.returncode,wall_seconds=time.time()-start,ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),memory_current_end=Path('/sys/fs/cgroup/memory.current').read_text().strip())
base.with_suffix('.result.json').write_text(json.dumps(record,indent=2)); print(json.dumps(record,indent=2));sys.exit(p.returncode)
