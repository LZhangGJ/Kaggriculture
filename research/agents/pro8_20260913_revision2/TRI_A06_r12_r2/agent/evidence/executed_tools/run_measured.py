"""Single-process, timeout-bounded task with unambiguous time and exit receipts."""
from pathlib import Path
import argparse,datetime,json,subprocess,time,os
os.environ["PYTHONDONTWRITEBYTECODE"]="1"
p=argparse.ArgumentParser();p.add_argument('--log',required=True);p.add_argument('--timeout',type=int,default=120);p.add_argument('cmd',nargs=argparse.REMAINDER);a=p.parse_args()
log=Path(a.log);log.parent.mkdir(parents=True,exist_ok=True)
cmd=a.cmd[1:] if a.cmd and a.cmd[0]=='--' else a.cmd
full=['/usr/bin/time','-v','-o',str(log)+'.time','timeout','--kill-after=5s',str(a.timeout)+'s',*cmd]
start=datetime.datetime.now(datetime.timezone.utc).isoformat();t=time.monotonic()
with open(str(log)+'.stdout','w') as out,open(str(log)+'.stderr','w') as err:
 r=subprocess.run(full,stdout=out,stderr=err)
record=dict(started_utc=start,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-t,command=full,returncode=r.returncode)
Path(str(log)+'.exit.json').write_text(json.dumps(record,indent=2))
raise SystemExit(r.returncode)
