"""Run one bounded local command, capture exit/RSS and cgroup samples.
Used only inside this response; no scheduled or persistent task is installed.
"""
import argparse,datetime,json,os,pathlib,subprocess,time
p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--logs',type=pathlib.Path,required=True);p.add_argument('--seconds',type=int,default=180);p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
if a.command and a.command[0]=='--':a.command=a.command[1:]
a.logs.mkdir(parents=True,exist_ok=True);stem=a.logs/a.name
start=time.perf_counter();utc=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
info={'command':a.command,'started_utc':utc(),'timeout_seconds':a.seconds,'parent_monitor_pid':os.getpid()}
pathlib.Path(str(stem)+'.command.json').write_text(json.dumps(info,indent=2)+'\n')
def read(name):
 try:return pathlib.Path('/sys/fs/cgroup/'+name).read_text().strip()
 except OSError:return None
samples=[]
env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
with pathlib.Path(str(stem)+'.stdout').open('w') as out,pathlib.Path(str(stem)+'.stderr').open('w') as err:
 proc=subprocess.Popen(['/usr/bin/time','-v','-o',str(stem)+'.time','timeout','-k','5s',str(a.seconds)+'s',*a.command],stdout=out,stderr=err,env=env)
 while proc.poll() is None:
  samples.append({'elapsed_seconds':time.perf_counter()-start,'memory_current':read('memory.current'),'memory_peak':read('memory.peak'),'cpu_max':read('cpu.max'),'memory_max':read('memory.max')});time.sleep(.5)
 code=proc.returncode
info.update(finished_utc=utc(),wall_seconds=time.perf_counter()-start,exit_code=code,cgroup_samples=samples)
pathlib.Path(str(stem)+'.receipt.json').write_text(json.dumps(info,indent=2)+'\n');pathlib.Path(str(stem)+'.exit').write_text(str(code)+'\n');print(json.dumps({k:v for k,v in info.items() if k!='cgroup_samples'}),flush=True)
raise SystemExit(code)
