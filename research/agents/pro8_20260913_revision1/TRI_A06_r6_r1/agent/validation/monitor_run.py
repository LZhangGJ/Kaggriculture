#!/usr/bin/env python3
"""Bounded command runner with persisted resource, time, stdout/stderr receipt."""
import argparse,datetime,json,os,pathlib,signal,subprocess,time,resource
p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--timeout',type=float,required=True);p.add_argument('--cwd');p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args();cmd=a.command
if cmd and cmd[0]=='--':cmd=cmd[1:]
root=pathlib.Path(__file__).resolve().parent;logs=root/'logs';logs.mkdir(exist_ok=True)
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
deadline=datetime.datetime.fromisoformat('2026-09-13T04:17:41.523+00:00').timestamp();allowed=min(a.timeout,deadline-time.time()-60)
if allowed<=0:raise SystemExit('Budget exhausted')
r={'name':a.name,'command':cmd,'cwd':a.cwd,'started_utc':utc(),'timeout_seconds':allowed,'peak_tree_rss_kb_sampled':0,'peak_cgroup_current':0,'timed_out':False};t=time.perf_counter()
with (logs/(a.name+'.stdout')).open('w') as out,(logs/(a.name+'.stderr')).open('w') as err:
 proc=subprocess.Popen(cmd,cwd=a.cwd,stdout=out,stderr=err,start_new_session=True);r['pid']=proc.pid;(logs/(a.name+'.running.json')).write_text(json.dumps(r,indent=2))
 while proc.poll() is None:
  try:
   stats={}
   for q in pathlib.Path('/proc').glob('[0-9]*/stat'):
    try:
     s=q.read_text().split(') ',1)[1].split();stats[int(q.parent.name)]=(int(s[1]),int(s[21])*os.sysconf('SC_PAGE_SIZE')//1024)
    except (OSError,ValueError,IndexError):pass
   ids={proc.pid};old=0
   while len(ids)!=old:
    old=len(ids);ids.update(i for i,(parent,_) in stats.items() if parent in ids)
   r['peak_tree_rss_kb_sampled']=max(r['peak_tree_rss_kb_sampled'],sum(stats.get(i,(0,0))[1] for i in ids));r['peak_cgroup_current']=max(r['peak_cgroup_current'],int(pathlib.Path('/sys/fs/cgroup/memory.current').read_text()))
  except OSError:pass
  if time.perf_counter()-t>allowed:
   r['timed_out']=True;os.killpg(proc.pid,signal.SIGTERM)
   try:proc.wait(timeout=3)
   except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL)
   break
  time.sleep(.10)
 r['returncode']=proc.wait();r['wall_seconds']=time.perf_counter()-t;r['finished_utc']=utc();r['ru_maxrss_children_kb']=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss;(logs/(a.name+'.receipt.json')).write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2),flush=True)
raise SystemExit(r['returncode'])
