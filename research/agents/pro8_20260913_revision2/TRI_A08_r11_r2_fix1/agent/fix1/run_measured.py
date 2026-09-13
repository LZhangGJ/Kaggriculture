import sys,pathlib,subprocess,json,datetime,time,os
root=pathlib.Path('/mnt/data/fix1_work/logs');name=sys.argv[1];seconds=int(sys.argv[2]);cmd=sys.argv[3:]
(root/(name+'.command.json')).write_text(json.dumps(cmd,indent=2)+'\n');beg=datetime.datetime.now(datetime.timezone.utc).isoformat();tick=time.monotonic()
with (root/(name+'.stdout')).open('w') as o,(root/(name+'.stderr')).open('w') as e:
 r=subprocess.run(['/usr/bin/time','-v','-o',str(root/(name+'.time')),'timeout','-k','5s',str(seconds)+'s',*cmd],stdout=o,stderr=e,env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1'))
rec={'name':name,'command':cmd,'start_utc':beg,'end_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'wall_seconds':time.monotonic()-tick,'exit_code':r.returncode};(root/(name+'.json')).write_text(json.dumps(rec,indent=2)+'\n');print(json.dumps(rec),flush=True);print((root/(name+'.stdout')).read_text()[-3000:]);print((root/(name+'.stderr')).read_text()[-3000:]);sys.exit(r.returncode)
