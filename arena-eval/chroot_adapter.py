"""Use the arena's guarded Vast sandbox for archived evaluation opponents."""
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import time

ACTIVE = {}


def install(config_path):
    from tools.arena import sandbox
    from tools.arena.chroot_worker import allocate, token, Process
    from tools.arena.store import read, write
    import tools.arena.chroot_worker as cw
    import tools.arena.worker as worker
    cfg = read(config_path)

    def args(_cfg, name, mounts, command, scratch_mb=512):
        base = Path(cfg['chroot_state'])
        if time.time()-read(base/'guard.json', {}).get('timestamp', 0)>5:
            raise RuntimeError('Sandbox guard is not healthy')
        receipt=read(Path(cfg['chroot_runtime']).parent/'runtime-receipt.json', {})
        if receipt.get('image')!=cfg['image'] or receipt.get('fingerprint')!=cfg['image_fingerprint']:
            raise RuntimeError('Sandbox runtime mismatch')
        uid=allocate(base); path=base/f'jail-{uid}'; record=base/'active'/f'{uid}.json'
        ACTIVE[name]=(path,record)
        write(record,dict(uid=uid,parent=os.getpid(),token=token(os.getpid()),path=str(path),
            deadline=time.time()+620,scratch_mb=scratch_mb,memory_mb=cfg['memory_mb']))
        subprocess.run(['cp','-al',cfg['chroot_runtime'],str(path)],check=True)
        for item in ['work','tmp']:
            p=path/item;p.mkdir(exist_ok=True);os.chown(p,uid,uid);p.chmod(0o700)
        (path/'dev').mkdir(exist_ok=True)
        for item,minor in [('null',3),('zero',5),('random',8),('urandom',9)]:
            p=path/'dev'/item
            if not p.exists():os.mknod(p,stat.S_IFCHR|0o666,os.makedev(1,minor))
        for source,target in mounts:
            p=path/target.lstrip('/');shutil.copyfile(source,p);p.chmod(0o644)
        cpus=sorted(os.sched_getaffinity(0))
        # league_eval adds Docker's '-i' at position 2. Our adapter removes it.
        return ['/usr/bin/setsid','/usr/bin/python3',str(Path(cw.__file__).with_name('chroot_launch.py')),
            str(path),str(uid),str(cpus[uid%len(cpus)]),str(cfg['memory_mb']),str(scratch_mb),
            '/usr/local/bin/python',*command[1:]]

    class GuardedProcess(Process):
        def __init__(self, command, timeout):
            if command[2]=='-i':command.pop(2)
            super().__init__(command,timeout)

    def cleanup(name):
        pair=ACTIVE.pop(name,None)
        if pair:
            path,record=pair
            violation=read(record,{}).get('violation')
            shutil.rmtree(path,ignore_errors=True);record.unlink(missing_ok=True)
            if violation:raise RuntimeError('Sandbox resource limit: '+str(violation))

    sandbox.docker_args=args;sandbox.cleanup=cleanup;worker.AgentProcess=GuardedProcess
