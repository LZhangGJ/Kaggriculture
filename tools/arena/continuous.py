"""CPU executor and private SSH transport for endless, centrally seeded rounds."""
import json
import shlex
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from . import controller, intake, schedule
from .store import digest, file_hash, ident, read, write, now, locked, event


def accept(root,bundle):
    root=Path(root);intake.inspect_zip(bundle)
    with zipfile.ZipFile(bundle) as z:
        m=json.loads(z.read('manifest.json'));rid=ident(m['id'])
        if m['kind']!='continuous' or digest(m['contract'])!=m['contract_hash']:raise ValueError('Invalid continuous contract')
        cfg=read(root/'config.json')
        if cfg['image']!=m['contract']['image']:raise ValueError('Executor image differs')
        old=read(root/'runs'/rid/'manifest.json')
        if old and old!=m:raise ValueError('Frozen round changed')
        for aid,a in m['agents'].items():
            ident(aid);sha=ident(a['archive']);p=root/'artifacts'/f'{sha}.zip'
            if not p.exists():
                data=z.read(sha+'.zip')
                import hashlib
                if hashlib.sha256(data).hexdigest()!=sha:raise ValueError('Artifact hash mismatch')
                p.write_bytes(data)
            if file_hash(p)!=sha:raise ValueError('Stored artifact corrupted')
            write(root/'agents'/f'{aid}.json',a)
        write(root/'runs'/rid/'manifest.json',m)


def export_results(root,rid):
    root=Path(root);rid=ident(rid)
    rows=[read(p) for p in sorted((root/'runs'/rid/'games').glob('*.json'))]
    destination=root/'outbox'/f'{rid}.json'
    write(destination,dict(run=rid,results=rows))


def sync(root):
    root=Path(root)
    cfg=read(root/'private/continuous-host.json',{})
    if not cfg.get('enabled'):return
    # Connection details and authentication remain solely in this private file / SSH config.
    host=cfg['ssh_host'];remote_root=cfg['root'];python=cfg['python'];repo=cfg['repository_path']
    def ssh(*args):
        command='cd '+shlex.quote(repo)+' && '+shlex.join([python,'-m','tools.arena.continuous',*args])
        subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',host,command],check=True,timeout=90,capture_output=True)
    def scp(source,destination):
        subprocess.run(['scp','-q','-o','BatchMode=yes','-o','ConnectTimeout=10',source,destination],check=True,timeout=180,capture_output=True)
    try:
        pending=[]
        for p in sorted((root/'runs').glob('continuous-*/manifest.json')):
            m=read(p)
            if schedule.complete(root,m):continue
            if (p.parent/'sent.json').exists():
                ssh('export',remote_root,m['id'])
                local=root/'private'/f"received-{m['id']}.json"
                scp(host+':'+remote_root+'/outbox/'+m['id']+'.json',str(local))
                packet=read(local)
                if packet['run']!=m['id']:raise ValueError('Wrong result batch')
                allowed={g['id']:g for g in m['games']}
                for result in packet['results']:
                    if result['game'] not in allowed:raise ValueError('Unexpected game')
                    schedule.save_result(root,m,allowed[result['game']],result)
            if not schedule.complete(root,m):pending.append(m)
        # Keep two finite rounds queued so the executor need not await each sync.
        while len(pending)<2:
            sequence=len(list((root/'runs').glob('continuous-*/manifest.json')))+1
            pending.append(schedule.plan(root,f'continuous-{sequence:08d}','continuous',cfg.get('seeds_per_round',4)))
        for m in pending:
            sent=root/'runs'/m['id']/'sent.json'
            if sent.exists():continue
            bundle=root/'private'/f"{m['id']}.zip"
            with zipfile.ZipFile(bundle,'w',compression=zipfile.ZIP_STORED) as z:
                z.writestr('manifest.json',json.dumps(m))
                for sha in sorted({a['archive'] for a in m['agents'].values()}):z.write(root/'artifacts'/f'{sha}.zip',sha+'.zip')
            target=remote_root+'/incoming/'+m['id']+'.zip'
            scp(str(bundle),host+':'+target)
            ssh('accept',remote_root,target)
            write(sent,dict(at=now()))
        from .elo import update
        update(root)
        write(root/'continuous-status.json',dict(at=now(),status='running',pending_rounds=len(pending)))
    except (OSError,ValueError,KeyError,subprocess.SubprocessError) as e:
        write(root/'continuous-status.json',dict(at=now(),status='connection_or_job_error'))
        event(root,'continuous_failed',now()[:13],{'reason':type(e).__name__})


def execute(root):
    root=Path(root)
    while True:
        with locked(root):
            cfg=read(root/'config.json')
            count=controller.work(root,cfg,remote_only=True)
            write(root/'executor-status.json',dict(at=now(),games_attempted=count))
        time.sleep(1 if count else 5)


if __name__=='__main__':
    command,root,*rest=sys.argv[1:]
    if command=='accept':accept(root,rest[0])
    elif command=='export':export_results(root,rest[0])
    elif command=='execute':execute(root)
    elif command=='sync':
        with locked(root):sync(root)
    else:raise ValueError('Unknown command')
