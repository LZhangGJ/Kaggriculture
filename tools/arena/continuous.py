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
        if (m['kind']!='continuous' and not (m['kind']=='daily' and m.get('parent_daily'))) or digest(m['contract'])!=m['contract_hash']:raise ValueError('Invalid executor contract')
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
    hosts=[('mini',read(root/'private/continuous-host.json',{}))]
    hosts += [(ident(name),cfg) for name,cfg in read(root/'private/continuous-extra-hosts.json',{}).items()]
    from .daily_dispatch import prepare
    prepare(root, hosts)
    statuses={}
    for name,cfg in hosts:
        if not cfg.get('enabled'):continue
        if name in statuses:raise ValueError('Duplicate executor name')
        statuses[name]=sync_host(root,name,cfg)
    if statuses:
        write(root/'continuous-status.json',dict(at=now(),status='running' if all(v['status']=='running' for v in statuses.values()) else 'connection_or_job_error',workers=statuses))


def owns_round(manifest,name):
    return manifest.get('executor','mini')==name


def sync_host(root,name,cfg):
    # Connection details and authentication remain solely in this private file / SSH config.
    host=cfg['ssh_host'];remote_root=cfg['root'];python=cfg['python'];repo=cfg['repository_path']
    def ssh(*args):
        command='cd '+shlex.quote(repo)+' && '+shlex.join([python,'-m','tools.arena.continuous',*args])
        subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',host,command],check=True,timeout=90,capture_output=True)
    def scp(source,destination):
        subprocess.run(['scp','-q','-o','BatchMode=yes','-o','ConnectTimeout=10',source,destination],check=True,timeout=180,capture_output=True)
    try:
        from .daily_dispatch import shard
        daily_pending = 0
        for path in sorted((root/'runs').glob('daily-*/manifest.json')):
            parent = read(path)
            if parent['kind'] != 'daily' or schedule.complete(root, parent):continue
            m = shard(root, parent, name)
            if not m:continue
            sent = root/'private/daily-sent'/f"{m['id']}.json"
            if sent.exists():
                ssh('export',remote_root,m['id'])
                local=root/'private'/f"received-{m['id']}.json"
                scp(host+':'+remote_root+'/outbox/'+m['id']+'.json',str(local))
                packet=read(local)
                if packet['run']!=m['id']:raise ValueError('Wrong daily result batch')
                allowed={g['id']:g for g in m['games']}
                for result in packet['results']:
                    if result['game'] not in allowed:raise ValueError('Unexpected daily game')
                    schedule.save_result(root,parent,allowed[result['game']],result)
            else:
                bundle=root/'private'/f"{m['id']}.zip"
                with zipfile.ZipFile(bundle,'w',compression=zipfile.ZIP_STORED) as z:
                    z.writestr('manifest.json',json.dumps(m))
                    for sha in sorted({a['archive'] for a in m['agents'].values()}):z.write(root/'artifacts'/f'{sha}.zip',sha+'.zip')
                target=remote_root+'/incoming/'+m['id']+'.zip'
                scp(str(bundle),host+':'+target)
                ssh('accept',remote_root,target)
                write(sent,dict(at=now()))
            daily_pending += sum(not read(path.parent/'games'/f"{g['id']}.json",{}).get('resolved') for g in m['games'])
        pending=[]
        for p in sorted((root/'runs').glob('continuous-*/manifest.json')):
            m=read(p)
            if not owns_round(m,name):continue
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
            rid=f'continuous-{sequence:08d}' if name=='mini' else f'continuous-{name}-{sequence:08d}'
            m=schedule.plan(root,rid,'continuous',cfg.get('seeds_per_round',4))
            m['executor']=name
            write(root/'runs'/rid/'manifest.json',m)
            pending.append(m)
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
        return dict(at=now(),status='running',pending_rounds=len(pending),daily_games_pending=daily_pending,
                    scheduling='daily tournament first; continuous resumes when assigned daily games finish')
    except (OSError,ValueError,KeyError,subprocess.SubprocessError) as e:
        event(root,'continuous_failed',name+'-'+now()[:13],{'reason':type(e).__name__})
        return dict(at=now(),status='connection_or_job_error')


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
