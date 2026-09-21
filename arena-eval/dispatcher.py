"""SSH dispatch to the arena's priority queues; never creates rated games."""
import copy
import concurrent.futures
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import time

from ppo import league_eval as core
from ppo.league_runtime import atomic, digest
from ppo.league import sha

CONFIG = Path(__file__).with_name('hosts.json')


def ssh(host, command):
    return subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',host,command],
                          check=True,capture_output=True,text=True,timeout=90).stdout


def copy_file(host, local, remote):
    subprocess.run(['scp','-q','-o','BatchMode=yes','-o','ConnectTimeout=10',str(local),host+':'+remote],
                   check=True,capture_output=True,timeout=180)


def remote_json(host, code):
    return json.loads(ssh(host,'python3 -c '+shlex.quote(code)))


def check_rows(rows, job):
    candidate,opp,seed,_,_=job
    if len(rows)!=2 or {r.get('seat') for r in rows}!={0,1}:raise ValueError('Incomplete remote seed pair')
    for row in rows:
        if row.get('seed')!=seed or row.get('opponent')!=opp['name']:raise ValueError('Remote assignment mismatch')
        if row.get('valid'):
            if (row.get('checkpoint_sha256')!=candidate['sha256'] or
                row.get('opponent_sha256')!=opp.get('sha256',opp.get('archive'))):
                raise ValueError('Remote checkpoint mismatch')
            cash=row.get('cash',[])
            if not(row.get('terminal') and row.get('reason')=='terminal' and row.get('statuses')==['DONE','DONE']
                   and len(cash)==2 and all(math.isfinite(v) for v in cash)):
                raise ValueError('Invalid remote terminal receipt')
            expected=.5 if cash[0]==cash[1] else float(cash[row['seat']]>cash[1-row['seat']])
            if row['score']!=expected:raise ValueError('Remote score mismatch')


def dispatch(host, jobs, output, request_id, stop):
    from cache_identity import engine_identity
    name=host['host'];root=host['root'];remote_out=root+'/results/'+request_id
    ssh(name,'mkdir -p '+shlex.quote(remote_out))
    host_cfg=remote_json(name,'import json;print(open('+repr(host['arena_root']+'/config.json')+').read())')
    files={};mapped=[]
    def asset(spec):
        spec=copy.deepcopy(spec)
        field='path' if spec.get('sha256') else 'file' if spec.get('archive') else None
        if field:
            identity=spec.get('sha256',spec.get('archive'));target=root+'/assets/'+identity+('.pt' if field=='path' else '.zip')
            if sha(spec[field])!=identity:raise ValueError('Source asset changed')
            exists=remote_json(name,'from pathlib import Path;import json;print(json.dumps(Path('+repr(target)+').exists()))')
            if not exists:
                copy_file(name,spec[field],target+'.tmp');ssh(name,'mv '+shlex.quote(target+'.tmp')+' '+shlex.quote(target))
            files[target]=identity;spec[field]=target
        return spec
    cached={}
    def once(spec):
        key=json.dumps(spec,sort_keys=True)
        if key not in cached:cached[key]=asset(spec)
        return copy.deepcopy(cached[key])
    for key,job in jobs:
        c,o,seed,cfg,_=job;cfg=copy.deepcopy(cfg)
        if cfg['image']!=host_cfg['image']:raise ValueError('Arena image differs')
        for k in ['image_runtime','image_fingerprint']:
            if k in host_cfg:cfg[k]=host_cfg[k]
        mapped.append((key,(once(c),once(o),seed,cfg,remote_out)))
    source=Path(core.__file__).resolve().parent.parent
    for p in source.rglob('*.py'):
        if '__pycache__' not in p.parts:files[root+'/source/'+str(p.relative_to(source))]=sha(p)
    # Pin the referee and sandbox implementation separately from model features.
    arena=Path(host['coordinator_arena'])
    for p in (arena/'tools/arena').glob('*.py'):files[root+'/arena/tools/arena/'+p.name]=sha(p)
    for filename in ['remote_worker.py','chroot_adapter.py']:
        files[root+'/'+filename]=sha(Path(__file__).with_name(filename))
    packet=dict(jobs=mapped,output=remote_out,files=files,engine=engine_identity())
    local=output/('dispatch-'+name+'.json')
    if local.exists() and json.loads(local.read_text())!=json.loads(json.dumps(packet)):
        raise ValueError('Frozen remote dispatch changed')
    atomic(local,packet)
    target=root+'/queue/'+request_id+'.json'
    # An unchanged queue request is safe to reattach to after a controller restart.
    copy_file(name,local,target+'.tmp');ssh(name,'mv '+shlex.quote(target+'.tmp')+' '+shlex.quote(target))
    expected=dict(jobs);start=time.monotonic()
    while True:
        if stop and Path(stop).exists():
            ssh(name,'touch '+shlex.quote(remote_out+'/STOP'));raise InterruptedError('Evaluation explicitly stopped')
        result=remote_json(name,"import json,pathlib; r=pathlib.Path("+repr(remote_out)+");print(json.dumps({p.stem:json.loads(p.read_text()) for p in r.glob('*.json')}))")
        for key,rows in result.items():
            if key in expected:
                check_rows(rows,expected[key]);atomic(output/(key+'.json'),rows)
        status=result.get('status',{})
        if status.get('state')=='failed':raise RuntimeError(name+': '+status.get('error','remote evaluation failed'))
        if all((output/(key+'.json')).exists() for key in expected):return
        if time.monotonic()-start>3000:raise TimeoutError('Remote evaluation did not finish: '+name)
        time.sleep(5)


def evaluate(candidate,panel,output,arena,stop=None):
    core.validate_panel(panel)
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    request=dict(candidate=candidate,panel=panel,decoding='greedy',runtime_sha=core.runtime_identity(arena))
    request_id=digest(request)
    if (output/'request.json').exists():
        if json.loads((output/'request.json').read_text())!=request:raise ValueError('Evaluation request mismatch')
    else:atomic(output/'request.json',request)
    jobs=[];all_jobs={}
    for opp in panel['opponents']:
        for seed in opp.get('seeds',panel.get('seeds',[])):
            key=digest([opp['name'],seed])
            job=(candidate,opp,seed,panel['config'],str(output))
            all_jobs[key]=(key,job)
            if (output/(key+'.json')).exists():check_rows(json.loads((output/(key+'.json')).read_text()),job)
            else:jobs.append((key,job))
    plan_path=output/'dispatch-plan.json'
    if plan_path.exists():
        plan=json.loads(plan_path.read_text());hosts=plan['hosts']
        if plan['request_id']!=request_id:raise ValueError('Dispatch request changed')
        assignments=[[all_jobs[key] for key in group] for group in plan['assignments']]
    else:
        hosts=json.loads(CONFIG.read_text())
        assignments=[[] for h in hosts]
        for job in jobs:
            eligible=[i for i,h in enumerate(hosts) if h.get('neural_opponents',True) or job[1][1].get('kind')!='checkpoint']
            i=min(eligible,key=lambda i:len(assignments[i])/hosts[i]['workers'])
            assignments[i].append(job)
        atomic(plan_path,dict(request_id=request_id,hosts=hosts,
                             assignments=[[key for key,job in group] for group in assignments]))
    atomic(output/'transport.json',dict(backend='arena-priority-v1',request_id=request_id,
        hosts={h['host']:len(a) for h,a in zip(hosts,assignments)},dispatcher_sha=sha(__file__),at=time.time()))
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(hosts)) as pool:
        futures=[pool.submit(dispatch,h,a,output,request_id,stop) for h,a in zip(hosts,assignments)
                 if any(not (output/(key+'.json')).exists() for key,job in a)]
        for f in futures:
            try:f.result()
            except (subprocess.SubprocessError, TimeoutError, OSError) as exc:
                raise ConnectionError('Arena evaluation transport failed: '+str(exc)) from exc
    rows=[]
    for opp in panel['opponents']:
        for seed in opp.get('seeds',panel.get('seeds',[])):
            rows.extend(json.loads((output/(digest([opp['name'],seed])+'.json')).read_text()))
    summary=core.summarize(rows,panel)
    summary.update(request_id=request_id,checkpoint_sha256=candidate['sha256'],panel=panel)
    atomic(output/'summary.json',summary)
    (output/'games.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    return summary,rows
