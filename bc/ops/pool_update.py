#!/home/keith/kaggriculture-rl-20260915/venv/bin/python
"""Daily public-agent pool update for the broad-PPO continuation (user-authorized 2026-09-21).

Runs at 04:30 UTC, after the arena's midnight-EST public refresh. Finds public-origin agents that are
active on the arena roster but absent from the training pool, builds a new hash-pinned pool version
(additions only; the dev panel is untouched), verifies it with the trainer's own load_pool, then does
the standard boundary migration: STOP at the update boundary, pin the latest full checkpoint, write the
arena-pool-update-v1 migration, run ONE bounded validation update, accept, import once, resume
indefinitely. Every step writes receipts under arena-mix/pool-updates/<UTC date>/.

  --dry-run                 detect and build/verify the pool, but never touch training
  --simulate-remove <id>    dry-run only: pretend this pool member is a new roster agent (exercises build + verify)
"""
import copy,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
R=Path('/home/keith/kaggriculture-ppo-production-20260919/broad-ppo-20260920');A=R/'arena-mix';ARENA=Path('/home/keith/kaggriculture-arena/.arena')
SOURCE=A/'source-v9';PY='/home/keith/kaggriculture-rl-20260915/venv/bin/python-guarded'
DRY='--dry-run' in sys.argv
SIM=sys.argv[sys.argv.index('--simulate-remove')+1] if '--simulate-remove' in sys.argv else None
if SIM and not DRY:raise SystemExit('--simulate-remove requires --dry-run')
day=time.strftime('%Y-%m-%d',time.gmtime());OUTDIR=A/'pool-updates'/day;OUTDIR.mkdir(parents=True,exist_ok=True)
h=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def atomic(p,v):t=Path(str(p)+'.tmp');t.write_text(json.dumps(v,indent=2));t.replace(p)
def alive(pid):return bool(pid) and Path('/proc',str(pid)).exists()
def log(**kw):kw.setdefault('at',time.time());print(json.dumps(kw),flush=True);(OUTDIR/'log.jsonl').open('a').write(json.dumps(kw)+'\n')
if (OUTDIR/'done.json').exists() and not DRY:log(stage='already_done_today');sys.exit(0)
active=json.loads((R/'trainer-active.json').read_text())
cmd=active['command'];cur_pool_path=Path(cmd[cmd.index('--arena-pool')+1]);cur_pool=json.loads(cur_pool_path.read_text())
if SIM:
    cur_pool['opponents']=[o for o in cur_pool['opponents'] if not o['id'].startswith(SIM)]
    if len(cur_pool['opponents'])==len(json.loads(cur_pool_path.read_text())['opponents']):raise SystemExit('simulate-remove: no such pool member')
# ---- 1. detect ----
roster=json.loads((ARENA/'roster.json').read_text());have={o['id'] for o in cur_pool['opponents']};excluded={e['id'] for e in cur_pool.get('excluded',[])}
new=[]
for e in roster:
    a=json.loads((ARENA/'agents'/(e['agent']+'.json')).read_text());m=a.get('manifest',{})
    if e['agent'] in have or e['agent'] in excluded:continue
    if m.get('origin',{}).get('kind')!='public' or a.get('status')!='active' or not a.get('build_verified'):continue
    if a.get('image')!=cur_pool['config']['image']:log(stage='skip_image_mismatch',agent=e['agent'][:12],name=m.get('name'));continue
    f=ARENA/'artifacts'/(a['archive']+'.zip')
    if not f.exists() or h(f)!=a['archive']:log(stage='skip_bad_artifact',agent=e['agent'][:12]);continue
    new.append(dict(id=a['id'],name=m.get('name'),family=m.get('author') or 'unknown',manifest=m,archive=a['archive'],file=str(f),image=a['image']))
log(stage='detected',new=[(x['id'][:12],x['name'],x['family']) for x in new],pool=str(cur_pool_path),pool_size=len(cur_pool['opponents']))
if not new:
    if not DRY:atomic(OUTDIR/'done.json',dict(result='no_new_public_agents',at=time.time()))
    sys.exit(0)
# ---- 2. build + verify new pool ----
version=int(json.loads((A/'pool-version.json').read_text())['version'])+1 if (A/'pool-version.json').exists() else 2
pool=copy.deepcopy(cur_pool);pool['opponents']=cur_pool['opponents']+new
pool['derived_from']=dict(path=str(cur_pool_path),sha256=h(cur_pool_path));pool['added']=dict(date=day,agents=[x['id'] for x in new])
new_path=(OUTDIR/f'pool-v{version}-simulated.json') if SIM else (A/f'pool-v{version}.json');new_path.write_text(json.dumps(pool,indent=2))
for p,x in pool['runtime_files'].items():
    if h(p)!=x:log(stage='abort_runtime_pin_mismatch',file=p);sys.exit(2)
v=subprocess.run([PY,'-c',f"import sys;sys.path.insert(0,'{SOURCE}');from ppo.arena_opponents import load_pool;p=load_pool('{new_path}');print(len(p['opponents']),len({{o['family'] for o in p['opponents']}}))"],cwd=SOURCE,capture_output=True,text=True,env=dict(os.environ,PYTHONPATH=str(SOURCE)))
if v.returncode!=0:log(stage='abort_load_pool_failed',err=v.stderr[-600:]);sys.exit(2)
families=sorted({o['family'] for o in pool['opponents']});log(stage='pool_verified',path=str(new_path),sha256=h(new_path),opponents=len(pool['opponents']),families=len(families),load_pool=v.stdout.strip())
exposure=json.loads((A/'practice-exposure.json').read_text());exposure['families']=families;exposure['opponents']=sorted({o['archive'] for o in pool['opponents']});exposure.setdefault('history',[]).append(dict(date=day,pool=str(new_path),added=[x['id'] for x in new]))
if DRY:log(stage='dry_run_complete');sys.exit(0)
# ---- 3. boundary migration ----
S=OUTDIR;launcher=int((R/'launcher.pid').read_text().strip())
if launcher!=active['pid'] or not alive(launcher):log(stage='abort_no_live_trainer');sys.exit(2)
if (R/'STOP').exists() or (R/'training/STOP').exists():log(stage='abort_stop_present');sys.exit(2)
code=hashlib.sha256()
for p in sorted((SOURCE/'ppo').glob('*.py')):code.update(p.name.encode());code.update(p.read_bytes())
atomic(S/'maintenance.json',dict(purpose=f'arena-pool-update-v1 {day}: add {len(new)} public agents',at=time.time(),owner='claude campaign owner (daily job)',phase='stopping-at-boundary',previous_active=active))
(R/'training/STOP').write_text(json.dumps(dict(owner=str(S/'maintenance.json'),purpose='daily pool update boundary',at=time.time())))
log(stage='stop_requested')
deadline=time.time()+1500
while time.time()<deadline:
    if not alive(launcher) and not subprocess.run(['pgrep','-f','python-guarded -u -m ppo[.]train'],capture_output=True,text=True).stdout.strip():break
    time.sleep(10)
else:log(stage='abort_trainer_did_not_exit');sys.exit(2)
time.sleep(5)
life=json.loads((R/'training/lifecycle.json').read_text())
if life.get('phase')!='stopped' or subprocess.run(['docker','ps','--filter','name=ppo-practice','-q'],capture_output=True,text=True).stdout.split():log(stage='abort_unclean_exit',lifecycle=life);sys.exit(2)
latest=json.loads((R/'training/latest.json').read_text());src=Path(latest['path'])
if h(src)!=latest['sha256']:log(stage='abort_latest_hash');sys.exit(2)
pin=S/'handoff.pt';shutil.copyfile(src,pin)
if h(pin)!=latest['sha256']:log(stage='abort_pin_hash');sys.exit(2)
import torch
saved=torch.load(pin,map_location='cpu',weights_only=True);old=saved['ppo_contract']
newc=copy.deepcopy(old);newc['code_sha256']=code.hexdigest();newc['arena_pool_sha256']=h(new_path);newc['config']['arena_pool']=str(new_path)
migration=dict(kind='arena-pool-update-v1',source_sha256=latest['sha256'],old_contract=old,new_contract=newc,arena_pool_sha256=h(new_path),old_arena_pool=str(cur_pool_path),new_arena_pool=str(new_path),
    added=[dict(id=x['id'],name=x['name'],family=x['family']) for x in new],preserve=['model','Adam','RNG','league','teacher','losses','mix','LRs','lambda','KL guard','reward shaping','worker count'],at=time.time())
atomic(S/'migration.json',migration)
vcmd=list(cmd)
def flag(c,name,value):
    if name in c:c[c.index(name)+1]=str(value)
    else:c.extend([name,str(value)])
flag(vcmd,'--output',S/'validation');flag(vcmd,'--resume',pin);flag(vcmd,'--updates',1);flag(vcmd,'--migration',S/'migration.json');flag(vcmd,'--arena-pool',new_path)
settings=dict(active['settings']);settings['PYTHONPATH']=str(SOURCE);env=os.environ.copy();env.update(settings)
(S/'validation').mkdir(exist_ok=True);vlog=(S/'validation.log').open('a')
proc=subprocess.Popen(vcmd,cwd=SOURCE,env=env,stdout=vlog,stderr=subprocess.STDOUT,start_new_session=True)
atomic(S/'validation-launch.json',dict(command=vcmd,settings=settings,cwd=str(SOURCE),pid=proc.pid,source=latest,code_sha256=code.hexdigest(),at=time.time()))
log(stage='validation_launched',pid=proc.pid,resume=latest['update'])
rc=proc.wait();log(stage='validation_exited',rc=rc)
V=S/'validation';ok=False;checks={}
try:
    vl=json.loads((V/'latest.json').read_text());row=json.loads((V/'metrics-000000.jsonl').read_text().splitlines()[0]);t=row['training'];ranks=row['ranks']
    seen=set();[seen.update(r.get('arena_families',{}).keys()) for r in ranks]
    ck=torch.load(vl['path'],map_location='cpu',weights_only=True)
    checks=dict(rc=rc==0,update=vl['update']==latest['update']+1,valid_games=row['valid_games']==512,full_seasons=all(r['full_seasons']==len(r['games']) for r in ranks),
        mix=all(r['families']=={'selfplay':128,'champion_slot':64,'arena':64} for r in ranks),families_union=seen==set(families),families_per_rank=all(len(r.get('arena_families',{}))>=len(families)-1 for r in ranks),
        steps=t['optimizer_steps']>0,league_error=row['league_error'] is None,density=max(x['max_density_error'] for rk in t['audit_before'] for x in rk)<=.002,
        contract_pool=ck['ppo_contract']['arena_pool_sha256']==h(new_path) and ck['ppo_contract']['config']['arena_pool']==str(new_path),code=ck['ppo_contract']['code_sha256']==code.hexdigest(),
        adam_finite=all(torch.isfinite(v).all() for st in ck['optimizer']['state'].values() for v in st.values() if torch.is_tensor(v)),migration_receipt=(V/'migration-provenance.json').exists())
    ok=all(checks.values())
except Exception as e:checks['exception']=repr(e)[:300]
atomic(S/'full-acceptance.json',dict(passed=ok,checks=checks,at=time.time()));log(stage='acceptance',passed=ok,checks=checks)
if not ok:
    # Roll forward on the OLD pool: resume the pinned checkpoint with the exact old contract so training does not stay stopped.
    (R/'training/STOP').unlink(missing_ok=True)
    rcmd=list(cmd);flag(rcmd,'--resume',str(src));env0=os.environ.copy();env0.update(active['settings'])
    tl=(R/'train.log').open('a');tl.write('\n'+json.dumps(dict(stage='pool_update_validation_failed_resume_old_pool',at=time.time(),resume=latest))+'\n');tl.flush()
    p2=subprocess.Popen(rcmd,cwd=active['cwd'],env=env0,stdout=tl,stderr=subprocess.STDOUT,start_new_session=True)
    atomic(R/'trainer-active.json',dict(active,command=rcmd,source=latest,pid=p2.pid,started_at=time.time()));(R/'launcher.pid').write_text(str(p2.pid))
    atomic(OUTDIR/'done.json',dict(result='validation_failed_resumed_old_pool',pid=p2.pid,at=time.time()));log(stage='resumed_old_pool',pid=p2.pid);sys.exit(3)
# ---- 4. import once + resume on the new pool ----
target=R/'training'/Path(vl['path']).name;shutil.copyfile(vl['path'],str(target)+'.tmp');Path(str(target)+'.tmp').replace(target)
tm=R/'training/metrics-000000.jsonl';existing={json.loads(s)['update'] for s in tm.read_text().splitlines()}
if vl['update'] not in existing:tm.open('a').write(json.dumps(row)+'\n')
for extra in ('migration-provenance.json',):
    if (V/extra).exists():shutil.copyfile(V/extra,S/('imported-'+extra))
for name in (f"retention-{vl['update']:06d}-rank-0.json",f"retention-{vl['update']:06d}-rank-1.json"):
    if (V/name).exists() and not (R/'training'/name).exists():shutil.copyfile(V/name,R/'training'/name)
newlatest=dict(vl,path=str(target));atomic(R/'training/latest.json',newlatest)
atomic(A/'practice-exposure.json',exposure);atomic(A/'pool-version.json',dict(version=version,path=str(new_path),sha256=h(new_path),at=time.time()))
ncmd=list(cmd);flag(ncmd,'--resume',str(target));flag(ncmd,'--arena-pool',new_path);flag(ncmd,'--updates',0)
if '--migration' in ncmd:i=ncmd.index('--migration');del ncmd[i:i+2]
settings=dict(active['settings']);settings['PYTHONPATH']=str(SOURCE);env=os.environ.copy();env.update(settings)
(R/'training/STOP').unlink(missing_ok=True)
tl=(R/'train.log').open('a');tl.write('\n'+json.dumps(dict(stage='arena_pool_update_continuation',at=time.time(),pool=str(new_path),added=[x['id'] for x in new],resume=newlatest))+'\n');tl.flush()
p3=subprocess.Popen(ncmd,cwd=SOURCE,env=env,stdout=tl,stderr=subprocess.STDOUT,start_new_session=True)
newactive=dict(command=ncmd,settings=settings,cwd=str(SOURCE),source=newlatest,pid=p3.pid,started_at=time.time(),paused=False,purpose=active.get('purpose','')+f' | pool v{version} ({day})',source_sha256=code.hexdigest())
atomic(R/'trainer-active.json',newactive);(R/'launcher.pid').write_text(str(p3.pid));atomic(S/'indefinite-launch.json',newactive)
atomic(OUTDIR/'done.json',dict(result='migrated',pool=str(new_path),version=version,added=[x['id'] for x in new],validation_update=vl['update'],pid=p3.pid,at=time.time()))
log(stage='resumed_new_pool',pid=p3.pid,pool=str(new_path),families=len(families))
