#!/home/keith/kaggriculture-rl-20260915/venv/bin/python
"""Build .arena/site/ppo.json for the arena dashboard's 'PPO run' tab. Read-only over the
training root; never touches training. Run by cron every 5 minutes on WRX90."""
import glob,json,os,sqlite3,subprocess,time
from pathlib import Path
R=Path('/home/keith/kaggriculture-ppo-production-20260919/broad-ppo-20260920');A=R/'arena-mix'
ARENA=Path('/home/keith/kaggriculture-arena/.arena');OUT=ARENA/'site'/'ppo.json';CACHE=Path('/home/keith/arena-maintenance/ppo-export-cache.json')
def sh(cmd,timeout=20):
    try:return subprocess.run(cmd,shell=True,capture_output=True,text=True,timeout=timeout).stdout.strip()
    except Exception:return ''
def rj(p,default=None):
    try:return json.loads(Path(p).read_text())
    except Exception:return default
def alive(pid):return pid and Path('/proc',str(pid)).exists()
cache=rj(CACHE,{}) or {}
now=time.time()
# ---- status ----
active=rj(R/'trainer-active.json',{});latest=rj(R/'training/latest.json',{});life=rj(R/'training/lifecycle.json',{})
launcher=int((R/'launcher.pid').read_text().strip()) if (R/'launcher.pid').exists() else None
ranks=[int(x) for x in sh(f"pgrep -P {launcher}").split()] if launcher else []
ctl=rj(R/'controller-active.json',{});ctl_pid=ctl.get('pid')
gpus=[l.split(', ') for l in sh("nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits").splitlines() if l]
disk={'wrx':sh("df -h --output=avail / | tail -1").strip()}
for host,key in (('arena-9975','vast_9975'),('arena-mini','mini')):
    v=sh(f"ssh -o BatchMode=yes -o ConnectTimeout=6 {host} 'df -h --output=avail / | tail -1' 2>/dev/null",15).strip()
    if v:disk[key]=v;cache['disk_'+key]=dict(value=v,at=now)
    elif cache.get('disk_'+key):disk[key]=cache['disk_'+key]['value']+' (cached)'
stops={k:(R/k).exists() for k in ('STOP','training/STOP','proxy-data/STOP')}
manifest=rj(R/'manifest.json',{})
status=dict(generated_at=now,trainer=dict(launcher_pid=launcher,launcher_alive=bool(alive(launcher)),rank_pids=ranks,ranks_alive=sum(alive(p) for p in ranks),source_sha256=active.get('source_sha256'),
    started_at=active.get('started_at'),purpose=active.get('purpose'),stored_update=latest.get('update'),stored_sha256=latest.get('sha256'),lifecycle=life),
    controller=dict(pid=ctl_pid,alive=bool(alive(ctl_pid))),stop_markers=stops,gpus=gpus,disk=disk,
    champion=dict(update=manifest.get('champion',{}).get('update'),sha256=manifest.get('champion',{}).get('sha256'),receipt=manifest.get('promotion_receipt'),version=manifest.get('version'),teacher_update=manifest.get('teacher',{}).get('update')))
# ---- per-update series ----
series=[]
for line in open(R/'training/metrics-000000.jsonl'):
    try:d=json.loads(line)
    except Exception:continue
    t=d.get('training',{});rk=d.get('ranks',[])
    fam={}
    for r in rk:
        for g in r.get('games',[]):
            if g.get('family')=='selfplay' or len(g.get('learner_seats',[]))!=1 or not g.get('cash'):continue
            l=g['learner_seats'][0];m=g['cash'][l]-g['cash'][1-l];f=fam.setdefault(g['family'],[0,0,0.0]);f[0]+=1;f[1]+=m>0;f[2]+=m
    series.append(dict(u=d['update'],at=d['at'],wall=round(d.get('wall_seconds',0),1),ppo=round(t.get('update_seconds',0),1),
        gpu=round(max((r.get('gpu_collection_seconds') or 0) for r in rk),1) if rk else None,official=round(max((r.get('official_collection_seconds') or 0) for r in rk),1) if rk else None,
        gps=round(d.get('end_to_end_games_per_second',0),3),kl=t.get('approx_kl'),clip=t.get('clipped'),ent=t.get('entropy'),vce=t.get('value'),huber=t.get('value_shaped'),pred=t.get('shaped_prediction'),
        gnorm=t.get('last_gradient_norm'),steps=t.get('optimizer_steps'),stop=bool(t.get('stopped_kl')),mem=max(t.get('rank_peak_allocated_gib') or [0]),valid=d.get('valid_games'),frozen=t.get('policy_frozen'),
        err=d.get('league_error'),practice={k:dict(n=v[0],wins=v[1],margin=round(v[2]/v[0]) if v[0] else None) for k,v in fam.items()}))
# ---- controller: panels, confirmations, promotions ----
c=sqlite3.connect(f"file:{R/'index.sqlite'}?mode=ro",uri=True)
panels=[];promotions=[];nominations=[]
for i,k,b in c.execute('select id,kind,body from events'):
    e=json.loads(b)
    if k=='development_evaluated':
        s=e.get('summary',{});opp=s.get('opponents',{});tg=sum(o.get('games',0) for o in opp.values())
        panels.append(dict(update=e.get('checkpoint',{}).get('update'),sha=e.get('checkpoint',{}).get('sha256','')[:12],wins=sum(o.get('wins',0) for o in opp.values()),games=tg,
            margin=round(sum((o.get('mean_margin') or 0)*o.get('games',0) for o in opp.values())/tg) if tg else None,
            opponents={n:dict(w=o.get('wins'),l=o.get('losses'),margin=round(o.get('mean_margin') or 0),cash=round(o.get('mean_cash') or 0),opp_cash=round(o.get('mean_opponent_cash') or 0)) for n,o in opp.items()}))
    elif k=='internal_champion_promoted':promotions.append(dict(id=i[:12],champion_update=e.get('champion',{}).get('update'),previous_update=e.get('previous',{}).get('update'),at=e.get('at')))
    elif k=='candidate_nominated':nominations.append(dict(id=i[:12],candidate_update=e.get('candidate',{}).get('update'),incumbent_update=e.get('incumbent',{}).get('update')))
panels.sort(key=lambda p:(p['update'] or 0))
confirmations=[]
for d in sorted(glob.glob(str(R/'confirmations/*/decision.json')),key=os.path.getmtime):
    x=rj(d,{});p=x.get('paired',{})
    confirmations.append(dict(id=Path(d).parent.name[:12],candidate_update=x.get('candidate',{}).get('update'),incumbent_update=x.get('incumbent',{}).get('update'),promoted=x.get('promoted'),
        delta=p.get('delta'),ci=p.get('ci95'),candidate_mean=x.get('candidate_scores',{}).get('mean'),incumbent_mean=x.get('incumbent_scores',{}).get('mean'),at=os.path.getmtime(d),
        families=x.get('candidate_scores',{}).get('families'),incumbent_families=x.get('incumbent_scores',{}).get('families')))
# ---- retention ----
retention=None
rets=sorted(glob.glob(str(R/'training/retention-*-rank-0.json')))
if rets:
    d=rj(rets[-1],{});rows=d.get('rows',[])
    retention=dict(update=int(Path(rets[-1]).name.split('-')[1]),read_only_verified=d.get('read_only_verified'),prefix_max=max((x.get('recurrent_reconstruction',{}).get('full_prefix_vs_recorded_max',0) for x in rows),default=None),
        bank_drift={f"{x.get('family')}@{x.get('retained_bank',{}).get('recorded_update')}":round(x.get('retained_bank',{}).get('mean_abs_logp_change',0),3) for x in rows})
# ---- epochs (objective/speed changes) ----
epochs=[dict(update=117,label='Broad branch begins (CP117, lambda 1)'),dict(update=307,label='Real-arena 50/25/25 mix'),dict(update=336,label='Shaped reward v4 (utility+residual baseline, 2 warm-up updates)'),dict(update=396,label='24 official actor processes per rank')]
# ---- arena Elo for team agents ----
team=[]
elo=rj(ARENA/'continuous-elo.json',{});ratings={r['agent']:r for r in elo.get('ratings',[])}
for f in glob.glob(str(ARENA/'agents/*.json')):
    a=rj(f,{});m=a.get('manifest',{})
    if m.get('author')=='keithtyser':
        r=ratings.get(a['id'],{});team.append(dict(name=m.get('name'),status=a.get('status'),elo=round(r.get('elo'),1) if r.get('elo') is not None else None,games=r.get('games'),retired=bool(a.get('retired_at'))))
out=dict(schema=1,updated=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime(now)),status=status,series=series,panels=panels,confirmations=confirmations,promotions=promotions,nominations=nominations,retention=retention,epochs=epochs,team_agents=team,
    notes=['Practice games use sampled decoding; dev panels and confirmations use greedy decoding.','Loss and utilization are not strength: only panels, confirmations and promotions are.','Branch updates = stored index - 116; games = updates x 512.'])
tmp=OUT.with_suffix('.tmp');tmp.write_text(json.dumps(out,separators=(',',':')));tmp.replace(OUT)
CACHE.write_text(json.dumps(cache))
# Ship directly to the dashboard host (web_sync.py is hash-pinned by the training pool and must stay untouched).
sh("scp -q -o BatchMode=yes -o ConnectTimeout=10 %s arena-mini:/home/keith/kaggriculture-continuous/.arena/web-incoming/ppo.json && ssh -o BatchMode=yes -o ConnectTimeout=10 arena-mini 'sudo -n /usr/local/sbin/arena-web-import'"%OUT,60)
print(json.dumps(dict(wrote=str(OUT),bytes=OUT.stat().st_size,series=len(series),panels=len(panels),confirmations=len(confirmations))))
