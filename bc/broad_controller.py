"""Bounded snapshot mailbox, official evaluations, and reversible internal promotions.

Run as a separate CPU process. No submission API and no trainer launch capability.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import sqlite3
import shutil
import secrets
import multiprocessing as mp
import fcntl
import time
import concurrent.futures
import subprocess
import sys
from contextlib import contextmanager
from ppo.league_runtime import atomic,digest,snapshot,admit,validate_manifest,rebuild_roster
from ppo.league import sha
from ppo.broad_curriculum import family_scores,paired_comparison,validate_registry
from ppo.league_eval import evaluate,paired_stats,regressions,runtime_identity,validate_panel


def screen_regressions(a,b,tolerance):
    if a['failures'] or b['failures']:return ['invalid_games']
    keys=set(a['opponents'])
    if keys!=set(b['opponents']):raise ValueError('Unmatched screen')
    return sorted(k for k in keys if a['opponents'][k]['score']-b['opponents'][k]['score'] < -tolerance)

@contextmanager
def manifest_lock(path):
    import fcntl
    with Path(str(path)+'.lock').open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX)
        try:yield
        finally:fcntl.flock(f,fcntl.LOCK_UN)

class Controller:
    def __init__(self,root,training,manifest,panel,arena):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        self.training=Path(training);self.manifest=Path(manifest);self.panel=json.loads(Path(panel).read_text());self.arena=arena
        self.db=sqlite3.connect(self.root/'index.sqlite',timeout=30)
        self.db.execute('create table if not exists events (id text primary key, kind text, body text)')
        self.db.execute('create table if not exists archive_metadata (sha text primary key, body text)')
        self.db.execute('create table if not exists state (key text primary key, body text)')
        self.db.execute('create table if not exists reserved_seeds (seed integer primary key)')
        self.db.commit()

    def get(self,key,default=None):
        row=self.db.execute('select body from state where key=?',(key,)).fetchone()
        return json.loads(row[0]) if row else default

    def put(self,key,value):
        self.db.execute('insert or replace into state values (?,?)',(key,json.dumps(value)));self.db.commit()

    def put_many(self,values):
        with self.db:
            self.db.executemany('insert or replace into state values (?,?)',[(k,json.dumps(v)) for k,v in values.items()])

    def event(self,key,kind,value):
        self.db.execute('insert or ignore into events values (?,?,?)',(key,kind,json.dumps(value)));self.db.commit()
        self.publish_event(key)

    def publish_event(self,key):
        kind,body=self.db.execute('select kind,body from events where id=?',(key,)).fetchone()
        try:atomic(self.root/'latest-event.json',dict(id=key,kind=kind,body=json.loads(body)))
        except OSError as exc:print(json.dumps(dict(reporting_error=repr(exc),committed_event=key)),flush=True)

    def seeds(self,count):
        reserved={180900001,180900002,180900003}|{v for o in self.panel['opponents'] for v in o.get('seeds',self.panel.get('seeds',[]))}
        values=[]
        with self.db:
            if self.db.execute('select count(*) from reserved_seeds').fetchone()[0]+len(reserved)+count>=10000000:
                raise ValueError('Evaluation seed namespace exhausted')
            while len(values)<count:
                value=180000000+secrets.randbelow(10000000)
                if value in reserved:continue
                if self.db.execute('insert or ignore into reserved_seeds values (?)',(value,)).rowcount:values.append(value)
        return values

    def fresh_panel(self,manifest,candidate):
        panel=copy.deepcopy(self.panel);seeds=self.seeds(64)
        for opp in panel['opponents']:opp['seeds']=seeds
        return panel

    def ingest(self):
        request=self.training/'snapshot-request.json'
        if not request.exists():return
        req=json.loads(request.read_text());key=req['sha256']
        if self.get('ingested')==key:return
        last=self.get('last_ingested_update',req['update']-10)
        if req['update']>last+10:
            path=self.training/'snapshots'/f'update-{last+10:06d}.pt'
            req=dict(path=str(path),sha256=sha(path),update=last+10);key=req['sha256']
        if sha(req['path'])!=key:raise ValueError('Published checkpoint changed')
        output=self.root/'archive'/f'{key}.pt'
        # The trainer exports model-only snapshots. Archive their exact bytes.
        output.parent.mkdir(parents=True,exist_ok=True)
        if not output.exists():
            temporary=output.with_suffix('.tmp');shutil.copyfile(req['path'],temporary);temporary.replace(output)
        if sha(output)!=key:raise ValueError('Archived snapshot changed')
        item=dict(path=str(output.resolve()),sha256=key,update=req['update'],family='ppo-lineage',pool='practice',decoding='sampled',eligible=True)
        with manifest_lock(self.manifest):
            manifest=validate_manifest(json.loads(self.manifest.read_text()))
            if manifest.get('last_snapshot_admission')!=key and not self.db.execute('select 1 from events where id=?',('snapshot-'+key,)).fetchone():
                manifest=admit(manifest,item);manifest['last_snapshot_admission']=key;atomic(self.manifest,manifest)
        old=self.get('pending_dev')
        if old:self.event('coalesced-'+old['sha256'],'evaluation_coalesced',old)
        event_key='snapshot-'+item['sha256']
        with self.db:
            self.db.execute('insert or ignore into archive_metadata values (?,?)',(key,json.dumps(item)))
            self.db.executemany('insert or replace into state values (?,?)',[(k,json.dumps(v)) for k,v in dict(pending_dev=item,ingested=key,last_ingested_update=req['update']).items()])
            self.db.execute('insert or ignore into events values (?,?,?)',(event_key,'snapshot_admitted',json.dumps(item)))
        self.publish_event(event_key)

    def develop(self,item):
        out=self.root/'development'/item['sha256']
        result,_=evaluate(item,self.panel,out,self.arena,self.root/'STOP')
        event_key='dev-'+item['sha256']
        with self.db:
            self.db.execute('insert or ignore into events values (?,?,?)',(event_key,'development_evaluated',json.dumps(dict(checkpoint=item,summary=result))))
            if not result['failures']:
                score=family_scores(result,self.panel)['mean']
                best=self.get('best_candidate')
                if not best or score>best['score'] or (score==best['score'] and item.get('update',-1)>best['checkpoint'].get('update',-1)):
                    self.db.execute('insert or replace into state values (?,?)',('best_candidate',json.dumps(dict(checkpoint=item,score=score))))
            # Ingestion may publish a newer mailbox while this evaluation runs.
            self.db.execute('delete from state where key=? and body=?',('pending_dev',json.dumps(item)))
        self.publish_event(event_key)
        self.audit(item)
        if item['update']+1-self.get('last_calibration',0)>=50:
            calibration=self.root/'value-calibration'/item['sha256']
            subprocess.run([sys.executable,'-u',str(self.root/'value_calibration.py'),'--plan',str(self.root/'proxy-plan.json'),'--checkpoint',item['path'],'--output',str(calibration)],env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1'),check=True)
            self.put('last_calibration',item['update']+1)
            self.event('calibration-'+item['sha256'],'value_calibration',json.loads((calibration/'summary.json').read_text()))

    def audit(self,item):
        if item['update']+1-self.get('last_audit',0)<50:return
        registry=validate_registry(json.loads((self.root/'opponents.json').read_text()))
        eligible=[r for r in registry['entries'] if r['pool']=='audit' and not r.get('audit_consumed')]
        if not eligible:
            atomic(self.root/'audit-status.json',dict(status='unavailable',reason='No unconsumed families with verified absence from BC and practice'))
            return
        out=self.root/'audits'/item['sha256'];out.mkdir(parents=True,exist_ok=True)
        assignment=out/'panel.json'
        if assignment.exists():panel=json.loads(assignment.read_text())
        else:
            seeds=self.seeds(64);panel=dict(config=self.panel['config'],opponents=[dict(r,seeds=seeds) for r in eligible[:8]])
            atomic(assignment,panel)
        result,_=evaluate(item,panel,out,self.arena,self.root/'STOP')
        self.event('audit-'+item['sha256'],'quarantined_audit',dict(checkpoint=item,summary=result,not_used_for_promotion=True))
        with manifest_lock(self.root/'opponents.json'):
            registry=validate_registry(json.loads((self.root/'opponents.json').read_text()))
            exposed={r['family'] for r in panel['opponents']}
            for row in registry['entries']:
                if row['family'] in exposed:row['audit_consumed']=dict(checkpoint=item['sha256'],reason='Results exposed; exclude from future unseen claims')
            atomic(self.root/'opponents.json',registry)
        self.put('last_audit',item['update']+1)

    def nominate(self):
        if self.get('active'):return
        candidate=self.get('best_candidate')
        if not candidate:return
        latest=json.loads((self.training/'latest.json').read_text())
        if latest['update']+1-self.get('last_nomination',-50)<50:return
        manifest=validate_manifest(json.loads(self.manifest.read_text()))
        c=candidate['checkpoint'];inc=manifest['champion']
        if c['sha256']==inc['sha256']:
            self.clear_candidate(c);return
        runtime=runtime_identity(self.arena)
        key=digest([c['sha256'],inc['sha256'],manifest['teacher']['sha256'],'broad-confirmation-v1',runtime])
        if self.db.execute('select 1 from events where id=?',(key,)).fetchone():
            self.clear_candidate(c);return
        panel=self.fresh_panel(manifest,c)
        active=dict(id=key,candidate=c,incumbent=inc,teacher=manifest['teacher'],panel=panel,seeds=self.seeds(32),phase='screen',runtime_sha=runtime,screen_protocol='broad-confirmation-v1')
        with self.db:
            self.db.executemany('insert or replace into state values (?,?)',[(k,json.dumps(v)) for k,v in dict(active=active,last_nomination=latest['update']+1).items()])
            self.db.execute('insert or ignore into events values (?,?,?)',('nomination-'+key,'candidate_nominated',json.dumps(active)))
        self.publish_event('nomination-'+key)

    def confirmation(self,job):
        out=self.root/'confirmations'/job['id'];out.mkdir(parents=True,exist_ok=True)
        if job['runtime_sha']!=runtime_identity(self.arena):raise ValueError('Confirmation runtime changed')
        if len(job['seeds'])!=32 or any(len(o['seeds'])!=64 or len(set(o['seeds']))!=64 for o in job['panel']['opponents']):raise ValueError('Wrong confirmation sample size')
        c,inc=job['candidate'],job['incumbent']
        a,ar=evaluate(c,job['panel'],out/'screen-candidate',self.arena,self.root/'STOP')
        b,br=evaluate(inc,job['panel'],out/'screen-incumbent',self.arena,self.root/'STOP')
        if job.get('screen_protocol')!='broad-confirmation-v1':raise ValueError('Unknown broad protocol')
        result=dict(candidate=c,incumbent=inc,promoted=False,protocol='broad-confirmation-v1',decoding='greedy',runtime_sha=job['runtime_sha'])
        if a['failures'] or b['failures']:
            result.update(comparison_accepted=False,reason='invalid_games')
        else:
            stats=paired_comparison(ar,br,job['panel'])
            sa,sb=family_scores(a,job['panel']),family_scores(b,job['panel'])
            bad=[k for k in sa['families'] if sa['families'][k]<sb['families'][k]-.20]
            accepted=stats['accepted'] and not bad and sa['lower_quartile']>=sb['lower_quartile']-.05
            result.update(comparison_accepted=accepted,paired=stats,candidate_scores=sa,incumbent_scores=sb,screen_regressions=bad,teacher=job['teacher'],advance_teacher=False)
            if accepted and job['teacher']['sha256']!=inc['sha256']:
                ts,tr=evaluate(job['teacher'],job['panel'],out/'screen-teacher',self.arena,self.root/'STOP')
                if ts['failures']:
                    result.update(comparison_accepted=False,reason='invalid_teacher_games');accepted=False
                else:
                    teacher_scores=family_scores(ts,job['panel']);teacher_stats=paired_comparison(ar,tr,job['panel'])
                    teacher_bad=[k for k in sa['families'] if sa['families'][k]<teacher_scores['families'][k]-.20]
                    result.update(teacher_scores=teacher_scores,teacher_paired=teacher_stats,
                        advance_teacher=teacher_stats['accepted'] and not teacher_bad and sa['lower_quartile']>=teacher_scores['lower_quartile']-.05)
            elif accepted:result['advance_teacher']=True
            if accepted:
                panel=dict(config=job['panel']['config'],opponents=[dict(inc,name='incumbent',kind='checkpoint',seeds=job['seeds'])])
                summary,rows=evaluate(c,panel,out/'head-to-head',self.arena,self.root/'STOP')
                result['head_to_head_support']=summary
                if summary['failures']:result.update(comparison_accepted=False,reason='invalid_support_games')
        atomic(out/'decision.json',result)
        self.finalize(job,result)

    def clear_candidate(self,candidate):
        best=self.get('best_candidate')
        if best and best['checkpoint']['sha256']==candidate['sha256']:
            self.db.execute('delete from state where key=? and body=?',('best_candidate',json.dumps(best)));self.db.commit()

    def finalize(self,job,result):
        terminal=self.db.execute('select kind,body from events where id=?',(job['id'],)).fetchone()
        if terminal:
            authoritative=json.loads(terminal[1])
            atomic(self.root/'confirmations'/job['id']/'decision.json',authoritative)
            self.publish_event(job['id'])
            return
        if result.get('comparison_accepted'):
            with manifest_lock(self.manifest):
                manifest=validate_manifest(json.loads(self.manifest.read_text()))
                if manifest['champion']['sha256']==result['incumbent']['sha256']:
                    if manifest['teacher']['sha256']!=job['teacher']['sha256']:raise ValueError('Teacher changed during confirmation')
                    teacher=result['candidate'] if result.get('advance_teacher') else manifest['teacher']
                    manifest.update(outgoing=result['incumbent'],champion=result['candidate'],teacher=teacher,retention=[manifest['retention'][0],teacher],champion_status='confirmed',promotion_receipt=job['id'])
                    manifest=rebuild_roster(manifest);atomic(self.manifest,manifest)
                result['promoted']=(manifest['champion']['sha256']==result['candidate']['sha256'] and manifest.get('promotion_receipt')==job['id'])
                if not result['promoted']:result['reason']='stale_incumbent'
        atomic(self.root/'confirmations'/job['id']/'decision.json',result)
        kind='internal_champion_promoted' if result['promoted'] else 'confirmation_failed'
        with self.db:
            self.db.execute('insert or ignore into events values (?,?,?)',(job['id'],kind,json.dumps(result)))
            active=self.get('active')
            if active and active['id']==job['id']:self.db.execute('delete from state where key=?',('active',))
            best=self.get('best_candidate')
            if best and best['checkpoint']['sha256']==job['candidate']['sha256']:self.db.execute('delete from state where key=?',('best_candidate',))
        self.publish_event(job['id'])

    def failed_attempt(self,kind,item,exc):
        if kind=='confirmation':
            decision=self.root/'confirmations'/item['id']/'decision.json'
            terminal=self.db.execute('select body from events where id=?',(item['id'],)).fetchone()
            if terminal or decision.exists():
                self.finalize(item,json.loads(terminal[0]) if terminal else json.loads(decision.read_text()))
                return
        key=item['id'] if kind=='confirmation' else item['sha256']
        count=self.get('attempt-'+key,0)+1;self.put('attempt-'+key,count)
        self.event('attempt-'+key+'-'+str(count),'evaluation_error',dict(error=repr(exc),attempt=count,request=key))
        if isinstance(exc,InterruptedError) and (self.root/'STOP').exists():return
        if isinstance(exc,(mp.TimeoutError,ConnectionError)) and count==1:return
        if kind=='confirmation':
            self.finalize(item,dict(candidate=item['candidate'],incumbent=item['incumbent'],promoted=False,reason='evaluation_error',error=repr(exc)))
        else:
            self.db.execute('delete from state where key=? and body=?',('pending_dev',json.dumps(item)));self.db.commit()

    def run(self):
        lock=self.root/'controller.lock'
        owner=lock.open('a+')
        fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
        owner.seek(0);owner.truncate();owner.write(str(os.getpid()));owner.flush()
        executor=concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future=None
        def task(kind,item):
            worker=Controller(self.root,self.training,self.manifest,self.root/'panel.json',self.arena)
            try:return worker.confirmation(item) if kind=='confirmation' else worker.develop(item)
            finally:worker.db.close()
        atomic(self.root/'panel.json',self.panel)
        try:
            while not (self.root/'STOP').exists():
                self.ingest()
                if future:
                    if not future.done():time.sleep(5);continue
                    try:future.result()
                    except Exception as exc:self.failed_attempt(*active_task,exc)
                    future=None
                self.nominate()
                active=self.get('active')
                if active:
                    decision=self.root/'confirmations'/active['id']/'decision.json'
                    if decision.exists():
                        result=json.loads(decision.read_text())
                        self.finalize(active,result)
                    else:
                        active_task=('confirmation',active);future=executor.submit(task,*active_task)
                elif self.get('pending_dev'):
                    active_task=('development',self.get('pending_dev'));future=executor.submit(task,*active_task)
                else:self.nominate();time.sleep(5)
        finally:executor.shutdown(wait=True);owner.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--training',required=True)
    p.add_argument('--manifest',required=True);p.add_argument('--panel',required=True);p.add_argument('--arena',required=True)
    args=p.parse_args();Controller(**vars(args)).run()
