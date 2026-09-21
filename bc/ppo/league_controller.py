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
from contextlib import contextmanager
from ppo.league_runtime import atomic,digest,snapshot,admit,validate_manifest,rebuild_roster
from ppo.league import sha
from ppo.league_eval import evaluate,paired_stats,regressions,runtime_identity,validate_panel

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
        panel=copy.deepcopy(self.panel)
        for opp in panel['opponents']:
            opp['seeds']=self.seeds(len(opp.get('seeds',panel.get('seeds',[]))))
        pool=([manifest['outgoing']] if manifest.get('outgoing') else [])+manifest.get('coverage',[])+list(reversed(manifest['roster']))
        unique={r['sha256']:r for r in pool if r['sha256'] not in (manifest['champion']['sha256'],candidate['sha256'])}
        sentinels=list(unique.values())[:2]
        for i,row in enumerate(sentinels):
            panel['opponents'].append(dict(row,name='history-'+str(i),kind='checkpoint',seeds=self.seeds(8)))
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
        item=dict(path=str(output.resolve()),sha256=key,update=req['update'])
        with manifest_lock(self.manifest):
            manifest=validate_manifest(json.loads(self.manifest.read_text()))
            if manifest.get('last_snapshot_admission')!=key and not self.db.execute('select 1 from events where id=?',('snapshot-'+key,)).fetchone():
                manifest=admit(manifest,item);manifest['last_snapshot_admission']=key;atomic(self.manifest,manifest)
        old=self.get('pending_dev')
        if old:self.event('coalesced-'+old['sha256'],'evaluation_coalesced',old)
        event_key='snapshot-'+item['sha256']
        with self.db:
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
                score=sum(r['score'] for k,r in result['opponents'].items() if k!='PASS')
                best=self.get('best_candidate')
                if not best or score>best['score'] or (score==best['score'] and item.get('update',-1)>best['checkpoint'].get('update',-1)):
                    self.db.execute('insert or replace into state values (?,?)',('best_candidate',json.dumps(dict(checkpoint=item,score=score))))
            # Ingestion may publish a newer mailbox while this evaluation runs.
            self.db.execute('delete from state where key=? and body=?',('pending_dev',json.dumps(item)))
        self.publish_event(event_key)

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
        key=digest([c['sha256'],inc['sha256'],'confirmation-v2',runtime])
        if self.db.execute('select 1 from events where id=?',(key,)).fetchone():
            self.clear_candidate(c);return
        panel=self.fresh_panel(manifest,c)
        active=dict(id=key,candidate=c,incumbent=inc,panel=panel,seeds=self.seeds(256),phase='screen',runtime_sha=runtime)
        with self.db:
            self.db.executemany('insert or replace into state values (?,?)',[(k,json.dumps(v)) for k,v in dict(active=active,last_nomination=latest['update']+1).items()])
            self.db.execute('insert or ignore into events values (?,?,?)',('nomination-'+key,'candidate_nominated',json.dumps(active)))
        self.publish_event('nomination-'+key)

    def confirmation(self,job):
        out=self.root/'confirmations'/job['id'];out.mkdir(parents=True,exist_ok=True)
        if job['runtime_sha']!=runtime_identity(self.arena):raise ValueError('Confirmation runtime changed')
        c,inc=job['candidate'],job['incumbent']
        a,ar=evaluate(c,job['panel'],out/'screen-candidate',self.arena,self.root/'STOP')
        b,br=evaluate(inc,job['panel'],out/'screen-incumbent',self.arena,self.root/'STOP')
        bad=regressions(a,b)
        if bad and bad!=['invalid_games']:
            if 'recheck' not in job:
                panel=copy.deepcopy(job['panel']);panel['opponents']=[o for o in panel['opponents'] if o['name'] in bad]
                for opp in panel['opponents']:opp['seeds']=self.seeds(32)
                job['recheck']=panel;self.put('active',job)
            ra,_=evaluate(c,job['recheck'],out/'recheck-candidate',self.arena,self.root/'STOP')
            rb,_=evaluate(inc,job['recheck'],out/'recheck-incumbent',self.arena,self.root/'STOP')
            bad=regressions(ra,rb)
        result=dict(candidate=c,incumbent=inc,screen_regressions=bad,promoted=False,protocol='confirmation-v2',
                    per_comparison_alpha=.05,decoding='greedy',runtime_sha=job['runtime_sha'])
        if not bad:
            panel=dict(config=self.panel['config'],opponents=[dict(inc,name='incumbent',kind='checkpoint',seeds=job['seeds'])])
            summary,rows=evaluate(c,panel,out/'head-to-head',self.arena,self.root/'STOP')
            stats=paired_stats(rows,'incumbent') if not summary['failures'] else dict(accepted=False,reason='invalid_games')
            result['head_to_head']=stats
            result['comparison_accepted']=stats['accepted']
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
                    manifest.update(outgoing=result['incumbent'],champion=result['candidate'],champion_status='confirmed',promotion_receipt=job['id'])
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
