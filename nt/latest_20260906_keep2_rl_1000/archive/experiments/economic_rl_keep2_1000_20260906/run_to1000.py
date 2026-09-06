"""Sequential training controller. Fixed plan; no score-based early stopping."""
from pathlib import Path
import fcntl,json,subprocess,sys,time,hashlib,os

ROOT=Path(__file__).resolve().parent

def read(p):return json.loads(p.read_text(encoding='utf8'))
def save(p,obj):
    tmp=p.with_suffix(p.suffix+'.writing');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf8');tmp.replace(p)
def execute(script,log,*args):
    with log.open('a')as stream:
        subprocess.run([sys.executable,str(script),*args],stdout=stream,stderr=subprocess.STDOUT,check=True)

def main():
    lock=(ROOT/'RUN.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError('Continuation already running; do not start a second copy.')
    (ROOT/'logs').mkdir(exist_ok=True)
    milestone_file=ROOT/'MILESTONES.json'
    milestones=read(milestone_file)if milestone_file.exists()else []
    if not (ROOT/'STARTED.json').exists():save(ROOT/'STARTED.json',dict(unix=time.time(),pid=os.getpid()))
    try:
        for end in range(400,1001,100):
            p=ROOT/'stages'/f'round{end:04}';log=ROOT/'logs'/f'stage_{end:04}.log'
            if not (p/'MANIFEST.json').exists():
                for phase,script,receipt in [
                    ('preflight',f'preflight{end}.py','G0_ACCEPTANCE.json'),
                    ('training_and_monitoring',f'run_all{end}.py','FINAL_RESULTS.json'),
                    ('audit',f'audit{end}.py','FINAL_AUDIT.json')]:
                    save(ROOT/'CURRENT_STATUS.json',dict(status='RUNNING',end_round=end,phase=phase,pid=os.getpid(),updated=time.time()))
                    if not (p/receipt).exists():execute(p/script,log)
                execute(ROOT/'summarize_stage.py',log,str(end))
                execute(p/f'freeze_manifest{end}.py',log)
            assert read(p/'FINAL_AUDIT.json')['status']=='PASS'
            if not any(m['round']==end for m in milestones):
                m=dict(round=end,report=str(p/'SUMMARY_ZH.md'),audit=str(p/'FINAL_AUDIT.json'),completed=time.time())
                milestones.append(m)
                # The persistent list is the handoff source; print wakes tool readers.
                save(milestone_file,milestones)
                print('MILESTONE_COMPLETE',json.dumps(m,ensure_ascii=False),flush=True)
        save(ROOT/'CURRENT_STATUS.json',dict(status='RUNNING',end_round=1000,phase='independent_holdout',pid=os.getpid(),updated=time.time()))
        if not (ROOT/'independent1000'/'ACCEPTANCE.json').exists():
            execute(ROOT/'independent1000.py',ROOT/'logs'/'independent1000.log')
        save(ROOT/'CURRENT_STATUS.json',dict(status='COMPLETE',end_round=1000,phase='all_accepted',updated=time.time()))
        index={}
        for f in [*ROOT.glob('*.py'),ROOT/'PLAN_ZH.md',ROOT/'CURRENT_STATUS.json',ROOT/'MILESTONES.json',ROOT/'FINAL_SUMMARY_ZH.md',ROOT/'independent1000'/'MANIFEST.json',*sorted((ROOT/'stages').glob('round*/MANIFEST.json'))]:
            index[str(f.relative_to(ROOT))]=dict(bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest())
        save(ROOT/'MANIFEST_INDEX.json',dict(files=index,stage_payloads='Recursively covered by each immutable stage MANIFEST.json'))
        print('GOAL_COMPUTATION_COMPLETE',flush=True)
    except Exception as exc:
        prior=read(ROOT/'CURRENT_STATUS.json')if (ROOT/'CURRENT_STATUS.json').exists()else {}
        save(ROOT/'CURRENT_STATUS.json',dict(**{k:v for k,v in prior.items()if k not in ('status','updated')},status='ERROR',error=str(exc),updated=time.time()))
        raise

if __name__=='__main__':main()
