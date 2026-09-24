"""Portable Python referee + Python policies. No Kaggle account or GPU required."""
from pathlib import Path
import argparse,concurrent.futures as cf,contextlib,copy,hashlib,importlib.util,inspect,io,json,multiprocessing as mp,os,random,sys,time,traceback
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from cpu_runtime import LocalGame,load_engine

class Policy:
    def __init__(self,path,name):
        self.path=Path(path).resolve();self.folder=self.path.parent;self.modules={};previous=dict(sys.modules)
        with self.context():
            spec=importlib.util.spec_from_file_location(name,self.path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module
            spec.loader.exec_module(module);self.fn=module.agent
            signature=inspect.signature(self.fn)
            try:signature.bind({},{});self.arity=2
            except TypeError:signature.bind({});self.arity=1
            for key,item in list(sys.modules.items()):
                filename=getattr(item,'__file__',None)
                if filename and Path(filename).resolve().is_relative_to(self.folder):self.modules[key]=item
        # Each policy keeps its own local helpers even when filenames overlap.
        for key in self.modules:
            if key in previous:sys.modules[key]=previous[key]
            else:sys.modules.pop(key,None)
    @contextlib.contextmanager
    def context(self):
        directory=Path.cwd();paths=list(sys.path);saved={k:sys.modules.get(k)for k in self.modules}
        try:
            os.chdir(self.folder);sys.path.insert(0,str(self.folder));sys.modules.update(self.modules);yield
        finally:
            for key,value in saved.items():
                if value is None:sys.modules.pop(key,None)
                else:sys.modules[key]=value
            os.chdir(directory);sys.path[:]=paths
    def __call__(self,observation,configuration):
        with self.context():return self.fn(observation,configuration)if self.arity==2 else self.fn(observation)

def play(job):
    opponent,candidate,seed,seat=job;started=time.monotonic();row=dict(opponent=opponent['id'],seed=seed,candidate_seat=seat,error=None)
    try:
        # Capture public policy diagnostics so concurrent stdout stays readable.
        with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
            own=Policy(candidate,'candidate_entry');rival=Policy(ROOT/opponent['working']/'main.py','opponent_entry')
            env=LocalGame(seed,load_engine())
            while not env.done:
                assert env.configuration.seed is None
                action=[None,None]
                action[seat]=own(env.observation(seat),copy.deepcopy(env.configuration))
                action[1-seat]=rival(env.observation(1-seat),copy.deepcopy(env.configuration))
                env.advance(action)
            cash=[f['money']for f in env.state[0].observation.farms];delta=cash[seat]-cash[1-seat]
            row.update(steps=env.t,candidate_cash=cash[seat],opponent_cash=cash[1-seat],margin=delta,win=delta>0,tie=delta==0)
    except Exception:row['error']=traceback.format_exc()
    row['seconds']=time.monotonic()-started;return row

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--candidate',type=Path,required=True);parser.add_argument('--seed-count',type=int,default=2)
    parser.add_argument('--seed-salt',type=int,default=2026091501);parser.add_argument('--seeds-json',type=Path);parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--opponent',action='append');parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    assert 1<=args.workers<=16 and args.seed_count>0
    candidate=args.candidate.resolve();assert candidate.is_file();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    pool=json.loads((ROOT/'POOL.json').read_text(encoding='utf8'))
    if args.opponent:
        assert set(args.opponent)<={p['id']for p in pool};pool=[p for p in pool if p['id']in args.opponent]
    seeds=json.loads(args.seeds_json.read_text(encoding='utf8'))if args.seeds_json else random.Random(args.seed_salt).sample(range(1_000_000_000,2_000_000_000),args.seed_count)
    assert isinstance(seeds,list)and len(set(seeds))==len(seeds)and all(isinstance(s,int)for s in seeds)
    jobs=[(op,str(candidate),s,seat)for op in pool for s in seeds for seat in(0,1)]
    (out/'PROTOCOL.json').write_text(json.dumps(dict(candidate=str(candidate),candidate_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),seeds=seeds,
        opponent_ids=[p['id']for p in pool],games=len(jobs),workers=args.workers,method='Official 1.32.7 live policy actions; no seed exposed. Not Kaggle timeout/schema sandbox.'),indent=2),encoding='utf8')
    rows=[];started=time.monotonic();print(f'Total games: {len(jobs)}',flush=True)
    with cf.ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('spawn'),max_tasks_per_child=1)as executor,(out/'games.jsonl').open('w',encoding='utf8')as log:
        for future in cf.as_completed([executor.submit(play,job)for job in jobs]):
            row=future.result();rows.append(row);log.write(json.dumps(row,ensure_ascii=False)+'\n');log.flush()
            print(json.dumps(dict(done=len(rows),total=len(jobs),opponent=row['opponent'],error=bool(row['error']))),flush=True)
    summary=dict(games=len(rows),errors=[r for r in rows if r['error']],seconds=time.monotonic()-started,opponents={})
    for op in pool:
        group=[r for r in rows if r['opponent']==op['id']and not r['error']]
        summary['opponents'][op['id']]=dict(games=len(group),wins=sum(r['win']for r in group),ties=sum(r['tie']for r in group),
            mean_margin=sum(r['margin']for r in group)/len(group)if group else None)
    (out/'SUMMARY.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(summary,ensure_ascii=False))
    assert not summary['errors']and all(r['steps']==719 for r in rows),'Errors are not counted as legitimate wins/losses.'
if __name__=='__main__':main()
