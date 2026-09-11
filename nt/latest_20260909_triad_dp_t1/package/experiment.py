"""Equal-weight, paired-seed/seat panel. Candidate only receives public View."""
from pathlib import Path
import argparse,ast,hashlib,json,time,zlib,statistics,gzip,shutil,zipfile
import _triad_panel
R=Path(__file__).parent;N=R/'arena/experiments/daily_dp_v7_20260903/native'
NAMES=['g001','g003','boatlee_v29','kaito_v58','lynn_v5','yhay81_six_day','yhay81_three_day','pass']
DEFAULTS=dict(competition=.8,supply=.85,future_shop=.7,capital_power=.4,labor_hours=15,work_price=1.2,animal_work=1.,reserve=120,max_animals=20,max_hands=14,max_land=4,feed_cover=2,rotation=1,preview=1,delivery=2,intraday=1,service=1,replant=.7,land_rent=2,discount=.015,tour_dp=0,layout=0,repeat=1,animal_bias=1,crop_bias=1,portfolio_passes=1,crop_fert=1,harvest_threshold=1,delay_sale=0,opening_budget=1,scenario=0,keep_commitments=1)
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2))
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def pool():
 assets={name:json.loads(zlib.decompress((N/(name+'_frozen.json.zlib')).read_bytes())) for name in NAMES[:5]}
 return _triad_panel.Pool(assets)
def baseline_config(arm):
 tree=ast.parse((R/'agent/agent.py').read_text());node=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PARAMS' for t in n.targets))
 cfg=eval(compile(ast.Expression(node.value),'<config>','eval'),{'dict':dict})
 cfg.update(json.loads((R/'agent/config.json').read_text()))
 if arm=='auto':cfg['new_limit']=30
 return cfg

def run(p,tag,overrides={},count=2,start=260908100,opponents=range(7),arm='triad',trace=False,threads=4):
 path=R/'runs'/tag;path.mkdir(parents=True,exist_ok=False)
 cfg=baseline_config(arm) if arm!='triad' else DEFAULTS.copy();unknown=set(overrides)-set(cfg)
 if unknown:raise ValueError(f'Unknown configuration keys: {unknown}')
 cfg.update(overrides);lib=R/(f'baseline_{arm}.so' if arm!='triad' else 'policy/agent.so')
 h=digest(lib);store=R/'snapshots';store.mkdir(exist_ok=True);frozen=store/(h+'.so')
 if not frozen.exists():shutil.copy2(lib,frozen)
 receipt=lib.with_suffix('.build.json')
 if receipt.exists():
  rec=json.loads(receipt.read_text());assert rec['binary_hash']==h,'build receipt mismatch'
  shashes=rec['source_files'];sh=rec['source_hash']
 else:
  files=list((R/('agent' if arm!='triad' else 'policy')).rglob('*.hpp'))+list((R/('agent' if arm!='triad' else 'policy')).rglob('*.cpp'))
  if arm!='triad':files.append(R/'src/baseline_bridge.cpp')
  shashes={str(f.relative_to(R)):digest(f) for f in files}
  sh=hashlib.sha256(json.dumps(shashes,sort_keys=True).encode()).hexdigest();srczip=store/(sh+'.zip')
  if not srczip.exists():
   with zipfile.ZipFile(srczip,'w',zipfile.ZIP_DEFLATED) as z:
    for f in files:z.write(f,f.relative_to(R))
 write(path/'protocol.json',dict(tag=tag,arm=arm,config=cfg,seeds=list(range(start,start+count)),opponents=list(opponents),seats=[0,1],source_hash=sh,binary_hash=h,source_files=shashes,opponent_hashes={name:digest(N/(name+'_frozen.json.zlib')) for name in NAMES[:5]},strict_cash_win=True,trace=trace))
 tic=time.perf_counter();rows=p.run(str(frozen),list(cfg.values()),list(range(start,start+count)),list(opponents),threads,trace)
 for r in rows:
  r['opponent_name']=NAMES[r['opponent']]
  if trace:
   body={'seed':r['seed'],'seat':r['seat'],'opponent':r['opponent_name'],'actions':r.pop('actions'),'days':r.pop('days')}
   (path/f'trace_{r["seed"]}_{r["opponent"]}_{r["seat"]}.json.gz').write_bytes(gzip.compress(json.dumps(body,separators=(',',':')).encode()))
  try:r['debug']=json.loads(r['debug'])
  except (json.JSONDecodeError,TypeError):pass
 write(path/'rows.json',rows)
 def summary(rs):
  return dict(games=len(rs),wins=sum(r['win'] for r in rs),ties=sum(r['cash']==r['opponent_cash'] and not r['error'] for r in rs),errors=sum(bool(r['error']) for r in rs),win_rate=statistics.mean(r['win'] for r in rs),cash=statistics.mean(r['cash'] for r in rs),margin=statistics.mean(r['margin'] for r in rs),max_ms=max(r['max_action_ms'] for r in rs),mean_policy_sec=statistics.mean(r['policy_seconds'] for r in rs),overflow=sum(r['overflow'] for r in rs))
 result=dict(tag=tag,seconds=time.perf_counter()-tic,overall=summary(rows),opponents={NAMES[o]:summary([r for r in rows if r['opponent']==o]) for o in opponents})
 write(path/'summary.json',result);print(json.dumps(result,ensure_ascii=False),flush=True);return result
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--tag',required=True);a.add_argument('--count',type=int,default=2);a.add_argument('--start',type=int,default=260908100);a.add_argument('--opponents',default='0,1,2,3,4,5,6');a.add_argument('--config',default='{}');a.add_argument('--arm',default='triad',choices=['auto','j7','triad']);a.add_argument('--trace',action='store_true');a.add_argument('--threads',type=int,default=4);x=a.parse_args()
 run(pool(),x.tag,json.loads(x.config),x.count,x.start,[int(v) for v in x.opponents.split(',')],x.arm,x.trace,x.threads)
