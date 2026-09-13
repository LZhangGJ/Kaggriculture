#!/usr/bin/env python3
"""Re-execute saved historical actions through frozen official code.
This audits factual trajectories only; it is NOT a counterfactual or new games.
"""
from pathlib import Path
import argparse,collections,gzip,hashlib,itertools,json,sys,time

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
 sys.path.insert(0,str(a.input/'referee'));from cpu_runtime import LocalGame,load_engine,compare_frame
 history=json.loads((a.input/'HISTORICAL_ALL_480_ROWS.json').read_text());assert len(history)==480 and len({r['id'] for r in history})==480
 opponents=sorted({r['opponent'] for r in history});seeds=sorted({r['seed'] for r in history});assert len(opponents)==12 and len(seeds)==20
 assert {(r['opponent'],r['seed'],r['seat']) for r in history}==set(itertools.product(opponents,seeds,[0,1]))
 for r in history:
  assert r['error'] is None and r['terminal'] and r['steps']==719 and r['status']==['DONE','DONE']
  assert r['cash'][r['seat']]==r['candidate_cash'] and r['candidate_cash']-r['opponent_cash']==r['margin']
 hist={'scope':'recalculation of all 480 historical rows, not new results','games':480,'strict_wins':sum(r['margin']>0 for r in history),'draws':sum(r['margin']==0 for r in history),'source_sha256':sha(a.input/'HISTORICAL_ALL_480_ROWS.json'),'by_opponent':{o:{'games':sum(r['opponent']==o for r in history),'strict_wins':sum(r['opponent']==o and r['margin']>0 for r in history)} for o in opponents},'complete_cartesian_panel':True}
 assert hist['strict_wins']==393 and hist['by_opponent']['submission_56149565']['strict_wins']==25
 (a.out/'HISTORICAL_480_RECOUNT.json').write_text(json.dumps(hist,indent=2));print(json.dumps(hist),flush=True)
 receipts=[]
 for fixture in sorted((a.input/'replays').glob('*.json.gz')):
  started=time.perf_counter();data=json.loads(gzip.decompress(fixture.read_bytes()));engine=load_engine();game=LocalGame(data['result']['seed'],engine);events=[];days=[];failed=[0,0];totals=[collections.Counter(),collections.Counter()]
  compare_frame(game,data['steps'][0]);assert len(data['actions'])==719 and len(data['steps'])==720
  def wrap(name):
   original=getattr(engine,name)
   def call(*args,**kwargs):
    farm=args[3] if name=='_commit_unit' else args[0];s=next(i for i in (0,1) if farm is game.state[0].observation.farms[i]);before=farm['money'];answer=original(*args,**kwargs)
    delta=farm['money']-before;op=args[0] if name=='_commit_unit' else name;item=args[1] if name=='_commit_unit' else None;price=args[2] if name=='_commit_unit' else None;success=bool(answer) if name=='_commit_unit' else delta!=0
    events.append({'player':s,'operation':op,'item':item,'price':price,'success':success,'cash_delta':delta})
    if name=='_commit_unit' and not answer:failed[s]+=1
    if op=='SELL' and success:
     totals[s][item+'_sold']+=1;totals[s][item+'_revenue']+=delta
     if price==1:totals[s][item+'_floor_sold']+=1
    return answer
   setattr(engine,name,call)
  for name in ('_commit_unit','_do_hire','_do_buy_land'):wrap(name)
  with gzip.open(a.out/(fixture.name.removesuffix('.json.gz')+'.audit.jsonl.gz'),'wt') as raw:
   for step,actions in enumerate(data['actions']):
    before=[f['money'] for f in game.state[0].observation.farms];events.clear();game.advance(actions);compare_frame(game,data['steps'][step+1]);after=[f['money'] for f in game.state[0].observation.farms]
    residual=[after[s]-before[s]-sum(e['cash_delta'] for e in events if e['player']==s) for s in (0,1)];assert residual==[0,0]
    raw.write(json.dumps({'step':step,'cash_before':before,'cash_after':after,'cash_identity_residual':residual,'events':events},separators=(',',':'))+'\n')
    if game.t%24==0 or game.done:
     for s in (0,1):days.append({'day':step//24,'player':s,'cash':after[s]})
  assert game.done and game.t==719 and len(days)==60
  receipt={'fixture':fixture.name,'fixture_sha256':sha(fixture),'scope':'saved-action factual audit only; zero new games','all719_transitions_match':True,'cash_rows':60,'cash_rows_all_match':True,'all_cash_identity_zero':True,'day_cash_rows':days,'cash_final':after,'failed_market_units_both':failed,'sales_totals_both':[dict(t) for t in totals],'seconds':time.perf_counter()-started}
  (a.out/(fixture.name.removesuffix('.json.gz')+'.summary.json')).write_text(json.dumps(receipt,indent=2));receipts.append(receipt);print(json.dumps({k:v for k,v in receipt.items() if k not in ('day_cash_rows','sales_totals_both')}),flush=True)
 (a.out/'SUMMARY.json').write_text(json.dumps({'fixtures':len(receipts),'matching_transitions':sum(719 for r in receipts),'matching_player_days':sum(r['cash_rows'] for r in receipts),'new_games':0,'passed':True,'receipts':receipts},indent=2))
if __name__=='__main__':main()
