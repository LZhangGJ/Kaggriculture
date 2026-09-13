"""Independent frozen-engine replay and full cash reconciliation of NEW game logs."""
import pathlib,json,gzip,sys,collections,hashlib,time
R=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(R/'input/referee'));from cpu_runtime import load_engine,LocalGame,compare_frame
results=json.loads((R/'diagnostics/closed_loop_results.json').read_text());engine=load_engine();audits=[]
assert results['completed_games']==results['planned_games']==16
for row in results['all_rows']:
 p=R/row['trace'];assert hashlib.sha256(p.read_bytes()).hexdigest()==row['trace_sha256'];g=LocalGame(row['seed'],engine);flows={};done_days=[];originals={k:getattr(engine,k) for k in ['_commit_unit','_do_hire','_do_buy_land']};ts=time.perf_counter();actions=hashlib.sha256()
 start=[g.state[0].observation.farms[i]['money'] for i in (0,1)]
 def seat_of(farm):return 0 if farm is g.state[0].observation.farms[0] else 1
 def add(farm,key,before):
  change=farm['money']-before
  if change:flows.setdefault((g.t//24,seat_of(farm)),collections.Counter())[key]+=change
 def commit(op,item,price,farm,private,market,shed_capacity=100):
  before=farm['money'];ok=originals['_commit_unit'](op,item,price,farm,private,market,shed_capacity);add(farm,op+':'+item,before);return ok
 def hire(farm,private,board_size,mult=engine.FARM_HAND_COST_MULT):
  before=farm['money'];ret=originals['_do_hire'](farm,private,board_size,mult);add(farm,'HIRE',before);return ret
 def land(farm,board_size):
  before=farm['money'];ret=originals['_do_buy_land'](farm,board_size);add(farm,'BUY_LAND',before);return ret
 engine._commit_unit=commit;engine._do_hire=hire;engine._do_buy_land=land
 try:
  with gzip.open(p,'rt') as f:
   meta=json.loads(next(f));compare_frame(g,meta['initial_state']);assert meta['seed']==row['seed'] and meta['candidate_seat']==row['candidate_seat']
   for line in f:
    x=json.loads(line);assert x['step']==g.t+1
    actions.update(json.dumps(x['actions'],sort_keys=True,separators=(',',':')).encode()+b'\n');g.advance(x['actions']);compare_frame(g,x['state'])
    if g.t%24==0 or g.done:
     day=(g.t-1)//24;end=[g.state[0].observation.farms[i]['money'] for i in (0,1)]
     for seat in (0,1):
      flow=flows.get((day,seat),{});assert sum(flow.values())==end[seat]-start[seat],(row['seed'],day,seat,flow,start,end)
      done_days.append({'day':day,'seat':seat,'cash_start':start[seat],'cash_end':end[seat],'flow':dict(flow),'net_cash_change':sum(flow.values())})
     start=end
  assert g.done and g.t==719 and len(done_days)==60
  assert [float(s.reward) for s in g.state]==row['final_cash'];assert actions.hexdigest()==row['actions_sha256']
  for day,record in zip(done_days,row['daily_cash']):
   for k in ['day','seat','cash_start','cash_end']:assert day[k]==record[k]
  result={'seed':row['seed'],'candidate_seat':row['candidate_seat'],'verified_transitions':719,'reconciled_player_days':60,'final_cash':row['final_cash'],'actions_sha256':actions.hexdigest(),'trace_sha256':row['trace_sha256'],'daily_cash':done_days,'wall_seconds':time.perf_counter()-ts};audits.append(result)
  (R/'diagnostics/new_games_independent_audit.json').write_text(json.dumps({'scope':'Independent replay of the actual NEW head-to-head logs through the frozen official engine. Includes every cash-mutating operation, all 719 transitions and 60 player-days per game. Does not add games to any denominator.','games':len(audits),'rows':audits},indent=2));print(json.dumps({k:v for k,v in result.items() if k!='daily_cash'}),flush=True)
 finally:
  for k,v in originals.items():setattr(engine,k,v)
