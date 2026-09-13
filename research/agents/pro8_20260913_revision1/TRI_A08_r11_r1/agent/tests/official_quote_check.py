"""Isolated official market primitives, not games and no policy input."""
from pathlib import Path
import argparse,ctypes,copy,gzip,hashlib,json,sys
p=argparse.ArgumentParser();p.add_argument('--referee',type=Path,required=True);p.add_argument('--fixture',type=Path,required=True);p.add_argument('--probe',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
sys.path.insert(0,str(a.referee));from cpu_runtime import AttrDict,load_engine
engine=load_engine()
def attr(x):
 if isinstance(x,dict):return AttrDict({k:attr(v) for k,v in x.items()})
 if isinstance(x,list):return [attr(v) for v in x]
 return x
fixture=json.loads(gzip.decompress(a.fixture.read_bytes()));base=fixture['steps'][0];lib=ctypes.CDLL(str(a.probe.resolve()));lib.quote_pair.argtypes=[ctypes.c_int]*5+[ctypes.POINTER(ctypes.c_double)];lib.quote_pair.restype=ctypes.c_int
items=['WHEAT','CARROT','TOMATO','STRAWBERRY','MELON','EGG','MILK','WOOL','FERTILIZER'];rows=[]
for item in range(1,8):
 lo=10000;hi=10001
 while engine.market_price(items[item],hi)>1:hi=10000+2*(hi-10000)
 while hi-lo>1:
  mid=(hi+lo)//2
  if engine.market_price(items[item],mid)>1:lo=mid
  else:hi=mid
 threshold=hi
 stocks={9900,9999,10000,10050,10200,20000}
 stocks.update(x for x in [threshold-2,threshold-1,threshold,threshold+1] if 0<=x<=20000)
 for stock in sorted(stocks):
  for ours in [0,1,2,6,17]:
   for rival in [0,1,2,6,17]:
    for order in range(3):
     state=attr(copy.deepcopy(base));farms=state[0].observation.farms;market=state[0].observation.market;market.inventory[items[item]]=stock;engine._refresh_prices(market)
     for s,n in enumerate([ours,rival]):farms[s]['money']=1000;state[s].observation.private.shed={items[item]:n}
     own=[['SELL',items[item],ours]] if ours else [['PASS']];other=[['SELL',items[item],rival]] if rival else [['PASS']]
     if order==0:own=[['PASS']]+own
     if order==2:other=[['PASS']]+other
     state[0].action={'market':own};state[1].action={'market':other}
     class Env:pass
     env=Env();env.configuration=attr(fixture['configuration']);engine._process_market(state,env)
     result=(ctypes.c_double*3)();assert lib.quote_pair(item,stock,ours,rival,order,result)==0
     actual=[farms[0]['money']-1000,farms[1]['money']-1000,market.inventory[items[item]]]
     assert list(result)==actual,(item,stock,ours,rival,order,list(result),actual)
     rows.append({'item':items[item],'stock':stock,'ours':ours,'rival':rival,'relative_order':order,'native':list(result),'official':actual})
a.out.parent.mkdir(parents=True,exist_ok=True);report={'scope':'isolated synthetic same-item official _process_market checks; no agent games','referee_sha256':hashlib.sha256((a.referee/'official/kaggriculture.py').read_bytes()).hexdigest(),'fixture_sha256':hashlib.sha256(a.fixture.read_bytes()).hexdigest(),'probe_sha256':hashlib.sha256(a.probe.read_bytes()).hexdigest(),'new_games':0,'stock_domain':'0..20000; extreme unreachable floor outside this domain omitted','checked':len(rows),'passed':True,'rows':rows};a.out.write_text(json.dumps(report,separators=(',',':'))+'\n');print('official quote matches',len(rows))
