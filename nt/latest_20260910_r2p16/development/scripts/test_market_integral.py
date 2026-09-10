"""Compare a conditional sequence of individual trades with official functions."""
from pathlib import Path
import json,subprocess,random,sys,time
HERE=Path(__file__).resolve().parent;OUT=HERE/'candidate_r2p7'
import run_panel as panel
panel.init_worker();engine=panel.ENGINE
ITEMS=('WHEAT','CARROT','TOMATO','STRAWBERRY','MELON','EGG','MILK','WOOL','FERTILIZER')
exe=OUT/'test_market'
subprocess.run(['g++-13','-std=c++20','-O2',str(OUT/'test_market.cpp'),'-o',str(exe)],check=True)
def query(cases):
 text=''.join(f'{i} {s:.17g} {q:.17g}\n' for i,s,q in cases)
 out=subprocess.run([str(exe)],input=text,text=True,capture_output=True,check=True)
 return [list(map(float,r.split())) for r in out.stdout.splitlines()]
thresholds=[r[-1] for r in query([(i,10000,0) for i in range(9)])]
cases=[];rng=random.Random(261010)
for i in range(9):
 stocks=[-4,0,1,8500,9700,10000,10100,19998,20002,25000]
 if thresholds[i]<100000:stocks+=list(range(int(thresholds[i])-4,int(thresholds[i])+5))
 for inv in stocks:
  for q in [0,1,2,3,8,30,100]+([-1,-2,-8,-100] if i in (0,8) else []):cases.append((i,inv,q))
 for _ in range(200):cases.append((i,rng.randint(8500,10500),rng.randint(-100 if i in (0,8) else 0,100)))
results=query(cases);failures=[];old_bad_stock=old_bad_cash=0;examples=[]
for case,native in zip(cases,results):
 i,inv,q=case;name=ITEMS[i];farm={'money':10**15};private={'shed':{name:max(0,q)}};market={'inventory':{name:inv}}
 before=farm['money']
 for _ in range(abs(q)):
  price=engine.market_price(name,market['inventory'][name]-(q<0))
  assert engine._commit_unit('SELL' if q>0 else 'BUY_PRODUCT',name,price,farm,private,market,100000)
 expected=(farm['money']-before,market['inventory'][name])
 if abs(native[3]-expected[0])>1e-7 or native[4]!=expected[1]:failures.append(dict(case=case,native=native,official=expected))
 old_bad_stock+=native[6]!=expected[1];old_bad_cash+=abs(native[5]-expected[0])>1e-7
 if i==7 and inv==10000 and q==100:examples.append(dict(item=name,initial=inv,quantity=q,old_cash=native[5],old_inventory=native[6],official_cash=expected[0],official_inventory=expected[1]))
# Integer exactness is distinct from the relaxed fractional-flow consistency.
fraction_cases=[(i,9999.25,12.75) for i in range(9)]
first=query([(i,s,q*.4) for i,s,q in fraction_cases]);second=query([(i,r[4],q*.6) for (i,s,q),r in zip(fraction_cases,first)])
whole=query(fraction_cases)
fraction_errors=[abs(a[3]+b[3]-c[3]) for a,b,c in zip(first,second,whole)]
assert max(fraction_errors)<1e-6
receipt=dict(status='PASS' if not failures else 'FAIL',cases=len(cases),failures=failures,old_stock_mismatches=old_bad_stock,
             old_cash_mismatches=old_bad_cash,saturation=thresholds,examples=examples,fraction_cases=len(fraction_cases),
             max_fraction_error=max(fraction_errors),official_sha256=panel.sha(panel.old.REF/'official/kaggriculture.py'),
             scope='Successive single-side trades; no funds/capacity limits in this primitive. Not simultaneous rival order matching. Fractional forecast flows are a relaxation.')
(OUT/'MARKET_UNIT_TESTS.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt),flush=True)
assert not failures
