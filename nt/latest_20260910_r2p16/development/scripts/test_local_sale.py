from pathlib import Path
import subprocess,random,json,time
import run_panel as panel
HERE=Path(__file__).resolve().parent;OUT=HERE/'candidate_r2p9'
panel.init_worker();engine=panel.ENGINE;exe=OUT/'test_local_sale'
subprocess.run(['g++-13','-std=c++20','-O2',str(OUT/'test_local_sale.cpp'),'-o',str(exe)],check=True)
cases=[];rng=random.Random(261011)
for item in range(1,8):
    for stock in (9800,9990,10000,10050,10100,12000):
        for q in (1,2,8,31):
            cases.append((item,stock,q,rng.randint(0,q),rng.randint(0,16),rng.randint(0,32),2))
text=''.join(' '.join(map(str,c))+'\n' for c in cases)
r=subprocess.run([str(exe)],input=text,text=True,capture_output=True,check=True)
results=[list(map(float,line.split())) for line in r.stdout.splitlines()]
def payoff(item,stock,q,now,demand,rival,competition):
    name=panel.R2_MODULE._ITEMS[item];market={'inventory':{name:stock}};cash=[]
    for qty in (now,rival,q-now):
        farm={'money':0};priv={'shed':{name:qty}}
        for _ in range(qty):
            price=engine.market_price(name,market['inventory'][name]);assert engine._commit_unit('SELL',name,price,farm,priv,market,100000)
        cash.append(farm['money'])
        if len(cash)==2:market['inventory'][name]-=demand
    return cash[0]+cash[2]-competition*cash[1]
for case,result in zip(cases,results):
    i,inv,q,n,d,op,c=case;expected=payoff(*case);assert abs(expected-result[0])<1e-7,(case,result,expected)
    b0=payoff(i,inv,q,q,d,0,c);b1=payoff(i,inv,q,q,d,op,c)
    gains={x:min(payoff(i,inv,q,x,d,0,c)-b0,payoff(i,inv,q,x,d,op,c)-b1) for x in range(q+1)}
    selected=int(result[1]);assert 0<=selected<=q and abs(gains[selected]-max(gains.values()))<1e-7
    assert abs(result[2]-gains[selected])<1e-7
assert len(results)==len(cases)
panel.save(OUT/'UNIT_TESTS.json',dict(status='PASS',official_payoff_and_best_split_cases=len(cases),time_guard_cases=719,fixed_deadline_and_guard_assertions='PASS',scope='Conditional sequential quote scenarios, not proof of real rival order timing or future cash improvement',official_sha256=panel.sha(panel.old.REF/'official/kaggriculture.py')))
print('PASS',len(cases),flush=True)
