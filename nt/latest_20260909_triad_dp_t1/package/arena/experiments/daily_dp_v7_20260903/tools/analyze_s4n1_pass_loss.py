"""Exact arithmetic decomposition of realized sales, not causal attribution."""
from pathlib import Path
import gzip,hashlib,json,statistics as st
E=Path(__file__).resolve().parents[1];root=E/'receipts/s4n1_pool_audit_O50_v1';labels=['all_intraday_insert','all_intraday_insert_workforce'];sets=[];hashes={}
for label in labels:
    path=root/f'{label}_pass.json.gz';hashes[str(path.relative_to(E))]=hashlib.sha256(path.read_bytes()).hexdigest()
    with gzip.open(path,'rt',encoding='utf8') as f:raw=json.load(f)['rows']
    sets.append({(r['seed'],r['seat']):r for r in raw})
records=[]
for key in sorted(sets[0]):
    a,b=sets[0][key],sets[1][key];seat=key[1];items=[]
    for i in range(9):
        q0=sum(d[seat]['sold'][i] for d in a['cash_ledger']);q1=sum(d[seat]['sold'][i] for d in b['cash_ledger'])
        c0=sum(d[seat]['sales'][i] for d in a['cash_ledger']);c1=sum(d[seat]['sales'][i] for d in b['cash_ledger'])
        if q0 and q1:volume=(q1-q0)*(c0/q0);price=q1*(c1/q1-c0/q0);new=0
        else:volume=price=0;new=c1-c0
        assert abs(volume+price+new-(c1-c0))<1e-6
        items.append(dict(sold_delta=q1-q0,sales_delta=c1-c0,quantity_at_old_price=volume,realized_unit_price_component=price,new_or_disappeared_sales=new))
    days=[]
    for da,db in zip(a['cash_ledger'],b['cash_ledger']):
        da,db=da[seat],db[seat]
        days.append(dict(sales=[y-x for x,y in zip(da['sales'],db['sales'])],sold=[y-x for x,y in zip(da['sold'],db['sold'])],products=[y-x for x,y in zip(da['products'],db['products'])],wages=db['hired']-da['hired'],cash=db['end']-da['end']))
    records.append(dict(seed=key[0],seat=seat,items=items,days=days,final_cash_delta=b['money'][seat]-a['money'][seat]))
items=[{k:st.fmean(r['items'][i][k] for r in records) for k in records[0]['items'][i]} for i in range(9)]
out=E/'receipts/s4n1_pass_O50_loss_decomposition_v1';out.mkdir(exist_ok=False)
(out/'summary.json').write_text(json.dumps(dict(status='COMPLETE_REALIZED_ARITHMETIC_NOT_CAUSAL_OR_ORACLE',games=len(records),cash_delta=st.fmean(r['final_cash_delta'] for r in records),items=items,input_hashes=hashes,caveat='Observed per-game unit prices include sale timing and market/quantity feedback; this is an exact accounting split, not the isolated effect of changing price or timing.',records=records),indent=2))
for name,row in zip(('wheat','carrot','tomato','strawberry','melon','egg','milk','wool','fertilizer'),items):print(json.dumps(dict(item=name,**row)))
