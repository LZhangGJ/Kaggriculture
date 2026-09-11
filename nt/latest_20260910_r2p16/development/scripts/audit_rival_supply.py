"""Retrospective forecast accounting; private labels never enter the agent."""
from pathlib import Path
from collections import defaultdict
import gzip,json,statistics,concurrent.futures as futures,multiprocessing,time

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
OUT=HERE/'rival_supply_audit'
ITEMS=('WHEAT','CARROT','TOMATO','STRAWBERRY','MELON','EGG','MILK','WOOL','FERTILIZER')

def load(p):return json.loads(gzip.decompress(p.read_bytes()))
def mean(v):return statistics.mean(v) if v else 0

def one(row):
    key=f"{row['opponent']}_{row['seed']}_{row['opponent_seat']}";seat=row['opponent_seat']
    trace=load(ROOT/row['trace']);forecast=load(HERE/'forecast_audit'/f'{key}.json.gz')
    audit=load(HERE/'baseline_audit'/key/'AUDIT.json.gz')
    assert row['joint_action_sha256']==forecast['source_action_hash']==trace['result']['joint_action_sha256']
    frames={d['step']:d['observations'][seat] for d in trace['days']}
    sales=defaultdict(int);harvests=defaultdict(int);buys=defaultdict(int);overflow=defaultdict(int)
    for tx in audit['transactions']:
        if tx['seat']!=seat:continue
        target=sales if tx['op']=='SELL' else buys if tx['op']=='BUY_PRODUCT' else None
        if target is not None:target[tx['step']//24,tx['item']]+=tx['quantity']
    for event in audit['unit_events']:
        if event['seat']==seat and event['effect'] and event['action'][0]=='HARVEST':
            for item,q in event['stock_delta'].items():
                if q>0:harvests[event['step']//24,item]+=q
    for event in audit['summary'][seat]['overflow']:
        for item,q in event['items'].items():overflow[event['step']//24,item]+=q
    records=[]
    for f in forecast['forecasts']:
        d=f['day'];initial=frames[d*24]['private'];ending=frames[min(24*(d+1),719)]['private']
        def held(p,k):return p['shed'].get(k,0)+sum(x.get(k,0) for x in p['inventories'])
        for i,item in enumerate(ITEMS):
            sold=sales[d,item];bought=buys[d,item];made=harvests[d,item];before=held(initial,item);after=held(ending,item)
            # Seven non-input commodities: exact material accounting validates labels.
            if 0<i<8:assert before+made==sold+after+overflow[d,item],(key,d,item,before,made,sold,after)
            pred=f['rival'][d][i];scaled=pred*f['supply']
            if pred or sold or bought or made or before or after:
                records.append(dict(day=d,item=item,forecast=pred,scaled=scaled,sold=sold,bought=bought,
                    harvested=made,initial_private=before,ending_private=after,
                    trade_error=scaled-sold+bought,harvest_error=pred-made))
    return dict(key=key,group='win_control' if row['r2_win'] else 'loss',records=records)

def main():
    OUT.mkdir(exist_ok=True);path=OUT/'RESULTS.json'
    if path.exists():data=json.loads(path.read_text())
    else:
        rows=json.loads((HERE/'baseline_audit/SELECTION.json').read_text())['rows'];started=time.perf_counter()
        with futures.ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn')) as pool:cases=list(pool.map(one,rows))
        buckets=defaultdict(list)
        for c in cases:
            for r in c['records']:buckets[c['group']+'/'+r['item']].append(r)
        summary={k:dict(samples=len(v),forecast=mean([r['forecast'] for r in v]),scaled=mean([r['scaled'] for r in v]),
                       net_sale=mean([r['sold']-r['bought'] for r in v]),harvest=mean([r['harvested'] for r in v]),
                       initial_private=mean([r['initial_private'] for r in v]),ending_private=mean([r['ending_private'] for r in v]),
                       trade_bias=mean([r['trade_error'] for r in v]),trade_mae=mean([abs(r['trade_error']) for r in v]),
                       harvest_bias=mean([r['harvest_error'] for r in v]),harvest_mae=mean([abs(r['harvest_error']) for r in v])) for k,v in buckets.items()}
        data=dict(cases=len(cases),summary=summary,records=cases,seconds=time.perf_counter()-started,
            boundary='Retrospective current-day material/market attribution, not causal rank or future prediction proof. Opponent private holdings used ONLY to validate labels; excluded from runtime input.')
        path.write_text(json.dumps(data,indent=2))
    lines=['# 对手供给：公开预测与真实收获/出售分解','',
        '固定原R2的754场轨迹（699败局、55胜局对照），不是均匀总体抽样。只做事后误差分解。私有库存仅用于核对物料守恒，不能作为实战可见特征。',
        'rival是当前公开地块推算的条件供应；scaled为短模拟实际采用的供给。净卖出=SELL减BUY_PRODUCT。收获与出售不同，未知仓库和日内延迟都可能造成差异。',
        '小麦和肥料还有自用，不能将HARVEST与净卖出当同一口径；因此分别报告。当前日误差不能直接证明候选排名有错。','',
        '| 商品 | 日样本 | 原预测 | 缩放预测 | 实际净卖出 | 实际收获 | 期初私有 | 期末私有 | 净卖出偏差/MAE |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for k,v in data['summary'].items():
        if k.startswith('loss/'):
            lines.append(f"|{k.split('/')[1]}|{v['samples']}|{v['forecast']:.2f}|{v['scaled']:.2f}|{v['net_sale']:.2f}|{v['harvest']:.2f}|{v['initial_private']:.2f}|{v['ending_private']:.2f}|{v['trade_bias']:+.2f}/{v['trade_mae']:.2f}|")
    (OUT/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(cases=data['cases'],summary=data['summary'],seconds=data['seconds'])),flush=True)
if __name__=='__main__':main()
