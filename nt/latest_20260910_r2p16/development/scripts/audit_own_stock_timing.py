"""Separate projected available product from cash realization; frozen traces only."""
from pathlib import Path
from collections import defaultdict
import concurrent.futures as futures
import gzip,json,multiprocessing,statistics,time

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
OUT=HERE/'own_stock_timing_audit'
ITEMS=('WHEAT','CARROT','TOMATO','STRAWBERRY','MELON','EGG','MILK','WOOL','FERTILIZER')
def load(p):return json.loads(gzip.decompress(p.read_bytes()))
def held(obs,item):return obs['private']['shed'].get(item,0)+sum(inv.get(item,0) for inv in obs['private']['inventories'])
def one(row):
    key=f"{row['opponent']}_{row['seed']}_{row['opponent_seat']}";own=1-row['opponent_seat']
    trace=load(ROOT/row['trace']);forecast=load(HERE/'forecast_audit'/f'{key}.json.gz')
    audit=load(HERE/'baseline_audit'/key/'AUDIT.json.gz')
    assert trace['result']['joint_action_sha256']==forecast['source_action_hash']==row['joint_action_sha256']
    assert audit['saved_frames_identical'] and audit['steps']==719
    frames={d['step']:d['observations'][own] for d in trace['days']}
    sales=defaultdict(int);overflow=defaultdict(int)
    for tx in audit['transactions']:
        if tx['seat']==own and tx['op']=='SELL':sales[(tx['step']//24,tx['item'])]+=tx['quantity']
    for event in audit['summary'][own]['overflow']:
        for item,q in event['items'].items():overflow[(event['step']//24,item)]+=q
    rows=[]
    for f in forecast['forecasts']:
        d=f['day'];before=frames[24*d];after=frames[min(24*(d+1),719)]
        for i in range(1,8):
            item=ITEMS[i];prediction=f['own'][d][i];sold=sales[(d,item)]
            # These seven items cannot be bought/consumed as feed. Counts that
            # end in private inventories are not cash, but not missing output.
            ending=held(after,item);lost=overflow[(d,item)]
            if prediction or sold or ending or lost:
                rows.append(dict(day=d,item=item,forecast_available=prediction,sold=sold,
                                 initial_private=held(before,item),ending_private=ending,overflow=lost,
                                 available_accounted=sold+ending+lost,
                                 sale_error=prediction-sold,availability_error=prediction-sold-ending-lost))
    return dict(case=key,group='win_control' if row['r2_win'] else 'loss',rows=rows)
def main():
    OUT.mkdir(exist_ok=True)
    if (OUT/'RESULTS.json').exists():
        data=json.loads((OUT/'RESULTS.json').read_text());return report(data)
    selected=json.loads((HERE/'baseline_audit/SELECTION.json').read_text())['rows'];started=time.perf_counter()
    with futures.ProcessPoolExecutor(max_workers=16,mp_context=multiprocessing.get_context('spawn')) as executor:
        cases=list(executor.map(one,selected))
    buckets=defaultdict(list)
    for c in cases:
        for r in c['rows']:buckets[c['group']+'/'+r['item']].append(r)
    summary={k:dict(samples=len(rows),forecast_mean=statistics.mean(r['forecast_available'] for r in rows),
                    sale_mean=statistics.mean(r['sold'] for r in rows),end_private_mean=statistics.mean(r['ending_private'] for r in rows),
                    overflow_mean=statistics.mean(r['overflow'] for r in rows),
                    sale_mae=statistics.mean(abs(r['sale_error']) for r in rows),
                    availability_mae=statistics.mean(abs(r['availability_error']) for r in rows),
                    exact_availability=sum(abs(r['availability_error'])<1e-8 for r in rows)) for k,rows in buckets.items()}
    data=dict(cases=len(cases),summary=summary,rows=cases,seconds=time.perf_counter()-started,
              boundary='Offline accounting only. Ending private holdings are known retrospectively, never a live forecasting feature. No claim of a delivery improvement.')
    (OUT/'RESULTS.json').write_text(json.dumps(data,indent=2));report(data)
def report(data):
    lines=['# 我方预计供给为何接近实际销量两倍：库存与现金分开','',
        '固定原R2的754场已核对轨迹。7种不能购买且不作饲料的成品，分开记期初私有库存、当天销售、日末私有留存和已记录溢出。',
        '只在事后做会计分解。期末留存不能倒灌为线上预测；相同日的“可获得产品”不等于同日现金回款。',
        '可解释数量 = 当天已售 + 日末仍持有 + 日末已核对溢出。预测减去这个数量的残差，才更接近生产/调度兑现误差；尚不证明一切正常。','',
        '|商品|样本|预测均量|实际卖量|日末留存|销量MAE|考虑留存/溢出后MAE|数量完全对齐|',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for key,x in data['summary'].items():
        if key.startswith('loss/'):
            lines.append(f"|{key.split('/')[1]}|{x['samples']}|{x['forecast_mean']:.2f}|{x['sale_mean']:.2f}|{x['end_private_mean']:.2f}|{x['sale_mae']:.2f}|{x['availability_mae']:.2f}|{x['exact_availability']}/{x['samples']}|")
    lines+=['','## 不能直接做出的结论','',
        '不能看到预测约为销量两倍，就说程序重复算了两份产品。反过来，即使产品数量都兑现，收集后留到次日才卖仍可能影响价格与再投资；是否应提前交付，要比较额外行走、雇工、其他义务和实际市场后果。',
        '旧P3已经试过统一推迟生产回款的近似，未有可靠强度改善；这里不重复启用，也不把事后留存直接当预测修正。下一步如需改，应绑定每个候选真实的交付安排，而不是再统一乘一个折扣。']
    (OUT/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'cases':data['cases'],'seconds':data['seconds'],'summary':data['summary']}),flush=True)
if __name__=='__main__':main()
