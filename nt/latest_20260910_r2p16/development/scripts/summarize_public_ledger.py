"""Read-only calibration of certified sale inference and uncertain inventory."""
from pathlib import Path
from collections import defaultdict
import gzip,json,statistics
HERE=Path(__file__).resolve().parent;OUT=HERE/'public_ledger_audit_v2'
ITEMS=('WHEAT','CARROT','TOMATO','STRAWBERRY','MELON','EGG','MILK','WOOL','FERTILIZER')
def load(path):return json.loads(gzip.decompress(path.read_bytes()))
def main():
    receipt=json.loads((OUT/'RESULTS.json').read_text());assert receipt['status']=='PASS'
    inventory=defaultdict(list);forecasts=defaultdict(list);sales_true=sales_covered=0;cases=0
    for path in sorted(OUT.glob('*.json.gz')):
        data=load(path);key=path.name.removesuffix('.json.gz');cases+=1
        fdata=load(HERE/'forecast_audit'/path.name);row=fdata['row'];rival=row['opponent_seat'];group='loss' if not row['r2_win'] else 'win_control'
        audit=load(HERE/'baseline_audit'/key/'AUDIT.json.gz');daily=defaultdict(int)
        for tx in audit['transactions']:
            if tx['seat']==rival and tx['op']=='SELL' and tx['item'] in ITEMS:daily[(tx['step']//24,tx['item'])]+=tx['quantity']
        byday={f['day']:f for f in fdata['forecasts']}
        for frame in data['frames']:
            step=frame['observation_step']
            for i in range(1,8):
                if step:
                    sales_true+=max(0,frame['actual_net'][i]);sales_covered+=frame['valid'][i]*max(0,frame['actual_net'][i])
                if step%24:continue
                upper=frame['upper'][i];actual=frame['actual_stock'][i];item=ITEMS[i]
                if upper or actual:inventory[(group,item)].append((upper,actual))
                d=step//24;f=byday[d];pred=f['rival'][d][i]*f['supply'];target=daily[(d,item)]
                # The original prediction concerns visible field flows. Adding
                # existing off-field stock is a hypothesis, not known intent.
                if pred or target or upper:
                    forecasts[(group,item)].append((pred,target,upper))
    assert cases==receipt['cases'];metrics={};stock_metrics={}
    for key,rows in inventory.items():
        stock_metrics['/'.join(key)]=dict(samples=len(rows),mean_upper=statistics.mean(r[0] for r in rows),mean_actual=statistics.mean(r[1] for r in rows),
                positive_upper_actual_zero=sum(u>0 and a==0 for u,a in rows),exact=sum(u==a for u,a in rows))
    for key,rows in forecasts.items():
        metrics['/'.join(key)]={str(w):dict(mae=statistics.mean(abs(p+w*u-a) for p,a,u in rows),bias=statistics.mean(p+w*u-a for p,a,u in rows),
                                 samples=len(rows)) for w in (0,.5,1)}
    result=dict(cases=cases,sale_checks=receipt['sale_checks'],bound_checks=receipt['bound_checks'],mismatches=receipt['mismatches'],
                sales_units= sales_true,certified_sales_units=sales_covered,inventory=stock_metrics,forecast=metrics,
                boundary='Stock is an upper bound. Held goods need not be sold today. Future target is offline only.')
    (OUT/'SUMMARY.json').write_text(json.dumps(result,indent=2))
    lines=['# 公开成交/待售库存上界：验收与校准','',
           f"{cases}局、{receipt['actions']:,}动作与原R2一致；{receipt['sale_checks']:,}项可判定成交和{receipt['bound_checks']:,}项私有库存上界检查，错误{receipt['mismatches']}。",
           f"在可见历史的step0–717中，目标7产品实际售出{sales_true:,}份，其中{sales_covered:,}份所在区间可以精确推断。日末与地板价区间保留未知，不硬猜。",
           '官方真实私有库存/成交只存在于Python审计标签；原生观察器未接收。这个结果证明本批观察器的推断一致性，不是证明未来必卖，更不是保证强度提升。','',
           '## 败局每日开头：库存上界的松紧','',
           '| 商品 | 非零日样本 | 上界均值 | 实际库存均值 | 上界>0但实际0 | 上界恰好等于实际 |','|---|---:|---:|---:|---:|---:|']
    for item in ITEMS[1:8]:
        x=stock_metrics.get('loss/'+item)
        if x:lines.append(f"| {item} | {x['samples']} | {x['mean_upper']:.2f} | {x['mean_actual']:.2f} | {x['positive_upper_actual_zero']} | {x['exact']} |")
    lines+=['','## 下一天出售量预测：加入上界的校准（MAE，越低越好）','',
            '| 商品 | 原预测 | +0.5上界 | +1上界 |','|---|---:|---:|---:|']
    for item in ITEMS[1:8]:
        x=metrics.get('loss/'+item)
        if x:lines.append(f"| {item} | {x['0']['mae']:.2f} | {x['0.5']['mae']:.2f} | {x['1']['mae']:.2f} |")
    lines+=['','不是预测修正降低MAE就能直接升级；仍需11对手实时、多seed、双座位实力复验。不能用真实私有库存替代区间，也不能按未来结果选择是否开启。']
    (OUT/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('inventory','forecast')}),flush=True)
if __name__=='__main__':main()
