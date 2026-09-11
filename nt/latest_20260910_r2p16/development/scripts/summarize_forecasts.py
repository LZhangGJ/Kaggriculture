"""Forecast calibration and within-day sale ordering; no policy optimization."""
from pathlib import Path
from collections import defaultdict
import gzip,json,statistics

HERE=Path(__file__).resolve().parent
ITEMS=('WHEAT','CARROT','TOMATO','STRAWBERRY','MELON','EGG','MILK','WOOL','FERTILIZER')


def load(path):return json.loads(gzip.decompress(path.read_bytes()))


def main():
    receipt=json.loads((HERE/'forecast_audit/RESULTS.json').read_text());assert receipt['status']=='PASS'
    buckets=defaultdict(list);timings=defaultdict(list)
    cases=[]
    for path in sorted((HERE/'forecast_audit').glob('*.json.gz')):
        data=load(path);row=data['row'];own=1-row['opponent_seat'];opponent=1-own
        key=path.name.removesuffix('.json.gz')
        audit=load(HERE/'baseline_audit'/key/'AUDIT.json.gz')
        daily=defaultdict(list)
        for t in audit['transactions']:
            if t['item'] in ITEMS and t['op'] in {'SELL','BUY_PRODUCT'}:
                daily[(t['step']//24,t['seat'],t['item'])].append(t)
        group='win_control' if row['r2_win'] else 'loss'
        for f in data['forecasts']:
            d=f['day'];assert f['step']==24*d
            for h in (1,3):
                if d+h>30:continue
                for side,seat,scale in [('own',own,1),('rival',opponent,f['supply'])]:
                    for i,item in enumerate(ITEMS):
                        pred=sum(f[side][t][i] for t in range(d,d+h))*scale
                        actual=sum((1 if tx['op']=='SELL' else -1)*tx['quantity'] for t in range(d,d+h)
                                   for tx in daily[(t,seat,item)])
                        # Exclude truly zero-zero observations from error averages,
                        # but report this denominator explicitly.
                        if abs(pred)>1e-9 or actual!=0:
                            buckets[(group,side,h,item)].append((pred,actual,d,row['seed']))
            for item in ITEMS:
                ours=[t for t in daily[(d,own,item)] if t['op']=='SELL']
                theirs=[t for t in daily[(d,opponent,item)] if t['op']=='SELL']
                nq=sum(t['quantity'] for t in ours);nr=sum(t['quantity'] for t in theirs)
                if not nq or not nr:continue
                prior=sum(x['quantity']*y['quantity']*(1 if y['step']<x['step'] else .5 if y['step']==x['step'] else 0)
                          for x in ours for y in theirs)/(nq*nr)
                own_hour=sum((t['step']%24)*t['quantity'] for t in ours)/nq
                rival_hour=sum((t['step']%24)*t['quantity'] for t in theirs)/nr
                timings[(group,item)].append(dict(prior=prior,own_hour=own_hour,rival_hour=rival_hour,day=d,seed=row['seed'],
                                                  own_qty=nq,rival_qty=nr,key=key))
        cases.append(key)
    assert len(cases)==receipt['cases']
    summary={}
    for key,data in buckets.items():
        errors=[p-a for p,a,*_ in data]
        summary['/'.join(map(str,key))]=dict(material_samples=len(data),pred_mean=statistics.mean(x[0] for x in data),
                                            actual_mean=statistics.mean(x[1] for x in data),bias=statistics.mean(errors),mae=statistics.mean(abs(x) for x in errors),
                                            predicted_positive_actual_zero=sum(p>0 and a==0 for p,a,*_ in data),
                                            predicted_nonpositive_actual_positive=sum(p<=0 and a>0 for p,a,*_ in data))
    order={}
    for key,data in timings.items():
        order['/'.join(key)]=dict(joint_sale_days=len(data),rival_precedes_own_share=statistics.mean(x['prior'] for x in data),
                                  own_mean_hour=statistics.mean(x['own_hour'] for x in data),rival_mean_hour=statistics.mean(x['rival_hour'] for x in data),
                                  median_hour_lag=statistics.median(x['own_hour']-x['rival_hour'] for x in data))
    result=dict(cases=len(cases),forecast_metrics=summary,sale_order=order,
                boundary='Conditional plans versus realized net market flow. Future replanning and stochastic events can explain forecast error; error is not automatically a rule bug.')
    (HERE/'forecast_audit/SUMMARY.json').write_text(json.dumps(result,indent=2))
    lines=['# R2预测校准审计','',
           f"{len(cases)}场原局面：699败局＋55胜局对照。探针每步输出必须与原版动作一致，共{receipt['actions']:,}动作；未来标签来自已核对的官方实际成交。",
           '预测只接收当时观察，实际未来只用于离线标签。预测按当日选中的经营参数，统计净市场供给（SELL减BUY_PRODUCT）；不是销量减饲料。只汇总预测或实际非零的样本。',
           '条件计划随后会重规划，随机事件也会变化，误差不自动等于实现错误。胜局对照是55例而非完整胜局总体。','',
           '## 败局：对手未来24步净供给','',
           '| 商品 | 非零样本 | 预测均值 | 实际均值 | 偏差 | MAE | 有预测却未成交 | 漏掉实际正供给 |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for item in ITEMS:
        x=summary.get(f'loss/rival/1/{item}');
        if x:lines.append(f"| {item} | {x['material_samples']} | {x['pred_mean']:.2f} | {x['actual_mean']:.2f} | {x['bias']:+.2f} | {x['mae']:.2f} | {x['predicted_positive_actual_zero']} | {x['predicted_nonpositive_actual_positive']} |")
    lines += ['','## 同日双方都有出售时：谁先进入市场','',
              '每个同日/商品计算随机一件对手货先于随机一件我方货出售的比例，同时点暂计0.5，不假装知道订单槽内先后。再等权平均。当前长线估值采用0.5这一固定比例。','',
              '| 商品 | 败局共同销售日 | 对手先卖比例 | 我方平均销售hour | 对手hour | 销售时差中位数 |',
              '|---|---:|---:|---:|---:|---:|']
    for item in ITEMS:
        x=order.get(f'loss/{item}')
        if x:lines.append(f"| {item} | {x['joint_sale_days']} | {x['rival_precedes_own_share']:.1%} | {x['own_mean_hour']:.2f} | {x['rival_mean_hour']:.2f} | {x['median_hour_lag']:+.2f} |")
    lines+=['','原始逐局预测：forecast_audit/*.json.gz；实际逐笔成交：baseline_audit/*/AUDIT.json.gz。',
            '同一天多个商品、同seed双座位以及相似对手均相关；不能把表中每个商品/日当独立对局证据。']
    (HERE/'forecast_audit/REPORT_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
