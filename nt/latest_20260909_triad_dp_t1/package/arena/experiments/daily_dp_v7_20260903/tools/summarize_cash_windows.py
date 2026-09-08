"""Summarize read-only S4X observations, never infer counterfactual profits."""
from pathlib import Path
from collections import Counter, defaultdict
import gzip, hashlib, json, statistics, sys
E=Path(__file__).resolve().parents[1]
source=E/'receipts/s4x_cash_window_audit_v1'
acceptance=json.loads((source/'acceptance.json').read_text())
assert acceptance['status']=='PASS_READ_ONLY_HARVEST_MARKET_AUDIT'
sys.path.insert(0,str(E/'native/build'))
import _dp7_native as n
out=E/'receipts/s4x_cash_window_summary_v1';out.mkdir(exist_ok=False)
names=['小麦','胡萝卜','番茄','草莓','瓜','蛋','奶','羊毛','肥料']
records=[];hashes={};forecast=defaultdict(list);signals=defaultdict(list)
for path in sorted(source.glob('*.json.gz')):
    hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
    game=json.load(gzip.open(path,'rt'));s=game['summary'];latent=Counter();phase=Counter();near=Counter();days=defaultdict(set)
    for r in game['rows']:
        for x in r['latent']:
            i=x['item'];latent[i]+=1;phase[r['phase']]+=1;days[i].add(r['step']//24)
            near[i]+=x['optimistic_harvest_drop_steps']<=23-r['step']%24
        if 'later_observed_inventory' not in r:continue
        relevant={x['item'] for x in r['latent']}|{i for i,q in enumerate(r['tactical'][:9]) if q}
        for i in relevant:
            if not 1<=i<8:continue
            current=n.price(i,r['inventory'][i]);pred=n.price(i,r['forecast_inventory'][i]);later=n.price(i,r['later_observed_inventory'][i])
            row=dict(seed=s['seed'],seat=s['seat'],opponent=s['opponent'],step=r['step'],pred_error=abs(pred-later),flat_error=abs(current-later),signed_error=pred-later,
                     predicted_change=pred-current,actual_change=later-current,own_net=r['own_net_orders_before_horizon'][i])
            forecast[s['label'],i].append(row)
            if r['tactical'][i]>0:signals[s['label'],i].append(row)
    records.append(dict(**s,latent_tile_observations=dict(latent),optimistic_within_day=dict(near),phase_tile_observations=dict(phase),latent_days={i:len(d) for i,d in days.items()}))
def stats(rows):
    if not rows:return dict(n=0)
    result=dict(n=len(rows),games=len({(r['opponent'],r['seed'],r['seat']) for r in rows}),mae=statistics.fmean(r['pred_error'] for r in rows),unchanged_price_mae=statistics.fmean(r['flat_error'] for r in rows),signed_error=statistics.fmean(r['signed_error'] for r in rows),observed_down=sum(r['actual_change']<0 for r in rows),observed_flat=sum(r['actual_change']==0 for r in rows),observed_up=sum(r['actual_change']>0 for r in rows))
    # Same-game observations correlate: also report per-game means, not only a
    # per-step average that favors longer inventory holding trajectories.
    grouped=defaultdict(list)
    for r in rows:grouped[r['opponent'],r['seed'],r['seat']].append(r)
    result['equal_game_mae']=statistics.fmean(statistics.fmean(r['pred_error'] for r in g) for g in grouped.values())
    result['equal_game_unchanged_mae']=statistics.fmean(statistics.fmean(r['flat_error'] for r in g) for g in grouped.values())
    return result
items=[dict(label=label,item=names[i],forecast=stats(rows),sell_signal=stats(signals[label,i])) for (label,i),rows in sorted(forecast.items())]
result=dict(status='COMPLETE_DESCRIPTIVE_AUDIT',games=records,items=items,hashes=hashes,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),policy_modified=False,causal_profit_estimate=False,independent_strength_test=False)
(out/'summary.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))
lines=['# S4X 成熟产品与出售预测只读结果','','64 完整局，两个已使用开发 seed、双座位、八对手、两个背景。所有动作原样执行，双方现金等于原面板。以下不是独立胜率提升验证。','','## 有成熟产品、但没有生成或安排当前收获的观察','','同一块地反复出现，计数是“地块×观察时点”，不是不同错失机会或可追回收入。近仓下界忽略原工作/库存/截止义务，不能证明值得提前收。','','|背景|产品|出现局数/32|地块观察次数|乐观可在日内收回比例|','|---|---|---:|---:|']
for label in ('all_intraday_insert','full_chain_autonomous'):
    subset=[r for r in records if r['label']==label]
    for i in (0,1,2,3,4,5,6,7):
        total=sum(r['latent_tile_observations'].get(i,0) for r in subset)
        if not total:continue
        count=sum(r['latent_tile_observations'].get(i,0)>0 for r in subset);close=sum(r['optimistic_within_day'].get(i,0) for r in subset)
        lines.append(f'|{label}|{names[i]}|{count}|{total}|{close/total:.1%}|')
lines+=['','## 原出售模型与实际后续价格','','仅统计当前有未排收获产品或原出售建议的商品/时点；每条是不同长度的原模型窗口，重叠观察并不独立。误差为单位报价，不是成交利润。比较“当前价格不变”只作预测基准，不能替代交易收益评测。','','|背景|商品|观察数|预测MAE|价格不变MAE|预测偏差（预测−实际）|','|---|---|---:|---:|---:|---:|']
for r in items:
    v=r['forecast'];lines.append(f"|{r['label']}|{r['item']}|{v['n']}|{v['mae']:.2f}|{v['unchanged_price_mae']:.2f}|{v['signed_error']:.2f}|")
lines+=['','## 已提出提前出售的窗口','','实际后续可能已受到本方卖出的影响，所以下降比例不能当作抢卖获利率。','','|背景|商品|出售建议观察数|后来跌/平/涨|','|---|---|---:|---|']
for r in items:
    v=r['sell_signal']
    if v['n']:lines.append(f"|{r['label']}|{r['item']}|{v['n']}|{v['observed_down']}/{v['observed_flat']}/{v['observed_up']}|")
lines+=['','## 解释边界','','成熟产品等待不是自动错误：集中收获可节省动作，货物留在田间也可能是合理库存。任何改动必须与被挤掉的生产/维护、融资和完整现金后果一起比较。','','本轮仅校准原策略预测、发现候选未覆盖的现象，没有替策略选用真实未来，也没有证明新增候选能盈利。']
(out/'TABLES_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
print('\n'.join(lines),flush=True)
