"""Separate PASS price compression from actual same-match competitive deficits."""
from pathlib import Path
import gzip,hashlib,json,statistics as st
E=Path(__file__).resolve().parents[1];out=E/'receipts/s4v_market_cash_gap_v1';out.mkdir(exist_ok=False)
hashes={}
def read(rel):
    p=E/rel;hashes[rel]=hashlib.sha256(p.read_bytes()).hexdigest();return json.loads(p.read_text())
source=read('receipts/s4l_pool_audit_N50_v1/summary.json');panel=read('profiles/s4u/s4t_partial_results.json')
assert source['status']=='PASS_OFFLINE_AUDIT_NOT_GOAL_ACCEPTANCE'
groups={r['opponent']:r for r in source['summary'] if r['label']=='all_intraday_causal_pickup'}
names=['小麦','胡萝卜','番茄','草莓','甜瓜','蛋','牛奶','羊毛','肥料']
for opp,g in groups.items():
    ref=panel['summary']['all_intraday'][opp]
    assert abs(ref['mean_cash']-g['own_cash']['cash'])<1e-6 and abs(ref['mean_opponent_cash']-g['rival_cash']['cash'])<1e-6 and g['wins']==ref['wins']
comparisons=[];daily=[]
for opp,g in groups.items():
    if opp=='pass':continue
    a=g['own_cash'];b=g['rival_cash'];air=groups['pass']['own_cash']
    price=qty=0.
    for i in range(9):
        pa=air['sales'][i]/air['sold'][i];pb=a['sales'][i]/a['sold'][i]
        price+=(pb-pa)*(air['sold'][i]+a['sold'][i])/2
        qty+=(a['sold'][i]-air['sold'][i])*(pa+pb)/2
    assert abs(price+qty-sum(a['sales'])+sum(air['sales']))<1e-6
    costs=lambda c:sum(c['products'])+sum(c['seeds'])+sum(c['animals'])+c['hired']+c['land']
    row=dict(opponent=opp,cash_margin=a['cash']-b['cash'],sales_margin=sum(a['sales'])-sum(b['sales']),
        extra_total_cost=costs(a)-costs(b),extra_supplies=sum(a['products'])-sum(b['products']),extra_wages=a['hired']-b['hired'],
        pass_to_live=dict(cash_delta=a['cash']-air['cash'],sales_delta=price+qty,symmetric_price_component=price,symmetric_quantity_component=qty),
        commodities=[dict(item=names[i],own_sold=a['sold'][i],rival_sold=b['sold'][i],
            own_unit_price=a['sales'][i]/a['sold'][i] if a['sold'][i] else None,
            rival_unit_price=b['sales'][i]/b['sold'][i] if b['sold'][i] else None,
            sales_margin=a['sales'][i]-b['sales'][i]) for i in range(9)])
    assert abs(row['cash_margin']-row['sales_margin']+row['extra_total_cost'])<1e-6
    comparisons.append(row)
    rel=f"receipts/s4l_pool_audit_N50_v1/all_intraday_causal_pickup_{opp}.json.gz"
    path=E/rel;hashes[rel]=hashlib.sha256(path.read_bytes()).hexdigest()
    with gzip.open(path,'rt') as f:raw=json.load(f)['rows']
    assert len(raw)==100
    for d in range(30):
        x={}
        for side in ('own','rival'):
            rec=[r['cash_ledger'][d][r['seat'] if side=='own' else 1-r['seat']] for r in raw]
            x[side]=dict(sold=[st.fmean(z['sold'][i] for z in rec) for i in range(9)],sales=[st.fmean(z['sales'][i] for z in rec) for i in range(9)],
                bought=[st.fmean(z['bought_products'][i] for z in rec) for i in range(9)],wages=st.fmean(z['hired'] for z in rec))
        daily.append(dict(opponent=opp,day_zero_based=d,**x))
    for i in range(9):
        for side in ('own','rival'):
            seq=[d for d in daily if d['opponent']==opp];n=sum(d[side]['sold'][i] for d in seq)
            row['commodities'][i][side+'_sale_day_zero_based']=sum(d['day_zero_based']*d[side]['sold'][i] for d in seq)/n if n else None
(out/'summary.json').write_text(json.dumps(dict(status='COMPLETE_ACCOUNTING_NOT_CAUSAL_INTERVENTION',comparisons=comparisons,daily=daily,input_hashes=hashes,
    caveats=['PASS vs live is not the same question as own vs opponent in the same match.','Quantity and timing/market price interact; symmetric decomposition is accounting, not recoverable profit.','Sale day is volume-weighted; it does not show whether holding, planting date or harvest date caused the difference.','900 historical baseline matches only; no new simulation or extra independent observations.']),indent=2))
lines=['# S4V竞争现金缺口：两种比较不能混为一谈','','这是复盘，不是新策略胜率验收。沿用已逐场对账的同N50双座位基线；与S4T的现金和胜局一致。',
    '','## 与空气比较','', '空气现金180,428；对G001为104,495，对G003为102,582。销售减少主要对应价格变化，但对手也承受同一市场，不能据此断言抢卖能补回这些钱。',
    '','## 同场双方比较','','|对手|现金差|销售收入差|采购额外支出|工资额外支出|','|---|---:|---:|---:|---:|']
for r in comparisons:lines.append('|'+r['opponent']+'|'+'|'.join(f'{r[k]:,.0f}' for k in ('cash_margin','sales_margin','extra_supplies','extra_wages'))+'|')
lines += ['','对G001我方总销售略多，但小麦等采购支出更多；对G003则多种商品产销均落后。不能把所有失败统一叫工人效率问题，也不能统一叫抢卖问题。',
    '','## 逐商品销售时点','','|对手|商品|我方数量|对手数量|我方成交均价|对手成交均价|我方加权出售日|对手加权出售日|','|---|---|---:|---:|---:|---:|---:|---:|']
fmt=lambda x:'—' if x is None else f'{x:.2f}'
for r in comparisons:
    for c in r['commodities']:
        lines.append('|'+r['opponent']+'|'+c['item']+'|'+'|'.join(fmt(c[k]) for k in ('own_sold','rival_sold','own_unit_price','rival_unit_price','own_sale_day_zero_based','rival_sale_day_zero_based'))+'|')
lines+=['','日期0-based，仅为描述统计，不代表提前到对手当天就能有相同价格。对方私有账本仅离线归因，策略不读取。',
    '','## 下一步可证伪假设','','1. 检查已有主动出售建议是否被仓容/准备阶段分支绕过；漏掉建议不等于损失，需实时开关对照。','2. 区分晚种、晚收和已入库后等待造成的售价差，不把所有差额归于销售排序。','3. 与生产、采购、人员配置联合看净现金，避免为抢卖打断更有价值的生产。','4. 保留成熟生产基座及全部能力开关；完整自主全开结果单独汇报，不能用18万基线为其背书。']
(out/'REVIEW_ZH.md').write_text('\n'.join(lines),encoding='utf8');print('COMPLETE',flush=True)
