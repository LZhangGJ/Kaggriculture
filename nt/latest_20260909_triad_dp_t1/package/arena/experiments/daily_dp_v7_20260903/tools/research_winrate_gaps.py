"""Read-only cross-opponent accounting research; no new simulations/Oracle."""
from pathlib import Path
import gzip,hashlib,json,statistics as st
E=Path(__file__).resolve().parents[1];out=E/'receipts/s4v_winrate_research_v1';out.mkdir(exist_ok=False);hashes={}
def read(p):
    hashes[str(p.relative_to(E))]=hashlib.sha256(p.read_bytes()).hexdigest()
    return json.load(gzip.open(p,'rt')) if p.suffix=='.gz' else json.loads(p.read_text())
names=('小麦','胡萝卜','番茄','草莓','瓜','蛋','牛奶','羊毛','肥料')
opps=('g001','g003','boatlee_v29','kaito_v58','lynn_v5','yhay81_six_day','yhay81_three_day','ecobot_v7')
sources=[('shared','s4m1_pool_audit_N50_v1','all_intraday_insert'),('autonomous','s4u_pool_audit_N50_v1','full_chain_autonomous')]
records=[]
for label,folder,variant in sources:
    summary=read(E/'receipts'/folder/'summary.json')
    for opp in opps:
        x=next(s for s in summary['summary'] if s['label']==variant and s['opponent']==opp)
        a,b=x['own_cash'],x['rival_cash'];goods=[]
        for i,name in enumerate(names):
            qa,qb=a['sold'][i],b['sold'][i];ra,rb=a['sales'][i],b['sales'][i]
            pa,pb=ra/qa if qa else None,rb/qb if qb else None
            # Symmetric arithmetic decomposition, NOT independent causal gains.
            volume=(qa-qb)*(pa+pb)/2 if pa is not None and pb is not None else None
            price=(pa-pb)*(qa+qb)/2 if pa is not None and pb is not None else None
            if volume is not None:assert abs(volume+price-(ra-rb))<1e-6
            goods.append(dict(item=name,own_quantity=qa,rival_quantity=qb,own_price=pa,rival_price=pb,revenue_gap=ra-rb,quantity_component=volume,price_component=price))
        costs={k:(sum(a[k])-sum(b[k]) if isinstance(a[k],list) else a[k]-b[k]) for k in ('products','seeds','animals','hired','land')}
        gap=a['cash']-b['cash'];assert abs(sum(g['revenue_gap'] for g in goods)-sum(costs.values())-gap)<1e-6
        raw=read(E/'receipts'/folder/f'{variant}_{opp}.json.gz')['rows']
        assert len(raw)==100
        daily=[]
        for d in (0,3,6,9,11,15,20,25,29):
            def avg(side,item):
                return st.fmean(sum(row['cash_ledger'][day][row['seat'] if side==0 else 1-row['seat']]['sold'][item] for day in range(d+1)) for row in raw)
            daily.append(dict(day_index=d,own_cash=st.fmean(r['cash_ledger'][d][r['seat']]['end'] for r in raw),rival_cash=st.fmean(r['cash_ledger'][d][1-r['seat']]['end'] for r in raw),own_strawberry_sold=avg(0,3),rival_strawberry_sold=avg(1,3),own_melon_sold=avg(0,4),rival_melon_sold=avg(1,4)))
        timing=[]
        for i in (3,4,6,7):
            sides=[]
            for side in (0,1):
                weights=[sum(r['cash_ledger'][d][r['seat'] if side==0 else 1-r['seat']]['sold'][i] for r in raw) for d in range(30)]
                sides.append(sum(d*q for d,q in enumerate(weights))/sum(weights) if sum(weights) else None)
            timing.append(dict(item=names[i],own_volume_weighted_sale_day=sides[0],rival_volume_weighted_sale_day=sides[1]))
        records.append(dict(background=label,variant=variant,opponent=opp,games=100,independent_seeds=50,wins=x['wins'],cash=a['cash'],rival_cash=b['cash'],gap=gap,goods=goods,cost_gap=costs,daily=daily,timing=timing,
            unassigned_job_count=x['planner_mean_unassigned_tasks'],degraded_job_count=x['planner_mean_degraded_tasks']))
result=dict(status='COMPLETE_DESCRIPTIVE_ACCOUNTING_RESEARCH',source_hashes=hashes,records=records,
    scope='1600 existing full matches, 50 paired development seeds; offline rival accounting not policy input',
    caveat='Goods/gap decompositions are arithmetic; averages are not paired causal interventions, timing includes changes in production. Wages/land shared, so no invented per-industry ROI.',final_goal_acceptance=False)
(out/'summary.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))
lines=['# 当前胜率差距：已执行对局的跨对手账目','','现有1,600条完整对局，仅分析，无新模拟。每背景/对手50个开发seed×双座位；不是1,600独立剧本。对手私有账只用于离线核账，禁止进入策略输入。','','## 共享版：亏在哪里','','|对手|胜/100|终盘资金差|草莓收入差|牛奶收入差|羊毛收入差|工资支出差|','|---|---:|---:|---:|---:|---:|---:|']
for r in records:
    if r['background']=='shared':lines.append(f"|{r['opponent']}|{r['wins']}|{r['gap']:,.0f}|{r['goods'][3]['revenue_gap']:,.0f}|{r['goods'][6]['revenue_gap']:,.0f}|{r['goods'][7]['revenue_gap']:,.0f}|{r['cost_gap']['hired']:,.0f}|")
lines+=['','负收入差表示我方更少，负工资差表示我方支出更少。商品之间会互相影响，不能把各负数相加当作可独立回收利润。','','## 草莓：数量与价格不能混为一谈','','|对手|我方量|对手量|我方均价|对手均价|量差算术项|价差算术项|','|---|---:|---:|---:|---:|---:|---:|']
for r in records:
    if r['background']=='shared':
        g=r['goods'][3];lines.append(f"|{r['opponent']}|{g['own_quantity']:.2f}|{g['rival_quantity']:.2f}|{g['own_price']:.2f}|{g['rival_price']:.2f}|{g['quantity_component']:,.0f}|{g['price_component']:,.0f}|")
lines+=['','## 同一对局内的早期兑现（G003，不是目标日历）','','|背景|day index|我方现金|对手现金|我方累计草莓卖出|对手累计草莓卖出|','|---|---:|---:|---:|---:|---:|']
for r in records:
    if r['opponent']=='g003':
        for d in r['daily']:lines.append(f"|{r['background']}|{d['day_index']}|{d['own_cash']:,.0f}|{d['rival_cash']:,.0f}|{d['own_strawberry_sold']:.2f}|{d['rival_strawberry_sold']:.2f}|")
lines+=['','这些是结果账目，不证明固定多种某作物、某天强制投资就能提高胜率。必须验证预测兑现和资源约束，并用实际对手响应的独立/组合实验检验。']
(out/'TABLES_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
print('\n'.join(lines),flush=True)
