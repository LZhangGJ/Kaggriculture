"""Exact cash-ledger decomposition, not causal profit attribution."""
from pathlib import Path
import json,statistics
import run_panel as panel
HERE=Path(__file__).resolve().parent
def main():
    rows=json.loads((HERE/'baseline_audit/RESULTS.json').read_text())['rows'];groups={}
    for name,selected in [('losses',[x for x in rows if not x['r2_win']]),('win_controls',[x for x in rows if x['r2_win']])]:
        keys=sorted({k for x in selected for s in x['summary'] for k in s['cash_ledger']});items=[]
        for key in keys:
            own=statistics.mean(x['summary'][1-x['opponent_seat']]['cash_ledger'].get(key,0) for x in selected)
            rival=statistics.mean(x['summary'][x['opponent_seat']]['cash_ledger'].get(key,0) for x in selected)
            items.append(dict(item=key,own=own,rival=rival,delta=own-rival))
        for x in selected:
            own=sum(x['summary'][1-x['opponent_seat']]['cash_ledger'].values());rival=sum(x['summary'][x['opponent_seat']]['cash_ledger'].values())
            assert abs(own-rival-x['r2_margin'])<1e-8
        total={}
        for kind in ['SELL','BUY_ANIMAL','BUY_PRODUCT','BUY_SEED','BUY_LAND','HIRE']:
            total[kind]={key:sum(x[key] for x in items if x['item'].startswith(kind+':')) for key in ['own','rival','delta']}
        groups[name]=dict(games=len(selected),items=items,totals=total,mean_margin=statistics.mean(x['r2_margin'] for x in selected))
    panel.save(HERE/'baseline_audit/CASH_COSTS.json',groups)
    r=groups['losses'];lines=['# 败局现金差距的会计分解','',f"原R2全部{r['games']}败局，双方现金科目逐局精确闭合；不是挑一个seed，也不是各产业独立因果收益。",'', '| 科目 | R2平均现金流 | 对手平均现金流 | 差额 |','|---|---:|---:|---:|']
    for x in r['items']:lines.append(f"| {x['item']} | {x['own']:+,.2f} | {x['rival']:+,.2f} | {x['delta']:+,.2f} |")
    lines+=['','## 结论与边界','',f"全部出售回款：R2 {r['totals']['SELL']['own']:,.2f}，对手 {r['totals']['SELL']['rival']:,.2f}；平均终局分差 {r['mean_margin']:+,.2f}。",'这些败局并非普遍卖不出产品：总回款相近，成本结构不同。小麦还用于喂养；不能把少买饲料和保留全部动物产出同时当作无代价收益。','R2源码把finite_fertilizer设为false，有限作物生命周期只模拟已有肥效而不提出新施肥。官方小麦、胡萝卜、甜瓜在有效生长期浇水可获得肥效增产；需要比较肥料售价/购买成本、额外动作、增产、缩短占地和实际后续任务，而非强制全施肥。','扩充这种规划能力前要先检验规则递推、资源预留、预测与执行一致，再做全局多seed对照。会计差额不保证策略改进幅度。']
    (HERE/'baseline_audit/CASH_COSTS_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({k:{'games':v['games'],'totals':v['totals']} for k,v in groups.items()}),flush=True)
if __name__=='__main__':main()
