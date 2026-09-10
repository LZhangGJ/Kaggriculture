"""Aggregate exact financial audits; correlations are not repair payoffs."""
from pathlib import Path
import json
import statistics

import run_panel as panel

HERE=Path(__file__).resolve().parent

def summarize(rows):
    own=[r['summary'][1-r['opponent_seat']] for r in rows]
    rivals=[r['summary'][r['opponent_seat']] for r in rows]
    def avg(seq):return statistics.mean(seq) if seq else None
    def median(seq):return statistics.median(seq) if seq else None
    def overflow(v):return sum(sum(e['items'].values()) for e in v['overflow'])
    result=dict(games=len(rows),cash_reject_games=sum(any(e['reason']=='no_cash' for e in v['rejects']) for v in own),
                zero_effect_hire_games=sum(any(e['op']=='HIRE' for e in v['rejects']) for v in own),
                overflow_games=sum(overflow(v)>0 for v in own),mean_overflow=avg([overflow(v) for v in own]),
                no_effect_unit_games=sum(sum(v['no_effect_units'].values())>0 for v in own),
                terminal_product_stock_games=sum(any(v['final_private']['shed'].get(k,0)>0 or any(b.get(k,0)>0 for b in v['final_private']['inventories']) for k in v['products']) for v in own),
                r2_margin_mean=avg([r['r2_margin'] for r in rows]),products={})
    products=list(own[0]['products']) if own else []
    for k in products:
        a=[v['products'][k] for v in own];b=[v['products'][k] for v in rivals]
        result['products'][k]=dict(r2_volume_mean=avg([x['quantity'] for x in a]),
                                  opponent_volume_mean=avg([x['quantity'] for x in b]),
                                  r2_revenue_mean=avg([x['revenue'] for x in a]),
                                  opponent_revenue_mean=avg([x['revenue'] for x in b]),
                                  r2_first_sale_median=median([x['first_sale_step'] for x in a if x['first_sale_step'] is not None]),
                                  opponent_first_sale_median=median([x['first_sale_step'] for x in b if x['first_sale_step'] is not None]))
    return result

def main():
    audit=json.loads((HERE/'baseline_audit/RESULTS.json').read_text())
    assert audit['status']=='PASS'
    rows=audit['rows'];losses=[r for r in rows if not r['r2_win']];wins=[r for r in rows if r['r2_win']]
    summary=dict(status='PASS',audited=len(rows),loss_count=len(losses),winner_controls=len(wins),
                 losses=summarize(losses),wins=summarize(wins),by_opponent={})
    for name in sorted({r['opponent'] for r in rows}):
        summary['by_opponent'][name]=dict(losses=summarize([r for r in losses if r['opponent']==name]),
                                         wins=summarize([r for r in wins if r['opponent']==name]))
    panel.save(HERE/'baseline_audit/SUMMARY.json',summary)
    a,b=summary['losses'],summary['wins']
    lines=['# 11公开对手：原R2全部败局审计','',
           f"官方1.32.7逐动作复演全部{len(losses)}场败局，加{len(wins)}场胜局对照。每场31个保存时点完整一致，719步与终局现金一致；双方现金账精确闭合。",
           '胜局对照每对手按seed跨度取5场座位0，不是随机比例样本；双座位相关。以下是描述性差距，不是补丁可追回收益。','',
           '| 指标 | 败局 | 胜局对照 |','|---|---:|---:|']
    for key,label in [('cash_reject_games','商品订单现金拒单局数'),('zero_effect_hire_games','无效果雇工局数（须继续核对原因）'),('overflow_games','日末溢出局数'),('mean_overflow','平均日末溢出件数'),('no_effect_unit_games','非移动/PASS动作无效果局数'),('terminal_product_stock_games','终局私有产品未清空局数')]:
        lines.append(f"| {label} | {a[key]} | {b[key]} |")
    lines += ['', '## 败局各商品已实现回款', '',
              '| 商品 | R2均销量 | 对手均销量 | R2均回款 | 对手均回款 | R2首次卖出中位step | 对手 |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for k,v in a['products'].items():
        lines.append(f"| {k} | {v['r2_volume_mean']:.1f} | {v['opponent_volume_mean']:.1f} | {v['r2_revenue_mean']:.1f} | {v['opponent_revenue_mean']:.1f} | {v['r2_first_sale_median']} | {v['opponent_first_sale_median']} |")
    lines += ['', '## 解释边界', '',
              '销量和回款差额不是各产业独立净利润；小麦承担饲料，工人和土地是共享成本。单项差额不能直接相加当作无代价改进。',
              '融资开关首轮已证明：改善成交也可能改变后续投资和对手路由，导致原胜局转负。不得只验证本步动作效果。',
              '未来商店序列和双方私有库存只用于离线审计，不可传入真实策略。',
              '每场完整证据在baseline_audit/<对手>_<seed>_<seat>/AUDIT.json.gz，含双方开局、每天产业、投资/雇工、逐笔成交、转产动作、溢出与终局。',
              '此文件是定量审计底表，尚需按模式逐组解释和验证策略修改；不宣称仅凭表格已修复问题。']
    (HERE/'baseline_audit/REPORT_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k not in {'by_opponent','wins'}}),flush=True)

if __name__=='__main__': main()
