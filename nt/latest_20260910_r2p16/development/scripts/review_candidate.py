"""Representative paired full-game audits, including adverse and positive cases."""
from pathlib import Path
from collections import Counter
import argparse
import gzip
import json

import audit_trace
import run_panel as panel

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


def gz(path):return json.loads(gzip.decompress(path.read_bytes()))


def audit(row,folder):
    if not (folder/'AUDIT.json.gz').exists():audit_trace.run_case(row,folder)
    return gz(folder/'AUDIT.json.gz')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('folder',type=Path);parser.add_argument('--baseline-folder',type=Path);args=parser.parse_args()
    folder=args.folder.resolve()
    baseline={(r['opponent'],r['seed'],r['opponent_seat']):r for r in json.loads((HERE/'baseline_rows_all11.json').read_text())}
    baseline_audit=HERE/'baseline_audit'
    if args.baseline_folder:
        base_folder=args.baseline_folder.resolve();baseline_audit=base_folder/'audit';baseline={}
        for row in json.loads((base_folder/'rows.json').read_text()):
            row=dict(row,trace=str((base_folder/row['trace']).relative_to(ROOT)))
            baseline[row['opponent'],row['seed'],row['opponent_seat']]=row
    rows=json.loads((folder/'rows.json').read_text())
    groups={name:[] for name in ['rescued','lost_win','still_lost','still_won']}
    for r in rows:
        b=baseline[(r['opponent'],r['seed'],r['opponent_seat'])]
        name=('still_won' if b['r2_win'] else 'rescued') if r['r2_win'] else ('lost_win' if b['r2_win'] else 'still_lost')
        groups[name].append((b,r))
    selected=[]
    for name,data in groups.items():
        # Distinct seeds and independent actual action traces prevent seven
        # copies of the same family/game from filling the representative set.
        data.sort(key=lambda pair:(abs(pair[1]['r2_margin']),pair[1]['seed'],pair[1]['opponent']))
        used=set();take=[]
        for b,r in data:
            if r['seed'] in used:continue
            used.add(r['seed']);take.append((b,r))
        if take:
            selected.append((name,*take[len(take)//2]))
            # With exactly two seeds the median above is already the last.
            # Select another case, not the same case a second time.
            if len(take)>1:selected.append((name,*(take[-1] if len(take)//2!=len(take)-1 else take[0])))
    report=[]
    lines=['# 候选修复：胜负转换全局复盘','',
           '每类取不同seed的代表，覆盖救回、丢旧胜、仍输、仍赢。只作机制解释，不以这几个样本估计总体胜率。',
           '每场均用原动作逐步重演官方1.32.7，核对全部保存帧和双边现金账。这里的未来商店只用于事后解释，不是Agent输入。',
           '重要：官方日末先按空地抽杂草，再用同一RNG抽商店。改变土地/作物状态可使同seed的后续商店变化。现金科目差异是整条轨迹的分解，不能全部归因于某个经营补丁；逐例救回不等于排除了商店随机影响。','']
    for group,b,r in selected:
        key=f"{r['opponent']}_{r['seed']}_{r['opponent_seat']}"
        own=1-r['opponent_seat']
        normalized=dict(r,trace=str((folder/r['trace']).relative_to(ROOT)))
        before=audit(b,baseline_audit/key)
        after=audit(normalized,folder/'audit'/key)
        bt=gz(ROOT/b['trace']);nt=gz(folder/r['trace'])
        first=next((i for i,(x,y) in enumerate(zip(bt['actions'],nt['actions'])) if x[own]!=y[own]),None)
        ledgers=[]
        for seat in (own,r['opponent_seat']):
            old=before['summary'][seat]['cash_ledger'];new=after['summary'][seat]['cash_ledger']
            ledgers.append({k:new.get(k,0)-old.get(k,0) for k in set(old)|set(new)})
        shop_changed=b['shops']!=r['shops']
        first_shop=next((i+1 for i,(x,y) in enumerate(zip(b['shops'],r['shops'])) if x!=y),None)
        payload=dict(group=group,baseline=b,candidate=r,first_own_difference=first,shop_sequence_changed=shop_changed,first_changed_shop=first_shop,
                     old_first_action=bt['actions'][first][own] if first is not None else None,
                     new_first_action=nt['actions'][first][own] if first is not None else None,
                     own_cash_delta=ledgers[0],opponent_cash_delta=ledgers[1],
                     old_summary=before['summary'],new_summary=after['summary'])
        report.append(payload)
        lines += [f"## {group} · {key}",'',
                  f"R2 {b['r2_cash']:,.0f} → {r['r2_cash']:,.0f}；对手 {b['opponent_cash']:,.0f} → {r['opponent_cash']:,.0f}；分差 {b['r2_margin']:+,.0f} → {r['r2_margin']:+,.0f}。",
                  f"首次我方动作变化：step {first}；首次变化的商店序号：{first_shop}。",
                  f"原版商店：{' → '.join(b['shops'])}。",
                  f"新版商店：{' → '.join(r['shops'])}。",'',
                  '| 天 | 原R2产业 | 新R2产业 | 对手产业（新对战） | 原/新雇工 | 原/新现金 |',
                  '|---|---|---|---|---:|---:|']
        for x,y in zip(before['daily_before_last_action'],after['daily_before_last_action']):
            if x['day'] not in {1,4,7,10,13,16,20,25,30}:continue
            def composition(f):return ', '.join(f'{k}:{v}' for k,v in f['counts'].items() if k in panel.old.ITEMS) if hasattr(panel.old,'ITEMS') else ', '.join(f'{k}:{v}' for k,v in f['counts'].items() if k not in {'LOCKED','EMPTY','WEED','PASTURE','COOP'})
            a=x['farms'][own];c=y['farms'][own];opp=y['farms'][1-own]
            lines.append(f"| {x['day']} | {composition(a)} | {composition(c)} | {composition(opp)} | {a['hands']}/{c['hands']} | {a['cash']:,.0f}/{c['cash']:,.0f} |")
        lines += ['','我方变化最大的现金科目（实际成交，不等同产业独立净利润）：','']
        for item,delta in sorted(ledgers[0].items(),key=lambda x:-abs(x[1]))[:8]:lines.append(f'- {item}：{delta:+,.0f}')
        lines += ['','对手变化最大的现金科目：','']
        for item,delta in sorted(ledgers[1].items(),key=lambda x:-abs(x[1]))[:4]:lines.append(f'- {item}：{delta:+,.0f}')
        lines.append('')
        panel.save(folder/'REVIEW_CASES.partial.json',report)
        print(json.dumps(dict(group=group,key=key,first_difference=first)),flush=True)
    panel.save(folder/'REVIEW_CASES.json',report)
    (folder/'REVIEW_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':main()
