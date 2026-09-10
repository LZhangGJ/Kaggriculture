"""Re-evaluate the frozen two-history-day gate without replaying or training."""
from pathlib import Path
from collections import defaultdict
import json,statistics

HERE=Path(__file__).resolve().parent

def main():
    out=HERE/'sale_clock_calibration'
    data=json.loads((out/'RESULTS.json').read_text())
    selected=json.loads((HERE/'baseline_audit/SELECTION.json').read_text())['rows']
    wins={f"{r['opponent']}_{r['seed']}_{r['opponent_seat']}":r['r2_win'] for r in selected}
    groups=defaultdict(list)
    for case in data['rows']:
        for r in case['rows']:
            groups[('win_control' if wins[case['case']] else 'loss',r['item'])].append(r)
    result={}
    for (group,item),rows in groups.items():
        joint=[r for r in rows if 'old_share_error' in r]
        def avg(field,old,items):
            return statistics.mean(r[field] if r['history_sale_days']>=2 else r[old] for r in items) if items else None
        result[f'{group}/{item}']={
            'days':len(rows),'active_days':sum(r['history_sale_days']>=2 for r in rows),
            'old_time_mae':statistics.mean(r['old_transport_error'] for r in rows),
            'gated_time_mae':avg('new_transport_error','old_transport_error',rows),
            'joint_days':len(joint),
            'old_early_share_mae':statistics.mean(r['old_share_error'] for r in joint) if joint else None,
            'gated_early_share_mae':avg('new_share_error','old_share_error',joint)}
    (out/'TWO_DAY_GATE.json').write_text(json.dumps(result,indent=2))
    lines=['# 至少两个销售日才更新的时刻预测','',
           '重用此前保存的过去历史与离线标签，不重新拟合。只评价“当日确实卖了货”时的时刻误差，不代表能预测是否卖、卖多少。',
           '这里先卖比例使用我方当天实际卖货时刻作条件评价；不是在线双方历史时钟的最终预测精度。','',
           '|商品|有销售日|启用日|旧时刻误差|新时刻误差|','|---|---:|---:|---:|---:|']
    for key,r in result.items():
        if key.startswith('loss/'):
            lines.append(f"|{key.split('/')[1]}|{r['days']}|{r['active_days']}|{r['old_time_mae']:.3f}|{r['gated_time_mae']:.3f}|")
    (out/'TWO_DAY_GATE_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
