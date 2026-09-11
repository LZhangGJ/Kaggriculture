"""Paired full-game score and failure-pattern diagnostics; no oracle labels."""
from pathlib import Path
import argparse
import gzip
import json
import statistics

import run_panel as panel

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]

def main():
    p=argparse.ArgumentParser()
    p.add_argument('folder',type=Path)
    args=p.parse_args()
    folder=args.folder.resolve()
    rows=json.loads((folder/'rows.json').read_text())
    baseline={(r['opponent'],r['seed'],r['opponent_seat']):r for r in json.loads((HERE/'baseline_rows_all11.json').read_text())}
    matches=[]
    for r in rows:
        b=baseline[(r['opponent'],r['seed'],r['opponent_seat'])]
        trace=json.loads(gzip.decompress((folder/r['trace']).read_bytes()))
        debug=trace['days'][-1]['r2_debug']
        matches.append(dict(opponent=r['opponent'],seed=r['seed'],opponent_seat=r['opponent_seat'],
                            old_win=b['r2_win'],new_win=r['r2_win'],old_margin=b['r2_margin'],new_margin=r['r2_margin'],
                            margin_delta=r['r2_margin']-b['r2_margin'],cash_delta=r['r2_cash']-b['r2_cash'],
                            same_actions=r['joint_action_sha256']==b['joint_action_sha256'],
                            finance_repairs=debug.get('preparation_finance_repairs',0),
                            finance_checks=debug.get('preparation_finance_checks',0)))
    def summary(data):
        changes=[r['margin_delta'] for r in data]
        return dict(games=len(data),old_wins=sum(r['old_win'] for r in data),new_wins=sum(r['new_win'] for r in data),
                    rescued=sum(not r['old_win'] and r['new_win'] for r in data),
                    lost_wins=sum(r['old_win'] and not r['new_win'] for r in data),
                    unchanged_actions=sum(r['same_actions'] for r in data),
                    margin_up=sum(x>0 for x in changes),margin_down=sum(x<0 for x in changes),
                    mean_margin_delta=statistics.mean(changes),median_margin_delta=statistics.median(changes),
                    mean_cash_delta=statistics.mean(r['cash_delta'] for r in data),
                    finance_activated_games=sum(r['finance_repairs']>0 for r in data),
                    finance_repairs=sum(r['finance_repairs'] for r in data))
    by={name:summary([r for r in matches if r['opponent']==name]) for name in sorted({r['opponent'] for r in rows})}
    result=dict(overall=summary(matches),by_opponent=by,rows=matches,
                boundary='Development paired comparison, not independent holdout or promotion')
    panel.save(folder/'PAIRED.json',result)
    v=result['overall']
    lines=['# 采购融资补丁：配对开发测试','',
           f"{v['games']}局，原R2 {v['old_wins']}胜 → 开启融资 {v['new_wins']}胜；救回{v['rescued']}场，丢掉旧胜局{v['lost_wins']}场。",
           f"平均现金变化{v['mean_cash_delta']:+,.1f}，平均分差变化{v['mean_margin_delta']:+,.1f}；分差增益中位数{v['median_margin_delta']:+,.1f}。",
           f"融资修复实际触发{v['finance_activated_games']}局/{v['finance_repairs']}次；动作序列完全不变{v['unchanged_actions']}局。",'',
           '| 对手 | 原胜 | 新胜 | 救回 | 丢旧胜 | 分差变化 |', '|---|---:|---:|---:|---:|---:|']
    for name,x in by.items():
        lines.append(f"| {name} | {x['old_wins']} | {x['new_wins']} | {x['rescued']} | {x['lost_wins']} | {x['mean_margin_delta']:+,.1f} |")
    lines += ['','所有对手原程序实时响应；同种子同座位完整重跑。现金和对手市场反应可联动改变，不能把单步多卖的钱等同终局增益。',
              '此表只用于筛选是否扩大测试，不等于平均90%验收。']
    (folder/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)

if __name__=='__main__':main()
