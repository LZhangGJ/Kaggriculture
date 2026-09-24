"""Report completed frozen matchups with seed-cluster paired uncertainty."""
from pathlib import Path
import csv
import html
import json
import random
import statistics

HERE=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def lines(p):return [json.loads(s)for s in p.read_text(encoding='utf-8').splitlines()if s]
def key(r):return r['candidate'],r['opponent'],r['seed'],r['seat']
def interval(values):
    rng=random.Random(2026092443)
    draws=sorted(statistics.fmean(rng.choices(values,k=len(values)))for _ in range(5000))
    return [draws[124],draws[4874]]
def save(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def main():
    result=dict(status='COMPLETED_MATCHUPS',opponents={},boundaries=[
        'Unmodified V1/V2 and teammate submission packages; no training or tuning.',
        'Official 1.32.7 local Python interpreter, not hosted sandbox or strict timeout enforcement.',
        '100 shared environment seeds, both seats, 200 games per candidate per opponent.',
        'Original sampled opponent inference retained; independent recorded policy RNG shared across V1/V2.',
        'Confidence intervals resample the 100 seed clusters, not the 200 seats independently.'])
    csv_rows=[]
    all_rows={}
    for path in sorted((HERE/'runs').glob('run_*/SUMMARY.json')):
        folder=path.parent
        summary=read(path)
        assert summary['games']==summary['expected']==summary['complete']==400 and summary['errors']==0
        rows=lines(folder/'games.jsonl');ident=rows[0]['opponent']
        protocol=read(HERE/f'PROTOCOL_{ident}.json')
        seeds=protocol['randomization']['seeds']
        expected={(c,ident,s,t)for c in ('V1','V2')for s in seeds for t in(0,1)}
        assert len(rows)==len({key(r)for r in rows})==400
        assert {key(r)for r in rows}==expected
        assert all(not r['error'] and r['steps']==719 and r['opponent_audit']['fallbacks']==0 and
                   r['opponent_audit']['illegal']==0 and r['opponent_audit']['days']==17 for r in rows)
        indexed={key(r):r for r in rows}
        all_rows[ident]=indexed
        by_candidate={}
        for c in ('V1','V2'):
            group=[r for r in rows if r['candidate']==c]
            wins=sum(r['win']for r in group);ties=sum(r['tie']for r in group)
            values=[statistics.fmean(indexed[c,ident,s,t]['win']for t in(0,1))for s in seeds]
            item=dict(games=200,wins=wins,ties=ties,losses=200-wins-ties,win_rate=wins/200,
                mean_cash=statistics.fmean(r['own_cash']for r in group),
                mean_opponent_cash=statistics.fmean(r['opponent_cash']for r in group),
                mean_margin=statistics.fmean(r['margin']for r in group),
                seed_cluster_ci95=interval(values),
                seat_wins={str(t):sum(r['win']for r in group if r['seat']==t)for t in(0,1)})
            by_candidate[c]=item
            csv_rows.append(dict(opponent=ident,candidate=c,**{k:v for k,v in item.items()if not isinstance(v,(dict,list))}))
        delta=[statistics.fmean(indexed['V2',ident,s,t]['win']-indexed['V1',ident,s,t]['win']for t in(0,1))for s in seeds]
        paired=dict(v2_minus_v1_win_rate=statistics.fmean(delta),seed_cluster_ci95=interval(delta),
            v1_win_to_v2_loss=sum(indexed['V1',ident,s,t]['win'] and not indexed['V2',ident,s,t]['win']for s in seeds for t in(0,1)),
            v1_loss_to_v2_win=sum(not indexed['V1',ident,s,t]['win'] and indexed['V2',ident,s,t]['win']for s in seeds for t in(0,1)),
            mean_margin_change=by_candidate['V2']['mean_margin']-by_candidate['V1']['mean_margin'])
        checks=[]
        for checkpath in (HERE/'runs').glob(f'*{ident}*/games.jsonl'):
            if checkpath.parent==folder:continue
            for row in lines(checkpath):
                base=indexed.get(key(row))
                if not base:continue
                exact=all(row.get(k)==base.get(k)for k in ('own_cash','opponent_cash','win','tie','steps','own_actions_sha256','opponent_actions_sha256')) and not row['error']
                checks.append(dict(run=checkpath.parent.name,candidate=row['candidate'],seed=row['seed'],seat=row['seat'],exact=exact,
                    max_own_noninitial_s=row.get('max_own_noninitial_s'),max_opponent_noninitial_s=row.get('max_opponent_noninitial_s')))
        assert all(r['exact']for r in checks)
        result['opponents'][ident]=dict(by_candidate=by_candidate,paired=paired,reproduction_checks=checks,
            games=400,complete=400,model_days=6800,fallbacks=0,illegal=0)
    assert result['opponents']
    if set(all_rows)=={'student_v306','student_v463'}:
        a,b='student_v306','student_v463'
        pa=read(HERE/f'PROTOCOL_{a}.json');pb=read(HERE/f'PROTOCOL_{b}.json')
        assert pa['randomization']==pb['randomization']
        assert pa['candidates']==pb['candidates'] and pa['referee']==pb['referee']
        result['opponent_upgrade_comparison']={}
        for c in ('V1','V2'):
            vals=[statistics.fmean(all_rows[b][c,b,s,t]['win']-all_rows[a][c,a,s,t]['win']for t in(0,1))for s in seeds]
            result['opponent_upgrade_comparison'][c]=dict(
                our_win_rate_vs_v463_minus_vs_v306=statistics.fmean(vals),
                seed_cluster_ci95=interval(vals),
                our_win_to_loss=sum(all_rows[a][c,a,s,t]['win']and not all_rows[b][c,b,s,t]['win']for s in seeds for t in(0,1)),
                our_loss_to_win=sum(not all_rows[a][c,a,s,t]['win']and all_rows[b][c,b,s,t]['win']for s in seeds for t in(0,1)),
                mean_margin_change=statistics.fmean(all_rows[b][c,b,s,t]['margin']-all_rows[a][c,a,s,t]['margin']for s in seeds for t in(0,1)))
        result['status']='ALL_REQUESTED_MATCHUPS_COMPLETE'
        result['total_main_games']=800
    save(HERE/'RESULTS.json',result)
    with (HERE/'RESULTS.csv').open('w',encoding='utf-8-sig',newline='')as f:
        w=csv.DictWriter(f,fieldnames=list(csv_rows[0]));w.writeheader();w.writerows(csv_rows)
    text=['# V1 / V2 对战队友提交版本','',
        '每组 100 个共享新种子 × 双座位 = 200 盘。胜率只计严格获胜；对手保留原包的采样行为，采样 RNG 与环境种子独立，并在 V1/V2 间配对。','',
        '| 队友版本 | 我方 | 胜 / 平 / 负 | 胜率 | 我方平均终局现金 | 对手平均终局现金 | 平均现金差 |',
        '|---|---|---|---|---|---|---|']
    for opponent, data in result['opponents'].items():
        for c,s in data['by_candidate'].items():
            text.append(f"| {opponent} | {c} | {s['wins']} / {s['ties']} / {s['losses']} | {s['win_rate']:.1%} | {s['mean_cash']:,.1f} | {s['mean_opponent_cash']:,.1f} | {s['mean_margin']:+,.1f} |")
    text+=['','## 配对比较','']
    for opponent,data in result['opponents'].items():
        p=data['paired'];ci=p['seed_cluster_ci95']
        text.append(f"- {opponent}：V2 − V1 = {p['v2_minus_v1_win_rate']*100:+.1f} 个百分点；按种子成组 bootstrap 的 95% 区间 [{ci[0]*100:+.1f}, {ci[1]*100:+.1f}] 个百分点。V1 胜→V2 负 {p['v1_win_to_v2_loss']} 局，V1 负→V2 胜 {p['v1_loss_to_v2_win']} 局。")
    if 'opponent_upgrade_comparison' in result:
        text+=['','## 队友 V463 相对 V306','']
        for c,p in result['opponent_upgrade_comparison'].items():
            ci=p['seed_cluster_ci95']
            text.append(f"- 固定我方 {c}，换成 V463 对手后，我方胜率改变 {p['our_win_rate_vs_v463_minus_vs_v306']*100:+.1f} 个百分点，95% 种子成组区间 [{ci[0]*100:+.1f}, {ci[1]*100:+.1f}]。我方胜→负 {p['our_win_to_loss']} 局，负→胜 {p['our_loss_to_win']} 局。")
        text.append('- 两个原包不仅权重不同，V463 还增加了 FundedReplayRoute 前期资金保护；不能将变化完全归因于模型训练。这里也没有直接进行 V306 对 V463 的对战。')
    text+=['','## 串行复核与耗时','']
    for opponent,data in result['opponents'].items():
        checks=data['reproduction_checks']
        serial=[r for r in checks if r['run'].startswith('serial_')]
        text.append(f"- {opponent}：{len(checks)}/{len(checks)} 个重复检查的双方完整动作哈希与终局现金一致，其中 {len(serial)} 个为串行复跑；重复局未计入正式胜率。")
        for candidate in ('V1','V2'):
            group=[r for r in serial if r['candidate']==candidate]
            if group:
                value=max(r['max_own_noninitial_s']for r in group)
                note='超过 1 秒单步预算，本次胜率没有按超时判负。'if value>1 else '仅代表抽查状态的本机耗时。'
                text.append(f'- {candidate} 抽查串行最大非首步耗时 {value:.3f} 秒；{note}')
    text+=['','## 验收与边界','',
        '- 所有已列对局均完整到 719 步，模型每局正常执行 17 个决策日，无 fallback 或非法类别。',
        '- V1/V2 均按上一轮冻结源码运行；队友对手按原始 Kaggle 包运行，未改成 argmax，也没有用普通非 NN 的 agent/main.py 替代。',
        '- 原包、下载哈希、来源提交与逐局数据保存在本目录，PROTOCOL 固定源码和裁判哈希。',
        '- 裁判采用官方 1.32.7 Python 规则；这是本地强度比较，不代表 Kaggle 超时沙箱验收。',
        '- 重复烟测及串行复跑结果见 RESULTS.json；重复局不计入 200 盘。',
        '- 本次种子与我方上一轮融合实验种子互斥；不能据此证明与队友全部历史训练种子互斥。','']
    (HERE/'REPORT_ZH.md').write_text('\n'.join(text),encoding='utf-8')
    header='<meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>V1 / V2 对战队友提交</title><style>body{max-width:1200px;margin:40px auto;padding:0 24px;font:16px/1.7 system-ui;color:#17212b}table{border-collapse:collapse;width:100%}th,td{padding:10px;border:1px solid #cbd5e1;text-align:right}th{background:#eef3fa}h1,h2{color:#163c66}code{background:#eef2f5;padding:2px 5px}</style>'
    body=['<h1>V1 / V2 对战队友提交</h1>','<p>'+html.escape(text[2])+'</p>',
          '<table><thead><tr>'+''.join('<th>'+html.escape(s)+'</th>'for s in ['队友版本','我方','胜/平/负','胜率','我方现金','对手现金','现金差'])+'</tr></thead><tbody>']
    for r in csv_rows:
        vals=[r['opponent'],r['candidate'],f"{r['wins']}/{r['ties']}/{r['losses']}",f"{r['win_rate']:.1%}",f"{r['mean_cash']:,.1f}",f"{r['mean_opponent_cash']:,.1f}",f"{r['mean_margin']:+,.1f}"]
        body.append('<tr>'+''.join('<td>'+html.escape(v)+'</td>'for v in vals)+'</tr>')
    body+=['</tbody></table>']
    for line in text[text.index('## 配对比较'):]:
        if line.startswith('## '):body.append('<h2>'+html.escape(line[3:])+'</h2>')
        elif line:body.append('<p>'+html.escape(line.lstrip('- '))+'</p>')
    body.append('<p><a href="RESULTS.json">详细数据</a> · <a href="RESULTS.csv">CSV</a></p>')
    (HERE/'REPORT_ZH.html').write_text('<!doctype html><html lang="zh"><head>'+header+'</head><body>'+''.join(body)+'</body></html>',encoding='utf-8')
    print(json.dumps({op:{c:s for c,s in data['by_candidate'].items()}for op,data in result['opponents'].items()},ensure_ascii=False))

if __name__=='__main__':main()
