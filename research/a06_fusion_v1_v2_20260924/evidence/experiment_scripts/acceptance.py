"""Audit the complete held-out schedule and report strict win rates."""
from pathlib import Path
import argparse
import collections
import csv
import hashlib
import html
import json
import random

HERE=Path(__file__).resolve().parent

DISPLAY = {
    'external/n14_prvsiyan': 'Soil Remembers Rain',
    'external/n20_wzhengbiao': 'V15Stack',
    'external/d24_20_haideptry': '2965 Master Hybrid Engine',
    'external/n28_ahmedberatozer': 'V57 Funding Order Invariant',
    'external/n69_hosen42': 'M4A MetaV4',
    'external/d24_05_tetsutani': 'Demand Preserving Sale Timing (Soil identical source)',
    'external/d24_02_guruprasaathas111': 'Master Engine V3',
    'external/n24_dmitriigluzdov': 'More Wheat Smarter Sales',
    'external/n40_lynnsakurai': 'Farmer John',
    'external/n42_nathanjacob': 'Pipe18',
}


def bootstrap(rows):
    totals=collections.defaultdict(lambda:[0,0])
    for r in rows:
        totals[r['seed']][0]+=r['local_win'];totals[r['seed']][1]+=1
    seeds=sorted(totals);rng=random.Random(2026092499);values=[]
    for _ in range(10000):
        sampled=rng.choices(seeds,k=len(seeds))
        values.append(sum(totals[s][0] for s in sampled)/sum(totals[s][1] for s in sampled))
    values.sort();return [values[250],values[9749]]


def metrics(rows):
    n=len(rows)
    return dict(games=n,wins=sum(r['local_win'] for r in rows),ties=sum(r['tie'] for r in rows),
        losses=sum(not r['local_win'] and not r['tie'] for r in rows),strict_win_rate=sum(r['local_win'] for r in rows)/n,
        point_rate=sum(r['local_win']+.5*r['tie'] for r in rows)/n,mean_margin=sum(r['margin'] for r in rows)/n,
        mean_cash=sum(r['local_cash'] for r in rows)/n,seed_bootstrap_95=bootstrap(rows),
        seat_win_rates={str(s):sum(r['local_win'] for r in rows if r['local_seat']==s)/sum(r['local_seat']==s for r in rows) for s in (0,1)})


def main():
    p=argparse.ArgumentParser();p.add_argument('run');a=p.parse_args();out=HERE/'runs'/a.run
    protocol=json.loads((out/'PROTOCOL.json').read_text());frozen=json.loads((HERE/'FINAL_FREEZE.json').read_text())
    config=json.loads((HERE/'candidates'/frozen['candidate']/'policy/config.json').read_text())
    overlay=json.loads((HERE/'candidates'/frozen['candidate']/'overlay.json').read_text())
    assert protocol['stage']=='holdout' and list(protocol['candidates'])==[frozen['candidate']]
    assert protocol['candidates'][frozen['candidate']]==frozen['files']
    for file,sha in frozen['files'].items():
        assert hashlib.sha256((HERE/'candidates'/frozen['candidate']/file).read_bytes()).hexdigest()==sha,file
    rows=[json.loads(s) for s in (out/'games.jsonl').read_text().splitlines() if s]
    assert len(rows)==protocol['total_games']==2300
    assert all(r['error'] is None and r['steps']==719 for r in rows)
    expected={(o['id'],s,seat) for o in protocol['opponents'] for s in protocol['seeds'] for seat in (0,1)}
    actual={(r['public'],r['seed'],r['local_seat']) for r in rows}
    assert actual==expected and len(actual)==len(rows)
    groups={g:metrics([r for r in rows if r['public'].startswith(g+'/')]) for g in ('internal','external')}
    opponents={o['id']:metrics([r for r in rows if r['public']==o['id']]) for o in protocol['opponents']}
    dedup={}
    for o in protocol['opponents']:
        if o['group']=='external':dedup.setdefault(o['files']['main.py'],o['id'])
    unique_rows=[r for r in rows if r['public'] in dedup.values()]
    unique=metrics(unique_rows)
    passed=all(g['strict_win_rate']>=.9 for g in groups.values())
    result=dict(status='PASS_FROZEN_TWO_PANEL_POINT_TARGET' if passed else 'NOT_MET',candidate=frozen['candidate'],
        definition='Each frozen panel average strict win rate >=90%; draws are not wins. This is not a per-opponent or statistical lower-bound guarantee.',
        groups=groups,per_opponent=opponents,display_names=DISPLAY,configuration=config,opening_overlay=overlay,external_unique_source_count=len(dedup),external_unique_source_metrics=unique,
        files_frozen=True,schedule_complete=True,errors=0)
    comparison=None
    if (HERE/'PAIRED_EXTERNAL.json').is_file():
        p=json.loads((HERE/'PAIRED_EXTERNAL.json').read_text())
        if p['v2_candidate']==frozen['candidate']:
            assert p['v2_wins']==groups['external']['wins']
            result['paired_external_comparison']=comparison=p
    (HERE/'ACCEPTANCE.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    with (HERE/'PER_OPPONENT.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f);writer.writerow(['id','display_name','games','wins','ties','losses','strict_win_rate','mean_cash_margin','seat0_win_rate','seat1_win_rate'])
        for name,g in opponents.items():writer.writerow([name,DISPLAY.get(name,name.split('/')[-1]),g['games'],g['wins'],g['ties'],g['losses'],g['strict_win_rate'],g['mean_margin'],g['seat_win_rates']['0'],g['seat_win_rates']['1']])
    label='两个冻结面板均达到 90%' if passed else '未达到双 90% 目标'
    lines=['# 内外双面板未见种子验收','',f"候选：`{frozen['candidate']}`；结果：**{label}**（{result['status']}）。",'',
        '50 个预留种子、双座位；内部 13 对手共 1,300 局，外部 10 对手共 1,000 局。全部完成 719 次状态转移，零运行错误。平局不算胜。','',
        '| 面板 | 胜-平-负 | 严格胜率 | 95% 种子重采样区间 | 平均终局现金差 |','|---|---:|---:|---:|---:|']
    for name,g in groups.items():
        lo,hi=g['seed_bootstrap_95'];lines.append(f"| {name} | {g['wins']}-{g['ties']}-{g['losses']} | {g['strict_win_rate']:.2%} | {lo:.2%}–{hi:.2%} | {g['mean_margin']:+,.1f} |")
    lines+=['',f"外部按 {len(dedup)} 份不同 main.py 去重后的严格胜率：{unique['strict_win_rate']:.2%}。",'',
        '目标是两个冻结面板各自的平均胜率，不等于每个对手都达到 90%，也不等于 95% 置信下界达到 90%。这批公开方案来自 2026-09-23/24 快照；尚非线上天梯或 Kaggle 沙箱认证。','',
        '| 对手 | 胜-平-负 / 100 | 严格胜率 | 平均现金差 |','|---|---:|---:|---:|']
    for name,g in opponents.items():lines.append(f"| {DISPLAY.get(name,name)} | {g['wins']}-{g['ties']}-{g['losses']} | {g['strict_win_rate']:.1%} | {g['mean_margin']:+,.1f} |")
    if comparison:
        p=comparison;lo,hi=p['seed_bootstrap_delta_95']
        lines+=['','## 同种子外战对照','',f"使用当前这批 50 个种子复跑冻结第一版：V1 {p['v1_wins']}/1000，V2 {p['v2_wins']}/1000；配对胜率差 {p['v2_minus_v1_win_rate']*100:+.2f} 个百分点，95% 种子区间 {lo*100:+.2f} 至 {hi*100:+.2f}。详见 PAIRED_EXTERNAL_ZH.md。此为冻结版本诊断，没有重新调参。"]
    (HERE/'ACCEPTANCE_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    cards=''.join(f'<article><small>{name}</small><strong>{g["strict_win_rate"]:.2%}</strong><span>{g["wins"]} 胜 / {g["games"]} 局</span><p>95% 种子区间 {g["seed_bootstrap_95"][0]:.2%}–{g["seed_bootstrap_95"][1]:.2%}</p></article>' for name,g in groups.items())
    table=''.join(f'<tr data-group="{name.split("/")[0]}"><td title="{html.escape(name)}">{html.escape(DISPLAY.get(name,name))}</td><td>{g["wins"]}-{g["ties"]}-{g["losses"]}</td><td><div class="bar"><i style="width:{g["strict_win_rate"]*100:.1f}%"></i><b>{g["strict_win_rate"]:.1%}</b></div></td><td>{g["mean_margin"]:+,.0f}</td></tr>' for name,g in opponents.items())
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>A06 双面板验收</title>
<style>body{max-width:1120px;margin:40px auto;font:16px/1.6 system-ui;padding:0 24px;color:#172033;background:#f8fafc}h1{margin:4px 0 12px}small{color:#526479}.cards{display:flex;gap:18px;margin:24px 0}article{flex:1;padding:24px;background:white;border:1px solid #dce3ed;border-radius:14px}article strong{display:block;font-size:42px;color:#15605d}article p{font-size:14px;color:#526479}table{border-collapse:collapse;width:100%;background:white;font-size:14px}th,td{text-align:left;padding:12px;border-bottom:1px solid #e2e8f0}td:first-child{word-break:break-all}button{padding:9px 18px;margin:0 6px 14px 0;border:1px solid #bccbdc;border-radius:8px;background:white;cursor:pointer}.bar{position:relative;min-width:110px;background:#eef2f7;border-radius:5px;overflow:hidden}.bar i{position:absolute;height:100%;background:#b8dfd7}.bar b{position:relative;padding-left:7px}.note{padding:20px;background:#edf3fa;border-radius:12px;margin:20px 0}a{color:#17578c}@media(max-width:650px){.cards{display:block}article{margin-bottom:12px}body{padding:0 10px}th,td{padding:7px}}</style>
<small>2026-09-24 · 官方 1.32.7 本地规则 · 未见种子</small><h1>A06 内外双面板验收</h1>'''
    page+=f'<p>候选 <code>{html.escape(frozen["candidate"])}</code> · <strong>{label}</strong></p><div class="cards">{cards}</div>'
    page+=f'<div class="note">50 个预留种子、双座位、2,300 局完整对战，零运行错误。胜率只计胜局。外部按 {len(dedup)} 份不同源码去重：<b>{unique["strict_win_rate"]:.2%}</b>。通过标准是两个冻结面板分别平均 ≥90%，不表示逐对手达标或置信下界达到 90%。</div>'
    page+=f'<h2>融合了什么</h2><p>保留 Cashflow 的回款估值与公开供给出售逻辑；加入有现金和仓储约束的开局小麦买卖（{overlay["opening_liquidity"]} 单位）；关闭日内新增项目准入/采购，雇工上限 {config["max_hands"]}，新动物候选排序权重 {config["animal_bias"]}。日初经济规划和底层动态执行仍保留，不使用 ML/RL 或对手身份分支。</p>'
    page+='<h2>逐对手结果</h2><button onclick="filterRows(\'all\')">全部</button><button onclick="filterRows(\'internal\')">内部 13</button><button onclick="filterRows(\'external\')">公开 10</button><table><thead><tr><th>对手</th><th>胜-平-负</th><th>严格胜率</th><th>平均现金差</th></tr></thead><tbody>'+table+'</tbody></table>'
    if comparison:
        p=comparison;lo,hi=p['seed_bootstrap_delta_95']
        page+=f'<h2>同种子外战对照</h2><p>同一批 50 种子、双座位：第一版 {p["v1_wins"]}/1000，当前版 {p["v2_wins"]}/1000。配对胜率差 {p["v2_minus_v1_win_rate"]*100:+.2f} 个百分点，95% 种子区间 {lo*100:+.2f} 至 {hi*100:+.2f}。这是冻结版本的诊断比较，没有重新调参。</p><p><a href="PAIRED_EXTERNAL_ZH.md">逐对手配对结果</a></p>'
    page+='<p>公开方案为 2026-09-23/24 冻结快照，尚非实时天梯或 Kaggle 沙箱认证。完整证据：<a href="ACCEPTANCE.json">验收 JSON</a> · <a href="DEVELOPMENT_LOG_ZH.md">开发记录</a> · <a href="FINAL_FREEZE.json">冻结哈希</a></p><script>function filterRows(group){document.querySelectorAll("tbody tr").forEach(r=>r.hidden=group!=="all"&&r.dataset.group!==group)}</script></html>'
    (HERE/'ACCEPTANCE_ZH.html').write_text(page,encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('status','candidate','groups','external_unique_source_metrics')},ensure_ascii=False))


if __name__=='__main__':main()
