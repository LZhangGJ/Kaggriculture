"""Matched-seed external comparison of the two already-frozen variants."""
from pathlib import Path
import collections
import json
import random

HERE=Path(__file__).resolve().parent


def main():
    def read(run):
        folder=HERE/'runs'/run
        summary=json.loads((folder/'SUMMARY.json').read_text())
        assert summary['games']==summary['expected']==summary['complete_719'] and not summary['errors']
        rows=[json.loads(s) for s in (folder/'games.jsonl').read_text().splitlines() if s]
        return json.loads((folder/'PROTOCOL.json').read_text()),[r for r in rows if r['public'].startswith('external/')]
    a,old=read('paired_external_v1_on_v2');b,new=read('holdout_v2')
    assert a['seeds']==b['seeds'] and a['engine_sha256']==b['engine_sha256']
    assert a['opponents']==[o for o in b['opponents'] if o['group']=='external']
    key=lambda r:(r['public'],r['seed'],r['local_seat'])
    old={key(r):r for r in old};new={key(r):r for r in new}
    assert old.keys()==new.keys() and len(new)==1000
    deltas=collections.defaultdict(list)
    for k,r in new.items():deltas[r['seed']].append(r['local_win']-old[k]['local_win'])
    seed_delta=[sum(v)/len(v) for v in deltas.values()];rng=random.Random(20260924999)
    draws=sorted(sum(rng.choices(seed_delta,k=len(seed_delta)))/len(seed_delta) for _ in range(10000))
    opponents={}
    for name in sorted({r['public'] for r in new.values()}):
        selected=[k for k in new if k[0]==name]
        opponents[name]=dict(v1_wins=sum(old[k]['local_win'] for k in selected),v2_wins=sum(new[k]['local_win'] for k in selected),games=len(selected),
            mean_margin_change=sum(new[k]['margin']-old[k]['margin'] for k in selected)/len(selected))
    result=dict(v1_candidate=next(iter(a['candidates'])),v2_candidate=next(iter(b['candidates'])),
        games_per_version=1000,seeds=50,both_seats=True,opponent_sources_identical=True,
        v1_wins=sum(r['local_win'] for r in old.values()),v2_wins=sum(r['local_win'] for r in new.values()),
        v1_ties=sum(r['tie'] for r in old.values()),v2_ties=sum(r['tie'] for r in new.values()),
        v2_minus_v1_win_rate=sum(seed_delta)/len(seed_delta),seed_bootstrap_delta_95=[draws[250],draws[9749]],
        v1_win_to_v2_loss=sum(old[k]['local_win'] and not new[k]['local_win'] and not new[k]['tie'] for k in new),
        v1_loss_to_v2_win=sum(not old[k]['local_win'] and not old[k]['tie'] and new[k]['local_win'] for k in new),
        mean_margin_change=sum(new[k]['margin']-old[k]['margin'] for k in new)/len(new),per_opponent=opponents,
        interpretation='Post-acceptance diagnosis with both source versions frozen. Same seeds and seats, live opposing policies; no retuning or new unseen-performance claim.')
    (HERE/'PAIRED_EXTERNAL.json').write_text(json.dumps(result,indent=2)+'\n')
    lo,hi=result['seed_bootstrap_delta_95']
    lines=['# 同种子外战对照','',
           '两个版本均已冻结。本补测使用 V2 的同一批 50 个种子、双座、10 个冻结公开入口，每版 1,000 局；双方实时决策。它用于诊断版本差异，没有重新调参，也不另称新一次未见验收。','',
           f"- 第一版：{result['v1_wins']}/1000 = {result['v1_wins']/1000:.2%}。",
           f"- 动物权重 0.6 版：{result['v2_wins']}/1000 = {result['v2_wins']/1000:.2%}。",
           f"- 配对胜率差：{result['v2_minus_v1_win_rate']*100:+.2f} 个百分点；按种子重采样的 95% 区间：{lo*100:+.2f} 至 {hi*100:+.2f} 个百分点。",
           f"- 第一版胜变败：{result['v1_win_to_v2_loss']} 局；第一版败变胜：{result['v1_loss_to_v2_win']} 局。",'',
           '| 对手 | V1 胜场 / 100 | V2 胜场 / 100 | V2 平均现金差变化 |','|---|---:|---:|---:|']
    for name,g in opponents.items():lines.append(f"| {name} | {g['v1_wins']} | {g['v2_wins']} | {g['mean_margin_change']:+,.1f} |")
    lines+=['','不要把 V1 第一批种子的 86.10% 与 V2 第二批种子的 83.20% 直接相减来归因；上表才控制了种子、座位和对手版本。','']
    (HERE/'PAIRED_EXTERNAL_ZH.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='per_opponent'}))


if __name__=='__main__':main()
