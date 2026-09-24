"""Write the selection trail separately from the independent acceptance result."""
from pathlib import Path
import json

HERE=Path(__file__).resolve().parent


def main():
    comparison=json.loads((HERE/'V2_DEVELOPMENT_COMPARISON.json').read_text())
    assert all(r['complete'] and r['errors']==0 for r in comparison)
    lines=['# 第二轮融合开发与选择','',
           'V1 正式验收：内战 1243/1300（95.62%），外战 861/1000（86.10%），未达双 90%。完整失败证据保存在 `versions/v1` 与 `runs/holdout_v1`。',
           '', '下表是同一批 16 个开发种子、双座位，2 内部难对手 + 3 外部代表的筛选结果。它刻意突出薄弱对手，不能把该表外战胜率当成公开十对手平均。所有参数选择都发生在 V2 新验收种子开跑前。','',
           '| 方案 | 内部胜场 / 64 | 外部胜场 / 96 | 平均现金差 |','|---|---:|---:|---:|']
    for r in comparison:
        lines.append(f"| {r['candidate']} | {r['internal_wins']} | {r['external_wins']} | {r['mean_margin']:+,.1f} |")
    lines+=['','重复对局检查：V1 完整开发与 V2 控制组之间有 160 局重叠，双方现金、胜负和步数 160/160 完全一致。开发与验收之间的落差是样本和策略泛化问题，未观察到这组重放的不确定性。','',
            '机制结论：开局买卖量 8 与 10 在筛选池终局完全相同，5 和 15 明显变差；增加运输压力触发未带来净胜场收益。调整新动物候选的排序权重有效，但 0.4 过度削弱动物竞争力，开局样例转成 20 甜瓜种子并导致整体崩溃。保持原生动态规划，用适中权重调整后续资产分配更可靠。','']
    full=HERE/'runs/development_v2_full/RESULTS.json'
    if full.is_file():
        r=json.loads(full.read_text());assert r['status']=='COMPLETE'
        lines+=['## 完整开发面板','', '16 开发种子 × 双座 × 23 对手。仍不是独立验收。','',
                '| 候选 | 内战 | 外战 |','|---|---:|---:|']
        for name,data in r['candidates'].items():
            a,b=data['internal'],data['external']
            lines.append(f"| {name} | {a['wins']}/{a['games']} = {a['win_rate']:.2%} | {b['wins']}/{b['games']} = {b['win_rate']:.2%} |")
    freeze=HERE/'FINAL_FREEZE_V2.json'
    if freeze.is_file():
        f=json.loads(freeze.read_text());lines+=['',f"选入 V2 验收的候选是 `{f['candidate']}`。{f['selection_reason']}",'',
            '全新 50 个种子记录在 `SEEDS_V2.json`，候选源文件哈希记录在 `FINAL_FREEZE_V2.json`。V1 用过的种子被排除。正式结论仅以新验收回执为准。']
        if (HERE/'PAIRED_EXTERNAL.json').is_file():
            p=json.loads((HERE/'PAIRED_EXTERNAL.json').read_text())
            lines+=['','## 最终没有晋级','',
                    f"V2 独立验收为内战 1242/1300（95.54%），外战 832/1000（83.20%）。同一批种子外战，V1 为 {p['v1_wins']}/1000，V2 为 {p['v2_wins']}/1000。保留 V1 作为基线；开发改善没有提供足够的晋级依据。最终完整说明见 FINAL_REPORT_ZH.md。"]
    (HERE/'V2_RESULTS_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Wrote V2_RESULTS_ZH.md')


if __name__=='__main__':main()
