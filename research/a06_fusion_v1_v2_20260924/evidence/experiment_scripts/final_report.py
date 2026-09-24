"""Create the final bilingual decision report without hiding failed trials."""
from pathlib import Path
import html
import json

HERE=Path(__file__).resolve().parent


def main():
    old=json.loads((HERE/'versions/v1/ACCEPTANCE.json').read_text())
    new=json.loads((HERE/'ACCEPTANCE.json').read_text())
    paired=json.loads((HERE/'PAIRED_EXTERNAL.json').read_text())
    timing=json.loads((HERE/'SERIAL_VERIFICATION.json').read_text())
    old_timing=json.loads((HERE/'versions/v1/SERIAL_VERIFICATION.json').read_text())
    cases=json.loads((HERE/'FAILURE_REVIEW.json').read_text())
    assert new['status']=='NOT_MET' and old['status']=='NOT_MET'
    assert timing['games']==timing['exact_cash_and_result_matches']
    assert paired['v2_minus_v1_win_rate']<0,'Reassess the final recommendation if V2 wins the paired comparison'
    summaries=[json.loads(p.read_text()) for p in (HERE/'runs').glob('*/SUMMARY.json')]
    assert all(s['games']==s['expected']==s['complete_719'] and not s['errors'] for s in summaries)
    total=sum(s['games'] for s in summaries)
    candidates=len(json.loads((HERE/'CANDIDATES.json').read_text()))
    result=dict(status='EXPERIMENT_COMPLETE_TARGET_NOT_MET',retained_baseline='cf_liq_h12_nointraday',
                rejected_promotion='v2_animal06',main_evaluation_games=total,includes_repeated_controls=True,
                candidate_definitions=candidates,v1_original_holdout=old['groups'],v2_new_holdout=new['groups'],
                paired_external=paired,serial_verification=timing,
                v1_serial_verification=old_timing,
                note='Neither version met 90% on both frozen panels. Keep V1 as the baseline; do not promote the development-only animal-weight improvement.')
    (HERE/'FINAL_REVIEW.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    a,b=old['groups'],new['groups'];lo,hi=paired['seed_bootstrap_delta_95']
    rows=[f"| V1：Cashflow + Liquidity 融合 | {a['internal']['wins']}/1300 = {a['internal']['strict_win_rate']:.2%} | {a['external']['wins']}/1000 = {a['external']['strict_win_rate']:.2%} | 第一批 50 种子 |",
          f"| V2：再加动物权重 0.6 | {b['internal']['wins']}/1300 = {b['internal']['strict_win_rate']:.2%} | {b['external']['wins']}/1000 = {b['external']['strict_win_rate']:.2%} | 全新第二批 50 种子 |"]
    zh=['# 内外融合最终结论','', '**本轮融合、复盘和验收已完成；对内对外平均都达到 90% 的目标未达成。保留 V1 作为基线，V2 不晋级。**','',
        '| 版本 | 内战：原 13 改良版 | 外战：冻结公开 10 入口 | 验收集 |','|---|---:|---:|---|',*rows,'',
        '每个对手 100 局，50 个共享种子交换座位；严格胜率只计胜局。两轮正式验收各 2,300 局，全部 719 步、零运行错误。公开池是 2026-09-23/24 快照。','',
        '## 同种子对照，决定是否晋级','',
        f"另外用第二批同一组种子复跑冻结 V1 的外战：**V1 {paired['v1_wins']}/1000（{paired['v1_wins']/1000:.2%}），V2 {paired['v2_wins']}/1000（{paired['v2_wins']/1000:.2%}）**。",
        f"V2 配对胜率差 {paired['v2_minus_v1_win_rate']*100:+.2f} 个百分点，95% 种子重采样区间 {lo*100:+.2f} 至 {hi*100:+.2f}。这控制了种子、座位和对手源码差异，没有再调参。",'',
        '同种子下，MetaV4 的胜场为 32→32；Master Engine V3 为 46→27。MetaV4 的平均现金差虽然改善，仍没有增加胜场。区间跨过零，因此不能宣称统计上证明 V2 普遍更差，但已经没有支持晋级的证据。','',
        '## 保留下来的改动','',
        'V1 保留 Cashflow 的回款估值和根据公开产出出售的逻辑，加入 Liquidity 的开局小麦买卖，关闭日内新增项目准入/采购，雇工上限设为 12。原 Cashflow C++ 规划器、执行器和原生库保持一致。未使用 ML/RL、对手身份分支、隐藏对手库存或测试种子。',
        'V2 将新增动物候选排序乘数改为 0.6。开发池看似改善，但在同种子独立面板里退化，撤回这项晋级。0.4 更激进的版本甚至改变开局采购并导致大幅退步。参数选择不能代替未见数据检验。','',
        '## 剩余问题','',
        '主要短板仍是 MetaV4 与 Master Engine V3。全局降低动物评分会改变资产组合、工人负担和市场相互作用，不能同时解决所有对手；已有数据不支持将其视为通用修复。下一步应针对失败阶段检查实际回款与规划估值的偏差，并依据当前可见盘面修正经营决策。','',
        '逐日复盘使用当前版本对主要弱点的中位败局，复跑现金与原记录完全一致；不把未卖库存估值当成已经赚到的终局现金。详见 `FAILURE_REVIEW_ZH.md` 与原始诊断 JSON。','',
        '## 文件与边界','',
        '- 建议保留的 V1：交付包 `agent_v1/main.py`；本地 `candidates/cf_liq_h12_nointraday/main.py`。',
        '- 未晋级 V2：交付包 `agent/main.py`；本地 `candidates/v2_animal06/main.py`。',
        '- 双语详细验收：`ACCEPTANCE_ZH.html` / `ACCEPTANCE_EN.html`；配对比较：`PAIRED_EXTERNAL_ZH.md`。',
        f'- 本轮共 {candidates} 个候选定义、{total:,} 局主评测执行（包含重复对照，不代表 {total:,} 份独立样本）。',
        f"- 当前版本 {timing['games']} 局串行核验全部精确复现，观测到的最大单次调用 {timing['max_serial_action_s']:.3f} 秒。官方规则本地核验不等同于 Kaggle 沙箱认证。",
        f"- V2 的最慢串行样例已超过 1 秒，不可声称满足严格单步 1 秒限制；V1 已测串行样例最大 {old_timing['max_serial_action_s']:.3f} 秒。这里只描述测过的样例，不作所有平台和所有局面的耗时保证。",
        '- 源码、原生运行库、官方本地规则、哈希和压缩逐局回执一并打包。对手从已上传的冻结对手包读取，不重复占用大文件空间。','']
    en=['# Final dual-panel fusion review','',
        '**The fusion study, diagnostics and acceptance are complete. The 90% average win-rate target on both panels was not met. Retain V1 as the baseline; do not promote V2.**','',
        '| Version | Internal 13 | External 10 entries | Test panel |','|---|---:|---:|---|',
        f"| V1 | {a['internal']['wins']}/1300 = {a['internal']['strict_win_rate']:.2%} | {a['external']['wins']}/1000 = {a['external']['strict_win_rate']:.2%} | First 50 seeds |",
        f"| V2, animal ranking multiplier 0.6 | {b['internal']['wins']}/1300 = {b['internal']['strict_win_rate']:.2%} | {b['external']['wins']}/1000 = {b['external']['strict_win_rate']:.2%} | New 50 seeds |",'',
        'Each acceptance panel used both seats and 100 games per opponent. Every game completed 719 transitions without execution errors. Draws are not wins. Public entries are the frozen September 23–24, 2026 snapshot.','',
        '## Matched-seed comparison','',
        f"V1 was also replayed on the exact V2 external panel: **V1 {paired['v1_wins']}/1000; V2 {paired['v2_wins']}/1000**. The paired difference is {paired['v2_minus_v1_win_rate']*100:+.2f} percentage points (95% seed-cluster interval {lo*100:+.2f} to {hi*100:+.2f}). Both source versions remained frozen; there was no retuning.",'',
        'Matched MetaV4 wins stayed at 32/100, while Master Engine V3 fell from 46 to 27 wins. An improved mean margin against MetaV4 did not translate into additional wins. The interval crosses zero, so universal statistical inferiority is not established; there is still no evidence supporting promotion.','',
        '## Retained changes','',
        'V1 preserves the Cashflow native planner/executor and public-output sale logic, adds the Liquidity opening wheat buy/sell intent, disables intraday admission/procurement of new projects, and caps hired workers at 12. No ML/RL, hidden rival state, opponent identity branches or test seed is used.',
        'V2 additionally lowered new-animal candidate ranking to 0.6. Its development improvement did not survive the matched independent panel, so this change is not promoted. The 0.4 trial changed the opening purchases and caused a large regression.','',
        '## Remaining bottleneck and handoff','',
        'MetaV4 and Master Engine V3 remain the principal weaknesses. A global animal preference changes capital allocation, worker load and market interaction; it is not a general repair. Investigate where realized cash diverges from plan valuation, and make state-conditioned economic corrections before further promotion.',
        'Use `agent_v1/main.py` for the retained baseline. `agent/main.py` is the rejected V2 comparison candidate. Both sources/runtimes, bilingual reports, per-game evidence and hashes are bundled. Detailed result and reproduction notes: ACCEPTANCE_EN.html / ACCEPTANCE_EN.md. The already-shared frozen opponent bundle is an external dependency.',
        f"There were {candidates} candidate definitions and {total:,} main game executions, including repeated controls. All {timing['games']} current-version serial reruns matched exact terminal cash; maximum observed serial call was {timing['max_serial_action_s']:.3f}s. This local referee/timing verification is not Kaggle sandbox certification.",'']
    en+= [f"V2 exceeded the strict one-second budget in the slowest serial case. V1's previously checked serial maximum was {old_timing['max_serial_action_s']:.3f}s. These are measured cases, not guarantees for all states or deployment platforms.",'']
    for lang,lines in [('ZH',zh),('EN',en)]:
        (HERE/f'FINAL_REPORT_{lang}.md').write_text('\n'.join(lines),encoding='utf-8')
        # A compact self-contained HTML view; Markdown remains the full text source.
        title='内外融合最终结论：未达双 90%' if lang=='ZH' else 'Fusion review: dual 90% target not met'
        intro=zh[2].strip('*') if lang=='ZH' else en[2].strip('*')
        html_rows=''.join(f'<tr><td>{name}</td><td>{g["internal"]["strict_win_rate"]:.2%}</td><td>{g["external"]["strict_win_rate"]:.2%}</td></tr>' for name,g in [('V1',a),('V2',b)])
        paired_text=(f'同种子外战：V1 {paired["v1_wins"]}/1000；V2 {paired["v2_wins"]}/1000。V2 差 {paired["v2_minus_v1_win_rate"]*100:+.2f} 个百分点。' if lang=='ZH' else f'Matched external panel: V1 {paired["v1_wins"]}/1000; V2 {paired["v2_wins"]}/1000. V2 difference {paired["v2_minus_v1_win_rate"]*100:+.2f} percentage points.')
        page='<!doctype html><html lang="'+('zh' if lang=='ZH' else 'en')+'"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+title+'</title><style>body{max-width:980px;margin:44px auto;padding:0 24px;font:17px/1.7 system-ui;color:#182337;background:#f8fafc}table{border-collapse:collapse;width:100%;background:white;margin:24px 0}td,th{text-align:left;padding:16px;border-bottom:1px solid #dae3ee}.note{background:#fff0db;border-left:5px solid #b4691b;padding:20px}a{color:#17578c}</style>'
        seed_note='上表两行使用不同的独立种子集；下面另列同种子对照。' if lang=='ZH' else 'The two rows use different independent seed panels; the matched-seed comparison is shown below.'
        timing_note=(f'V2 串行最慢样例 {timing["max_serial_action_s"]:.3f}s，超过单步 1 秒；V1 已测样例最大 {old_timing["max_serial_action_s"]:.3f}s。均非 Kaggle 沙箱认证。' if lang=='ZH' else f'V2 serial maximum {timing["max_serial_action_s"]:.3f}s exceeded one second. Previously tested V1 maximum: {old_timing["max_serial_action_s"]:.3f}s. Neither is Kaggle sandbox certification.')
        page+=f'<h1>{title}</h1><p class="note">{html.escape(intro)}</p><table><thead><tr><th>Version</th><th>Internal 13</th><th>External 10</th></tr></thead><tbody>{html_rows}</tbody></table><p>{seed_note}</p><p>{paired_text}</p><p>{timing_note}</p>'
        page+=f'<p><a href="FINAL_REPORT_{lang}.md">'+('完整结论、原因和使用说明' if lang=='ZH' else 'Full conclusions, causes and handoff notes')+f'</a> · <a href="ACCEPTANCE_{lang}.html">'+('逐对手验收' if lang=='ZH' else 'Per-opponent acceptance')+'</a></p>'
        page+='<p>V1 baseline: <code>agent_v1/main.py</code><br>V2 comparison: <code>agent/main.py</code></p></html>'
        (HERE/f'FINAL_REPORT_{lang}.html').write_text(page,encoding='utf-8')
    print(json.dumps(dict(status=result['status'],baseline=result['retained_baseline'],main_games=total,candidates=candidates)))


if __name__=='__main__':main()
