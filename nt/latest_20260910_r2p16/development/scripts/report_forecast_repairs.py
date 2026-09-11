"""Summarize measured tests, preserving the distinction between scan/full/holdout."""
from pathlib import Path
import json,statistics
HERE=Path(__file__).resolve().parent
def main():
    paths=[('交付估值0.5',HERE/'shipment_screen8/half/shipment_half'),
           ('交付估值1',HERE/'shipment_screen8/one/shipment_one'),
           ('销售先验25/75',HERE/'timing_screen8/quarter/timing_quarter'),
           ('销售先验＋交付0.5',HERE/'timing_screen8/quarter_half/timing_quarter_half'),
           ('销售先验＋交付1',HERE/'timing_screen8/quarter_one/timing_quarter_one')]
    lines=['# 预测修正：开发实验验收','',
           '原发布R2仍为保留best，2,200基线1501胜699负（68.23%）。最终独立90%目标尚未达成。','',
           '## 工程检查','',
           '- 754场只读预测审计、542,126动作逐步一致。预测之外的未来只作离线标签。',
           '- R2P3/R2P4开关关闭各8局、5,752动作与原版完全相同。原发布目录不变。',
           '- P3三档各7合成断言；P4四档各151净流守恒样例与2估值断言通过。',
           '- 新策略只改条件估值/预测，不修改官方动作、规则、随机序列，不使用真实对手库存、ID或未来。','',
           '## 五项小筛：同一8个分散种子×双座位×11实时对手','',
           '| 方法 | 胜/176 | 救回 | 丢旧胜 | 平均现金变化 | 平均分差变化 |',
           '|---|---:|---:|---:|---:|---:|','| 原R2 | 118 | — | — | 0 | 0 |']
    payload=[]
    for name,folder in paths:
        if not (folder/'RESULTS.json').exists():continue
        result=json.loads((folder/'RESULTS.json').read_text());o=result['overall'];assert o['errors']==0
        lines.append(f"| {name} | {o['r2_wins']} | {result['rescued']} | {result['lost_wins']} | {result['mean_cash_delta']:+,.1f} | {result['mean_margin_delta']:+,.1f} |")
        payload.append(dict(name=name,folder=str(folder.relative_to(HERE)),result=result))
    lines+=['','这些是开发筛选，不是总体或线上胜率，也不是5次独立证明。相似对手、同seed双座位均相关。',
            '销售先验单项和交付1组合胜率同为140/176，组合分差较高，先扩大组合，不按小样本直接升级。',
            '官方杂草/商店RNG受空地状态影响；同seed改策略可能改变后来商店，胜负转化不能全部归因于局部市场操作。','',
            '## 扩样','']
    full=HERE/'timing_full100/quarter_one/timing_quarter_one/RESULTS.json'
    if full.exists():
        r=json.loads(full.read_text());o=r['overall']
        lines.append(f"组合100开发seed×双座位×11：{o['r2_wins']}/{o['games']}，{o['r2_win_rate']:.2%}；救{r['rescued']}，丢{r['lost_wins']}，平均分差变化{r['mean_margin_delta']:+,.1f}。")
        lines += ['','| 对手 | 新版胜/200 | 新版胜率 |','|---|---:|---:|']
        for opponent,data in r['by_opponent'].items():
            lines.append(f"| {opponent} | {data['r2_wins']} | {data['r2_win_rate']:.1%} |")
        lines+=['','这是开发面板，不是独立未见seed终验。原发布与候选都保留，正式best晋升另记录，不能按本报告默默替换。']
        payload.append(dict(name='full_quarter_one',result=r))
    else:lines.append('quarter_one正在扩大至100开发seed×双座位×11=2,200局；尚无最终结果。')
    solo=HERE/'timing_full100/quarter/timing_quarter/RESULTS.json'
    if solo.exists():
        r=json.loads(solo.read_text());o=r['overall']
        lines+=['',f"销售先验单项也完成2,200局：{o['r2_wins']}胜，{o['r2_win_rate']:.2%}，原版1501胜；救{r['rescued']}、丢{r['lost_wins']}。无平均胜率增益，仍保留原版。"]
        payload.append(dict(name='full_quarter',result=r))
    lines+=['','## 尚未解决','',
            '1. 当前田块已消失但产品仍在隐藏物流中的对手供给，纯田块模型可能漏掉。需要合法、可解释的公开历史估计及误差边界。',
            '2. 交付滞后不是所有产品固定一天，按统一时差估值并没有单独提高胜率；精确交付预测还需结合实时任务与容量。',
            '3. 不能把“符合方向的预测修正”直接称为强度修复，最终以实时独立多seed结果为准。']
    (HERE/'FORECAST_REPAIRS_ACCEPTANCE_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (HERE/'FORECAST_REPAIRS_ACCEPTANCE.json').write_text(json.dumps(payload,indent=2))
    print('Forecast repair report written',flush=True)
if __name__=='__main__':main()
