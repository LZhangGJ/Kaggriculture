from pathlib import Path
import json,gzip,statistics,random
from collections import defaultdict
import run_panel as panel
HERE=Path(__file__).resolve().parent
def main():
    result={};lines=['# P9：短窗口分批出售验收','',
        '原R2仍为保留发布版；本报告把局部规则兑现、完整开发胜率和独立确认分开。不将同seed造成的后续商店变化冒充直接销售收益。','',
        '## 工程检查','',
        '- 7种非饲料/非肥料商品，窗口≤4步、同日、非最后一天；当前融资/准备订单、低现金、仓储压力回退原出售。',
        '- 固定到期步，不能无限延期；0..q数量枚举，官方逐份价格及1元库存规则。',
        '- 168组条件交易/最优分批数量对照官方PASS；719个起始step时间上界和资金/容量/截止保护断言PASS。仅条件顺序成交，不声称预测真实对手同索引订单。',
        '- 关闭新开关22个实时官方局，15,818步联合动作哈希与原R2相同。独立上下文复位验证。',
        '- 模式1无对手窗口供给；模式2以零供给和公开产量/过去成交的供给场景最差相对收益选分批。不是使用对手私有库存。','',
        '## 实际双座位原程序对战','',
        '| 面板 | 模式 | 局数 | 原胜→新胜 | 胜率 | 救回/丢旧胜 | 我方现金变化 | 分差变化 | 完整局/s |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for seeds in (8,100):
        for mode in (1,2):
            name=f'local_sale_{mode}';folder=HERE/f'local_sale_screen{seeds}'/name/name;file=folder/'RESULTS.json';key=f'{seeds}_seeds_mode_{mode}'
            if not file.exists():
                lines.append(f'| {seeds}开发seed | {mode} | 未结束 | — | — | — | — | — | — |');continue
            r=json.loads(file.read_text());o=r['overall'];result[key]=r
            lines.append(f"| {seeds}开发seed | {mode} | {o['games']} | {r['base_wins']}→{o['r2_wins']} | {o['r2_win_rate']:.2%} | {r['rescued']}/{r['lost_wins']} | {r['mean_cash_delta']:+.1f} | {r['mean_margin_delta']:+.1f} | {o['games']/r['seconds']:.2f} |")
    confirmation=HERE/'local_sale_validation32/RESULTS.json'
    if confirmation.exists():
        r=json.loads(confirmation.read_text());result['independent_32_seeds']=r
        lines+=['','## 预先冻结的新32seed确认','',
            '2609122000–2609122031，和开发100seed/P8确认32seed不重叠。两档都报告；这不是最终100新seed验收。',
            '| 版本 | 胜/704 | 胜率 | 救回/丢旧胜 | 我方现金差 | 分差变化 | 配对胜率变化95%区间 |',
            '|---|---:|---:|---:|---:|---:|---|',
            f"| 原R2 | {r['original']['r2_wins']} | {r['original']['r2_win_rate']:.2%} | — | — | — | — |"]
        for name in ('mode1','mode2'):
            x=r[name];o=x['overall'];paired=json.loads((confirmation.parent/f'PAIRED_{name}.json').read_text());groups=defaultdict(list)
            for a in paired:groups[a['seed']].append(int(a['new_win'])-int(a['old_win']))
            values=[statistics.mean(v) for v in groups.values()];rng=random.Random(9122)
            boot=sorted(statistics.mean(rng.choices(values,k=len(values))) for _ in range(10000));ci=[boot[250],boot[9749]]
            x['paired_seed_delta95']=ci
            lines.append(f"| {name} | {o['r2_wins']} | {o['r2_win_rate']:.2%} | {x['rescued']}/{x['lost_wins']} | {x['mean_cash_delta']:+.1f} | {x['mean_margin_delta']:+.1f} | [{ci[0]:+.2%}, {ci[1]:+.2%}] |")
        lines+=['','区间按seed整块重采样，保留同seed对手和座位相关性。正向点估计不等于已证明稳定提高，更不等于90%。']
    lines+=['','## 首次改动的直接证据','',
        '真实共同前缀后，Thomas seed2609110014 seat0：step242的8蛋改为先卖3、消费后step245卖5。双方其余动作、成交量相同；这段实际只多赚2元，但终局分差改善4931元。',
        '同样，MarketSmart seed2609110000的首次改动只多赚1元，终局分差+14563；AuraxReactive相同首次改动也多赚1元，终局却-9468。后续规划和官方随机商店耦合影响很大。',
        '忽略对手的模式1有明确反例：Ahmed seed2609110014 step266–269延迟9奶，我方少50、对手多83。不能认定消费后卖必赚。',
        '每条证据来自真实原程序对战，不是冻结对手反事实；FIRST_SALE_WINDOW_ZH.md区分单位/对手动作/成交量是否一致。','',
        '## 接受标准','',
        '8个seed初筛不能晋升；完整100开发seed结果仍不等于独立最终90%验收。新方案若增加旧败局同时损失更多旧胜局，不替换原版。最终100新seed×双座位×11对手尚未使用。']
    (HERE/'LOCAL_SALE_ACCEPTANCE_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    panel.save(HERE/'LOCAL_SALE_ACCEPTANCE.json',result)
    print(json.dumps({k:dict(games=v['overall']['games'],wins=v['overall']['r2_wins'],rescued=v['rescued'],lost_wins=v['lost_wins']) for k,v in result.items() if 'overall' in v}),flush=True)
if __name__=='__main__':main()
