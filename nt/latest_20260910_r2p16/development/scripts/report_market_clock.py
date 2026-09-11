"""Refresh P7/P8 evidence without modifying a promoted binary or test traces."""
from pathlib import Path
import json

HERE=Path(__file__).resolve().parent
CASES={
    '仅修底价库存（P7）':'market_screen100/market_1/market_1',
    '库存与报价一起修（P7）':'market_screen100/market_2/market_2',
    '历史时刻只改短模拟（P8）':'clock_screen8/clock_mpc_market0/clock_mpc_market0',
    '历史时刻与长期估值联动（P8小筛）':'clock_screen8/clock_coupled_market0/clock_coupled_market0',
    '历史时刻与长期估值联动（P8完整开发）':'clock_screen100/clock_coupled_market0/clock_coupled_market0',
}

def main():
    values={};lines=['# R2 P7/P8：修正预测与实际胜率验收','',
        '原R2基线：11个冻结原程序实时对手，各100开发seed×双座位，总计1501胜/2200局（68.23%）。原版仍保留；没有使用最终留出seed，没有提交或推送。','',
        '## 工程与经济结论必须分开','',
        '- P7：官方1元售价区间卖出不增加库存，旧长期估值会多推进。2987整数交易样例对照官方，修正版无差异；22条关闭回归15818动作一致。数学修正成立，但不是模拟器全规则改写，也不是强度保证。',
        '- P7的前后半对手卖出顺序仍是假设，私有库存未知；精确积分只解决条件单边交易，不能称完全预测真实并发市场。',
        '- P8：根据真实已发生的、公开可认证的成交来估计日内卖货时刻。至少两个销售日后才启用，最近3日加默认先验；不能认证时回退。不能拿待售上界冒充对手库存。',
        '- P8只读probe：22场、15818动作相同、208976成交检查和221760历史分布检查无错误。对应生产开关关闭；是否更强看下面实际对局。','',
        '## 同开发面板实际比赛','',
        '|版本|局数|原胜数→新胜数|胜率|救回/丢旧胜|我方现金变化|分差变化|完整局/s|',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name,relative in CASES.items():
        path=HERE/relative/'RESULTS.json'
        if not path.exists():
            lines.append(f'|{name}|进行中|—|—|—|—|—|—|');continue
        x=json.loads(path.read_text());values[name]=x;o=x['overall'];n=o['games']
        lines.append(f"|{name}|{n}|{x['base_wins']}→{o['r2_wins']}|{o['r2_win_rate']:.2%}|{x['rescued']}/{x['lost_wins']}|{x['mean_cash_delta']:+,.1f}|{x['mean_margin_delta']:+,.1f}|{n/x['seconds']:.2f}|")
    lines+=['','## 逐对手（完整开发面板）','',
        '|对手|原R2 /200|P7库存 /200|P7完整 /200|P8联动 /200|','|---|---:|---:|---:|---:|']
    base_rows=json.loads((HERE/'baseline_rows_all11.json').read_text())
    by={r['opponent']:sum(x['r2_win'] for x in base_rows if x['opponent']==r['opponent']) for r in base_rows}
    full=[values.get(n) for n in ['仅修底价库存（P7）','库存与报价一起修（P7）','历史时刻与长期估值联动（P8完整开发）']]
    for opp,w in by.items():
        cells=[str(x['by_opponent'][opp]['r2_wins']) if x else '进行中' for x in full]
        lines.append(f"|{opp}|{w}|{'|'.join(cells)}|")
    validation_path=HERE/'clock_validation32/RESULTS.json'
    if validation_path.exists():
        v=json.loads(validation_path.read_text());values['P8 frozen new32 confirmation']=v
        lines+=['','## 冻结后的32新seed配对确认','',
            '两组各704局：32个之前未用seed×双座位×11个原程序实时对手。不是最终预留的100seed；本次看过结果以后不能再当下一次调参的盲测。','',
            f"原版 {v['original']['r2_wins']}/704（{v['original']['r2_win_rate']:.2%}），P8 {v['clock']['r2_wins']}/704（{v['clock']['r2_win_rate']:.2%}）；救回{v['rescued']}局，丢旧胜{v['lost_wins']}局。",
            f"我方现金变化 {v['mean_cash_delta']:+,.1f}，分差变化 {v['mean_margin_delta']:+,.1f}。",
            '结论：旧开发集的优势未在本批新seed复现，不能晋升。保留候选源码、完整轨迹和旧开发最佳记录，但原R2仍是保留发布版。不能把这32seed的反向结果说成统计证明该思路永远无效。']
    lines+=['','## 解释边界与下一步','',
        '小筛从118升至126并不等于稳定上分。P7已证明初筛有增益、完整100seed却持平或下降；必须报告丢掉的原胜局，不能只列挽回的败局。',
        '同seed修改产业与地块后，官方杂草抽样次数会变化，并可能改变未来商店。这是官方规则本身的状态耦合；事后逐项现金差不是严格排除随机因素的补丁因果效应。',
        '所有新强度测试都是原对手程序对当前真实盘面响应，不是冻结其原Replay。前后案例复盘用官方原动作重演核对现金账，未来只作离线解释。',
        'P7完整开发两档都没有晋升；P8也必须通过新seed确认，未达到目标不能宣称修好。最终目标仍是独立100seed×双座位×11对手，等权平均胜率至少90%。',
        '后续另行冻结短模拟2/3天与维护边际价值的机制对照，见HORIZON_SERVICE_PLAN_ZH.md。不是继续搜索单seed最优路线。']
    (HERE/'MARKET_CLOCK_ACCEPTANCE_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (HERE/'MARKET_CLOCK_ACCEPTANCE.json').write_text(json.dumps(values,indent=2),encoding='utf-8')
    print(json.dumps({k:{'wins':v['overall']['r2_wins'],'games':v['overall']['games']} for k,v in values.items() if 'overall' in v}),flush=True)

if __name__=='__main__':main()
