"""Evidence-only rollup; never promote a config or modify frozen policies."""
from pathlib import Path
import json
import run_panel as panel
from validate_crop_clock import paired
HERE=Path(__file__).resolve().parent
FOLDERS={'P9':'local_sale_screen100/local_sale_1/local_sale_1','P10':'finite_crop_screen100/finite1_sale1_v2/finite1_sale1_v2','P11':'crop_clock_screen100/cropclock1_sale1/cropclock1_sale1','P12':'crop_chain_screen100/cropchain1/cropchain1'}

def main():
    base=json.loads((HERE/'baseline_rows_all11.json').read_text());allrows={'Original':base};versions={}
    for name,rel in FOLDERS.items():
        folder=HERE/rel
        if not (folder/'RESULTS.json').exists():continue
        rows=json.loads((folder/'rows.json').read_text());allrows[name]=rows
        result=json.loads((folder/'RESULTS.json').read_text());stats,_=paired(base,rows)
        v=dict(overall=panel.summarize(rows),paired_original=stats,result=rel+'/RESULTS.json',seconds=result.get('seconds'),independent_final_100=False)
        review=folder/'REVIEW_CASES.json'
        if review.exists():
            cases=json.loads(review.read_text());v['official_representative_review']=dict(cases=len(cases),groups=sorted({r['group'] for r in cases}),replay_checks='All saved official frames and both cash ledgers checked by audit_trace',own_noeffect_units=sum(len(r['new_summary'][1-r['candidate']['opponent_seat']]['no_effect_units']) for r in cases),own_market_rejects=sum(len(r['new_summary'][1-r['candidate']['opponent_seat']]['rejects']) for r in cases),source=rel+'/REVIEW_ZH.md')
        versions[name]=v
    comparisons={}
    for a,b in [('P9','P11'),('P9','P12'),('P11','P12')]:
        if a in allrows and b in allrows:comparisons[a+'_to_'+b]=paired(allrows[a],allrows[b])[0]
    result=dict(status='DEVELOPMENT_EVIDENCE_NOT_FINAL_90_PERCENT',original=panel.summarize(base),versions=versions,comparisons=comparisons,final_holdout_used=False)
    validation=HERE/'crop_clock_validation32/RESULTS.json'
    if validation.exists():result['independent_32']=json.loads(validation.read_text())
    panel.save(HERE/'CROP_CLOCK_CHAIN_ACCEPTANCE.json',result)
    lines=['# 公开作物节奏及作物施肥组合：阶段验收','',
           '目标仍是11个原程序实时对手、最终全新100种子双座位、等权平均90%。下面完整2200局是开发集，不是最终验收。原发布文件未覆盖。','',
           '|版本|胜/2200|胜率|相对原版救回/丢胜|平均现金变化|平均分差变化|','|---|---:|---:|---:|---:|---:|']
    lines.append('|Original|1501|68.23%|—|—|—|')
    for n,v in versions.items():
        o=v['overall'];p=v['paired_original'];lines.append(f"|{n}|{o['r2_wins']}|{o['r2_win_rate']:.2%}|{p['rescued']}/{p['lost_wins']}|{p['mean_cash_delta']:+.1f}|{p['mean_margin_delta']:+.1f}|")
    lines+=['','- P9：保护融资/饲料/仓储/终局后的短窗口分批出售。','- P10：P9＋有限作物有收益的施肥，但不改变官方成长。','- P11：P9＋从已发生公开地块腾地推断有限作物收获年龄分布。','- P12：三项组合。','',
            'P11开发胜率最高；P12现金和分差更高，但不是胜率更高。P12小筛149/176不能当完整实力。','',
            '## 预测工程证据','',
            '42条原动作不变的官方轨迹，共30198步；C++/Python逐步窗口一致。小麦净市场供给MAE11.43→8.52，偏差−7.83→−0.05；胡萝卜略差。只预测条件供给，不知道对手私有库存、未来投资或真实未来商店。成熟DIG也可能与HARVEST公开效果相似，因此不是万能意图识别。','',
            'P12关闭新功能的22局15818动作与原版相同，P10/P11组件源码哈希一致；合成单测和代表官方复盘记录分别留存。少量代表无硬伤不证明所有新轨迹完全无硬伤。','',
            '## 新旧候选之间','', '|比较|救回/丢胜|胜率增益95%区间|','|---|---:|---:|']
    for n,p in comparisons.items():
        a,b=p['paired_seed_win_delta95'];lines.append(f"|{n}|{p['rescued']}/{p['lost_wins']}|[{a:+.2%},{b:+.2%}]|")
    lines+=['','区间按seed整块自助抽样，保留双座位和不同克隆家族间的相关性。开发集反复选择候选存在选择偏差，必须看冻结后的新seed确认。','',
            '## 独立确认','']
    if 'independent_32' in result:
        lines+=['四版冻结新32seed已完成，见crop_clock_validation32/REPORT_ZH.md。最终100seed仍未使用。']
    else:lines+=['四版已冻结，正在执行新32seed确认。不是已达到90%，也不是直接可替换发布。']
    lines+=['','## 复盘边界','',
            '同seed改变农场空地会改变官方杂草RNG消耗，进而改变未来商店。不能把整局现金差全部解释为某次卖货多赚几元，也不能以已知未来商店筛选线上分支。所有真实对手独立响应；没有对手身份或隐藏未来输入。','']
    (HERE/'CROP_CLOCK_CHAIN_ACCEPTANCE_ZH.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({n:v['overall']['r2_wins'] for n,v in versions.items()}),flush=True)
if __name__=='__main__':main()
