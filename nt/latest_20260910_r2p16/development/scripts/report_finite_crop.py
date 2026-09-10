from pathlib import Path
import json
HERE=Path(__file__).resolve().parent

def main():
    lines=['# P10：有限作物施肥及分批出售组合验收','',
           '结果：没有超过P9开发最佳，暂不晋升。不是官方硬规则失败；经济组合的小样本优势未在完整开发面板维持。',
           '11712组官方跨日单作物规则校验通过；关闭开关22条实时局15818动作与原版完全一致。单作物收益判据不等于全农场资金与调度可行性保证。','',
           '|版本|样本|胜率|救回/丢旧胜|平均自身现金变化|平均分差变化|', '|---|---:|---:|---:|---:|---:|']
    results={}
    for seeds in (8,100):
        for sale in (0,1):
            name=f'finite1_sale{sale}_v2';folder=HERE/f'finite_crop_screen{seeds}'/name/name
            if not (folder/'RESULTS.json').exists():continue
            r=json.loads((folder/'RESULTS.json').read_text());results[f'{seeds}/{sale}']=r
            lines.append(f"|施肥+分批出售{sale}|{r['overall']['games']}|{r['overall']['r2_win_rate']:.2%}|{r['rescued']}/{r['lost_wins']}|{r['mean_cash_delta']:+.1f}|{r['mean_margin_delta']:+.1f}|")
            if (folder/'REVIEW_CASES.json').exists():
                reviews=json.loads((folder/'REVIEW_CASES.json').read_text())
                noops=sum(sum(v['new_summary'][1-v['candidate']['opponent_seat']]['no_effect_units'].values()) for v in reviews)
                rejects=sum(len(v['new_summary'][1-v['candidate']['opponent_seat']]['rejects']) for v in reviews)
                results[f'{seeds}/{sale}']['representative_audit']=dict(cases=len(reviews),no_effect_unit_actions=noops,market_rejects=rejects)
    lines+=['','## 解释','',
            '- 施肥单项8seed仅98/176；组合146/176，看似超过原118和P9单项138，但完整100seed组合只有1646/2200，低于P9的1666。不能用初筛83%宣称新最好。',
            '- 组合相对原R2救324、丢179，现金+2615，分差+1294；净胜145局，但相对P9仍少20胜。保留全部胜负转换和原版，不强行升级。',
            '- 各8例代表覆盖救回、丢旧胜、仍输、仍赢，官方逐步重演与双边现金账核对。可能出现市场雇工未成交，不因此误称全部事务成功；无效单位动作逐例另列。',
            '- 同seed并不保持相同未来商店：经营改动影响空地杂草随机调用，因此终局差不能全部归因于施肥或销售价差。',
            '- 未进行该候选独立留出；最终100seed仍未用。当前发布版不变，开发最佳仍P9模式1，且其32新seed增益置信区间跨0。','']
    (HERE/'FINITE_CROP_ACCEPTANCE_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (HERE/'FINITE_CROP_ACCEPTANCE.json').write_text(json.dumps(results,indent=2))
    print('Saved finite-crop report; development best unchanged.')
if __name__=='__main__':main()
