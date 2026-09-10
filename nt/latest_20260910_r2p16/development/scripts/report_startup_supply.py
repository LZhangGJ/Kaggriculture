"""Read-only experiment analysis. Does not promote or rewrite any policy."""
from pathlib import Path
import json
import run_panel as panel
from validate_crop_clock import paired
HERE=Path(__file__).resolve().parent

def main():
    folders={'original':None,'p12':HERE/'crop_chain_screen100/cropchain1/cropchain1',
             'p16':HERE/'startup_supply_screen100/startupsupply2/startupsupply2'}
    rows={n:json.loads(((p/'rows.json') if p else HERE/'baseline_rows_all11.json').read_text()) for n,p in folders.items()}
    assert all(len(r)==2200 and not any(x['runtime_error'] for x in r) for r in rows.values())
    result={'status':'FULL_DEVELOPMENT_NOT_UNSEEN_ACCEPTANCE','versions':{},'paired':{},'final_holdout_used':False}
    for n,r in rows.items():
        measured=[x['latency']['r2'] for x in r if 'latency' in x and 'r2' in x['latency']]
        result['versions'][n]={'overall':panel.summarize(r),
            'by_opponent':{o:panel.summarize([x for x in r if x['opponent']==o]) for o in sorted({x['opponent'] for x in r})},
            'latency_measured_games':len(measured),
            'maximum_observed_own_call_s':max((x['maximum'] for x in measured),default=None),
            'over_1s_calls':sum(x['over_1s'] for x in measured) if measured else None}
    for a in ('original','p12'):
        stats,data=paired(rows[a],rows['p16']);result['paired'][a+'_to_p16']=stats
        panel.save(HERE/f'candidate_r2p16/PAIRED_{a}_FULL.json',data)
    panel.save(HERE/'candidate_r2p16/FULL_DEVELOPMENT.json',result)
    lines=['# P16全量开发验证','',
           '100个开发seed×双座位×11个实时原程序对手；各2200局。不是最终未见100seed。','',
           '|版本|胜/2200|胜率|平均现金|平均分差|','|---|---:|---:|---:|---:|']
    for n,v in result['versions'].items():
        o=v['overall'];lines.append(f"|{n}|{o['r2_wins']}|{o['r2_win_rate']:.2%}|{o['r2_mean_cash']:.2f}|{o['mean_margin']:.2f}|")
    lines+=['','|配对|救回/丢旧胜|现金差|分差差|胜率变化95%区间|','|---|---:|---:|---:|---:|']
    for n,v in result['paired'].items():
        a,b=v['paired_seed_win_delta95'];lines.append(f"|{n}|{v['rescued']}/{v['lost_wins']}|{v['mean_cash_delta']:+.2f}|{v['mean_margin_delta']:+.2f}|[{a:+.2%},{b:+.2%}]|")
    lines+=['','|对手|原R2胜/200|P12胜/200|P16胜/200|','|---|---:|---:|---:|']
    for o in result['versions']['p16']['by_opponent']:
        vals=[result['versions'][n]['by_opponent'][o]['r2_wins'] for n in folders];lines.append(f"|{o}|"+'|'.join(map(str,vals))+'|')
    lines+=['','## 机制与限制','',
            '只在第0步用同一自主规划器构造假想对手的未来投资供给。实际对手可能不是这样投资；本先验不是已知未来。下一天清掉，使用真实公开盘面。',
            'MPC复制/重新配置后仍保留先验，并改变候选估值；单测已验证。首个检查样例的起手订单与P12相同，但step10放置牛/羊次序不同，因此不能称为已证明大幅改进了宏观产业选择。',
            '收益统计含后续连锁效果。同seed下，不同空地影响官方杂草随机数消耗，可能改变商店；不可把全部终局收益归为本地一步净改善。',
            '区间以seed为整块重采样，保留座位与相近公开家族相关性。多次开发筛选存在选择偏差；不替换原发布版，不宣布90%。']
    v=result['versions']['p16'];lines+=['',f"本机测得最慢单次调用{v['maximum_observed_own_call_s']:.6f}秒，超过1秒{v['over_1s_calls']}次；不是官方线上硬件保证。"]
    (HERE/'candidate_r2p16/FULL_DEVELOPMENT_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({n:v['overall']['r2_wins'] for n,v in result['versions'].items()}),flush=True)
if __name__=='__main__':main()
