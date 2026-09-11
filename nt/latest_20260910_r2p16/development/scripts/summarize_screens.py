"""Report paired seed-level uncertainty and action-dependent shop sequences."""
from collections import defaultdict
from pathlib import Path
import json
import random
import statistics

HERE=Path(__file__).resolve().parent


def main():
    baseline={(r['opponent'],r['seed'],r['opponent_seat']):r for r in json.loads((HERE/'baseline_rows_all11.json').read_text())}
    folders=[p.parent for parent in ['economic_screen8','competitive_screen8','competitive3_100']
             for p in (HERE/parent).glob('*/RESULTS.json')]
    folders += [p.parent for parent in ['shipment_screen8','timing_screen8','timing_full100','startup_screen8','market_screen8','market_screen100','clock_screen8','clock_screen100','horizon_screen8','horizon_screen100','local_sale_screen8','local_sale_screen100','local_sale_combination_screen8','local_sale_combination_screen100','finite_crop_screen8','finite_crop_screen100','crop_clock_screen8','crop_clock_screen100','crop_chain_screen8','crop_chain_screen100','crop_portfolio_screen8','crop_portfolio_screen100','fert_portfolio_screen8','fert_portfolio_screen100','crop_continuation_screen8','crop_continuation_screen100','startup_supply_screen8','startup_supply_screen100']
                for p in (HERE/parent).glob('*/*/RESULTS.json')]
    result={}
    lines=['# R2强化：实际对战消融结果','',
           '每项均冻结一个二进制和配置，使用11个原程序实时对手，不按seed或对手身份选择策略。代码修正只在独立副本，原版R2发布目录未修改。',
           '注意开发小筛不是最终验收；同seed改策略后，官方空地杂草抽样可改变后续商店序列，不能把逐局差额当纯确定性修复效果。',
           '不确定性按seed整组重采样，保持同seed的双座位和不同对手相关性；只有8个seed的区间本身也很不稳定。','',
           '| 开发面板/配置 | 局数 | 原胜率 | 新胜率 | 救回/丢旧胜 | 分差变化 | 商店序列改变 |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for folder in folders:
        rows=json.loads((folder/'rows.json').read_text());groups=defaultdict(list)
        saved=lost=changed=0;delta=[]
        for r in rows:
            b=baseline[(r['opponent'],r['seed'],r['opponent_seat'])]
            saved+=not b['r2_win'] and r['r2_win'];lost+=b['r2_win'] and not r['r2_win']
            changed+=b['shops']!=r['shops'];delta.append(r['r2_margin']-b['r2_margin'])
            groups[r['seed']].append((int(b['r2_win']),int(r['r2_win'])))
        old=statistics.mean(v[0] for g in groups.values() for v in g)
        new=statistics.mean(v[1] for g in groups.values() for v in g)
        seed_deltas=[statistics.mean(y-x for x,y in g) for g in groups.values()]
        rng=random.Random(218314);boot=[]
        for _ in range(10000):boot.append(statistics.mean(rng.choices(seed_deltas,k=len(seed_deltas))))
        boot.sort();interval=[boot[250],boot[9749]]
        name=str(folder.relative_to(HERE))
        result[name]=dict(games=len(rows),seeds=len(groups),old_rate=old,new_rate=new,rescued=saved,lost_wins=lost,
                          shop_changed_games=changed,mean_margin_delta=statistics.mean(delta),
                          paired_seed_bootstrap_delta_interval95=interval)
        lines.append(f"| {name} | {len(rows)} | {old:.2%} | {new:.2%} | {saved}/{lost} | {statistics.mean(delta):+,.1f} | {changed}/{len(rows)} |")
        if len(groups)>=100:
            lines += ['',f"完整开发面板 {name}：胜率变化的seed级95% bootstrap区间 [{interval[0]:+.2%}, {interval[1]:+.2%}]。这不是未见种子验收。",'']
    lines += ['','## 尚未达到最终目标','',
              '最终要求仍是冻结版本在100个未见seed×双座位×11对手中等权平均≥90%；未使用2609130000—2609130099。',
              '新增R2P2中途交付触发器在首轮176局胜负不变，只有小额现金变化，不能宣布上分修复成功。',
              '每场轨迹、配置、原程序哈希与官方核对证据均保留在对应子目录。']
    (HERE/'ABLATION_RESULTS.json').write_text(json.dumps(result,indent=2))
    (HERE/'ABLATION_RESULTS_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
