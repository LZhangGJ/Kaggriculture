from common import *

def paired(a,b):
    key=lambda r:(r['seed'],r['opponent'],r['seat'])
    a=sorted(a,key=key);b=sorted(b,key=key);assert [key(r)for r in a]==[key(r)for r in b]
    seeds=[r['seed']for r in a]
    return dict(win=bootstrap([int(x['win'])-int(y['win'])for x,y in zip(a,b)],seeds),
                margin=bootstrap([x['margin']-y['margin']for x,y in zip(a,b)],seeds),
                rescued=sum(x['win']and not y['win']for x,y in zip(a,b)),
                damaged=sum(y['win']and not x['win']for x,y in zip(a,b)))

def main():
    assert read(P/'RUN_COMPLETE.json')['status']=='PASS'
    base=P/'final';dirs=sorted(d for d in base.iterdir()if (d/'games.json').exists())
    summaries={d.name:read(d/'summary.json') for d in dirs};games={d.name:read(d/'games.json')for d in dirs}
    comparisons={}
    for rep in (0,1):
        for label in ('selected','last_greedy','last_sample'):
            a=f'aux_r{rep}_{label}';b=f'control_r{rep}_{label}'
            comparisons[f'aux_minus_control_r{rep}_{label}']=paired(games[a],games[b])
    versus_keep={n:paired(rows,games['keep']) for n,rows in games.items()if n!='keep'}
    # Average paired differences across two training repeats inside each seed;
    # this interval describes environment seeds, not broad training-seed uncertainty.
    combined={}
    for label in ('selected','last_greedy','last_sample'):
        a=games[f'aux_r0_{label}']+games[f'aux_r1_{label}'];b=games[f'control_r0_{label}']+games[f'control_r1_{label}']
        combined[label]=paired(a,b)
    totals={}
    for key in ('control_r0','aux_r0','control_r1','aux_r1'):
        hist=read(P/'training'/key/'COMPLETE.json')['history']
        ts=sum(h['update_seconds']for h in hist);rs=sum(h['rollout']['call_seconds']for h in hist)
        totals[key]=dict(training_seconds=ts+rs,update_seconds=ts,rollout_seconds=rs,
             full_games_per_second=4480/(ts+rs),environment_steps_per_second=4480*719/(ts+rs),
             mean_aux_loss_last=hist[-1]['ppo']['aux_loss'],max_gpu_mib=max(h['ppo']['peak_gpu_allocated_mib']for h in hist))
    result=dict(status='COMPLETE',summaries=summaries,paired_comparisons=comparisons,versus_keep=versus_keep,
                mean_repeats=combined,training_cost=totals,forecast=read(P/'FORECAST_RESULTS.json'))
    save(P/'FINAL_RESULTS.json',result)
    lines=['# G3 新种子完整对战验收','',
      '100个未用于训练或选模型的seed × 7个实时C++对手 × 双座位，每版本1,400局。只有开发集选择checkpoint，最终结果不回流改模型。', '',
      '|版本|胜率|平均现金|平均分差|非KEEP次数|完整对局/s|','|---|---:|---:|---:|---:|---:|']
    for n,s in summaries.items():
        o=s['overall'];lines.append(f"|{n}|{o['win_rate']:.2%}|{o['mean_cash']:.0f}|{o['mean_margin']:.0f}|{s['non_keep']}|{s['games_per_second']:.1f}|")
    lines+=['','## 主比较：开发集选定 greedy，辅助组 − 普通PPO','']
    for rep in (0,1):
        c=comparisons[f'aux_minus_control_r{rep}_selected']
        lines.append(f"- 重复{rep}：胜率差 {c['win']['mean']*100:+.2f}个百分点；95% seed区间 {[round(v*100,2)for v in c['win']['ci95']]}；平均分差增益 {c['margin']['mean']:+.0f}；救回 {c['rescued']}，损失 {c['damaged']}。")
    c=combined['selected'];lines.append(f"- 两次重复平均胜率差 {c['win']['mean']*100:+.2f}个百分点，95% seed区间 {[round(v*100,2)for v in c['win']['ci95']]}。区间只描述环境seed，不替代更多训练重复。")
    lines+=['','## 逐对手胜率（开发选定模型）','','|对手|F3 KEEP|普通r0|辅助r0|普通r1|辅助r1|','|---|---:|---:|---:|---:|---:|']
    for opponent in NAMES:
        keys=['keep','control_r0_selected','aux_r0_selected','control_r1_selected','aux_r1_selected']
        lines.append('|'+opponent+'|'+'|'.join(f"{summaries[k]['per_opponent'][opponent]['win_rate']:.2%}"for k in keys)+'|')
    lines+=['','## 训练成本','','|运行|C++采样秒|GPU更新秒|含更新完整局/s|','|---|---:|---:|---:|']
    for n,s in totals.items():lines.append(f"|{n}|{s['rollout_seconds']:.1f}|{s['update_seconds']:.1f}|{s['full_games_per_second']:.1f}|")
    if c['win']['ci95'][0]>0:
        verdict='本固定预算下辅助组对普通PPO主比较出现正向证据；仍需同时看是否超过KEEP、逐对手退化及更多训练重复，不能宣称普遍有效。'
    elif c['win']['ci95'][1]<0:
        verdict='本固定预算下辅助组真实对战更差，不采用。不能由此证明所有辅助目标或训练预算均无效。'
    else:
        verdict='本固定预算下未证明辅助预测提升真实对战，暂不作为必要训练环节。不能据此断言更长训练或其他辅助目标永远无效。'
    lines+=['','## 结论','',verdict,'',
       '预测误差见 FORECAST_RESULTS.json：未来头预测状态变化，分别与“状态保持不变”对照。即使预测更准，也不能替代真实胜率。',
       '只测试修账本F3、现有日级单项目候选和这套辅助损失，未扩展跨日候选组合；没有把事后分支选择作为线上策略。']
    (P/'G3_ACCEPTANCE_ZH.md').write_text('\n'.join(lines),encoding='utf8')
    print('FINAL',verdict,'main',combined['selected'],flush=True)

if __name__=='__main__':main()
