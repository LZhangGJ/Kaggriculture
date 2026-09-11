from run100_common import *

def compare(a,b):
    key=lambda r:(r['seed'],r['opponent'],r['seat']);a=sorted(a,key=key);b=sorted(b,key=key)
    assert [key(r)for r in a]==[key(r)for r in b];g=[r['seed']for r in a]
    return dict(win=bootstrap([int(x['win'])-int(y['win'])for x,y in zip(a,b)],g),
                margin=bootstrap([x['margin']-y['margin']for x,y in zip(a,b)],g),
                rescued=sum(x['win']and not y['win']for x,y in zip(a,b)),damaged=sum(y['win']and not x['win']for x,y in zip(a,b)))

def main():
    assert read(P/'RUN_COMPLETE.json')['status']=='PASS';check_hashes()
    paths=sorted(d for d in (P/'final').iterdir()if (d/'games.json').exists())
    summary_by={d.name:read(d/'summary.json')for d in paths};rows={d.name:read(d/'games.json')for d in paths}
    chosen=read(P/'FINAL_SELECTION_FROZEN.json');paired={};training={}
    for name in chosen:
        for mode in ('greedy','sample'):
            paired[f'{name}_100_minus_20_{mode}']=compare(rows[f'{name}_step100_{mode}'],rows[f'{name}_step20_{mode}'])
        for suffix in ('selected','step100_greedy','step100_sample'):
            paired[f'{name}_{suffix}_minus_keep']=compare(rows[f'{name}_{suffix}'],rows['keep'])
        receipt=read(P/'training'/name/'COMPLETE.json');hist=receipt['history']
        totalroll=sum(r['rollout']['call_seconds']for r in hist);totalupdate=sum(r['update_seconds']for r in hist)
        training[name]=dict(best_step=receipt['best']['step'],games=receipt['training_games'],rollout_seconds=totalroll,update_seconds=totalupdate,
            games_per_second=22400/(totalroll+totalupdate),environment_steps_per_second=22400*719/(totalroll+totalupdate),
            effective_decisions=sum(r['rollout']['effective_actor_records']for r in hist),
            developments=[dict(step=s['step'],win=s['summary']['overall']['win_rate'],nonkeep=s['summary']['non_keep'])for s in read(P/'training'/name/'selection.json')])
    for rep in (0,1):
        for suffix in ('selected','step100_greedy','step100_sample'):
            paired[f'aux_minus_control_r{rep}_{suffix}']=compare(rows[f'aux_r{rep}_{suffix}'],rows[f'control_r{rep}_{suffix}'])
    save(P/'FINAL_RESULTS.json',dict(status='COMPLETE',summaries=summary_by,paired=paired,training=training,
                                   forecasts=read(P/'FORECAST_RESULTS.json')))
    lines=['# KEEP=2、100轮训练验收','',
       '训练和推理都使用2；同一修账本F3、同一候选空间、两个算法各两次重复。每模型22,400训练局；最终100新seed×7实时对手×双座位，每项1,400局。', '',
       '|模型|20轮greedy|100轮greedy|20轮sample|100轮sample|开发选定轮数|选定greedy|',
       '|---|---:|---:|---:|---:|---:|---:|']
    for name,c in chosen.items():
        keys=[f'{name}_step20_greedy',f'{name}_step100_greedy',f'{name}_step20_sample',f'{name}_step100_sample']
        lines.append('|'+name+'|'+'|'.join(f"{summary_by[k]['overall']['win_rate']:.2%}"for k in keys)+f"|{c['best_step']}|{summary_by[name+'_selected']['overall']['win_rate']:.2%}|")
    lines+=['',f"原F3 KEEP在同一批新seed的胜率：{summary_by['keep']['overall']['win_rate']:.2%}。不能把它与旧seed面板直接当版本强弱比较。",'',
            '## 100轮相对20轮的配对变化','', '|模型/方式|胜率差百分点|95% seed区间|平均分差增益|救回/损失|','|---|---:|---|---:|---|']
    for name in chosen:
        for mode in ('greedy','sample'):
            c=paired[f'{name}_100_minus_20_{mode}']
            lines.append(f"|{name}/{mode}|{c['win']['mean']*100:+.2f}|{[round(v*100,2)for v in c['win']['ci95']]}|{c['margin']['mean']:+.0f}|{c['rescued']}/{c['damaged']}|")
    lines+=['','## 逐对手：开发选定模型','', '|对手|KEEP|普通r0|辅助r0|普通r1|辅助r1|','|---|---:|---:|---:|---:|---:|']
    for op in NAMES:
        keys=['keep','control_r0_selected','aux_r0_selected','control_r1_selected','aux_r1_selected']
        lines.append('|'+op+'|'+'|'.join(f"{summary_by[k]['per_opponent'][op]['win_rate']:.2%}"for k in keys)+'|')
    lines+=['','## 所有评测的行为与金额','', '|版本|现金|分差|非KEEP数|完整局/s|','|---|---:|---:|---:|---:|']
    for key,s in summary_by.items():lines.append(f"|{key}|{s['overall']['mean_cash']:.0f}|{s['overall']['mean_margin']:.0f}|{s['non_keep']}|{s['games_per_second']:.1f}|")
    lines+=['','## 训练成本','', '|运行|采样秒|GPU更新秒|含更新局/s|有效决策记录|','|---|---:|---:|---:|---:|']
    for name,t in training.items():lines.append(f"|{name}|{t['rollout_seconds']:.1f}|{t['update_seconds']:.1f}|{t['games_per_second']:.1f}|{t['effective_decisions']}|")
    lines+=['','## 边界','',
      '20→100轮是同一KEEP=2设置下的训练量对照；本次与旧KEEP=3.58/20轮同时改变先验、预算和种子，不能单独归因给先验。',
      '开发选择完成并冻结后才看最终种子；预测拟合、训练胜率或事后最优都不作为真实对战改善。区间按seed整组重采样，仍只有2次训练重复，不能据此证明对所有训练条件有效/无效。',
      '未改变旧模型与官方规则，未提交Kaggle，未推Git。逐对手和未见seed结果应优先于只挑一个最高数值。']
    (P/'ACCEPTANCE_ZH.md').write_text('\n'.join(lines),encoding='utf8')
    print('FINAL_SUMMARY', {k:dict(win=v['overall']['win_rate'],nonkeep=v['non_keep'])for k,v in summary_by.items()},flush=True)

if __name__=='__main__':main()
