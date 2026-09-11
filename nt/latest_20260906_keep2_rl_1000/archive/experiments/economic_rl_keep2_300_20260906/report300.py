from common300 import *
from report100 import compare


def main():
    assert read(P/'RUN_COMPLETE.json')['status']=='PASS';check_hashes()
    chosen=read(P/'FINAL_SELECTION_FROZEN.json')
    directories=sorted((P/'final').iterdir())
    ss={d.name:read(d/'summary.json')for d in directories}
    rows={d.name:read(d/'games.json')for d in directories};paired={};training={}
    for name in chosen:
        for mode in ('greedy','sample'):
            paired[f'{name}_300_minus_200_{mode}']=compare(rows[f'{name}_step300_{mode}'],rows[f'{name}_step200_{mode}'])
        for suffix in ('selected','step300_greedy','step300_sample'):
            paired[f'{name}_{suffix}_minus_keep']=compare(rows[f'{name}_{suffix}'],rows['keep'])
        receipt=read(P/'training'/name/'COMPLETE.json');hist=receipt['history']
        roll=sum(h['rollout']['call_seconds']for h in hist);update=sum(h['update_seconds']for h in hist)
        training[name]=dict(new_games=22400,total_games=67200,rollout_seconds=roll,update_seconds=update,
            included_training_games_per_second=22400/(roll+update),elapsed_seconds=receipt['seconds'],
            effective_records=sum(h['rollout']['effective_actor_records']for h in hist),
            developments=[dict(step=h['step'],win=h['summary']['overall']['win_rate'],nonkeep=h['summary']['non_keep'])for h in read(P/'training'/name/'selection.json')])
    for rep in (0,1):
        for mode in ('greedy','sample'):
            paired[f'aux_minus_control_r{rep}_{mode}']=compare(rows[f'aux_r{rep}_step300_{mode}'],rows[f'control_r{rep}_step300_{mode}'])
    save(P/'FINAL_RESULTS.json',dict(summaries=ss,paired=paired,training=training,forecasts=read(P/'FORECAST_RESULTS.json')))
    lines=['# KEEP=2：200→300轮续训验收','',
      '四组均从各自200轮模型与优化器状态继续，候选/特征/执行器/奖励不变。每组新增22,400训练局，累计67,200局；终测70300000起100个新seed，每版本7对手×双座位=1400局。','',
      '|运行|200轮确定性|300轮确定性|200轮采样|300轮采样|开发选定轮数|选定版本确定性|',
      '|---|---:|---:|---:|---:|---:|---:|']
    for name,c in chosen.items():
        keys=[f'{name}_step200_greedy',f'{name}_step300_greedy',f'{name}_step200_sample',f'{name}_step300_sample']
        lines.append('|'+name+'|'+'|'.join(f"{ss[k]['overall']['win_rate']:.2%}"for k in keys)+f"|{c['best_step']}|{ss[name+'_selected']['overall']['win_rate']:.2%}|")
    lines+=['',f"本次同seed KEEP胜率：{ss['keep']['overall']['win_rate']:.2%}。上次200轮终测数字来自另一批seed，不能直接和本次300轮相减。",'',
      '## 300轮相对200轮：配对结果','', '|运行/评估方式|胜率差百分点|95% seed区间|分差增益|救回/损失|','|---|---:|---|---:|---|']
    for name in chosen:
        for mode in ('greedy','sample'):
            v=paired[f'{name}_300_minus_200_{mode}']
            lines.append(f"|{name}/{mode}|{100*v['win']['mean']:+.2f}|{[round(100*x,2)for x in v['win']['ci95']]}|{v['margin']['mean']:+.0f}|{v['rescued']}/{v['damaged']}|")
    for title,suffix in (('300轮确定性','step300_greedy'),('开发选定确定性','selected')):
        lines+=['',f'## 逐对手：{title}','', '|对手|KEEP|普通1|普通2|辅助1|辅助2|','|---|---:|---:|---:|---:|---:|']
        for op in NAMES:
            keys=['keep',f'control_r0_{suffix}',f'control_r1_{suffix}',f'aux_r0_{suffix}',f'aux_r1_{suffix}']
            lines.append('|'+op+'|'+'|'.join(f"{ss[k]['per_opponent'][op]['win_rate']:.2%}"for k in keys)+'|')
    lines+=['','## 金额、行为和运行速度','', '|版本|我方现金|分差|非KEEP|有效决策记录|完整局/s|','|---|---:|---:|---:|---:|---:|']
    for name,s in ss.items():
        lines.append(f"|{name}|{s['overall']['mean_cash']:.0f}|{s['overall']['mean_margin']:.0f}|{s['non_keep']}|{s['effective_actor_records']}|{s['games_per_second']:.1f}|")
    lines+=['','## 续训成本','', '|运行|新增局数|累计局数|采样秒|GPU更新秒|训练+更新局/s|新增有效选择|','|---|---:|---:|---:|---:|---:|---:|']
    for name,t in training.items():lines.append(f"|{name}|22400|67200|{t['rollout_seconds']:.1f}|{t['update_seconds']:.1f}|{t['included_training_games_per_second']:.1f}|{t['effective_records']}|")
    lines+=['','## 边界','',
      '从step200续训，不是从开发最佳版本续训。阶段标签、候选数量、KEEP强度都没有修改。恢复Adam moment和计数，不重启学习率或优化器。',
      '200与300轮均在同一批新seed公平评估；选模只用开发集。每组只有两个训练重复，bootstrap按环境seed整组，不是按几十万决策记录伪装独立样本。',
      '没有新模型完整官方Python逐步复验；没有Kaggle提交或Git推送；原基座及前200轮证据保留。']
    (P/'ACCEPTANCE_ZH.md').write_text('\n'.join(lines),encoding='utf8')
    print('FINAL_WIN_RATES',{k:v['overall']['win_rate']for k,v in ss.items()},flush=True)


if __name__=='__main__':main()
