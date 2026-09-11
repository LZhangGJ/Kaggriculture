"""Candidate value audit, explicitly not an online agent or training run."""
from pathlib import Path
import sys,json,hashlib
from collections import defaultdict
import numpy as np
P=Path(__file__).resolve().parent
def read(f):return json.loads(Path(f).read_text())
def save(f,x):Path(f).write_text(json.dumps(x,indent=2,ensure_ascii=False))
NAMES=['KEEP','WHEAT','CARROT','TOMATO','STRAWBERRY','MELON','GOOSE','COW','SHEEP','DEFER']

def aggregate(records):
    active=[r for r in records if r['candidate_count']>1]
    def mean(key):return float(np.mean([r[key]for r in active]))if active else 0.
    def count(key):return sum(bool(r[key])for r in active)
    alts=[a for r in active for a in r['alternatives']]
    return dict(states=len(records),active_states=len(active),
        positive_states=count('has_positive'),positive_state_rate=mean('has_positive'),
        economic_states=count('has_economic'),economic_state_rate=mean('has_economic'),
        keep_losing_states=sum(not r['keep_win']for r in active),rescuable_states=count('can_rescue'),
        keep_winning_states=sum(r['keep_win']for r in active),vulnerable_states=count('can_lose'),
        nonkeep_options=len(alts),positive_options=sum(a['reward_delta']>1e-8 for a in alts),
        negative_options=sum(a['reward_delta']<-1e-8 for a in alts),
        identical_terminal_options=sum(a['identical_terminal']for a in alts),
        mean_best_margin_gain=mean('best_margin_gain'),median_best_margin_gain=float(np.median([r['best_margin_gain']for r in active])),
        mean_model_margin_gain=mean('model_margin_gain'),mean_model_win_gain=mean('model_win_gain'),
        mean_model_reward_gain=mean('model_reward_gain'),mean_uniform_reward_gain=mean('uniform_reward_gain'),
        mean_uniform_margin_gain=mean('uniform_margin_gain'),
        mean_model_nonkeep_reward_gain=mean('model_nonkeep_reward_gain'),
        mean_uniform_nonkeep_reward_gain=mean('uniform_nonkeep_reward_gain'),
        mean_model_positive_probability=mean('positive_probability'),
        mean_keep_probability=mean('keep_probability'))

def main():
    done=read(P/'RUN_COMPLETE.json');assert done['status']=='PASS'
    protocol=read(P/'PROTOCOL.json')
    for f,h in protocol['hashes'].items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==h,f
    states=read(P/'STATES.json');by=defaultdict(dict)
    for f in sorted((P/'branches').glob('*.json')):
        for row in read(f):
            key=(row['state_id'],row['tail']);assert row['choice']not in by[key]
            by[key][row['choice']]=row
    summaries=[]
    for state in states:
        for tail in (0,2):
            group=by[state['id'],tail];assert set(group)==set(state['available'])
            keep=group[0];p=np.array(state['probability']);assert abs(p.sum()-1)<1e-5
            choices=state['available'];dr=np.array([group[c]['reward']-keep['reward']for c in choices])
            dm=np.array([group[c]['margin']-keep['margin']for c in choices])
            dw=np.array([int(group[c]['win'])-int(keep['win'])for c in choices])
            probs=p[choices];non=np.array(choices)!=0
            pmass=max(float(probs[non].sum()),1e-20)
            best=max(choices,key=lambda c:(group[c]['reward'],group[c]['margin'],-c))
            alternatives=[]
            for c in choices:
                if c==0:continue
                row=group[c];alternatives.append(dict(choice=c,name=NAMES[c],
                    reward_delta=row['reward']-keep['reward'],margin_delta=row['margin']-keep['margin'],
                    cash_delta=row['cash']-keep['cash'],win_delta=int(row['win'])-int(keep['win']),
                    identical_terminal=row['cash']==keep['cash']and row['opponent_cash']==keep['opponent_cash'],
                    model_probability=float(p[c])))
            summaries.append(dict(state_id=state['id'],source=state['source'],cohort=state['cohort'],seed=state['seed'],
                opponent=state['opponent'],seat=state['seat'],day=state['day'],tail=tail,candidate_count=len(choices),
                keep_win=bool(keep['win']),keep_cash=keep['cash'],keep_margin=keep['margin'],best_choice=best,
                has_positive=bool((dr>1e-8).any()),has_economic=bool(((dm>=1000)|(dw>0)).any()),
                can_rescue=not keep['win']and any(r['win']for r in group.values()),
                can_lose=keep['win']and any(not r['win']for r in group.values()),
                best_margin_gain=float(dm.max()),best_reward_gain=float(dr.max()),
                keep_probability=float(p[0]),positive_probability=float(probs[dr>1e-8].sum()),
                model_reward_gain=float(probs@dr),model_margin_gain=float(probs@dm),model_win_gain=float(probs@dw),
                uniform_reward_gain=float(dr.mean()),uniform_margin_gain=float(dm.mean()),
                model_nonkeep_reward_gain=float(probs[non]@dr[non]/pmass)if non.any()else 0.,
                uniform_nonkeep_reward_gain=float(dr[non].mean())if non.any()else 0.,alternatives=alternatives))
    save(P/'STATE_VALUES.json',summaries)
    grouped={}
    for cohort in ('train','new'):
        for tail in (0,2):
            rr=[r for r in summaries if r['cohort']==cohort and r['tail']==tail]
            grouped[f'{cohort}_tail{tail}']=aggregate(rr)
    per_source={}
    for source in protocol['sources']:
        for tail in (0,2):
            per_source[f'{source["id"]}_tail{tail}']=aggregate([r for r in summaries if r['source']==source['id']and r['tail']==tail])
    periods={}
    for tail in (0,2):
        for label,lo,hi in [('early',0,6),('middle',7,18),('late',19,28)]:
            periods[f'{label}_tail{tail}']=aggregate([r for r in summaries if r['cohort']=='new'and r['tail']==tail and lo<=r['day']<=hi])
    opps={}
    for opp in sorted({s['opponent']for s in states}):
        for tail in (0,2):opps[f'{opp}_tail{tail}']=aggregate([r for r in summaries if r['cohort']=='new'and r['tail']==tail and r['opponent']==opp])
    tail_compare={}
    for cohort in ('train','new'):
        pairs=defaultdict(dict)
        for r in summaries:
            if r['cohort']==cohort:pairs[r['state_id']][r['tail']]=r
        rows=[x for x in pairs.values()if x[0]['candidate_count']>1]
        tail_compare[cohort]=dict(active_states=len(rows),positive_both=sum(x[0]['has_positive']and x[2]['has_positive']for x in rows),
            positive_safe_only=sum(x[0]['has_positive']and not x[2]['has_positive']for x in rows),
            positive_rl_only=sum(x[2]['has_positive']and not x[0]['has_positive']for x in rows),
            neither=sum(not x[0]['has_positive']and not x[2]['has_positive']for x in rows))
    species={}
    for tail in (0,2):
        for c in range(1,10):
            aa=[a for r in summaries if r['cohort']=='new'and r['tail']==tail for a in r['alternatives']if a['choice']==c]
            species[f'{NAMES[c]}_tail{tail}']=dict(n=len(aa),positive=sum(a['reward_delta']>1e-8 for a in aa),
                negative=sum(a['reward_delta']<-1e-8 for a in aa),neutral=sum(abs(a['reward_delta'])<=1e-8 for a in aa),
                mean_margin_delta=float(np.mean([a['margin_delta']for a in aa]))if aa else None)
    # Examples are hindsight diagnostics only, not policy templates.
    cases=sorted([r for r in summaries if r['cohort']=='new'and r['tail']==0],key=lambda r:r['best_reward_gain'],reverse=True)
    examples=[]
    for r in cases:
        if r['best_choice']==0:continue
        if any(e['opponent']==r['opponent']for e in examples):continue
        examples.append(r)
        if len(examples)==7:break
    result=dict(status='COMPLETE',engineering='PASS',learning_performed=False,states=len(states),unique_seeds=len({s['seed']for s in states}),
        days=sorted({s['day']for s in states}),branch_games=done['branch_games'],groups=grouped,per_source=per_source,
        new_periods=periods,new_opponents=opps,tail_comparison=tail_compare,new_species=species,examples=examples,
        seconds=done['seconds'],native_games_per_second=done['branch_games']/done['native_seconds'])
    save(P/'RESULTS.json',result)
    text=['# F3 RL 候选价值审计结果','',
        '## 验证范围','',
        f'从112场来源对局抽取{len(states)}个日节点，覆盖0–28天、7个实时对手、双座位和8个不同seed。训练来源4seed；新增来源4seed。每场8个日期，抽样不看胜负。全部可行选项×两种后续，共{done["branch_games"]:,}场完整719步分支对局。', '',
        'KEEP=只在当前节点不做RL修改。安全后续=之后每一天都交给原F3；RL后续=之后每一天仍由同一个冻结MLP采样。两个后续类型使用相同干预前缀；对手均实时响应。', '',
        '## 1. 有没有比KEEP更有价值的候选？','',
        '|数据/后续|有选择节点|存在更优候选|有明显增益候选|KEEP失败可救回|KEEP胜局存在搞输选项|',
        '|---|---:|---:|---:|---:|---:|']
    for key,g in grouped.items():
        text.append(f'|{key}|{g["active_states"]}|{g["positive_states"]} ({g["positive_state_rate"]:.1%})|{g["economic_states"]} ({g["economic_state_rate"]:.1%})|{g["rescuable_states"]}/{g["keep_losing_states"]}|{g["vulnerable_states"]}/{g["keep_winning_states"]}|')
    text+=['','更优以原PPO终局奖励为准：胜负优先，随后小幅资金差塑形。明显增益指救回败局或分差增加至少1000；1000只是报告阈值，不是新规则。同一场不同日期相关，表中节点不能当独立比赛胜率。','',
        '## 2. 好选项和坏选项各有多少？','',
        '|数据/后续|非KEEP候选|有利|有害|双方终盘现金完全相同|事后最佳平均分差增益|',
        '|---|---:|---:|---:|---:|---:|']
    for key,g in grouped.items():
        text.append(f'|{key}|{g["nonkeep_options"]}|{g["positive_options"]}|{g["negative_options"]}|{g["identical_terminal_options"]}|{g["mean_best_margin_gain"]:+,.0f}|')
    text+=['','终盘相同不等于动作未执行，也可能被后续重规划或市场过程抵消。本轮没有将所有中性分支逐原子动作做效果归因，不能全部判为执行器bug。事后最佳只是单次编辑的条件上限，不是可提交胜率。','',
        '## 3. 模型是否把概率给对了？','',
        '|数据/后续|KEEP概率均值|有利候选总概率|按模型概率的平均分差增益|按模型概率的胜率变化|非KEEP模型奖励增益|非KEEP均匀奖励增益|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for key,g in grouped.items():
        text.append(f'|{key}|{g["mean_keep_probability"]:.1%}|{g["mean_model_positive_probability"]:.1%}|{g["mean_model_margin_gain"]:+,.0f}|{g["mean_model_win_gain"]*100:+.2f}pp|{g["mean_model_nonkeep_reward_gain"]:+.4f}|{g["mean_uniform_nonkeep_reward_gain"]:+.4f}|')
    text+=['','这些是同一节点全部候选的实测后果，再按该节点模型概率加权；不是新训练/新在线Agent实测胜率，也没有对未知未来取完整期望。非KEEP列去除了KEEP概率，帮助区分“偏向不改”与“在改动选项里排序较差”。','',
        '## 4. 新seed：早中晚期','',
        '|时期/后续|有选择节点|有更优候选比例|可救回KEEP失败节点|模型平均分差增益|',
        '|---|---:|---:|---:|---:|']
    for key,g in periods.items():
        text.append(f'|{key}|{g["active_states"]}|{g["positive_state_rate"]:.1%}|{g["rescuable_states"]}/{g["keep_losing_states"]}|{g["mean_model_margin_gain"]:+,.0f}|')
    text+=['','early=0–6天；middle=7–18天；late=19–28天。分对手、项目类型与两种后续的交叉表见RESULTS.json；不是只检查6/12/18天。','',
        '## 5. 可靠性与局限','',
        '- 新接口未干预时，与原接口14局/10,066步完整动作一致；强制原选择后仍全程一致；1/16线程一致。',
        f'- 全部分支检查干预前日级状态、候选、概率和动作选择一致；强制原选择后继续RL的{done["identity_suffix_checks"]}条分支，全部后续日级记录与终局也一致。',
        '- 使用真实完整前缀和同一随机状态，避免把不同开局误当成同一局面；对手没有被换成冻结动作回放。',
        '- 该对照使用同一真实未来随机实现，属于离线反事实诊断。线上看不到未来，因此不能直接选本报告事后最佳。',
        '- 每节点每候选每种后续仅一个共同随机实现；8个seed不足以估计精确长期胜率。能够证明一些候选确有价值，不能证明任何候选普遍必胜/必亏，更不能排除跨日组合价值。',
        '- 本次仅评估已经交给MLP的固定单项目候选，不代表完整F3或官方全部行动空间；未新增跨项目联动能力。',
        '- 固定候选、模型、账本和奖励均未改变；没有学习、选新模型、提交Kaggle或推Git。','',
        '## 性能与材料','',
        f'审计总墙钟{done["seconds"]:.1f}秒，分支对局原生吞吐{result["native_games_per_second"]:.1f}局/秒（完整前缀重建+续跑；不含编译）。',
        '`STATES.json`为抽样节点；`baselines/`保存来源状态与概率；`JOB_MANIFEST.json`保存预定候选；`branches/`保存逐局结果；`STATE_VALUES.json`为每节点价值表；`RESULTS.json`为汇总。']
    (P/'ACCEPTANCE_ZH.md').write_text('\n'.join(text)+'\n')
    print(json.dumps(dict(groups=grouped,tail_comparison=tail_compare,performance=result['native_games_per_second']),indent=2))

if __name__=='__main__':main()
