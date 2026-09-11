"""Paired holdout evidence; no additional policy selection or tuning."""
from common import *
from collections import defaultdict

def paired(a, b):
    key = lambda r: (r['seed'], r['opponent'], r['seat'])
    aa, bb = {key(r): r for r in a}, {key(r): r for r in b}
    assert len(aa) == len(a) and len(bb) == len(b) and aa.keys() == bb.keys()
    by = defaultdict(list)
    rescued = lost = 0
    for k, r in aa.items():
        q = bb[k]
        by[r['seed']].append([int(r['win'])-int(q['win']), r['cash']-q['cash'], r['margin']-q['margin']])
        rescued += r['win'] and not q['win']
        lost += q['win'] and not r['win']
    values = np.array([np.mean(by[s], axis=0) for s in sorted(by)])
    rng = np.random.default_rng(314159)
    bs = values[rng.integers(0, len(values), (10000, len(values)))].mean(1)
    ci = np.percentile(bs, [2.5, 97.5], axis=0)
    return dict(win_delta=float(values[:, 0].mean()), win_ci=ci[:, 0].tolist(),
                cash_delta=float(values[:, 1].mean()), cash_ci=ci[:, 1].tolist(),
                margin_delta=float(values[:, 2].mean()), margin_ci=ci[:, 2].tolist(),
                rescued=int(rescued), lost=int(lost), seed_blocks=len(values))

def behavior(folder):
    with np.load(folder / 'decisions.npz') as d:
        active = d['mask'].sum(1) > 1
        p = d['probability'][active]
        return dict(non_keep=int((d['choice'] != 0).sum()), effective=int(active.sum()),
                    non_keep_rate=float((d['choice'][active] != 0).mean()),
                    mean_keep_probability=float(p[:, 0].mean()),
                    changed=int(d['changed'].sum()))

def main():
    protocol = read(P / 'PROTOCOL.json')
    check_hashes(protocol['hashes'])
    assert read(P / 'RUN_COMPLETE.json')['status'] == 'PASS'
    final = P / 'final'
    groups = {}
    for f in sorted(final.iterdir()):
        rows = read(f / 'games.json')
        assert len(rows) == 1400
        assert all(not r['error'] and r['steps'] == 719 and r['reference_calls'] == 0 for r in rows)
        assert sorted({r['seed'] for r in rows}) == list(range(FINAL_START, FINAL_START + 100))
        groups[f.name] = dict(summary=read(f / 'summary.json'), behavior=behavior(f), rows=rows)
    keep = groups['keep_fixed']['rows']
    keep_old = groups['keep_old']['rows']
    assert all((a['seed'], a['seat'], a['opponent'], a['cash'], a['opponent_cash']) ==
               (b['seed'], b['seat'], b['opponent'], b['cash'], b['opponent_cash']) for a, b in zip(keep, keep_old))
    comparisons, runs, total_games, total_actor = {}, [], 0, 0
    for name, g in groups.items():
        comparisons[f'{name}_vs_KEEP'] = paired(g['rows'], keep)
    for rep in (0, 1):
        def rows(name): return groups[name]['rows']
        for label, a, b in (
            ('new_vs_old_sample', f'new_r{rep}_last_sample', f'old_r{rep}_last_sample'),
            ('inference_fix_only', f'old_r{rep}_fixed_sample', f'old_r{rep}_last_sample'),
            ('retraining_on_fixed', f'new_r{rep}_last_sample', f'old_r{rep}_fixed_sample'),
            ('new_vs_initial_sample', f'new_r{rep}_last_sample', 'initial_sample_fixed')):
            comparisons[f'r{rep}_{label}'] = paired(rows(a), rows(b))
        folder = P / 'training' / f'f3_r{rep}'
        done = read(folder / 'TRAINING_COMPLETE.json')
        assert done['status'] == 'PASS' and done['training_games'] == 4480
        for step in range(1, 21):
            oldgames = read(OLD / 'training' / f'f3_r{rep}' / f'rollout_{step:03}/games.json')
            games = read(folder / f'rollout_{step:03}/games.json')
            keys = lambda rr: [(r['seed'], r['seat'], r['opponent']) for r in rr]
            assert keys(games) == keys(oldgames)
            assert len(games) == 224 and all(not r['error'] and r['steps'] == 719 for r in games)
            total_games += len(games)
        hist = done['history']
        rollout_s = sum(h['rollout']['call_seconds'] for h in hist)
        update_s = sum(h['update_seconds'] for h in hist)
        effective = sum(h['rollout']['effective_actor_records'] for h in hist)
        total_actor += effective
        assert digest(folder / 'step000.bin') == digest(OLD / 'training' / f'f3_r{rep}' / 'step000.bin')
        weight_changed = digest(folder / 'step020.bin') != digest(folder / 'step000.bin')
        assert weight_changed
        runs.append(dict(run=rep, selected_step=done['best_step'], effective_actor_records=effective,
            initialized_identically=True, weights_changed=weight_changed,
            first_training=behavior(folder / 'rollout_001'), last_training=behavior(folder / 'rollout_020'),
            games_per_second=4480 / (rollout_s + update_s),
            environment_steps_per_second=4480 * 719 / (rollout_s + update_s),
            rollout_seconds=rollout_s, update_seconds=update_s,
            update_fraction=update_s / (rollout_s + update_s), training_plus_dev_seconds=done['seconds']))
    primary_gain = all(comparisons[f'new_r{r}_selected_greedy_vs_KEEP']['win_ci'][0] > 0 for r in (0, 1))
    sampled_gain = all(comparisons[f'new_r{r}_last_sample_vs_KEEP']['win_ci'][0] > 0 for r in (0, 1))
    all_keep = all(groups[f'new_r{r}_last_greedy']['behavior']['non_keep'] == 0 for r in (0, 1))
    result = dict(status='COMPLETE', engineering='PASS',
        learning='REPLICATED_GREEDY_GAIN' if primary_gain else 'NO_PROVEN_PRIMARY_GAIN',
        replicated_sampled_gain=sampled_gain, all_final_greedy_keep=all_keep,
        training_games=total_games, effective_actor_records=total_actor,
        final_evaluation_games=sum(len(g['rows']) for g in groups.values()), development_games=1792,
        runs=runs, groups={n:{k:v for k,v in g.items() if k != 'rows'} for n,g in groups.items()}, comparisons=comparisons)
    assert total_games == 8960 and result['final_evaluation_games'] == 21000
    save(P / 'FINAL_RESULTS.json', result)
    rate = lambda name: groups[name]['summary']['overall']['win_rate'] * 100
    cash = lambda name: groups[name]['summary']['overall']['mean_cash']
    margin = lambda name: groups[name]['summary']['overall']['mean_margin']
    ci = lambda p: f"[{p['win_ci'][0]*100:+.2f}, {p['win_ci'][1]*100:+.2f}]pp"
    text = ['# F3账本修复后RL复跑验收 — 2026-09-06', '',
        '## 结论', '',
        '本轮修复后训练在两个重复的主要确定性评测中均出现可信净提升，仍需外部对手和更广独立复验。' if primary_gain else
        '账本修复及两次RL复跑完成，但本预算下仍未证明RL超过原版F3；不晋升、不替换原版。',
        ('两次结束模型贪心推理仍全部KEEP，不能称学出了新的确定性经营策略。' if all_keep else
         '结束贪心策略已出现非KEEP选择；是否有效必须看配对胜负，不能仅依据动作改变判断。'), '',
        '## 控制条件', '',
        '唯一训练改动是F3日级预测账本。原70,402参数MLP、PPO更新函数、奖励、KEEP先验、单项目/日权限和7个实时对手未改变。两次各4,480局；初始权重与旧对应run逐字节一致，训练seed/座位/对手顺序一致。CPU16线程完整对局，RTX3090更新模型。', '',
        '最终100个新seed66000000–66000099，每版本1,400局；开发16seed沿用旧集，只负责预先选择checkpoint。所有训练结束、checkpoint选定后才开始最终评测。未使用旧最终胜率直接作差，旧模型也在本轮新seed重测。', '',
        '## 新seed总体比较', '',
        '|策略|胜率|我方现金|资金差|非KEEP次数|', '|---|---:|---:|---:|---:|']
    names = [('keep_fixed','F3 KEEP'), ('initial_sample_fixed','初始MLP采样')]
    for rep in (0, 1):
        names += [(f'old_r{rep}_last_sample',f'旧RL r{rep} /旧账本/采样'),
                  (f'old_r{rep}_fixed_sample',f'旧RL r{rep} /新账本/采样'),
                  (f'new_r{rep}_last_sample',f'重训RL r{rep} /新账本/采样'),
                  (f'new_r{rep}_selected_greedy',f'重训RL r{rep} /开发选择step{runs[rep]["selected_step"]}/贪心'),
                  (f'new_r{rep}_last_greedy',f'重训RL r{rep} /step20/贪心')]
    for name, label in names:
        text.append(f'|{label}|{rate(name):.2f}%|{cash(name):,.0f}|{margin(name):+,.0f}|{groups[name]["behavior"]["non_keep"]}|')
    text += ['', '## 配对增益：不是不同随机局的裸胜率比较', '',
             '|比较|胜率差|95% seed块区间|救回局/新丢局|资金差变化|', '|---|---:|---|---:|---:|']
    for rep in (0, 1):
        for label, key in (
            ('重训采样 vs KEEP',f'new_r{rep}_last_sample_vs_KEEP'),
            ('重训采样 vs 旧RL旧账本',f'r{rep}_new_vs_old_sample'),
            ('旧权重仅换账本',f'r{rep}_inference_fix_only'),
            ('同新账本，重训 vs 旧权重',f'r{rep}_retraining_on_fixed')):
            d = comparisons[key]
            text.append(f'|r{rep} {label}|{d["win_delta"]*100:+.2f}pp|{ci(d)}|{d["rescued"]}/{d["lost"]}|{d["margin_delta"]:+,.0f}|')
    text += ['', '区间使用10,000次seed块重采样；保留每seed的7对手和双座位相关性。多个次要对比为探索性诊断，未作多重比较校正，不据此挑胜率较好的版本。随机策略在每个seed/座位只有一条固定采样序列，并非精确期望胜率。', '',
             '## 七对手最终胜率', '',
             '|对手|KEEP|旧RL r0采样|旧RL r1采样|重训r0采样|重训r1采样|', '|---|---:|---:|---:|---:|---:|']
    for opp in NAMES:
        cols = ['keep_fixed','old_r0_last_sample','old_r1_last_sample','new_r0_last_sample','new_r1_last_sample']
        rates = [f'{groups[n]["summary"]["per_opponent"][opp]["win_rate"]*100:.1f}%' for n in cols]
        text.append('|' + opp + '|' + '|'.join(rates) + '|')
    text += ['', '每格200局；确定性模型的完整逐对手表保存在FINAL_RESULTS.json，不能把采样结果当作贪心提交结果。', '',
             '## 训练与性能', '', '|重复|有效actor样本|训练局/秒|环境步/秒|PPO耗时占比|开发选中step|',
             '|---|---:|---:|---:|---:|---:|']
    for r in runs:
        text.append(f'|{r["run"]}|{r["effective_actor_records"]:,}|{r["games_per_second"]:.1f}|{r["environment_steps_per_second"]:.0f}|{r["update_fraction"]:.1%}|{r["selected_step"]}|')
    text += ['', '吞吐包含C++双方决策、规则推进、MLP推理、数组返回及PPO更新，不含文件压缩保存与独立评测；不是线上1秒限时认证。完整总墙钟见RUN_COMPLETE.json。', '',
             '## 验收与解释边界', '',
             f'- 新训练{total_games:,}局，开发1,792局，最终比较21,000局，均完整719步且无记录到的运行/账本异常；这些评测复用100个seed，不能宣称22,792个独立随机样本。',
             '- KEEP两库在全部1,400局中双方现金逐局一致。新旧初始化一致；首批C++与PyTorch数值一致；PPO确有梯度/权重变化。原源文件、二进制、checkpoint及已验收修复版哈希未变。',
             '- 账本修复解决了虚假预测流，不保证候选本身有经济价值，也不保证终局奖励可以在20批内教会跨日联合经营。',
             '- 本轮仍只允许日级单个预选项目替换/扩产/暂缓；不是让RL完全自主规划整块农场。',
             '- 候选质量/权限、强KEEP先验和训练预算可能是后续调查方向；本次没有消融，不将任意一项判为已证明的根因。',
             '- 不修改本轮最终种子上的策略，不继续加训直到碰巧变强，不进行事后Oracle、不提交Kaggle、不推Git。', '',
             '## 文件', '',
             '`PROTOCOL.json`冻结输入；`training/f3_r0`和`f3_r1`保存21个checkpoint、全部rollout、开发选择；`final/`保存15个组的逐局与每日模型数据；`FINAL_RESULTS.json`保存胜率、配对区间、行为与计时。运行命令见PLAN_ZH.md。']
    (P / 'ACCEPTANCE_ZH.md').write_text('\n'.join(text) + '\n')
    print('REPORT', result['learning'], 'greedy_all_KEEP', all_keep, flush=True)

if __name__ == '__main__':
    main()
