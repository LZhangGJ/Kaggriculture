"""Describe what was actually learned; do not retune against final seeds."""
from runtime import *

def behavior(folder):
    with np.load(folder/'decisions.npz') as d:
        active=d['mask'].sum(1)>1
        p=d['probability'][active]
        return dict(records=len(active),effective=int(active.sum()),
            non_keep=int((d['choice']!=0).sum()),
            non_keep_among_effective=float((d['choice'][active]!=0).mean()),
            mean_keep_probability=float(p[:,0].mean()),
            mean_entropy=float(-(p*np.log(np.maximum(p,1e-30))).sum(1).mean()),
            choice_counts=np.bincount(d['choice'],minlength=10).tolist())

def main():
    result=[]
    for arm in ('c3auto','f3','c3j7'):
        baseline=P/'baselines'/f'{arm}_initial_stochastic'
        for rep in (0,1):
            root=P/'training'/f'{arm}_r{rep}'
            final=read(root/'FINAL.json')
            result.append(dict(arm=arm,run=rep,
                first_training=behavior(root/'rollout_001'),
                last_training=behavior(root/'rollout_020'),
                initial_stochastic=behavior(baseline),
                final_stochastic=behavior(root/'final_last_stochastic'),
                final_greedy=behavior(root/'final_last' if (root/'final_last').exists() else root/'final_best'),
                selected_step=final['best_step']))
    save(P/'LEARNING_DIAGNOSTICS.json',result)
    lines=['# 首轮RL实际学到了什么','',
        '本报告只读本轮冻结训练和评测数据，不追加模拟、不调阈值、不做事后Oracle。','',
        '|底座|重复|初始采样非KEEP比例|结束采样非KEEP比例|结束贪心非KEEP次数|结束采样平均KEEP概率|',
        '|---|---:|---:|---:|---:|---:|']
    for r in result:
        a=r['initial_stochastic'];b=r['final_stochastic'];c=r['final_greedy']
        lines.append(f"|{r['arm']}|{r['run']}|{a['non_keep_among_effective']:.2%}|{b['non_keep_among_effective']:.2%}|{c['non_keep']} / {c['effective']}有效选择|{b['mean_keep_probability']:.2%}|")
    lines+=['','比例分母为真正有至少两个可行选项的日级状态。不同模型会走到不同状态，所以概率均值变化仅为行为描述，不能直接归因某类动作正确。','',
        '## 已有证据与不能下的结论','',
        '- PPO梯度、模型权重、概率和实际采样动作都有变化，训练接口不是空转；见G2数值/动作回归。',
        '- 模型初始给KEEP固定logit先验3.5835：十个候选全可选时KEEP约80%；候选更少时KEEP概率更高。贪心只有当另一个候选分数真正超过KEEP才会改变执行。',
        '- 若结束贪心仍全部KEEP，且开发选中step000，明确只能结论为“本轮未学出超过原版的确定性经营调整”；不能称已经产生新拳法。',
        '- 采样策略超过自己的随机初始化，不等于超过原版；必须同时看G3报告中与KEEP的配对胜率/分差。',
        '- 单个新项目/天是本轮人为缩小的动作空间，不是官方动作空间，也不是C3/F3能够表达的全部经营策略。',
        '- 先验较强、单项目调整范围小、训练量和日级终局回报都可能影响结果，但本轮没有独立消融，不能指定其中一项为已证实根因。',
        '- 六次短训练不能证明RL不可行，也不能证明简单扩大网络或训练局数必然成功。后续若继续，应另开实验，分别检验先验/探索、候选实际价值与训练时长；不得依据本轮最终种子继续调参。']
    (P/'LEARNING_DIAGNOSTICS_ZH.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
