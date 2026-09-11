"""No model selection here: summarize preselected policies and paired intervals."""
from runtime import *
import collections
def paired(a,b):
 key=lambda r:(r['seed'],r['opponent'],r['seat'])
 aa={key(r):r for r in a};bb={key(r):r for r in b};assert aa.keys()==bb.keys()
 by=collections.defaultdict(list);rescued=lost=0
 for k,r in aa.items():
  q=bb[k];by[r['seed']].append([int(r['win'])-int(q['win']),r['cash']-q['cash'],r['margin']-q['margin']])
  rescued+=r['win']and not q['win'];lost+=q['win']and not r['win']
 values=np.array([np.mean(by[s],axis=0)for s in sorted(by)])
 rng=np.random.default_rng(314159);bs=values[rng.integers(0,len(values),(10000,len(values)))].mean(1)
 ci=np.percentile(bs,[2.5,97.5],axis=0)
 return dict(win_rate_delta=float(values[:,0].mean()),win_rate_ci=ci[:,0].tolist(),cash_delta=float(values[:,1].mean()),cash_ci=ci[:,1].tolist(),margin_delta=float(values[:,2].mean()),margin_ci=ci[:,2].tolist(),rescued=int(rescued),lost=int(lost))
def fmt(p):return f'{100*p:.2f}%'
def main():
 assert read(P/'G1_RECEIPT.json')['status']=='PASS'and read(P/'G2_RECEIPT.json')['status']=='PASS'
 results=[];performance=[];details={};total_train=total_eval=0
 for arm in ('c3auto','f3','c3j7'):
  keep=read(P/'baselines'/f'{arm}_keep/games.json');rnd=read(P/'baselines'/f'{arm}_random/games.json');initial=read(P/'baselines'/f'{arm}_initial_stochastic/games.json')
  for rep in (0,1):
   folder=P/'training'/f'{arm}_r{rep}';receipt=read(folder/'FINAL.json');assert receipt['status']=='PASS'
   protocol=read(folder/'protocol.json');assert all(digest(f)==h for f,h in protocol['hashes'].items())
   best=read(folder/'final_best/games.json');stoch=read(folder/'final_last_stochastic/games.json')
   c=paired(best,keep);s=paired(stoch,initial)
   row=dict(arm=arm,run=rep,best_step=receipt['best_step'],baseline=read(P/'baselines'/f'{arm}_keep/summary.json')['overall'],best=receipt['best']['overall'],last=receipt['last']['overall'],stochastic=receipt['stochastic']['overall'],vs_keep=c,vs_random=paired(best,rnd),stochastic_vs_initial=s,stochastic_vs_keep=paired(stoch,keep),stochastic_per_opponent=receipt['stochastic']['per_opponent'])
   results.append(row);details[f'{arm}_r{rep}']=receipt['best']['per_opponent'];total_train+=receipt['training_games']
   hist=receipt['history'];rollout=sum(h['rollout']['call_seconds']for h in hist);update=sum(h['update_seconds']for h in hist);compute_only=sum(h['rollout']['wall_seconds']for h in hist)
   effective=sum(h['rollout']['effective_actor_records']for h in hist);count=receipt['training_games']
   perf=dict(arm=arm,run=rep,training_games=count,rollout_seconds=rollout,native_seconds=compute_only,update_seconds=update,
    host_to_device_seconds=sum(h['ppo'].get('host_to_device_seconds',0)for h in hist),
    training_games_per_second=count/(rollout+update),environment_steps_per_second=count*719/(rollout+update),effective_actor_records=effective,effective_actor_per_second=effective/(rollout+update),
    update_percent=100*update/(rollout+update),peak_gpu_mib=max(h['ppo'].get('peak_gpu_allocated_mib',0)for h in hist),
    full_experiment_seconds=receipt['seconds'],max_action_ms=max(h['rollout']['max_action_ms']for h in hist))
   performance.append(perf)
 for f in (P/'baselines').glob('*/games.json'):
  rr=read(f);assert all(not r['error']and r['steps']==719 for r in rr);total_eval+=len(rr)
 for f in (P/'training').glob('*/*/games.json'):
  if f.parent.name.startswith('rollout_'):continue
  rr=read(f);assert all(not r['error']and r['steps']==719 for r in rr);total_eval+=len(rr)
 improved=any(r['vs_keep']['win_rate_ci'][0]>0 or r['stochastic_vs_keep']['win_rate_ci'][0]>0 for r in results)
 result=dict(status='COMPLETE',engineering='PASS',learning='GAIN_REQUIRES_REPLICATION'if improved else'NO_PROVEN_GAIN',training_games=total_train,evaluation_games=total_eval,results=results,performance=performance,per_opponent=details)
 save(P/'FINAL_RESULTS.json',result)
 text=['# 三组经济决策RL首轮结果','',
 '## 简明结论','',
 '实现、训练和性能验证已经完成；本轮没有证实RL带来胜率净提升，不能替换原版。六次训练的开发选择均为step000，结束模型贪心执行也仍全部KEEP。随机采样虽实际改变了经营选择，但没有重复出现可信增益，F3第一轮相对原版反而下降4.64个百分点。'if all(r['best_step']==0 for r in results)and not improved else'请以以下配对结果和两次独立重复判断，不按最终测试集挑选模型。','',
 '因此：学习链路可继续用于后续实验，但尚未学出更强经营策略。先验、候选范围和训练时长可能影响结果；没有独立消融，不能把无提升归罪于某一项，更不能推导RL整体不可行。详细行为统计见LEARNING_DIAGNOSTICS_ZH.md。','',
 '## 验收口径','',
 'C3_AUTO/F3_AUTO均从第0天自主，不调用J7配方；J7_C3保留原C3条件经营参考。三组RL都只学习每日一个新增投资的替换/扩产或当天暂缓，底层动态执行不训练。本实验不是原始动作端到端RL，也不能据此判断完整RL的理论上限。','',
 f'每组两次独立训练，各4480局，合计{total_train:,}训练局。7个固定实时对手，多seed、双座位；开发16seed与最终100seed分离。70,402参数MLP，PPO。每个最终结果1400局。评测记录合计{total_eval:,}局，全部完成719步，无记录到的运行异常。','',
 '## 主要结果：开发集选出的贪心模型','',
 '|底座|训练重复|原版KEEP胜率|训练模型胜率|差值|95%配对区间|开发选中批次|',
 '|---|---:|---:|---:|---:|---|---:|']
 for r in results:
  c=r['vs_keep'];text.append(f"|{r['arm']}|{r['run']}|{fmt(r['baseline']['win_rate'])}|{fmt(r['best']['win_rate'])}|{100*c['win_rate_delta']:+.2f}pp|[{100*c['win_rate_ci'][0]:+.2f}, {100*c['win_rate_ci'][1]:+.2f}]pp|{r['best_step']}|")
 text+=['','批次0表示开发集没有选出超过初始策略的模型，不能称为学会更好的经营。区间以完整seed为重采样块，保留7对手和双座位；是当前固定对手池证据，不是天梯泛化保证。','',
 '## 训练结束随机采样策略 vs 初始随机采样策略','',
 '|底座|重复|结束模型胜率|相对初始化变化|95%区间|','|---|---:|---:|---:|---|']
 for r in results:
  s=r['stochastic_vs_initial'];text.append(f"|{r['arm']}|{r['run']}|{fmt(r['stochastic']['win_rate'])}|{100*s['win_rate_delta']:+.2f}pp|[{100*s['win_rate_ci'][0]:+.2f}, {100*s['win_rate_ci'][1]:+.2f}]pp|")
 text+=['','两种执行方式分开报告，不能把减少随机探索损失解释为超过原版。冻结的随机候选对照、逐对手现金/分差、救回/丢失胜局均见FINAL_RESULTS.json及baselines。','',
 '## 结束采样策略 vs 原版KEEP','',
 '|底座|重复|胜率变化|95%区间|我方现金变化|分差变化|','|---|---:|---:|---|---:|---:|']
 for r in results:
  s=r['stochastic_vs_keep'];text.append(f"|{r['arm']}|{r['run']}|{100*s['win_rate_delta']:+.2f}pp|[{100*s['win_rate_ci'][0]:+.2f}, {100*s['win_rate_ci'][1]:+.2f}]pp|{s['cash_delta']:+.0f}|{s['margin_delta']:+.0f}|")
 text+=['','## 七对手：原版/结束贪心策略','',
 '|对手|C3自主|F3自主|J7+C3|','|---|---:|---:|---:|']
 for name in NAMES:
  rates=[fmt(details[f'{arm}_r0'][name]['win_rate'])for arm in ('c3auto','f3','c3j7')]
  text.append('|'+name+'|'+'|'.join(rates)+'|')
 text+=['','每格200局＝100seed×双座位。两次重复的贪心策略均为KEEP，在本测试集上与原版完全一致，故不重复列两次。采样模型逐对手结果另见JSON。']
 text+=['',
 '## 实际性能','',
 '|底座|重复|训练对局/秒|环境步/秒|有效actor样本/秒|更新耗时占比|GPU张量峰值MiB|','|---|---:|---:|---:|---:|---:|---:|']
 for r in performance:text.append(f"|{r['arm']}|{r['run']}|{r['training_games_per_second']:.1f}|{r['environment_steps_per_second']:.0f}|{r['effective_actor_per_second']:.0f}|{r['update_percent']:.1f}%|{r['peak_gpu_mib']:.1f}|")
 text+=['','训练吞吐包含C++完整双方策略、规则推进、CPU模型推理、跨语言数组返回及PPO更新，不包括编译、保存文件和独立评测。完整实验墙钟另见JSON。GPU张量峰值不是整张卡的系统总占用。没有等价全GPU C3/F3实现，因此不能声称相对全GPU快/慢若干倍。','',
 '各次4480局训练、开发评测和最终评测合计墙钟：'+ '；'.join(f"{r['arm']}/r{r['run']} {r['full_experiment_seconds']:.1f}秒"for r in performance)+'。不含首次编译和前置官方一致性验收。','',
 '计时是本机16线程运行；不是Kaggle容器CPU限时认证，也未生成可提交封装。','',
 '## 解释与限制','',
 '- 同样游戏预算不等于同样有效动作数；C3和F3的候选可用日期不同，已另报有效actor数量。',
 '- 原生规划器仍决定基础组合和布局，RL只做有限的单项目日级调整；不足以验证大规模联合经营/人员调度的学习能力。',
 '- 只允许合法当前信息，未调用事后Oracle、未来seed/事件、对手私有状态或基于对手ID的路由。',
 '- 经济候选能改变目标，不意味着都能当天成交和投产；计划兑现由底层与实际市场共同决定。',
 '- KEEP逐步一致、官方60,396步、线程隔离、概率/价值误差、PPO数值与DEFER修复见G0/G1/G2报告。',
 '- 不根据最终测试集更改权重、阈值或挑选新候选；本轮未提交Kaggle/未推送Git。','',
 '## 复现','',
 'WSL中使用`.venv_wsl_cpp/bin/python build.py`构建。训练/报告使用已安装PyTorch的`/home/mitubant/vllm-env/bin/python`。先checks.py、recheck_features.py、learn_checks.py，再run_all.py。所有脚本路径相对此实验目录。输出目录默认禁止覆盖，重跑必须在新的实验副本/新输出目录中进行，不能直接覆盖已有证据。']
 (P/'G3_FINAL_REPORT_ZH.md').write_text('\n'.join(text)+'\n')
 print(json.dumps(results,indent=2),flush=True)
if __name__=='__main__':main()
