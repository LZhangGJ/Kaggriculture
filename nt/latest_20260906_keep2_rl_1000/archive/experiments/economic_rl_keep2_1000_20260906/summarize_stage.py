from pathlib import Path
import sys,json

ROOT=Path(__file__).resolve().parent
end=int(sys.argv[1]);start=end-100;p=ROOT/'stages'/f'round{end:04}'
read=lambda f:json.loads(f.read_text(encoding='utf8'))
assert read(p/'FINAL_AUDIT.json')['status']=='PASS'
r=read(p/'FINAL_RESULTS.json');ss=r['summaries'];chosen=read(p/'FINAL_SELECTION_FROZEN.json')
lines=[f'# KEEP=2：{end}轮阶段汇报','',f'已从{start}轮模型及Adam状态继续到总计{end}轮，每模型累计{end*224:,}局。工程验收PASS。训练与阶段评测用时{read(p/"RUN_COMPLETE.json")["seconds"]/60:.1f}分钟。','',
 '以下为两次训练重复的平均胜率，并非逐局挑模型或组合Agent。固定监测seed与300轮相同，只用于汇报，不用于调参。','',
 f'|方式|决策|{start}轮|{end}轮|变化百分点|','|---|---|---:|---:|---:|']
for arm,title in [('control','普通PPO'),('aux','辅助PPO')]:
    for mode,label in [('greedy','确定性'),('sample','采样')]:
        avg=lambda n:sum(ss[f'{arm}_r{i}_step{n}_{mode}']['overall']['win_rate']for i in (0,1))/2
        a,b=avg(start),avg(end);lines.append(f'|{title}|{label}|{a:.2%}|{b:.2%}|{100*(b-a):+.2f}|')
lines+=['',f'同批KEEP基准：{ss["keep"]["overall"]["win_rate"]:.2%}。','',
 f'|运行|{end}轮确定性|{end}轮采样|开发选定轮数|','|---|---:|---:|---:|']
for name,c in chosen.items():lines.append(f'|{name}|{ss[f"{name}_step{end}_greedy"]["overall"]["win_rate"]:.2%}|{ss[f"{name}_step{end}_sample"]["overall"]["win_rate"]:.2%}|{c["best_step"]}|')
lines+=['','## 配对差异','',f'|运行/方式|{end}减{start}百分点|95% seed区间|','|---|---:|---|']
for name in chosen:
    for mode in ('greedy','sample'):
        c=r['paired'][f'{name}_{end}_minus_{start}_{mode}']['win']
        lines.append(f'|{name}/{mode}|{100*c["mean"]:+.2f}|{[round(100*x,2)for x in c["ci95"]]}|')
lines+=['','所有模型与旧最佳均保留。单个最高分不是预先选定的组合策略；是否稳定变强要结合训练重复、逐对手结果和区间，而不只看均值。1000轮后另做未见种子复验。完整结果见ACCEPTANCE_ZH.md、FINAL_RESULTS.json和FINAL_AUDIT.json。']
(p/'SUMMARY_ZH.md').write_text('\n'.join(lines),encoding='utf8')
print('\n'.join(lines),flush=True)
