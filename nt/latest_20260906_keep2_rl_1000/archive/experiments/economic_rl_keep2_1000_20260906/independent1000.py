"""Predeclared final holdout; never used for training or checkpoint selection."""
from pathlib import Path
import sys,json,time,hashlib
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'stages'/'round1000'))
from common1000 import *
from audit100 import check_games,check_records
from report100 import compare


def main():
    assert read(P/'FINAL_AUDIT.json')['status']=='PASS' and (P/'MANIFEST.json').exists()
    check_hashes();configure_torch()
    out=ROOT/'independent1000';out.mkdir(exist_ok=True);chosen=read(P/'FINAL_SELECTION_FROZEN.json')
    configs={'keep':dict(model='',mode=0)}
    original=ROOT.parent/'economic_rl_keep2_300_20260906'
    for name,c in chosen.items():
        configs[f'{name}_selected']=dict(model=c['selected'],mode=1)
        for step,directory in [(300,original),(900,ROOT/'stages'/'round0900'),(1000,P)]:
            for mode,label in [(1,'greedy'),(2,'sample')]:
                configs[f'{name}_step{step}_{label}']=dict(model=str(directory/'training'/name/f'step{step:03}.bin'),mode=mode)
    assert len(configs)==29
    frozen=dict(configs=configs,hashes={c['model']:digest(c['model'])for c in configs.values()if c['model']},seed_start=71100000,seeds=100,sample=9931)
    f=out/'SELECTION_FROZEN.json'
    if f.exists():assert read(f)==frozen
    else:save(f,frozen)
    pool=runtime.make_pool();games={};ss={};checks={}
    for name,c in configs.items():
        dest=out/name
        if not (dest/'summary.json').exists():evaluate(pool,dest,c['model'],c['mode'],71100000,100,9931)
        assert f.stat().st_mtime<=(dest/'games.json').stat().st_mtime
        rows=check_games(dest,71100000,100);checks[name]=check_records(dest,rows)
        prov=read(dest/'provenance.json')
        assert prov['checkpoint']==c['model'] and prov['mode']==c['mode']
        assert prov['seed_start']==71100000 and prov['seeds']==100 and prov['sample']==9931
        assert prov['keep_bonus']==2 and prov['library_sha']==digest(NATIVE/'build/keep2.so')
        if c['model']:assert prov['sha256']==digest(c['model'])
        games[name]=rows;ss[name]=read(dest/'summary.json')
    assert sum(x['games']for x in checks.values())==40600
    paired={}
    for name in chosen:
        for mode in ('greedy','sample'):
            a=games[f'{name}_step1000_{mode}']
            for earlier in (300,900):paired[f'{name}_1000_minus_{earlier}_{mode}']=compare(a,games[f'{name}_step{earlier}_{mode}'])
            paired[f'{name}_1000_minus_keep_{mode}']=compare(a,games['keep'])
        paired[f'{name}_selected_minus_keep']=compare(games[f'{name}_selected'],games['keep'])
    save(out/'RESULTS.json',dict(summaries=ss,paired=paired,configs=configs))
    lines=['# KEEP=2：1000轮连续训练总报告','',
       '四模型均从300轮模型及Adam状态继续到1000轮，未调整经营策略、候选、特征、网络、奖励或优化器设置。每模型累计224,000训练局，四模型累计896,000局，本次新增627,200局。','',
       '## 每100轮固定监测趋势','', '|轮数|普通确定性|普通采样|辅助确定性|辅助采样|KEEP|','|---|---:|---:|---:|---:|---:|']
    for end in range(300,1001,100):
        base=original if end==300 else ROOT/'stages'/f'round{end:04}'
        summaries=read(base/'FINAL_RESULTS.json')['summaries'];vals=[]
        for arm in ('control','aux'):
            for mode in ('greedy','sample'):vals.append(sum(summaries[f'{arm}_r{i}_step{end}_{mode}']['overall']['win_rate']for i in (0,1))/2)
        vals.append(summaries['keep']['overall']['win_rate'])
        lines.append(f'|{end}|'+'|'.join(f'{v:.2%}'for v in vals)+'|')
    lines+=['','监测集固定70300000起100seed，与300轮相同，不是八次独立测试；未用于改参数或选模。以下使用另外71100000起100个未见seed。','',
       '## 最终独立复验','', '|运行|300确定性|900确定性|1000确定性|300采样|900采样|1000采样|开发选择|选定确定性|','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name,c in chosen.items():
        vals=[ss[f'{name}_step{s}_{mode}']['overall']['win_rate']for mode in ('greedy','sample')for s in (300,900,1000)]
        lines.append('|'+name+'|'+'|'.join(f'{v:.2%}'for v in vals)+f'|{c["best_step"]}|{ss[name+"_selected"]["overall"]["win_rate"]:.2%}|')
    lines+=['',f'独立集KEEP胜率：{ss["keep"]["overall"]["win_rate"]:.2%}。每配置1,400局，共40,600局。','',
       '## 1000轮相对300轮与KEEP的配对差异','', '|比较|胜率差百分点|95% seed区间|平均分差增益|救回/损失|','|---|---:|---|---:|---|']
    for key,c in paired.items():
        lines.append(f'|{key}|{100*c["win"]["mean"]:+.2f}|{[round(100*x,2)for x in c["win"]["ci95"]]}|{c["margin"]["mean"]:+.0f}|{c["rescued"]}/{c["damaged"]}|')
    lines+=['','## 1000轮确定性逐对手','', '|对手|KEEP|普通1|普通2|辅助1|辅助2|','|---|---:|---:|---:|---:|---:|']
    for op in NAMES:
        keys=['keep','control_r0_step1000_greedy','control_r1_step1000_greedy','aux_r0_step1000_greedy','aux_r1_step1000_greedy']
        lines.append('|'+op+'|'+'|'.join(f'{ss[k]["per_opponent"][op]["win_rate"]:.2%}'for k in keys)+'|')
    lines+=['','## 解释边界','',
       '均值为两次训练重复的描述性均值，不是逐局Oracle组合。配对区间按seed分组；两次训练重复不足以精确估计训练随机性的总体分布。没有根据终测选择新参数或自动替换基座。所有检查点和历史最佳保留。',
       '工程审计通过不等于竞技能力达标。阶段报告各有独立文件清单；本轮未做新模型完整官方Python逐步复验，未自动提交Kaggle或推送Git。']
    (ROOT/'FINAL_SUMMARY_ZH.md').write_text('\n'.join(lines),encoding='utf8')
    check_hashes()
    save(out/'ACCEPTANCE.json',dict(status='PASS',games=40600,configs=29,seed_start=71100000,seeds=100,selection_before_holdout=True,train_seed_end_by_run=[69615999,69715999],holdout_not_used_for_training_or_selection=True,checks=checks))
    entries={}
    for f in sorted(out.rglob('*')):
        if not f.is_file()or f.name=='MANIFEST.json':continue
        h=hashlib.sha256()
        with f.open('rb')as stream:
            for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
        entries[str(f.relative_to(out))]=dict(bytes=f.stat().st_size,sha256=h.hexdigest())
    save(out/'MANIFEST.json',dict(files=entries,count=len(entries)))
    print('INDEPENDENT1000_PASS',40600,flush=True)

if __name__=='__main__':main()
