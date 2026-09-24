"""Reproduce the median losing game against each major remaining weakness."""
from pathlib import Path
import json
import subprocess
import sys

HERE=Path(__file__).resolve().parent


def main():
    accepted=json.loads((HERE/'ACCEPTANCE.json').read_text())
    frozen=json.loads((HERE/'FINAL_FREEZE.json').read_text())
    rows=[json.loads(s) for s in (HERE/'runs'/frozen.get('run','holdout_v1')/'games.jsonl').read_text().splitlines() if s]
    names=['internal/r14_cashflow','external/n69_hosen42','external/d24_02_guruprasaathas111']
    cases=[]
    for name in names:
        losses=sorted([r for r in rows if r['public']==name and r['margin']<0],key=lambda r:r['margin'])
        if not losses:continue
        selected=losses[len(losses)//2]
        output='heldout_'+name.split('/')[-1]+'_median_loss.json'
        subprocess.run([sys.executable,str(HERE/'trace.py'),'--candidate',accepted['candidate'],
                        '--opponent',name,'--seed',str(selected['seed']),'--seat',str(selected['local_seat']),
                        '--out',output],check=True)
        trace=json.loads((HERE/'diagnostics'/output).read_text())
        assert (trace['own_cash'],trace['rival_cash'])==(selected['local_cash'],selected['public_cash'])
        checkpoints=[]
        for day in (5,10,15,20,25,29):
            row=next(x for x in trace['daily'] if x['step']==day*24)
            own,rival=row['own'],row['rival']
            checkpoints.append(dict(day_zero_based=day,own_cash=own['cash'],rival_cash=rival['cash'],
                                    margin=own['cash']-rival['cash'],own_crops=own['crops'],rival_crops=rival['crops'],
                                    own_animals=own['animals'],rival_animals=rival['animals']))
        cases.append(dict(opponent=name,seed=selected['seed'],seat=selected['local_seat'],
                          selection='median cash margin among losing holdout games',
                          own_terminal_cash=trace['own_cash'],rival_terminal_cash=trace['rival_cash'],
                          final_margin=selected['margin'],exact_reproduction=True,checkpoints=checkpoints,trace=output))
    (HERE/'FAILURE_REVIEW.json').write_text(json.dumps(cases,ensure_ascii=False,indent=2)+'\n')
    lines=['# 融合版剩余败局复盘','',
           '正式验收后，只诊断不修改已冻结 Agent。对主要弱点分别选取败局终局分差的中位数场次；不是挑最差极端案例，也不代表所有败局都有相同原因。下表日期沿用 observation 的 day = step // 24（0 开始）。现金不包含未售库存和待收产出。','']
    for case in cases:
        name=accepted['display_names'].get(case['opponent'],case['opponent'])
        lines += [f'## {name}','',f"种子 {case['seed']}，座位 {case['seat']}。终局 {case['own_terminal_cash']:,} 对 {case['rival_terminal_cash']:,}，差 {case['final_margin']:+,}；与原回执完全一致。",'',
                  '| day（0 起） | 本方现金 | 对手现金 | 分差 |','|---|---:|---:|---:|']
        for point in case['checkpoints']:
            lines.append(f"| {point['day_zero_based']} | {point['own_cash']:,} | {point['rival_cash']:,} | {point['margin']:+,} |")
        early=case['checkpoints'][0]
        lines += ['',f"day 5 本方作物：`{json.dumps(early['own_crops'])}`；对手：`{json.dumps(early['rival_crops'])}`。完整动物、作物、库存、公开市场与前两天动作见 `diagnostics/{case['trace']}`。",'']
    lines += ['## 判断边界','',
              '逐日现金能定位缺口出现的阶段，但不能单独证明是扩地、资产配置、收运、售价或雇工造成。投资更积极也会暂时压低现金；需要结合对应盘面或受控拆分才能归因。',
              '本轮配对拆分已支持：保留 Cashflow 的回款估值/出售层、加入开局市场操作、关闭日内新增项目。最后六天终局搜索被实际胜负淘汰；不把事后库存估值加回终局现金冒充胜利。',
              '若继续迭代，优先从这些败局的前中期资产组合和可兑现回款差距查起，再检验晚期售出和作物更换。仍只依据当前可见盘面生成规则；这批种子此后不再是未见验收集。','']
    (HERE/'FAILURE_REVIEW_ZH.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps([dict(opponent=c['opponent'],final_margin=c['final_margin'],day15_margin=c['checkpoints'][2]['margin']) for c in cases]))


if __name__=='__main__':main()
