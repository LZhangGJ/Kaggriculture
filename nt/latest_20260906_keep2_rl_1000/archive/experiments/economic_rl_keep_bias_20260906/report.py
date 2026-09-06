from pathlib import Path
import sys,json
import numpy as np
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P.parent/'economic_rl_future_aux_20260906'))
from common import read,save,digest,NAMES,bootstrap

def compare(a,b):
    key=lambda r:(r['seed'],r['opponent'],r['seat'])
    a=sorted(a,key=key);b=sorted(b,key=key);assert [key(r)for r in a]==[key(r)for r in b]
    groups=[r['seed']for r in a]
    return dict(win=bootstrap([int(x['win'])-int(y['win'])for x,y in zip(a,b)],groups),
                margin=bootstrap([x['margin']-y['margin']for x,y in zip(a,b)],groups),
                rescued=sum(x['win']and not y['win']for x,y in zip(a,b)),damaged=sum(y['win']and not x['win']for x,y in zip(a,b)))

def main():
    assert read(P/'RUN_COMPLETE.json')['status']=='PASS'
    protocol=read(P/'PROTOCOL.json');results=read(P/'RESULTS.json');comparisons={}
    base=P/'evaluation';keep=read(base/'keep/games.json');data={k:read(base/k/'games.json')for k in results}
    for key in protocol['models']:
        for bi in range(1,4):comparisons[f'{key}_b{bi}']=compare(data[f'{key}_b{bi}'],data[f'{key}_b0'])
    for key in protocol['models']:
        assert [(r['cash'],r['opponent_cash'])for r in data[f'{key}_b0']]==[(r['cash'],r['opponent_cash'])for r in keep]
    averaged={}
    for bi in range(4):
        rows=sum([data[f'{key}_b{bi}']for key in protocol['models']],[])
        averaged[str(bi)]=dict(win_rate=float(np.mean([r['win']for r in rows])),mean_margin=float(np.mean([r['margin']for r in rows])))
    save(P/'PAIRED_RESULTS.json',dict(comparisons=comparisons,averaged=averaged))
    lines=['# KEEP加分降低测试：只改推理，不重训','',
       '100个全新seed，7个实时C++对手，双座位，每格1,400局；固定四个训练末版模型和四个加分档。完整评测23,800局。', '',
       '|模型|原3.58|降至2|降至1|取消0|','|---|---:|---:|---:|---:|']
    for key in protocol['models']:
        lines.append('|'+key+'|'+'|'.join(f"{results[f'{key}_b{bi}']['overall']['win_rate']:.2%}"for bi in range(4))+'|')
    lines.append('|四模型平均|'+'|'.join(f"{averaged[str(bi)]['win_rate']:.2%}"for bi in range(4))+'|')
    lines+=['','## 实际修改比例','', '|模型|原3.58|降至2|降至1|取消0|','|---|---:|---:|---:|---:|']
    for key in protocol['models']:
        lines.append('|'+key+'|'+'|'.join(f"{results[f'{key}_b{bi}']['active_nonkeep_rate']:.2%}"for bi in range(4))+'|')
    lines+=['','比例只在有其他合法选项的节点上计算；KEEP是交回F3正常经营，不是PASS。','',
       '## 配对差异（相对同权重原加分）','', '|模型/档位|胜率差百分点|95% seed区间|分差增益|救回/损失|','|---|---:|---|---:|---|']
    for key,c in comparisons.items():
        lines.append(f"|{key}|{100*c['win']['mean']:+.2f}|{[round(x*100,2)for x in c['win']['ci95']]}|{c['margin']['mean']:+.0f}|{c['rescued']}/{c['damaged']}|")
    lines+=['','## 逐对手胜率','', '|配置|'+'|'.join(NAMES)+'|','|---|'+'---:|'*7]
    for key,r in results.items():lines.append('|'+key+'|'+'|'.join(f"{r['per_opponent'][o]['win_rate']:.2%}"for o in NAMES)+'|')
    best=max(results,key=lambda k:(results[k]['overall']['win_rate'],results[k]['overall']['mean_margin']))
    lines+=['','## 结论边界','',
      '降低加分只是改变已经训练好的策略在推理时的取舍阈值；不会让它重新学会好坏，也不能把去掉先验后的原始分数解释为无偏经济价值。',
      '同一批测试比较了多个设置，探索性区间未进行多重比较校正，不能挑最高一格直接部署。如果出现可用候选，需要新的独立seed确认。',
      '原加分完整动作与原模块一致，C++/Torch概率一致，1/16线程完整动作一致，原源码和模型未修改。未运行低先验重新训练实验、未提交Kaggle、未推Git。',
      '本批得分最高配置：'+best+'；是否真正优于原基座应看相应配对差异及新的独立验证，而不是只看名次。']
    (P/'ACCEPTANCE_ZH.md').write_text('\n'.join(lines),encoding='utf8')
    save(P/'MANIFEST.json',{str(f.relative_to(P)):digest(f)for f in P.rglob('*')if f.is_file()and f.suffix in ('.py','.cpp','.hpp','.so','.md','.json')and f.name!='MANIFEST.json'})
    print('SUMMARY',averaged,'best',best,flush=True)

if __name__=='__main__':main()
