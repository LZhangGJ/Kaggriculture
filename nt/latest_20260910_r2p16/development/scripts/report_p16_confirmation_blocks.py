"""Fair common-seed comparison; no policy selection or deployment side effects."""
from pathlib import Path
import json
import run_panel as panel
from validate_startup_supply import BINS
from validate_crop_clock import paired
HERE=Path(__file__).resolve().parent
BLOCKS=('startup_supply_validation32','startup_supply_validation32_b2')
NAMES=('original','p12','p16')

def summarize_set(rows):
    result=dict(versions={},paired={})
    keys=[{(r['seed'],r['opponent'],r['opponent_seat']) for r in rows[n]} for n in NAMES]
    assert all(k==keys[0] and len(k)==len(rows[NAMES[0]]) for k in keys)
    for n,r in rows.items():
        assert not any(x['runtime_error'] for x in r)
        result['versions'][n]=dict(overall=panel.summarize(r),by_opponent={o:panel.summarize([x for x in r if x['opponent']==o]) for o in sorted({x['opponent'] for x in r})})
    for a,b in (('original','p12'),('original','p16'),('p12','p16')):
        result['paired'][a+'_to_'+b]=paired(rows[a],rows[b])[0]
    return result

def main():
    allrows={n:[] for n in NAMES};blocks={};used=set()
    for b in BLOCKS:
        out=HERE/b;protocol=json.loads((out/'PROTOCOL.json').read_text())
        for n,(path,h) in BINS.items():assert protocol['binaries'][n]==h and panel.sha(path)==h
        seeds=set(range(protocol['seed_start'],protocol['seed_start']+32));assert not used&seeds;used|=seeds
        rows={n:json.loads((out/n/'rows.json').read_text()) for n in NAMES}
        for n,r in rows.items():
            assert len(r)==704 and {x['seed'] for x in r}==seeds
            allrows[n]+=r
        blocks[b]=summarize_set(rows)
    combined=summarize_set(allrows)
    result=dict(blocks=blocks,common64=combined,final_holdout_used=False,
        boundary='Identical frozen policies on two nonoverlapping common32 seed blocks; not separate best-of-version seed subsets; no final100 use or automatic promotion')
    earlier=HERE/'crop_clock_validation32';early_protocol=json.loads((earlier/'PROTOCOL.json').read_text())
    p12_all={}
    for n in ('original','p12'):
        assert early_protocol['binary_sha256'][n]==BINS[n][1]
        old=json.loads((earlier/n/'rows.json').read_text())
        assert len(old)==704 and not {x['seed'] for x in old}&used
        p12_all[n]=old+allrows[n]
    result['p12_all96_separate_reference']=dict(versions={n:panel.summarize(r) for n,r in p12_all.items()},
        paired_original_to_p12=paired(p12_all['original'],p12_all['p12'])[0],
        boundary='P16 was not tested in the earlier32; this is NOT a fair three-version comparison')
    out=HERE/'startup_supply_validation32_b2';panel.save(out/'COMMON64.json',result)
    lines=['# 用户追加新种子：本批与共同64种子结果','',
           '原R2、P12、P16二进制和配置保持不变。每版每批32种子×双座位×11原程序实时对手=704局；两批共同64种子共1408局/版。','',
           '本批是2609125000–5031，上一批为2609124000–4031。P12更早独有的32种子不混入这张三版公平比较表。','',
           '|范围|原R2胜率|P12胜率|P16胜率|','|---|---:|---:|---:|']
    for label,x in [('上一批32',blocks[BLOCKS[0]]),('本次新32',blocks[BLOCKS[1]]),('合计共同64',combined)]:
        values=[x['versions'][n]['overall'] for n in NAMES]
        lines.append('|'+label+'|'+'|'.join(f"{v['r2_wins']}/{v['games']} ({v['r2_win_rate']:.2%})" for v in values)+'|')
    lines+=['','## 本次新32现金、胜负与运行','', '|版本|胜/负/平|平均现金|平均分差|运行异常|','|---|---:|---:|---:|---:|']
    for n,v in blocks[BLOCKS[1]]['versions'].items():
        o=v['overall'];lines.append(f"|{n}|{o['r2_wins']}/{o['losses']}/{o['ties']}|{o['r2_mean_cash']:.1f}|{o['mean_margin']:.1f}|{o['errors']}|")
    lines+=['','## 共同64种子的配对结果','', '|比较|救回/丢旧胜|胜率增益|seed级95%区间|现金差|分差差|','|---|---:|---:|---:|---:|---:|']
    for n,p in combined['paired'].items():
        lo,hi=p['paired_seed_win_delta95'];lines.append(f"|{n}|{p['rescued']}/{p['lost_wins']}|{p['win_delta']:+.2%}|[{lo:+.2%},{hi:+.2%}]|{p['mean_cash_delta']:+.1f}|{p['mean_margin_delta']:+.1f}|")
    lines+=['','## 本次新32：逐对手胜率','', '|对手|原R2|P12|P16|','|---|---:|---:|---:|']
    for o in blocks[BLOCKS[1]]['versions']['original']['by_opponent']:
        values=[blocks[BLOCKS[1]]['versions'][n]['by_opponent'][o] for n in NAMES]
        lines.append('|'+o+'|'+'|'.join(f"{v['r2_wins']}/64 ({v['r2_win_rate']:.2%})" for v in values)+'|')
    lines+=['','区间按seed整块重采样，保留双座位和相近公开路线间相关性；1408局不等于1408个独立样本。同seed改变动作会改变杂草及后续商店抽样，不以个案现金差直接宣称纯局部因果。',
            '没有改参数、按seed/对手身份切策略，也没有public提交或Git推送。最终100个预留seed未使用；本次不自动晋升版本。']
    v=result['p12_all96_separate_reference']['versions']
    lines+=['','## P12额外累计记录（不是三版公平比较）','',
        f"加上P12更早独有的第一批32seed，P12三批累计{v['p12']['r2_wins']}/2112={v['p12']['r2_win_rate']:.2%}，同批原R2 {v['original']['r2_wins']}/2112={v['original']['r2_win_rate']:.2%}。P16未测更早那批，不以此直接断定P12/P16谁更强。"]
    (out/'COMMON64_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    # Retain the existing candidates and original published binary. Append
    # this evaluation instead of relabeling a past favorable block as final.
    for filename,name in [('DEVELOPMENT_BEST.json','p16'),('INDEPENDENT_CANDIDATE.json','p12')]:
        file=HERE/filename;record=json.loads(file.read_text())
        assert record['binary_sha256']==BINS[name][1]
        record['status']='FROZEN_CANDIDATE_CONFIRMATIONS_RECORDED_NOT_FINAL_90_ACCEPTANCE'
        record['additional_new32_5000']=dict(results='startup_supply_validation32_b2/RESULTS.json',
            seeds=[2609125000,2609125031],overall=blocks[BLOCKS[1]]['versions'][name]['overall'])
        record['fair_common64_p12_p16']=dict(results='startup_supply_validation32_b2/COMMON64.json',
            overall=combined['versions'][name]['overall'],paired_original=combined['paired']['original_to_'+name])
        if name=='p12':record['all_three_new32_blocks']=result['p12_all96_separate_reference']
        panel.save(file,record)
    print(json.dumps(dict(new32={n:v['overall'] for n,v in blocks[BLOCKS[1]]['versions'].items()},common64={n:v['overall'] for n,v in combined['versions'].items()},paired=combined['paired'])),flush=True)
if __name__=='__main__':main()
