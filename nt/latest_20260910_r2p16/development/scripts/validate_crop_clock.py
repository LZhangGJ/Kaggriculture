"""Pre-frozen new-seed comparison: original, P9, P11, and P12."""
from pathlib import Path
from collections import defaultdict
import json,statistics,subprocess,sys,random
import run_panel as panel
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=HERE/'crop_clock_validation32'
BINARIES={
 'original':(panel.old.R2/'agent.so','877f196113692722edc8a5b3e30d2d3dd1d4700b0772c13f1e3f2ad7a9ac0ef2'),
 'p9':(HERE/'candidate_r2p9/policy/local_sale_1.so','51a8e08c675eead688520d444f2cd1a1f7e50e342bfd5a8f4689b2d38472ba0c'),
 'p11':(HERE/'candidate_r2p11/policy/cropclock1_sale1.so','6ce9e670163a7f6fcea7f449a152d70f758d0f0984eb60c22e7e81446d5610d1'),
 'p12':(HERE/'candidate_r2p12/policy/cropchain1.so','f62c0a303133d28eb93380f52aaead97ff9e7b0ca2613668ac859b638f9adf63')}

def paired(before,after):
    old={(r['opponent'],r['seed'],r['opponent_seat']):r for r in before};pairs=[];blocks=defaultdict(list)
    for r in after:
        b=old[r['opponent'],r['seed'],r['opponent_seat']]
        x=dict(opponent=r['opponent'],seed=r['seed'],opponent_seat=r['opponent_seat'],old_win=b['r2_win'],new_win=r['r2_win'],cash_delta=r['r2_cash']-b['r2_cash'],margin_delta=r['r2_margin']-b['r2_margin'])
        pairs.append(x);blocks[r['seed']].append(int(x['new_win'])-int(x['old_win']))
    v=[statistics.mean(b) for b in blocks.values()];rng=random.Random(9123)
    boot=sorted(statistics.mean(rng.choices(v,k=len(v))) for _ in range(10000))
    return dict(rescued=sum(not x['old_win'] and x['new_win'] for x in pairs),lost_wins=sum(x['old_win'] and not x['new_win'] for x in pairs),mean_cash_delta=statistics.mean(x['cash_delta'] for x in pairs),mean_margin_delta=statistics.mean(x['margin_delta'] for x in pairs),win_delta=statistics.mean(v),paired_seed_win_delta95=[boot[250],boot[9749]]),pairs

def main():
    for path,h in BINARIES.values():assert panel.sha(path)==h
    assert (HERE/'crop_clock_screen100/cropclock1_sale1/cropclock1_sale1/RESULTS.json').exists()
    assert (HERE/'crop_chain_screen100/cropchain1/cropchain1/RESULTS.json').exists()
    protocol=dict(names=list(BINARIES),seed_start=2609123000,seeds=32,seats=[0,1],binary_sha256={n:x[1] for n,x in BINARIES.items()},pool_sha256=panel.sha(HERE/'pool_all11.json'),rule_sha256=panel.sha(panel.old.REF/'official/kaggriculture.py'),original_config_sha256=panel.sha(panel.old.R2/'config.json'),boundary='Frozen public-only policies before confirmation; live original opponents; no tuning on these results; final100 holdout untouched')
    file=OUT/'PROTOCOL.json'
    if file.exists():assert json.loads(file.read_text())==protocol
    else:panel.save(file,protocol)
    for name in protocol['names']:
        folder=OUT/name
        if (folder/'RESULTS.json').exists():continue
        cmd=[sys.executable,str(HERE/'run_panel.py'),'--pool',str(HERE/'pool_all11.json'),'--output',str(folder),'--seed-start',str(protocol['seed_start']),'--seeds','32','--workers','16','--binary',str(BINARIES[name][0])]
        if not (folder/'PILOT.json').exists():subprocess.run(cmd+['--pilot'],check=True,cwd=ROOT)
        subprocess.run(cmd,check=True,cwd=ROOT)
    rows={name:json.loads((OUT/name/'rows.json').read_text()) for name in protocol['names']}
    result=dict(seed_start=protocol['seed_start'],final_holdout_used=False,versions={},paired={})
    for name,data in rows.items():result['versions'][name]=dict(overall=panel.summarize(data),by_opponent={o:panel.summarize([r for r in data if r['opponent']==o]) for o in {r['opponent'] for r in data}})
    for a,b in [('original','p9'),('original','p11'),('original','p12'),('p9','p11'),('p9','p12'),('p11','p12')]:
        stats,pairs=paired(rows[a],rows[b]);key=f'{a}_to_{b}';result['paired'][key]=stats;panel.save(OUT/f'PAIRED_{key}.json',pairs)
    panel.save(OUT/'RESULTS.json',result)
    lines=['# P11冻结新32seed确认','', '2609123000–2609123031，11实时原程序对手，各32seed双座位=704局。非最终100seed验收。','', '|版本|胜/704|胜率|平均现金|平均分差|','|---|---:|---:|---:|---:|']
    for n,v in result['versions'].items():
        o=v['overall'];lines.append(f"|{n}|{o['r2_wins']}|{o['r2_win_rate']:.2%}|{o['r2_mean_cash']:.1f}|{o['mean_margin']:.1f}|")
    lines+=['','|配对|救回/丢旧胜|现金变化|分差变化|胜率变化95%区间|','|---|---:|---:|---:|---:|']
    for n,v in result['paired'].items():
        ci=v['paired_seed_win_delta95'];lines.append(f"|{n}|{v['rescued']}/{v['lost_wins']}|{v['mean_cash_delta']:+.1f}|{v['mean_margin_delta']:+.1f}|[{ci[0]:+.2%},{ci[1]:+.2%}]|")
    lines+=['','区间按seed整块重采样，不把同seed的座位/路线家族当独立样本；不能仅看现金变多宣称胜率提高。平均90%目标仍须最终未见100seed验收。']
    (OUT/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(versions={n:v['overall'] for n,v in result['versions'].items()},paired=result['paired'])),flush=True)
if __name__=='__main__':main()
