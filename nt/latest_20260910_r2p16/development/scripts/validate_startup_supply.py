"""Freeze a new seed block before seeing it. Original / parent / candidate."""
from pathlib import Path
import argparse,json,subprocess,sys
import run_panel as panel
from validate_crop_clock import paired
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=HERE/'startup_supply_validation32'
BINS={
 'original':(panel.old.R2/'agent.so','877f196113692722edc8a5b3e30d2d3dd1d4700b0772c13f1e3f2ad7a9ac0ef2'),
 'p12':(HERE/'candidate_r2p12/policy/cropchain1.so','f62c0a303133d28eb93380f52aaead97ff9e7b0ca2613668ac859b638f9adf63'),
 'p16':(HERE/'candidate_r2p16/policy/startupsupply2.so','4d7ef55bb79e4b312445a67628dd15609421e8c9afff21e8192a9b1b055f875b')}

def main():
    global OUT
    p=argparse.ArgumentParser();p.add_argument('--freeze-only',action='store_true')
    p.add_argument('--seed-start',type=int,default=2609124000)
    p.add_argument('--output',type=Path,default=OUT)
    a=p.parse_args();OUT=a.output.resolve()
    assert not set(range(a.seed_start,a.seed_start+32)).intersection(range(2609130000,2609130100)), 'Final holdout stays untouched'
    for path,h in BINS.values():assert panel.sha(path)==h
    protocol=dict(seed_start=a.seed_start,seeds=32,seats=[0,1],names=list(BINS),
        binaries={n:h for n,(_,h) in BINS.items()},pool_sha256=panel.sha(HERE/'pool_all11.json'),
        rule_sha256=panel.sha(panel.old.REF/'official/kaggriculture.py'),config_sha256=panel.sha(panel.old.R2/'config.json'),
        boundary='Frozen before new seed runs; equal opponent weights; unchanged realtime original opponents; final100 unused')
    f=OUT/'PROTOCOL.json'
    if f.exists():assert json.loads(f.read_text())==protocol
    else:panel.save(f,protocol)
    if a.freeze_only:print('Frozen; no games launched',flush=True);return
    dev=json.loads((HERE/'candidate_r2p16/FULL_DEVELOPMENT.json').read_text())
    assert dev['versions']['p16']['overall']['r2_wins']>dev['versions']['p12']['overall']['r2_wins'],'No full-development gate improvement'
    for name,(path,_) in BINS.items():
        folder=OUT/name
        if (folder/'RESULTS.json').exists():continue
        cmd=[sys.executable,str(HERE/'run_panel.py'),'--pool',str(HERE/'pool_all11.json'),'--output',str(folder),'--seed-start',str(protocol['seed_start']),'--seeds','32','--workers','16','--binary',str(path)]
        if not (folder/'PILOT.json').exists():subprocess.run(cmd+['--pilot'],check=True,cwd=ROOT)
        subprocess.run(cmd,check=True,cwd=ROOT)
    rows={n:json.loads((OUT/n/'rows.json').read_text()) for n in BINS}
    assert all(len(r)==704 and not any(x['runtime_error'] for x in r) for r in rows.values())
    result=dict(final_holdout_used=False,versions={n:dict(overall=panel.summarize(r),by_opponent={o:panel.summarize([x for x in r if x['opponent']==o]) for o in sorted({x['opponent'] for x in r})}) for n,r in rows.items()},paired={})
    for x,y in [('original','p12'),('original','p16'),('p12','p16')]:
        stats,data=paired(rows[x],rows[y]);result['paired'][x+'_to_'+y]=stats;panel.save(OUT/f'PAIRED_{x}_to_{y}.json',data)
    panel.save(OUT/'RESULTS.json',result)
    lines=['# P16冻结新32seed确认','',
           f"{protocol['seed_start']}–{protocol['seed_start']+31}；每版704局，11实时对手、双座位；不是最终100seed验收。",'',
           '|版本|胜/704|胜率|平均现金|平均分差|','|---|---:|---:|---:|---:|']
    for n,v in result['versions'].items():
        o=v['overall'];lines.append(f"|{n}|{o['r2_wins']}|{o['r2_win_rate']:.2%}|{o['r2_mean_cash']:.1f}|{o['mean_margin']:.1f}|")
    lines+=['','|配对|救回/丢旧胜|胜率差95%区间|','|---|---:|---:|']
    for n,v in result['paired'].items():
        lo,hi=v['paired_seed_win_delta95'];lines.append(f"|{n}|{v['rescued']}/{v['lost_wins']}|[{lo:+.2%},{hi:+.2%}]|")
    lines+=['','32个seed整块重采样；座位和相近路线不当独立样本。只用于冻结候选确认，不按这些seed逐例调参。未使用最终预留100seed。']
    (OUT/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({n:v['overall']['r2_wins'] for n,v in result['versions'].items()}),flush=True)
if __name__=='__main__':main()
