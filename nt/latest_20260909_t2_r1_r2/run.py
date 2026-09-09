"""Relocatable live C++ arena for the exact Public R1/R2 binaries."""
from pathlib import Path
import argparse
import gzip
import hashlib
import importlib.util
import json
import statistics
import sys
import time
import zlib

ROOT=Path(__file__).resolve().parent
NAMES=['g001','g003','boatlee_v29','kaito_v58','lynn_v5','yhay81_six_day','yhay81_three_day']
FIELDS=['steps','cash','opponent_cash','margin','win','error','overflow','action_hash_fnv64','ops']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text())

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--version',choices=['R1','R2'],required=True)
    ap.add_argument('--suite',choices=['seven','710'],default='seven');ap.add_argument('--count',type=int,default=1)
    ap.add_argument('--start',type=int);ap.add_argument('--threads',type=int,default=4);ap.add_argument('--tag',required=True)
    ap.add_argument('--reference',action='store_true',help='Use previous frozen seeds and verify exact rows, not new independent games')
    args=ap.parse_args();assert 1<=args.threads<=16 and 1<=args.count<=100
    sys.path.insert(0,str(ROOT/'arena'));import _triad_panel
    ns='t2'+args.version.lower()+'_runtime';folder=ROOT/'public'/args.version/ns
    source=ROOT/'agents'/args.version/'policy'
    spec=importlib.util.spec_from_file_location('runtime_policy',folder/'policy.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    config=module.DEFAULTS.copy();config.update(load(folder/'config.json'));order=list(module._ORDER) if hasattr(module,'_ORDER') else list(config)
    assert len(order)==(37 if args.version=='R1' else 38)
    receipt=load(ROOT/'public'/args.version/'BUILD_RECEIPT.json');assert sha(folder/'agent.so')==receipt['binary_sha256']
    assert sha(folder/'policy.py')==sha(source/'agent.py') and sha(folder/'config.json')==sha(source/'config.json')
    assets={n:json.loads(zlib.decompress((ROOT/'arena/assets'/(n+'_frozen.json.zlib')).read_bytes())) for n in NAMES[:5]}
    if args.suite=='710':assets['g001']=json.loads(zlib.decompress((ROOT/'arena/assets/710_frozen.json.zlib').read_bytes()))
    pool=_triad_panel.Pool(assets);opponents=list(range(7)) if args.suite=='seven' else [0]
    reference=json.loads(gzip.decompress((ROOT/'evidence'/args.suite/(args.version+'_rows.json.gz')).read_bytes()))
    if args.reference:
        assert args.start is None,'Reference mode chooses frozen seeds'
        seeds=sorted({r['seed'] for r in reference})[:args.count]
    else:
        assert args.start is not None,'Supply a new --start seed, or use --reference'
        seeds=list(range(args.start,args.start+args.count))
    output=ROOT/'runs'/args.tag;output.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter();rows=pool.run(str(folder/'agent.so'),[config[k] for k in order],seeds,opponents,args.threads,False)
    differences=[]
    if args.reference:
        old={(r['seed'],r['opponent'],r['seat']):r for r in reference};seen=set()
        for r in rows:
            key=(r['seed'],r['opponent'],r['seat']);assert key not in seen;seen.add(key)
            diff={f:[r[f],old[key][f]] for f in FIELDS if r[f]!=old[key][f]}
            if diff:differences.append(dict(key=key,differences=diff))
        assert len(seen)==len(seeds)*len(opponents)*2
    result=dict(version=args.version,suite=args.suite,games=len(rows),wins=sum(r['win'] for r in rows),
        win_rate=statistics.mean(r['win'] for r in rows),mean_cash=statistics.mean(r['cash'] for r in rows),
        mean_margin=statistics.mean(r['margin'] for r in rows),errors=sum(bool(r['error']) for r in rows),
        incomplete=sum(r['steps']!=719 for r in rows),reference_comparison=args.reference,differences=differences,
        binary_sha256=sha(folder/'agent.so'),seconds=time.perf_counter()-start,seeds=seeds,threads=args.threads)
    result['status']='PASS' if not differences and not result['errors'] and not result['incomplete'] else 'FAIL'
    for name,obj in [('rows.json',rows),('summary.json',result)]: (output/name).write_text(json.dumps(obj,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True);assert result['status']=='PASS'

if __name__=='__main__':main()
