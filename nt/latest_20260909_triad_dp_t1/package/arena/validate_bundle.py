"""Reproduce frozen receipts after a fresh portable build. No strategy tuning."""
from pathlib import Path
import argparse,hashlib,json,subprocess,sys,time

ROOT=Path(__file__).resolve().parent
EXP=ROOT/'experiments/daily_dp_v7_20260903'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    started=time.perf_counter()
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',default=str(ROOT/'bundle_validation'))
    args=parser.parse_args()
    out=Path(args.out).resolve()
    out.mkdir(exist_ok=False)
    common=[sys.executable,str(ROOT/'run_arena.py'),'--count','1','--workers','4','--official','all']
    commands=[common+['--agents','v1','v2','--seed-start','61020','--out',str(out/'gpt')],
              common+['--agents','ours_j7','--seed-start','20270401','--out',str(out/'j7')],
              common+['--agents','ours_base','ours_autonomous','--opponents','pass','--seed-start','61020','--out',str(out/'bases')]]
    for i,cmd in enumerate(commands):
        with (out/f'run{i}.log').open('w',encoding='utf-8') as log:
            subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        print('Completed validation stage',i,flush=True)
    old=ROOT/'gpt_review/codex/gpt6_daily_dp_review_20260905/arena7_v1/games'
    compared=0
    for p in (out/'gpt/games').glob('*.json'):
        a,b=read(p),read(old/p.name)
        for field in ('cash','opponent_cash','margin','win','steps'):
            assert a[field]==b[field],('gpt',p.name,field,a[field],b[field])
        compared+=1
    names=['g001','g003','boatlee_v29','kaito_v58','lynn_v5','yhay81_six_day','yhay81_three_day']
    historical=read(EXP/'strategy_switch7_20260905/switch_learning_v1/final_test/J7_03.json')['rows']
    expected={(names[r['opponent_index']],r['seed'],r['seat']):r for r in historical}
    j7_compared=0
    for p in (out/'j7/games').glob('*.json'):
        a=read(p);b=expected[(a['opponent'],a['seed'],a['seat'])]
        for field in ('cash','opponent_cash','margin','win','steps'):
            assert a[field]==b[field],('j7',p.name,field,a[field],b[field])
        j7_compared+=1
    results=[read(out/f'{kind}/result.json') for kind in ('gpt','j7','bases')]
    assert compared==28 and j7_compared==14
    assert all(r['status']=='PASS' for r in results)
    provenance=read(ROOT/'SOURCE_PROVENANCE.json')['files']
    for rel,h in provenance.items(): assert sha(ROOT/rel)==h,('source drift',rel)
    allrows=[read(p) for kind in ('gpt','j7','bases') for p in (out/kind/'games').glob('*.json')]
    result=dict(status='PASS_PORTABLE_COMBINED_BUNDLE',gpt_original_cash_pairs=compared,j7_historical_cash_pairs=j7_compared,
                completed_games=len(allrows),official_checked_steps=sum(r['official_checked_steps'] for r in allrows),
                unchanged_original_files=len(provenance),build=read(EXP/'native/build/portable_build_receipt.json')['seconds'],
                validation_seconds=time.perf_counter()-started,
                caveats=['Finite regression, not an all-state proof.','No new strength promotion.','Not Kaggle sandbox validation.',
                         'GPT v1/v2 remain Python policies.','Portable validation performed on local machine with four workers, not on GPT host.'])
    (ROOT/'PACKAGE_ACCEPTANCE.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
