"""Compare rerun prior-stage monitors with frozen originals; no new games."""
from pathlib import Path
import sys,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
end=int(sys.argv[1]);start=end-100
target=ROOT/'stages'/f'round{end:04}'/'final'
source=(ROOT.parent/'economic_rl_keep2_300_20260906'if start==300 else ROOT/'stages'/f'round{start:04}')/'final'
read=lambda p:json.loads(p.read_text(encoding='utf8'))
names=['keep']+[f'{arm}_r{rep}_step{start}_{mode}'for arm in ('control','aux')for rep in (0,1)for mode in ('greedy','sample')]
checks={}
for name in names:
    a=source/name;b=target/name
    if not (b/'provenance.json').exists():raise RuntimeError(f'Not ready: {name}')
    old=read(a/'games.json');new=read(b/'games.json');assert len(old)==len(new)==1400
    for x,y in zip(old,new):
        for k in ('seed','seat','opponent','cash','margin','win','steps','error','plan_calls','execute_calls','reference_calls'):
            assert x[k]==y[k],(name,k,x,y)
    for k in ('checkpoint','sha256','library_sha','seed_start','seeds','sample','mode','keep_bonus'):
        assert read(a/'provenance.json')[k]==read(b/'provenance.json')[k],(name,k)
    with np.load(a/'decisions.npz')as x,np.load(b/'decisions.npz')as y:
        fields=('global','candidates','mask','probability','choice','game','day','reward','changed','value','logp')
        for k in fields:assert np.array_equal(x[k],y[k]),(name,k)
    checks[name]=dict(games=1400,decision_records=42000,all_checked_arrays_identical=True)
out=ROOT/'diagnostics'/f'MONITOR_REPRODUCTION_{end:04}.json'
out.write_text(json.dumps(dict(status='PASS',source_round=start,target_round=end,checks=checks),indent=2),encoding='utf8')
print('MONITOR_REPRODUCTION_PASS',end,len(names)*1400,flush=True)
