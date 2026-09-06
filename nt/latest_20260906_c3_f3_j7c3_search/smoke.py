"""Small delivery/parity checks, not a strength or universal-legality claim."""
import argparse, hashlib, json, subprocess, tempfile
from pathlib import Path
import numpy as np
from runtime import P, ARMS, make_pool, jobs, batch, summary, save_json

def trace_hash(trace):
    return hashlib.sha256(json.dumps(trace,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def game_hashes(result):
    return [dict(seed=r['seed'],seat=r['seat'],opponent=r['opponent'],cash=r['cash'],
        opponent_cash=r['opponent_cash'],trace_sha256=trace_hash(t)) for r,t in zip(result['rows'],result['traces'])]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    pool=make_pool(); js=jobs(67000000,1)
    receipt=dict(status='PASS',games=0,arms={},dependencies={})
    import pybind11
    receipt['dependencies']=dict(numpy=np.__version__,pybind11=pybind11.__version__)
    for arm in ARMS:
        base=batch(pool,arm,js,threads=1,trace=True)
        keep=batch(pool,arm,js,P/'examples/keep_day0_28.txt',threads=16,trace=True)
        mixed1=batch(pool,arm,js,P/'examples/mixed_day0_28.txt',threads=1,trace=True)
        mixed16=batch(pool,arm,js,P/'examples/mixed_day0_28.txt',threads=16,trace=True)
        again=batch(pool,arm,js,P/'examples/mixed_day0_28.txt',threads=16,trace=True)
        for result in (base,keep,mixed1,mixed16,again):
            assert summary(result)['status']=='PASS'
            assert len(result['choice'])==30*len(js)
            assert np.all(result['mask'][np.arange(len(result['choice'])),result['choice']]==1)
            receipt['games']+=len(result['rows'])
        assert game_hashes(base)==game_hashes(keep)
        assert game_hashes(mixed1)==game_hashes(mixed16)==game_hashes(again)
        for key in ('global','candidates','mask','choice','changed','kind','pos'):
            assert np.array_equal(base[key],keep[key]),(arm,key)
            assert np.array_equal(mixed1[key],mixed16[key]),(arm,key)
        assert np.sum(mixed16['changed'])>0
        expected=np.where(mixed16['mask'][np.arange(len(mixed16['choice'])),mixed16['requested']]>0,mixed16['requested'],0)
        assert np.array_equal(expected,mixed16['choice'])
        assert np.all(mixed16['choice'][mixed16['day']==29]==0)
        candidates=[i for i,d in enumerate(base['day']) if 1<=d<=28 and np.sum(base['mask'][i])>1]
        idx=candidates[len(candidates)//2];day=int(base['day'][idx]);game=int(base['game'][idx])
        choice=int(np.flatnonzero(base['mask'][idx,1:])[0]+1)
        with tempfile.TemporaryDirectory(prefix='kaggri-plan-smoke-') as tmp:
            plan=Path(tmp)/'plan.txt';values=[0]*29;values[day]=choice
            plan.write_text(' '.join(map(str,values)))
            switched=batch(pool,arm,[js[game]],plan,threads=1,trace=True)
        receipt['games']+=1
        assert summary(switched)['status']=='PASS'
        # Official day0 has 23 actions; later day d starts at step 24*d-1.
        prefix=24*day-1
        assert switched['traces'][0][:prefix]==base['traces'][game][:prefix]
        assert switched['choice'][day]==choice and switched['changed'][day]==1
        receipt['arms'][arm]=dict(baseline=summary(base),mixed=summary(mixed16),
            baseline_games=game_hashes(base),all_keep_identical=True,threads_1_16_identical=True,
            repeated_batch_identical=True,late_day_single_edit=dict(day=day,choice=choice,prefix_steps=prefix))
        print(arm,'PASS',flush=True)
    unit=subprocess.run([str(P/'build/test_ledger')],text=True,capture_output=True,check=True)
    receipt['ledger_test']=unit.stdout
    # Malformed schedules must fail closed; no silent defaults.
    with tempfile.TemporaryDirectory(prefix='kaggri-invalid-plan-') as tmp:
        bad=Path(tmp)/'bad.txt';bad.write_text('0 0')
        try:batch(pool,'f3',js,bad)
        except ValueError:pass
        else:raise AssertionError('bad plan accepted')
    save_json(args.out/'PORTABLE_ACCEPTANCE.json',receipt)
    print('PASS',receipt['games'],'full games',flush=True)

if __name__=='__main__':main()
