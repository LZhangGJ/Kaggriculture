"""Replay two seed pairs for all 29 predeclared configs, compare original receipts."""
from pathlib import Path
import gzip,json
from evaluate import P,runtime,evaluate

def main():
    saved=P/'results/independent1000'
    frozen=json.loads((saved/'SELECTION_FROZEN.json').read_text())
    models={r['bin_sha256']:P/r['bin']for r in json.loads((P/'MODEL_INDEX.json').read_text())}
    pool=runtime.make_pool();checks={}
    for name,c in frozen['configs'].items():
        model=models[frozen['hashes'][c['model']]]if c['model']else None
        result=evaluate(pool,model,c['mode'],71100000,2)
        with gzip.open(saved/name/'games.json.gz','rt',encoding='utf8')as f:expected=json.load(f)
        expected=[r for r in expected if r['seed']<71100002]
        assert len(result['rows'])==len(expected)==28
        for a,b in zip(result['rows'],expected):
            assert a['opponent_name']==b['opponent']
            for k in ('seed','seat','cash','opponent_cash','margin','win','steps','plan_calls','execute_calls','reference_calls','error'):
                assert a[k]==b[k],(name,k,a[k],b[k])
        checks[name]={'games':28,'cash_win_and_execution_counters_exact':True}
    out=P/'PORTABLE_ACCEPTANCE.json'
    out.write_text(json.dumps(dict(status='PASS',games=812,configs=29,seeds=2,checks=checks,
        boundary='Frozen two-seed receipt reproduction after include-path-only C++ relocation. Not a new strength estimate or full state/action formal equivalence proof.'),indent=2),encoding='utf8')
    print('PORTABLE_PARITY_PASS',812,flush=True)
if __name__=='__main__':main()
