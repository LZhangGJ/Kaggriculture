"""Validate public development panels without opening a holdout file."""
import argparse
import json
from pathlib import Path
from audit_seeds import sha
from seed_sets import sample,select


def validate(root,reproduce=False):
    protocol=json.loads((root/'selection_protocol.json').read_text())
    exclusions=set(json.loads((root/'audit/exclusions.json').read_text())['seeds'])
    assert sha(root/'audit/exclusions.json')==protocol['exclusion_sha256']
    manifests={n:json.loads((root/'manifests'/f'{n}.json').read_text()) for n in ('representative','stress','stress_pool')}
    for n,count in [('representative',256),('stress',128),('stress_pool',4096)]:
        seeds=manifests[n]['seeds']
        assert len(seeds)==len(set(seeds))==manifests[n]['count']==count
        assert all(type(s)is int and 0<=s<2**31 for s in seeds)
        assert not set(seeds)&exclusions
    assert not set(manifests['representative']['seeds'])&set(manifests['stress_pool']['seeds'])
    assert set(manifests['stress']['seeds'])<=set(manifests['stress_pool']['seeds'])
    rep,_=sample(protocol['keys']['representative'],'representative-v1',256,exclusions)
    pool,_=sample(protocol['keys']['stress_pool'],'stress-pool-v1',4096,exclusions|set(rep))
    assert rep==manifests['representative']['seeds'] and pool==manifests['stress_pool']['seeds']
    rows=[json.loads(line) for line in (root/'features/stress_pool.jsonl').open()]
    assert [r['seed'] for r in rows]==pool
    chosen,_=select(rows);assert chosen==manifests['stress']['seeds']
    if reproduce:
        from economy_features import characterize
        for name in ('representative','stress_pool'):
            for line in (root/'features'/f'{name}.jsonl').open():
                row=json.loads(line);assert characterize(row['seed'])==row
    result=dict(status='PASS',representative=256,stress=128,stress_pool=4096,draws_and_selection_reproduce=True,
                feature_rows_recomputed=4352 if reproduce else 0,holdout_opened=False)
    print(json.dumps(result));return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--reproduce',action='store_true');a=p.parse_args()
    validate(Path(__file__).resolve().parents[1],a.reproduce)
