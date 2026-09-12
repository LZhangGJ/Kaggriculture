"""Create and reproduce seed manifests; never evaluate a candidate."""
from __future__ import annotations
import argparse
from bisect import bisect_left, bisect_right
import hashlib
import json
import math
from pathlib import Path
import secrets
import sys
import time
from audit_seeds import dump,sha
from economy_features import characterize, selection_dimensions, NAMES

DOMAIN=2**31
POOL_SIZE=4096
ROOT=Path(__file__).resolve().parents[1]


def sample(key, label, count, excluded):
    selected=[]; used=set(excluded); counter=0
    while len(selected)<count:
        raw=hashlib.sha256(bytes.fromhex(key)+b'\0'+label.encode('ascii')+counter.to_bytes(8,'big')).digest()
        seed=int.from_bytes(raw[:4],'big') & (DOMAIN-1);counter+=1
        if seed in used: continue
        selected.append(seed);used.add(seed)
    return selected,counter


def freeze_draws(root):
    audit=json.loads((root/'audit/exclusions.json').read_text())
    excluded=set(audit['seeds'])
    keys={label:secrets.token_hex(32) for label in ('representative','stress_pool','holdout')}
    representative,rc=sample(keys['representative'],'representative-v1',256,excluded)
    pool,pc=sample(keys['stress_pool'],'stress-pool-v1',POOL_SIZE,excluded|set(representative))
    # Do not characterize the holdout, even to check whether it looks representative.
    holdout,hc=sample(keys['holdout'],'holdout-v1',256,excluded|set(representative)|set(pool))
    protocol=dict(schema_version=1,domain=[0,DOMAIN],domain_end_exclusive=True,
        entropy='Independent 256-bit OS-random keys, drawn once before characterization',
        sampler='SHA256(key_bytes || NUL || ASCII_label || uint64_be(counter)), first 32 bits masked to 31; reject exclusions and duplicates',
        uniformity='Uniform pseudorandom sample without replacement over the eligible 31-bit domain; no modulo bias, screening or economic stratification for representative/holdout.',
        exclusion_sha256=sha(root/'audit/exclusions.json'),keys={k:v for k,v in keys.items() if k!='holdout'},
        counters=dict(representative=rc,stress_pool=pc),stress_pool_size=POOL_SIZE,
        characterization='Fixed before selection: potential RNG table and PASS/PASS conditional features; no price or candidate-result inputs.',
        code_sha256={name:sha(root/'tools'/name) for name in ('seed_sets.py','economy_features.py')},
        selection='26 coordinate extrema (low/high), eight first-shop witnesses, contrasting milk/wool and early/late cells, then rank-distance farthest-first to 128.',
        release_note='Holdout excludes the entire characterized pool, not just the 128 selected stress seeds.')
    for name,values in (('representative',representative),('stress_pool',pool)):
        dump(root/'manifests'/f'{name}.json',dict(schema_version=1,set=name,count=len(values),seeds=values))
    dump(root/'sealed/holdout.json',dict(schema_version=1,set='holdout',count=256,seeds=holdout,
         status='sealed_unevaluated',separation='Procedural only; local agents can read this file.'))
    dump(root/'sealed/generator.json',dict(key=keys['holdout'],label='holdout-v1',counter=hc,
         exclusion_sha256=protocol['exclusion_sha256'],representative_sha256=sha(root/'manifests/representative.json'),stress_pool_sha256=sha(root/'manifests/stress_pool.json')))
    dump(root/'selection_protocol.json',protocol)
    dump(root/'holdout_receipt.json',dict(count=256,manifest='sealed/holdout.json',sha256=sha(root/'sealed/holdout.json'),
        generator_sha256=sha(root/'sealed/generator.json'),status='sealed_unevaluated',economy_inspected=False,candidate_games=0,
        release='Freeze candidate files, settings, evaluator and analysis first; re-audit exclusions; use tools/release_holdout.py.'))


def ranks(rows,dims):
    columns={d:sorted(r['features'][d] for r in rows) for d in dims}
    # Integer midranks avoid platform-dependent distances and ties.
    vectors=[[bisect_left(columns[d],r['features'][d])+bisect_right(columns[d],r['features'][d]) for d in dims] for r in rows]
    return vectors,columns


def select(rows):
    dims=selection_dimensions(); vec,cols=ranks(rows,dims);chosen=[];reasons={}
    def add(i,reason):
        reasons.setdefault(i,[]).append(reason)
        if i not in chosen:chosen.append(i)
    for d in dims:
        add(min(range(len(rows)),key=lambda i:(rows[i]['features'][d],rows[i]['seed'])),'minimum:'+d)
        add(min(range(len(rows)),key=lambda i:(-rows[i]['features'][d],rows[i]['seed'])),'maximum:'+d)
    for name in NAMES:
        eligible=[i for i,r in enumerate(rows) if r['reference_first_shop']==name]
        add(min(eligible,key=lambda i:rows[i]['seed']),'reference_first_shop:'+name)
    cells=[]
    for x,y,label in [('ref_shop_units_MILK','ref_shop_units_WOOL','reference_milk_wool'),
                      ('ref_early_milk_minus_wool','ref_late_milk_minus_wool','reference_early_late_balance'),
                      ('potential_units_sum_MILK','potential_units_sum_WOOL','potential_milk_wool')]:
        lowx,highx=cols[x][len(rows)//4],cols[x][3*len(rows)//4]
        lowy,highy=cols[y][len(rows)//4],cols[y][3*len(rows)//4]
        for a in ('low','high'):
            for b in ('low','high'):
                eligible=[i for i,r in enumerate(rows) if
                    (r['features'][x]<=lowx if a=='low' else r['features'][x]>=highx) and
                    (r['features'][y]<=lowy if b=='low' else r['features'][y]>=highy)]
                cell=dict(name=label+':'+a+':'+b,pool_count=len(eligible),x=x,y=y,
                          x_cutoff=lowx if a=='low' else highx,y_cutoff=lowy if b=='low' else highy)
                if eligible:
                    i=min(eligible,key=lambda i:rows[i]['seed']);add(i,'contrast:'+cell['name']);cell['witness_seed']=rows[i]['seed']
                cells.append(cell)
    if len(chosen)>128:raise ValueError('Anchors exceed stress budget')
    def dist(a,b):return sum((x-y)**2 for x,y in zip(vec[a],vec[b]))
    nearest=[min(dist(i,j) for j in chosen) for i in range(len(rows))]
    while len(chosen)<128:
        picked=set(chosen)
        i=min((i for i in range(len(rows)) if i not in picked),key=lambda i:(-nearest[i],rows[i]['seed']))
        add(i,'rank_distance_farthest_first')
        for j in range(len(rows)):nearest[j]=min(nearest[j],dist(i,j))
    stress=[rows[i]['seed'] for i in chosen]
    return stress,dict(dimensions=dims,contrast_cells=cells,anchors_and_reasons={str(rows[i]['seed']):reasons[i] for i in chosen},
        coordinate_extrema_covered=all(min(rows[i]['features'][d] for i in chosen)==min(r['features'][d] for r in rows) and
             max(rows[i]['features'][d] for i in chosen)==max(r['features'][d] for r in rows) for d in dims),
        distance='Squared Euclidean distance on integer doubled midranks over the 4096 pool; ties use numeric seed ascending.',
        features_constant_in_pool=[k for k in rows[0]['features'] if len({r['features'][k] for r in rows})==1])


def build(root):
    if (root/'selection_protocol.json').exists():raise FileExistsError('Draws already frozen; use reproduce, never redraw')
    freeze_draws(root)
    values={name:json.loads((root/'manifests'/f'{name}.json').read_text())['seeds'] for name in ('representative','stress_pool')}
    outputs={}
    for name,seeds in values.items():
        rows=[];start=time.time()
        path=root/'features'/f'{name}.jsonl';path.parent.mkdir(exist_ok=True)
        with path.open('w',encoding='utf-8') as f:
            for i,seed in enumerate(seeds):
                row=characterize(seed); rows.append(row);f.write(json.dumps(row,sort_keys=True)+'\n')
                if (i+1)%256==0:print(json.dumps(dict(set=name,characterized=i+1,seconds=round(time.time()-start,2))),flush=True)
        outputs[name]=rows
    stress,report=select(outputs['stress_pool'])
    dump(root/'manifests/stress.json',dict(schema_version=1,set='stress',count=128,seeds=stress))
    dump(root/'selection_result.json',report)
    source={r['seed']:r for r in outputs['stress_pool']}
    with (root/'features/stress.jsonl').open('w',encoding='utf-8') as f:
        for seed in stress:f.write(json.dumps(source[seed],sort_keys=True)+'\n')
    print(json.dumps(dict(representative=256,stress=128,holdout=256,holdout_characterized=0,candidate_games=0)))


def reproduce(root):
    protocol=json.loads((root/'selection_protocol.json').read_text())
    assert sha(root/'audit/exclusions.json')==protocol['exclusion_sha256']
    excluded=set(json.loads((root/'audit/exclusions.json').read_text())['seeds'])
    rep,rc=sample(protocol['keys']['representative'],'representative-v1',256,excluded)
    pool,pc=sample(protocol['keys']['stress_pool'],'stress-pool-v1',POOL_SIZE,excluded|set(rep))
    assert rep==json.loads((root/'manifests/representative.json').read_text())['seeds']
    assert pool==json.loads((root/'manifests/stress_pool.json').read_text())['seeds']
    generator=json.loads((root/'sealed/generator.json').read_text())
    held,hc=sample(generator['key'],'holdout-v1',256,excluded|set(rep)|set(pool))
    assert held==json.loads((root/'sealed/holdout.json').read_text())['seeds']
    assert rc==protocol['counters']['representative'] and pc==protocol['counters']['stress_pool'] and hc==generator['counter']
    assert sha(root/'sealed/holdout.json')==json.loads((root/'holdout_receipt.json').read_text())['sha256']
    for name,seeds in (('representative',rep),('stress_pool',pool)):
        rows=[json.loads(line) for line in (root/'features'/f'{name}.jsonl').open()]
        assert [r['seed'] for r in rows]==seeds
        for r in rows: assert characterize(r['seed'])==r
    poolrows=[json.loads(line) for line in (root/'features/stress_pool.jsonl').open()]
    stress,report=select(poolrows)
    assert stress==json.loads((root/'manifests/stress.json').read_text())['seeds']
    assert report==json.loads((root/'selection_result.json').read_text())
    result=dict(representative_exact=True,stress_pool_exact=True,stress_selection_exact=True,holdout_numeric_reproduction_only=True,
                all_4352_feature_rows_exact=True,holdout_characterization_calls=0)
    dump(root/'reproducibility.json',result);print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=('build','reproduce'));p.add_argument('--root',type=Path,default=ROOT)
    a=p.parse_args();(build if a.command=='build' else reproduce)(a.root)
