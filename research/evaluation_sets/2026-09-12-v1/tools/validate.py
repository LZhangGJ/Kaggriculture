"""Validate seed sets and bounded simulator-only characterization. No policies."""
import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
import random
import sys
import time
from audit_seeds import dump,sha
from economy_features import characterize,day_table,NAMES,PRODUCTS,SHOPS
from seed_sets import ROOT,DOMAIN


def load_host(root):
    path=root/'source_snapshot/nt/latest_20260911_p16_jointafs_r1/agent/referee/cpu_runtime.py'
    spec=importlib.util.spec_from_file_location('isolated_cpu_runtime',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def validate(root):
    t=time.time()
    manifests={n:json.loads((root/'manifests'/f'{n}.json').read_text()) for n in ('representative','stress','stress_pool')}
    manifests['holdout']=json.loads((root/'sealed/holdout.json').read_text())
    counts={'representative':256,'stress':128,'stress_pool':4096,'holdout':256}
    excluded=set(json.loads((root/'audit/exclusions.json').read_text())['seeds'])
    current_excluded=set(excluded);latest_audit_utc=None
    for name in ('audit_refresh.json','audit_refresh_final.json'):
        if (root/name).exists():
            refreshed=json.loads((root/name).read_text());assert refreshed['status']=='PASS'
            for record in refreshed['records']:current_excluded.update(record['seeds'])
            latest_audit_utc=refreshed['utc']
    for n,m in manifests.items():
        assert m['set']==n and m['count']==counts[n] and len(m['seeds'])==counts[n]
        assert len(set(m['seeds']))==counts[n]
        assert all(type(s) is int and 0<=s<DOMAIN for s in m['seeds'])
        assert not set(m['seeds']) & current_excluded
    sets={n:set(m['seeds']) for n,m in manifests.items()}
    for a,b in (('representative','stress'),('representative','holdout'),('stress','holdout'),('representative','stress_pool'),('holdout','stress_pool')):
        assert not sets[a]&sets[b],(a,b)
    assert sets['stress']<=sets['stress_pool']
    characterized=set()
    for name in ('representative','stress_pool','stress'):
        rows=[json.loads(l) for l in (root/'features'/f'{name}.jsonl').open()]
        assert [r['seed'] for r in rows]==manifests[name]['seeds']
        characterized.update(r['seed'] for r in rows)
    assert not characterized&sets['holdout'] and len(characterized)==4352
    protocol=json.loads((root/'selection_protocol.json').read_text())
    for n,digest in protocol['code_sha256'].items():assert sha(root/'source_snapshot/selection_tools'/n)==digest
    # The shipped verifier fixes only an unordered metadata-list comparison.
    import ast
    frozen=ast.parse((root/'source_snapshot/selection_tools/seed_sets.py').read_text())
    current=ast.parse((root/'tools/seed_sets.py').read_text())
    for name in ('sample','freeze_draws','ranks','select','build'):
        a=next(n for n in frozen.body if isinstance(n,ast.FunctionDef) and n.name==name)
        b=next(n for n in current.body if isinstance(n,ast.FunctionDef) and n.name==name)
        assert ast.dump(a)==ast.dump(b),name
    assert sha(root/'tools/economy_features.py')==protocol['code_sha256']['economy_features.py']
    provenance=json.loads((root/'provenance.json').read_text())
    for entry in provenance['files']:assert sha(root/'source_snapshot'/entry['path'])==entry['sha256']
    assert protocol['exclusion_sha256']==sha(root/'audit/exclusions.json')
    receipt=json.loads((root/'holdout_receipt.json').read_text())
    assert receipt['sha256']==sha(root/'sealed/holdout.json') and receipt['generator_sha256']==sha(root/'sealed/generator.json')
    pins=json.loads((root/'opponents.json').read_text())['opponents']
    assert len(pins)==len({o['id'] for o in pins})==16
    host=load_host(root);engine=host.load_engine()
    assert {s:tuple(v) for s,v in engine.SHOPS.items()}==SHOPS
    assert engine.MAX_SHOP_INSTANCES==8
    fixtures=manifests['representative']['seeds'][:4]+manifests['stress']['seeds'][:4]
    checks=0;games=[]
    for seed in fixtures:
        for day in range(29):
            uniforms,choices=day_table(seed,day);rng=random.Random((seed*1000003)^day)
            assert uniforms==[rng.random() for _ in range(200)];checks+=200
            for k,s in enumerate(choices):
                rng=random.Random((seed*1000003)^day)
                for _ in range(k):rng.random()
                assert rng.choice(NAMES)==s;checks+=1
        baseline=None
        for controller in ('pass_pass','buy_ne_once_then_pass'):
            game=host.LocalGame(seed,engine);initial=game.observation(0)
            assert game.configuration.seed is None
            starts=dict(initial['market']['inventory']);shops=[]
            while not game.done:
                action={'farmer':['PASS'],'hands':[],'market':[['BUY_LAND']] if controller=='buy_ne_once_then_pass' and game.t==0 else []}
                game.advance([action,action])
                if game.t%24==0:
                    town=game.state[0].observation.town['unlocked_shops']
                    if len(town)>len(shops):shops=list(town)
            assert game.t==719
            assert len(shops)==8
            obs=game.observation(0);assert 'seed' not in obs
            if controller=='pass_pass':
                expected=characterize(seed)
                assert shops==expected['reference_pass_shops']
                for product in PRODUCTS:
                    assert obs['market']['inventory'][product]==starts[product]-30-expected['features']['ref_shop_units_'+product],(product,game.t)
                weeds=sum(isinstance(tile,dict) and tile.get('kind')=='WEED' for farm in obs['farms'] for row in farm['tiles'] for tile in row)
                assert weeds==expected['features']['ref_weeds_total']
                baseline=shops
            games.append(dict(seed=seed,controller=controller,transitions=game.t,terminal=True,
                 shops_differ_from_pass=shops!=baseline,shop_sequence=shops))
    divergence=sum(r['shops_differ_from_pass'] for r in games if r['controller']!='pass_pass')
    assert divergence>0,'Expected documented action-dependent RNG coupling witness'
    selection=json.loads((root/'selection_result.json').read_text())
    assert selection['coordinate_extrema_covered']
    result=dict(status='PASS',counts={k:v for k,v in counts.items() if k!='stress_pool'},stress_pool=4096,
        counts_unique_valid_disjoint=True,audited_exclusions=len(excluded),overlap_with_audited_exclusions=0,holdout_overlap_with_all_characterized=0,
        latest_audit_exclusions=len(current_excluded),latest_audit_utc=latest_audit_utc,overlap_with_latest_audit=0,
        source_and_protocol_checksums_exact=True,opponents=16,
        scheduled_games_per_candidate={'representative':8192,'stress':4096,'holdout':8192},
        candidate_evaluation_games_run=0,holdout_simulator_calls=0,holdout_economic_features=0,
        raw_rng_values_and_shop_offsets_checked=checks,simulator_only_full_seasons=16,simulator_transitions=16*719,
        reference_demand_and_weeds_exact_in_8_seasons=True,controller_changes_shop_sequence_in_cases=divergence,
        stress_coordinate_extrema_covered=len(selection['dimensions']),contrast_cells_covered=sum(c['pool_count']>0 for c in selection['contrast_cells']),
        contrast_cells_total=len(selection['contrast_cells']),seconds=time.time()-t,
        caveat='No policy panel ran; raw RNG/official interpreter checks do not validate native policy runtime performance.')
    dump(root/'validation.json',result);dump(root/'simulator_validation.json',dict(games=games,source='Isolated frozen official interpreter, default settings; legal scripted actions only.'))
    print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT);a=p.parse_args();validate(a.root)
