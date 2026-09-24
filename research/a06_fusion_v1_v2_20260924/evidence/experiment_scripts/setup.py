"""Freeze source pool, unused seeds and first transparent fusion candidates."""
from pathlib import Path
import hashlib
import json
import random
import shutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
INTERNAL = ROOT / 'experiments/a06_r12_gptpro_round_robin_20260924'
PUBLIC = ROOT / 'experiments/r14_cashflow_vs_public_top10_20260924/PROTOCOL.json'


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def save(p, data):
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    if (HERE / 'SOURCE_POOL.json').exists():
        raise SystemExit('Already frozen; add candidates explicitly instead of overwriting.')
    pool = read(INTERNAL / 'POOL.json')
    public = read(PUBLIC)
    opponents = []
    for row in pool:
        folder = INTERNAL / 'agents' / row['id']
        for name, value in row['files'].items():
            assert digest(folder / name) == value
        opponents.append(dict(id='internal/' + row['id'], group='internal',
                              entry=str(folder / 'main.py'), files=row['files']))
    for row in public['opponents']:
        folder = Path(row['entry']).parent
        if not folder.exists():
            folder = Path(row['entry'].replace('/mnt/e/', 'E:/')).parent
        for name, value in row['files'].items():
            assert digest(folder / name) == value
        opponents.append(dict(id='external/' + row['id'], group='external',
                              ref=row['ref'], version=row['version'], entry=str(folder / 'main.py'), files=row['files']))
    seen = set()
    def collect(x):
        if isinstance(x, dict):
            for key, value in x.items():
                if key == 'seed' and isinstance(value, int): seen.add(value)
                elif key == 'seeds' and isinstance(value, list): seen.update(v for v in value if isinstance(v, int))
                else: collect(value)
        elif isinstance(x, list):
            for v in x: collect(v)
    for path in (ROOT / 'experiments').glob('*/*PROTOCOL*.json'):
        try: collect(read(path))
        except (ValueError, OSError): pass
    rng = random.Random(202609249031)
    candidates = rng.sample(range(2_030_000_000, 2_100_000_000), 200)
    seeds = [s for s in candidates if s not in seen][:66]
    assert len(seeds) == 66
    save(HERE / 'SEEDS.json', dict(development=seeds[:16], sealed_holdout=seeds[16:],
        excluded_known_protocol_seed_count=len(seen), policy='Holdout must not run before candidate hash freeze.'))
    save(HERE / 'SOURCE_POOL.json', dict(opponents=opponents, original_pool_sha256=digest(INTERNAL / 'POOL.json'),
        public_protocol_sha256=digest(PUBLIC), engine_sha256=public['engine_sha256']))
    definitions = [
        ('cf_control','r14_cashflow',{},False,False),
        ('liq_control','r14_liquidity',{},False,False),
        ('cf_liq','r14_cashflow',{},True,False),
        ('cf_liq_direct','r14_cashflow',{'scenario':0},True,False),
        ('cf_liq_conservative','r14_cashflow',{'scenario':0,'intraday':0,'max_hands':12},True,False),
        ('cf_liq_nointraday','r14_cashflow',{'intraday':0},True,False),
        ('cf_liq_h12','r14_cashflow',{'max_hands':12},True,False),
        ('cf_liq_nodelay','r14_cashflow',{'delivery_calendar':0},True,False),
        ('liq_sale','r14_liquidity',{},False,True),
        ('liq_search_sale','r14_liquidity',{'scenario':1,'intraday':1,'max_hands':14},False,True),
        ('cf_liq_h12_nointraday','r14_cashflow',{'max_hands':12,'intraday':0},True,False),
    ]
    manifest = []
    for ident, parent, config, opening, sale in definitions:
        folder = HERE / 'candidates' / ident
        shutil.copytree(INTERNAL / 'agents' / parent, folder, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        cfg = read(folder / 'policy/config.json'); cfg.update(config); save(folder / 'policy/config.json',cfg)
        if sale:
            shutil.copy2(folder / 'main.py', folder / 'native_entry.py')
            shutil.copy2(INTERNAL / 'agents/r14_cashflow/main.py', folder / 'main.py')
        if opening:
            shutil.copy2(folder / 'main.py', folder / 'base_entry.py')
            shutil.copy2(HERE / 'opening_overlay.py', folder / 'main.py')
            save(folder / 'overlay.json',dict(opening_liquidity=10))
        manifest.append(dict(id=ident,parent=parent,config_changes=config,opening_overlay=opening,sale_overlay=sale))
    save(HERE / 'CANDIDATES.json',manifest)
    print(json.dumps(dict(candidates=len(manifest),opponents=len(opponents),development_seeds=16,holdout_seeds=50)))


if __name__ == '__main__': main()
