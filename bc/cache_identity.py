"""Cache and initialization compatibility checks, without loading replay arrays."""
import hashlib
import json
from pathlib import Path
from importlib.metadata import distribution, version
from kgrl import contracts
from kgrl.commitments import mechanics
from label_contract import LABEL_CONTRACT

NUMERIC_SCHEMA='numeric-macro-bc-v3.2-compact'


def file_sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def engine_identity():
    root=Path(distribution('kaggle-environments').locate_file('kaggle_environments/envs/kaggriculture'))
    rules=contracts.RuleSpec(engine_sha256=file_sha(root/'kaggriculture.py'),
        specification_sha256=file_sha(root/'kaggriculture.json'),
        configuration=dict(contracts.DEFAULT_CONFIG),package_version=version('kaggle-environments'))
    mechanics.check_rules(rules)
    return dict(engine_sha=rules.engine_sha256,spec_sha=rules.specification_sha256,
                version=rules.package_version,configuration=rules.configuration)


def make_identity(root,chunk,quantities,entries,sample_seed=None,limit=None):
    root=Path(root).resolve();source=json.loads((root/'identity.json').read_text())
    if list(quantities)!=sorted(map(int,json.loads((root/'training-quantity-vocabulary.json').read_text()))):
        raise ValueError('Cache quantity vocabulary mismatch')
    engine=engine_identity()
    if source.get('engine')!=engine['version'] or source.get('scope')!='all_replays':
        raise ValueError('Expected all-replay current-engine source dataset')
    here=Path(__file__).resolve().parent
    files={name:file_sha(here/name) for name in (
        'fast_bc_v2.py','features_v2.py','policy_v2.py','bc_runtime.py','bc_policy.py',
        'label_contract.py','train_v2.py','cache_bc_v2.py','cache_identity.py')}
    files['worker_mechanics']=file_sha(mechanics.__file__)
    files['worker_contracts']=file_sha(contracts.__file__)
    inventory=[]
    for line in (root/'manifest.jsonl').read_text().splitlines():
        for path in json.loads(line)['files']:
            st=Path(path).stat()
            inventory.append((path,st.st_size,st.st_mtime_ns))
    if entries!=(min(limit,len(inventory)) if limit is not None else len(inventory)):
        raise ValueError('Cache source entry count mismatch')
    result=dict(schema=NUMERIC_SCHEMA,label_contract=LABEL_CONTRACT,source_root=str(root),
        source_sha=file_sha(root/'manifest.jsonl'),dataset_identity_sha=file_sha(root/'identity.json'),
        vocabulary_sha=file_sha(root/'training-quantity-vocabulary.json'),
        trajectory_inventory_sha=digest(inventory),
        code=files,engine=engine,chunk=chunk,quantities=list(quantities),entries=entries,
        sample_seed=sample_seed,limit=limit,all_replays=limit is None)
    return dict(result,digest=digest(result))


def validate_identity(actual):
    if actual.get('schema')!=NUMERIC_SCHEMA:raise ValueError('Numeric cache schema mismatch')
    expected=make_identity(actual['source_root'],actual['chunk'],actual['quantities'],
                           actual['entries'],actual.get('sample_seed'),actual.get('limit'))
    if actual!=expected:raise ValueError('Numeric cache source/code/engine identity mismatch')
    return expected['digest']


def validate_entries(entries):
    names=[Path(e['file']).name+'.bc.zst' for e in entries]
    if len(names)!=len(set(names)):raise ValueError('Duplicate numeric cache output filenames')


def validate_initialization(old,model,quantities):
    if old.get('architecture')!='MacroPolicyV2' or old.get('schema')!='macro-bc-v2':
        raise ValueError('Checkpoint architecture/schema mismatch')
    if list(old.get('quantities',[]))!=list(quantities):raise ValueError('Checkpoint quantity vocabulary mismatch')
    expected=model.state_dict();actual=old['model']
    if expected.keys()!=actual.keys() or any(expected[k].shape!=actual[k].shape for k in expected):
        raise ValueError('Checkpoint model state mismatch')
    return dict(source_label_contract=old.get('label_contract','legacy-v2'),
                source_cache_digest=old.get('cache_digest'),source_data_identity=old.get('data_identity'),
                optimizer_loaded=True,exact_resume=False)
