"""Content identity for exact replay labels, features, and decoder contract."""
import hashlib,json
from pathlib import Path
from cache_identity import file_sha,engine_identity
from exact_model import SCHEMA
from kgrl import contracts
from kgrl.commitments import mechanics

SOURCES=('exact_actions.py','exact_decoder.py','exact_features.py','exact_records.py',
         'worker_phase.py','exact_training.py','features_v2.py','bc_runtime.py','bc_policy.py',
         'exact_model.py','model_v2.py','cache_exact.py','exact_identity.py')


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def sources():
    result={name:file_sha(Path(__file__).parent/name) for name in SOURCES}
    result['cache_identity.py']=file_sha(Path(__file__).parent/'cache_identity.py')
    result['worker_mechanics']=file_sha(mechanics.__file__)
    result['worker_contracts']=file_sha(contracts.__file__)
    return result


def make_identity(root,audit,chunk,limit):
    status=json.loads((audit/'status.json').read_text())
    if not status['complete'] or not status['all_replays']:raise ValueError('Complete all-game audit required')
    if status['source_manifest_sha']!=file_sha(root/'manifest.jsonl'):raise ValueError('Audit source changed')
    value=dict(schema=SCHEMA,engine=engine_identity(),sources=sources(),chunk=chunk,
        audit_sha=file_sha(audit/'status.json'),source_manifest_sha=status['source_manifest_sha'],
        worker_quantities=json.loads((audit/'worker-quantities.json').read_text()),
        market_quantities=json.loads((audit/'market-quantities.json').read_text()),
        all_replays=limit is None,limit=limit,loss='per-turn means: worker gate/ACT/quantity + market gate/ACT/quantity + 0.05 WDL')
    return dict(value,digest=digest(value))


def validate_identity(value):
    unsigned={k:v for k,v in value.items() if k!='digest'}
    if digest(unsigned)!=value['digest']:raise ValueError('Cache identity digest mismatch')
    if value['schema']!=SCHEMA or value['engine']!=engine_identity() or value['sources']!=sources():
        raise ValueError('Code or engine changed since cache creation')
    return value['digest']
