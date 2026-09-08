"""Check complete downloaded artifacts without executing this new opponent."""
from pathlib import Path
import hashlib
import json
import tarfile
from export_native_boatlee_v29 import put

EXP=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    base=EXP/'opponents/yhay81_three_day';output=base/'output'
    meta=json.loads((base/'source/kernel-metadata.json').read_text())
    assert meta['id']=='yhay81/three-day-shop-router' and meta['id_no']==132926004
    manifest=json.loads((output/'submission-manifest.json').read_text())
    hashes={p.relative_to(base).as_posix():sha(p) for p in sorted(base.rglob('*')) if p.is_file() and 'inspection' not in p.parts}
    for rel,key in [('main.py','main_py_sha256'),('agent.so','agent_so_sha256'),('shopstate-router-agent.tar.gz','submission_archive_sha256'),('source/tape.inc','tape_include_sha256')]:
        assert sha(output/rel)==manifest[key],rel
    with tarfile.open(output/manifest['submission_archive']) as t:
        assert sorted(x.name for x in t.getmembers())==['agent.so','main.py']
        for name in ['agent.so','main.py']:assert t.extractfile(name).read()==(output/name).read_bytes()
    for rel in ['source/policy.cpp','source/include/policy_plugin_abi.hpp','source/include/runtime_types.hpp','source/include/six_day_budget_guard.hpp','submission_bridge.cpp']:
        assert (output/rel).is_file()
    receipt=dict(status='PASS_ACQUISITION_NOT_NATIVE_ACCEPTANCE',url='https://www.kaggle.com/code/'+meta['id'],kernel_id=meta['id_no'],
                 files=hashes,manifest=manifest,archive_matches_loose_files=True,source_executed=False,
                 native_integration='PENDING',replaces_fieldbook=False,required_real_opponents=8)
    put(EXP/'receipts/three_day_source_acquisition_v1.json',json.dumps(receipt,indent=2).encode())
    print(json.dumps({k:v for k,v in receipt.items() if k not in ('files','manifest')}),flush=True)

if __name__=='__main__':main()
