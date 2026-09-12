"""Copy small source evidence and verify existing runtime pins without imports."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
from audit_seeds import dump, sha


def freeze(repo, out):
    repo, out = repo.resolve(), out.resolve()
    if repo == out or repo in out.parents: raise ValueError('Use isolated output')
    snap = out/'source_snapshot'; snap.mkdir(exist_ok=False)
    core = 'nt/latest_20260911_p16_jointafs_r1/agent'
    official = core+'/referee'
    paths = [official+'/cpu_runtime.py', *[official+'/official/'+n for n in ('kaggriculture.py','kaggriculture.json','seed_utils.py')],
             'research/robust90/POOL.json','research/robust90/PANEL_SOURCES.json','research/robust90/BASELINE.json',
             'research/robust90/RNG_COUPLING_AUDIT.json','research/robust90/PERFORMANCE_LEADER.json',
             'research/robust90/evaluate.py','research/robust90/native_env.py','research/robust90/pin_panel_sources.py',
             'research/robust90/run_connected_day3.py','research/robust90/future_sweep.py']
    records=[]
    for rel in paths:
        p=repo/rel; target=snap/rel; target.parent.mkdir(parents=True,exist_ok=True)
        before=sha(p); shutil.copyfile(p,target)
        if sha(target)!=before or sha(p)!=before: raise ValueError('Changed source: '+rel)
        records.append(dict(path=rel,sha256=before,bytes=p.stat().st_size))
    pool=json.loads((repo/'research/robust90/POOL.json').read_text())['primary_opponents']
    pins=json.loads((repo/'research/robust90/PANEL_SOURCES.json').read_text())['opponents']
    assert len(pool)==len(set(pool))==16
    opponents=[]
    for name in pool:
        mapping=pins[name]
        for rel,digest in mapping.items():
            if sha(repo/rel)!=digest: raise ValueError('Opponent source drift: '+rel)
        if name.startswith('native:'):
            folder=repo/'nt/latest_20260911_afs_workflow_repair_r1_r2'/name.split(':')[1]/'policy'
            expected={folder/'agent.so',folder/'config.json',repo/core/'policy/agent.py'}
        else:
            folder=repo/'research/robust90/opponents'/name.split(':')[1] if name.startswith('extra:') else repo/core/'opponents'/name
            expected={p for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'}
        if expected!={repo/rel for rel in mapping}: raise ValueError('Runtime file list changed: '+name)
        treehash=hashlib.sha256(json.dumps(mapping,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        opponents.append(dict(id=name,runtime_files=mapping,runtime_tree_sha256=treehash,verified=True))
    dump(out/'opponents.json',dict(schema_version=1,opponents=opponents,excluded_diagnostic='stress:delayed_seller',
         source='research/robust90/POOL.json and PANEL_SOURCES.json',tree_hash_rule='SHA256 of UTF-8 canonical JSON runtime_files (sorted keys; separators comma/colon)'))
    candidates=[]
    for name in ('connected-day3-v1','expected-selector-v1'):
        folder=repo/'research/robust90/packages'/name
        manifest=json.loads((folder/'MANIFEST.json').read_text())
        for rel,digest in manifest['files'].items():
            if sha(folder/rel)!=digest: raise ValueError('Candidate package drift: '+name+'/'+rel)
        # Preserve exact package metadata and all small runtime sources; binary identity stays pinned.
        full={p.name:sha(p) for p in folder.iterdir() if p.is_file() and p.suffix!='.pyc'}
        for rel in full:
            if (folder/rel).suffix!='.so':
                target=snap/'research/robust90/packages'/name/rel; target.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(folder/rel,target)
        candidates.append(dict(id=name,path=folder.relative_to(repo).as_posix(),runtime_files=full,
             package_manifest_sha256=sha(folder/'MANIFEST.json'),verified=True))
    r2=pins['native:r2']
    candidates.append(dict(id='original-afs-r2',path='nt/latest_20260911_afs_workflow_repair_r1_r2/r2',runtime_files=r2,verified=True))
    dump(out/'candidates.json',dict(candidates=candidates,scope='Version pins only. No evaluation or promotion. Re-freeze before holdout.'))
    head=subprocess.check_output(['git','--no-optional-locks','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    config=json.loads((repo/official/'official/kaggriculture.json').read_text())['configuration']
    config={k:v.get('default') if isinstance(v,dict) else v for k,v in config.items()}
    dump(out/'provenance.json',dict(source_root=str(repo),source_commit=head,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        files=records,default_configuration=config,engine_source_claim='Frozen Kaggriculture 1.32.7 interpreter from campaign CPU host; raw source hashes authoritative.',
        caveat='HEAD identifies provenance only; source checkout may contain uncommitted campaign work. No Git state changed.',
        native_evaluation_host=dict(path='research/robust90/native_direct_host.so',sha256=sha(repo/'research/robust90/native_direct_host.so'))))
    print(json.dumps(dict(opponents_verified=len(opponents),candidates_verified=len(candidates),source_commit=head)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();freeze(a.repo,a.out)
