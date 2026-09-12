"""Record and verify every runtime file for the frozen comparison panel."""
import argparse
import hashlib
import json
from pathlib import Path
import time

root=Path(__file__).resolve().parent
repo=root.parents[1]
package=repo/'nt/latest_20260911_p16_jointafs_r1/agent'
baseline=repo/'nt/latest_20260911_afs_workflow_repair_r1_r2'


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def files_for(name):
    if name.startswith('native:'):
        folder=baseline/name.split(':',1)[1]/'policy'
        return [folder/'agent.so',folder/'config.json',package/'policy/agent.py']
    if name.startswith('stress:'):
        return files_for('native:r2')+[root/'stress.py']
    folder=root/'opponents'/name.split(':',1)[1] if name.startswith('extra:') else package/'opponents'/name
    return sorted(p for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--verify',action='store_true');args=parser.parse_args()
    path=root/'PANEL_SOURCES.json'
    if args.verify:
        document=json.loads(path.read_text())
        for opponent,files in document['opponents'].items():
            actual={p.relative_to(repo).as_posix():digest(p) for p in files_for(opponent)}
            assert actual==files,opponent
        print(json.dumps(dict(verified_opponents=len(document['opponents']),all_runtime_sources_exact=True)))
    else:
        names=json.loads((root/'SCREEN_WITH_DELAYED.json').read_text())['primary_opponents']
        mappings={name:{p.relative_to(repo).as_posix():digest(p) for p in files_for(name)} for name in names}
        assert all(mappings.values())
        for entry in json.loads((root/'opponents/MANIFEST.json').read_text()):
            assert digest(repo/entry['path'])==entry['sha256'],entry
        result=dict(time=time.time(),opponents=mappings,scope='Runtime source hashes. Family diversity requires separate behavioral evidence.')
        with path.open('x') as stream:json.dump(result,stream,indent=2)
        print(json.dumps(dict(pinned_opponents=len(mappings),unique_files=len({p for group in mappings.values() for p in group}))))
