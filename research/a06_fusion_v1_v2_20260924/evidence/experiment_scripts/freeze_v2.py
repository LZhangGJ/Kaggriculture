"""Freeze a promoted V2 candidate and reserve an entirely new heldout panel."""
from pathlib import Path
from datetime import datetime,timezone
import argparse
import hashlib
import json
import random

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--candidate',required=True);p.add_argument('--reason',required=True)
    p.add_argument('--evidence',action='append',required=True);a=p.parse_args()
    assert not (HERE/'FINAL_FREEZE_V2.json').exists(),'Do not silently overwrite a frozen version'
    sources={};tested_files=[]
    for run in a.evidence:
        protocol=json.loads((HERE/'runs'/run/'PROTOCOL.json').read_text())
        summary=json.loads((HERE/'runs'/run/'SUMMARY.json').read_text())
        assert protocol['stage']!='holdout' and summary['errors']==0 and summary['games']==summary['expected']
        assert a.candidate in protocol['candidates']
        tested_files.append(protocol['candidates'][a.candidate])
        sources[f'runs/{run}/games.jsonl']=hashlib.sha256((HERE/'runs'/run/'games.jsonl').read_bytes()).hexdigest()
    seen=set()
    def collect(x):
        if isinstance(x,dict):
            for k,v in x.items():
                if k=='seed' and isinstance(v,int):seen.add(v)
                elif k in ('seeds','development','sealed_holdout') and isinstance(v,list):seen.update(s for s in v if isinstance(s,int))
                else:collect(v)
        elif isinstance(x,list):
            for v in x:collect(v)
    paths=list((ROOT/'experiments').glob('*/*PROTOCOL*.json'))+list(HERE.glob('SEEDS*.json'))+list((HERE/'runs').glob('*/PROTOCOL.json'))
    for path in paths:
        try:collect(json.loads(path.read_text()))
        except (ValueError,OSError):pass
    selected=[s for s in random.Random(202609249034).sample(range(2030000000,2100000000),1000) if s not in seen][:50]
    assert len(selected)==len(set(selected))==50
    seeds=dict(development=json.loads((HERE/'SEEDS.json').read_text())['development'],sealed_holdout=selected,
               excluded_known_seed_count=len(seen),policy='New V2 holdout, never used for V1 or V2 development.')
    folder=HERE/'candidates'/a.candidate
    files={p.relative_to(folder).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(folder.rglob('*'))
           if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc' and 'build' not in p.relative_to(folder).parts}
    assert all(prior==files for prior in tested_files),'Candidate changed since its development games'
    result=dict(candidate=a.candidate,run='holdout_v2',frozen_utc=datetime.now(timezone.utc).isoformat(),files=files,
                selection_reason=a.reason,selection_sources=sources,seeds_file='SEEDS_V2.json')
    (HERE/'SEEDS_V2.json').write_text(json.dumps(seeds,indent=2)+'\n')
    (HERE/'FINAL_FREEZE_V2.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(candidate=a.candidate,files=len(files),new_holdout_seeds=len(selected))))


if __name__=='__main__':main()
