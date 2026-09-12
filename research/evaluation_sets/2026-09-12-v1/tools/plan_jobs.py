"""Write a job matrix only. No simulator or agent is imported."""
import argparse
import json
from pathlib import Path
from audit_seeds import dump,sha
from seed_sets import ROOT


def plan(root,name,candidate,out,release_receipt=None):
    if out.exists():raise FileExistsError('Use a new job-plan directory')
    candidates={c['id']:c for c in json.loads((root/'candidates.json').read_text())['candidates']}
    if candidate not in candidates:raise ValueError('Candidate is not pinned')
    opponents=json.loads((root/'opponents.json').read_text())['opponents']
    if name=='holdout':
        if release_receipt is None:raise ValueError('Holdout needs a release receipt')
        release=json.loads(release_receipt.read_text())
        if release['status']!='released_for_one_frozen_matched_comparison' or candidate not in release['candidates']:raise ValueError('Candidate not released')
        if sha(root/'candidates.json')!=release['candidate_contract_sha256'] or sha(root/'opponents.json')!=release['opponents_sha256']:raise ValueError('Released contract changed')
        freeze_path=Path(release['freeze_record'])
        if sha(freeze_path)!=release['freeze_sha256']:raise ValueError('Freeze record changed')
        from release_holdout import verify_files
        frozen=json.loads(freeze_path.read_text())
        for files in frozen['candidates'].values():verify_files(files)
        verify_files(frozen['evaluator_files']);verify_files(frozen['analysis_files'])
        source=release_receipt.parent/'holdout.json'
        if sha(source)!=release['holdout_sha256']:raise ValueError('Released holdout differs')
    else:source=root/'manifests'/f'{name}.json'
    seeds=json.loads(source.read_text())['seeds']
    expected={'representative':8192,'stress':4096,'holdout':8192}[name]
    if len(seeds)*len(opponents)*2!=expected:raise ValueError('Wrong matrix size')
    out.mkdir(parents=True)
    with (out/'jobs.jsonl').open('w',encoding='utf-8') as f:
        for seed in seeds:
            for opponent in opponents:
                for own_seat in (0,1):
                    f.write(json.dumps(dict(set=name,candidate=candidate,seed=seed,opponent=opponent['id'],
                                           candidate_seat=own_seat,opponent_seat=1-own_seat))+'\n')
    dump(out/'manifest.json',dict(set=name,candidate=candidate,scheduled_games=expected,seed_manifest_sha256=sha(source),
         opponent_manifest_sha256=sha(root/'opponents.json'),candidate_manifest_sha256=sha(root/'candidates.json'),
         jobs_sha256=sha(out/'jobs.jsonl'),evaluated_games=0,execution='No evaluation launched; use a separately reviewed main-campaign runner.'))
    print(json.dumps(dict(jobs=expected,evaluated_games=0,manifest=str((out/'manifest.json').resolve()))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--set',dest='name',choices=('representative','stress','holdout'),required=True)
    p.add_argument('--candidate',required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--release-receipt',type=Path)
    a=p.parse_args();plan(a.root,a.name,a.candidate,a.out,a.release_receipt)
