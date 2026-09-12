"""Procedural holdout release. This never launches an evaluation.

The freeze record must list the three finalists' absolute runtime file hashes,
the exact runner/evaluator files, analysis-plan files, and the opponent manifest.
A new full source audit, completed after that freeze, is required.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from audit_seeds import dump,sha
from seed_sets import ROOT


def verify_files(mapping):
    if not mapping:raise ValueError('Empty frozen file group')
    for name,digest in mapping.items():
        p=Path(name)
        if not p.is_absolute() or not p.is_file() or sha(p)!=digest:raise ValueError('Frozen file mismatch: '+name)


def release(root,freeze_path,audit_dir,out):
    freeze=json.loads(freeze_path.read_text());receipt=json.loads((root/'holdout_receipt.json').read_text())
    if out.exists():raise FileExistsError('Use a new release directory')
    if freeze.get('finalists_frozen') is not True:raise ValueError('Freeze finalists first')
    known=json.loads((root/'candidates.json').read_text())['candidates']
    if set(freeze['candidates'])!={c['id'] for c in known}:raise ValueError('Freeze all three matched finalists')
    provenance=json.loads((root/'provenance.json').read_text());repo=Path(provenance['source_root'])
    for c in known:
        files=freeze['candidates'][c['id']];verify_files(files)
        # All pinned runtime files must appear with the recorded bytes.
        for rel,digest in c['runtime_files'].items():
            path=repo/rel if c['id']=='original-afs-r2' else repo/c['path']/rel
            if files.get(str(path))!=digest:raise ValueError('Candidate version differs from this contract: '+c['id'])
    verify_files(freeze['evaluator_files']);verify_files(freeze['analysis_files'])
    if freeze['opponents_sha256']!=sha(root/'opponents.json'):raise ValueError('Opponent contract changed')
    for opp in json.loads((root/'opponents.json').read_text())['opponents']:
        for rel,digest in opp['runtime_files'].items():
            if sha(repo/rel)!=digest:raise ValueError('Opponent source drift: '+opp['id'])
    audit=json.loads((audit_dir/'summary.json').read_text())
    if audit['source_root']!=str(repo):raise ValueError('Audit source differs')
    frozen_time=datetime.fromisoformat(freeze['frozen_utc'].replace('Z','+00:00'))
    audit_time=datetime.fromisoformat(audit['inventory_started_utc'].replace('Z','+00:00'))
    if frozen_time.tzinfo is None or audit_time.tzinfo is None or audit_time<frozen_time or audit_time>datetime.now(timezone.utc):
        raise ValueError('Run a fresh full audit after a valid UTC freeze timestamp')
    if audit['changed_files'] or audit['new_files_after_inventory']:raise ValueError('Source changed during audit; resolve coverage first')
    if not audit.get('npz_seed_arrays_recovered'):raise ValueError('Run supplement_audit.py too')
    if sha(audit_dir/'exclusions.json')!=audit['exclusions_sha256']:raise ValueError('Audit checksum mismatch')
    if any(g['reason'].startswith(('Python AST parse','Changed binary')) or 'Binary content' not in g['reason'] for g in audit['gaps']):
        raise ValueError('Resolve text audit errors before release')
    exclusions=set(json.loads((audit_dir/'exclusions.json').read_text())['seeds'])
    held=json.loads((root/'sealed/holdout.json').read_text())
    if sha(root/'sealed/holdout.json')!=receipt['sha256']:raise ValueError('Holdout changed')
    if len(held['seeds'])!=256 or len(set(held['seeds']))!=256 or set(held['seeds'])&exclusions:
        raise ValueError('Holdout count/uniqueness/contamination failure; retire and replace it')
    out.mkdir(parents=True)
    (out/'holdout.json').write_bytes((root/'sealed/holdout.json').read_bytes())
    dump(out/'release_receipt.json',dict(status='released_for_one_frozen_matched_comparison',holdout_sha256=receipt['sha256'],
        freeze_sha256=sha(freeze_path),freeze_record=str(freeze_path.resolve()),audit_exclusions_sha256=audit['exclusions_sha256'],
        candidates=sorted(freeze['candidates']),candidate_contract_sha256=sha(root/'candidates.json'),opponents_sha256=sha(root/'opponents.json'),
        count=256,policy='Once this feedback informs development, retire this holdout; do not use it again for confirmation.'))
    print(json.dumps(dict(released=True,count=256,receipt=str((out/'release_receipt.json').resolve()))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--freeze',type=Path,required=True)
    p.add_argument('--fresh-audit',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();release(a.root,a.freeze,a.fresh_audit,a.out)
