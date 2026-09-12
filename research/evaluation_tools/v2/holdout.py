"""Release a fresh holdout only after the original comparison has finished."""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import time

from contracts import (DEFAULT_BUNDLE, read, require, require_complete, seeds, sha,
                       verify_freeze, write)


def release_new(freeze_path, bundle, holdout, audit_path, retired_holdout, completed_status, output):
    freeze_path, bundle = Path(freeze_path).resolve(), Path(bundle).resolve()
    record, plan, _, _ = verify_freeze(freeze_path, bundle)
    output = Path(output).resolve()
    require(not output.exists(), 'Release directory already exists')
    old_release = read(bundle / 'evaluation/HOLDOUT_RELEASE.json')
    old_count = read(bundle / 'holdout_receipt.json')['count']
    old_roster = read(bundle / 'evaluation/roster.json')
    status = read(completed_status)
    require_complete(status, old_count * 16 * 2 * len(old_roster['candidates']))
    require(status.get('phase') == 'holdout' and status.get('roster_sha256') == old_release['roster_sha256'],
            'Original holdout comparison has not finished with its frozen roster')
    require(sha(retired_holdout) == old_release['holdout_sha256'], 'Supply the exact retired original holdout')
    require(sha(holdout) != old_release['holdout_sha256'], 'Original holdout is reserved for its original comparison')
    audit_path = Path(audit_path)
    audit = read(audit_path)
    require(audit.get('status') == 'PASS' and not audit.get('errors'), 'Fresh campaign audit must pass')
    started = datetime.fromisoformat(audit['created_utc'].replace('Z', '+00:00'))
    ended = datetime.fromisoformat(audit['completed_utc'].replace('Z', '+00:00'))
    require(started.tzinfo is not None and ended.tzinfo is not None
            and record['created_unix'] <= started.timestamp() <= ended.timestamp() <= time.time(),
            'Run a fresh campaign audit after freezing the candidate runtime')
    require(audit.get('npz_files_checked', 0) > 0 and audit.get('remote_registry_hashes_rechecked', 0) > 0,
            'Fresh audit must include NPZ and remote seed stores')
    exclusions_path = audit_path.parent / 'exclusions.json'
    excluded = set(seeds(exclusions_path))
    excluded.update(seeds(retired_holdout))
    for name in ('representative', 'stress', 'stress_pool'):
        excluded.update(seeds(bundle / f'manifests/{name}.json'))
    proposed = seeds(holdout)
    require(len(proposed) == 256, 'Fresh holdout requires 256 unique seeds')
    require(not set(proposed) & excluded, 'Fresh holdout overlaps reserved or historical seeds')
    output.mkdir(parents=True)
    copies = {'holdout.json': Path(holdout), 'audit.json': audit_path,
              'exclusions.json': exclusions_path, 'retired_holdout.json': Path(retired_holdout),
              'original_completion.json': Path(completed_status)}
    for name, source in copies.items():
        with (output / name).open('xb') as stream:
            stream.write(source.read_bytes())
    receipt = dict(schema='kaggriculture-holdout-release-v2',
                   status='released_for_one_frozen_comparison', freeze_sha256=sha(freeze_path),
                   experiment_id=plan['experiment_id'], created_unix=time.time(),
                   files={name: sha(output / name) for name in copies},
                   retirement='Fresh confirmation is exhausted after this fixed comparison; future work needs new seeds')
    write(output / 'RELEASE.json', receipt, exclusive=True)
    return receipt


def verify_release(release_path, freeze_path, bundle):
    release_path, freeze_path = Path(release_path).resolve(), Path(freeze_path).resolve()
    receipt = read(release_path)
    require(receipt.get('schema') == 'kaggriculture-holdout-release-v2',
            'The original release cannot authorize a new comparison')
    require(receipt.get('status') == 'released_for_one_frozen_comparison'
            and receipt.get('freeze_sha256') == sha(freeze_path), 'Holdout belongs to another frozen experiment')
    from contracts import inside
    expected_files = {'holdout.json', 'audit.json', 'exclusions.json', 'retired_holdout.json', 'original_completion.json'}
    require(set(receipt['files']) == expected_files, 'Incomplete holdout release evidence')
    for name, expected in receipt['files'].items():
        require(sha(inside(release_path.parent, name)) == expected, f'Released file changed: {name}')
    old = read(Path(bundle) / 'evaluation/HOLDOUT_RELEASE.json')['holdout_sha256']
    require(receipt['files']['holdout.json'] != old and receipt['files']['retired_holdout.json'] == old,
            'Original holdout cannot be reused')
    freeze=read(freeze_path)
    plan=read(freeze_path.parent/'PLAN.json')
    require(receipt.get('experiment_id')==plan['experiment_id'], 'Released experiment identity changed')
    status=read(release_path.parent/'original_completion.json')
    old_release=read(Path(bundle)/'evaluation/HOLDOUT_RELEASE.json')
    count=read(Path(bundle)/'holdout_receipt.json')['count']*16*2*len(read(Path(bundle)/'evaluation/roster.json')['candidates'])
    require_complete(status,count)
    require(status.get('phase')=='holdout' and status.get('roster_sha256')==old_release['roster_sha256'],
            'Original comparison completion is invalid')
    audit=read(release_path.parent/'audit.json')
    started=datetime.fromisoformat(audit['created_utc'].replace('Z','+00:00'))
    ended=datetime.fromisoformat(audit['completed_utc'].replace('Z','+00:00'))
    require(audit.get('status')=='PASS' and not audit.get('errors')
            and started.tzinfo is not None and ended.tzinfo is not None
            and freeze['created_unix']<=started.timestamp()<=ended.timestamp()<=time.time()
            and audit.get('npz_files_checked',0)>0 and audit.get('remote_registry_hashes_rechecked',0)>0,
            'Released audit is stale, incomplete or invalid')
    proposed = seeds(release_path.parent / 'holdout.json')
    reserved = set(seeds(release_path.parent / 'retired_holdout.json'))
    reserved.update(seeds(release_path.parent / 'exclusions.json'))
    for name in ('representative', 'stress', 'stress_pool'):
        reserved.update(seeds(Path(bundle) / f'manifests/{name}.json'))
    require(len(proposed) == 256 and not set(proposed) & reserved, 'Holdout seed overlap or size changed')
    return proposed


def claim_release(release_path, output, manifest):
    claim = dict(release_sha256=sha(release_path), output=str(Path(output).resolve()),
                 freeze_sha256=manifest['freeze_sha256'], jobs_sha256=manifest['jobs_sha256'])
    path = Path(release_path).parent / 'USE.json'
    if path.exists():
        require(read(path) == claim, 'Holdout already used by another run; obtain a new holdout')
    else:
        write(path, claim, exclusive=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, default=DEFAULT_BUNDLE)
    for name in ('freeze', 'holdout', 'audit', 'retired-holdout', 'completed-status', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    release_new(args.freeze, args.bundle, args.holdout, args.audit,
                args.retired_holdout, args.completed_status, args.out)
    print('Fresh holdout released for this frozen comparison only.')


if __name__ == '__main__':
    main()
