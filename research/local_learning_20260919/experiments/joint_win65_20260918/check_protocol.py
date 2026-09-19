"""Quick executable check of seed isolation, balanced pool coverage, and pure-win threshold."""
from collections import Counter
import hashlib
import json
from pathlib import Path
from evaluate import summary

root = Path(__file__).resolve().parent
plan = json.loads((root / 'PLAN.json').read_text(encoding='utf8'))
audit_file = root / 'SEED_AUDIT.json'
assert hashlib.sha256(audit_file.read_bytes()).hexdigest() == plan['seed_audit_sha256']
audit = json.loads(audit_file.read_text(encoding='utf8'))
excluded = set(audit['excluded_positive_integers'])
repair_path = root / 'EVALUATION_REPAIR.json'
replacement = json.loads(repair_path.read_text(encoding='utf8'))['replacement_seeds'] if repair_path.exists() else {}
original_seeds = {s for jobs in [*plan['development_panels'], plan['final_jobs']] for s, _, _ in jobs}
assert not set(replacement.values()).intersection(original_seeds | excluded)
seen = set()
for index, (jobs, repeats) in enumerate(zip([*plan['development_panels'], plan['final_jobs']], [4, 4, 16])):
    if index == 0:
        jobs = [(replacement.get(str(s), s), o, p) for s, o, p in jobs]
    seeds = {s for s, _, _ in jobs}
    assert len(seeds) == len(jobs) and not seeds.intersection(seen | excluded)
    assert Counter((o, p) for _, o, p in jobs) == Counter({(o, p): repeats for o in plan['opponents'] for p in (0, 1)})
    seen.update(seeds)
assert len(seen) == 720
assert not set(map(int, replacement)).intersection(seen)


def row(win=False, tie=False, error=None):
    return dict(opponent='test', win=win, tie=tie, error=error)


result = summary([row(True), row(tie=True)], 2, 'test')
assert result['strict_win_rate'] == .5 and result['score_rate'] == .75 and not result['meets_65']
assert summary([row(True)] * 312 + [row(tie=True)] * 168, 480, 'test')['meets_65']
assert not summary([row(True)] * 311 + [row(tie=True)] * 169, 480, 'test')['meets_65']
assert not summary([row(True)] * 312 + [row(tie=True)] * 167 + [row(error='failure')], 480, 'test')['meets_65']
assert not summary([row(True)] * 312, 480, 'test')['meets_65']
print(json.dumps(dict(status='PASS', unique_seeds=len(seen), balanced_full_pool=True,
                      strict_win_threshold='312/480', draws_not_wins=True)))
