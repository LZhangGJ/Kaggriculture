"""Freeze the new experiment and reserve a unique, previously unregistered seed per match."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent
SOURCE = Path('F:/Kaggriculture/experiments/local_teacher_bc_20260917')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def positive_integers(value):
    # Conservatively exclude every positive integer in seed/protocol metadata,
    # including embedded lists of jobs whose first column is an unnamed seed.
    if isinstance(value, bool):
        return
    if isinstance(value, int) or isinstance(value, str) and value.isdecimal():
        if 0 < int(value) < 2**31:
            yield int(value)
    elif isinstance(value, dict):
        for item in value.values():
            yield from positive_integers(item)
    elif isinstance(value, list):
        for item in value:
            yield from positive_integers(item)


assert not (ROOT / 'PLAN.json').exists(), 'Do not replace registered evaluation seeds'
command = ['rg', '--files']
for glob in ('!**/deps/**', '!**/.venv/**', '!**/__pycache__/**', '*PROTOCOL*.json',
             '*PLAN*.json', '*SEEDS*.json', '*seeds*.json', 'episodes.json', 'sources.jsonl', '*SPLIT*.json'):
    command.extend(['-g', glob])
command.append('F:/Kaggriculture/experiments')
paths = subprocess.run(command, capture_output=True, text=True, encoding='utf8', check=True).stdout.splitlines()
excluded, manifest = set(), []
for filename in sorted(paths):
    path = Path(filename)
    raw = path.read_text(encoding='utf-8-sig')
    values = [json.loads(line) for line in raw.splitlines() if line.strip()] if path.suffix == '.jsonl' else [json.loads(raw)]
    numbers = set()
    for value in values:
        numbers.update(positive_integers(value))
    excluded.update(numbers)
    manifest.append(dict(path=str(path), sha256=digest(path), excluded_integers=len(numbers)))
assert len(paths) > 100 and len(excluded) > 1000
pool = json.loads((SOURCE / 'frozen/pool/POOL.json').read_text(encoding='utf8'))
opponents = [row['id'] for row in pool]
assert len(set(opponents)) == 15
for row in pool:
    assert digest(SOURCE / f"frozen/pool/opponents/{row['id']}/main.py") == row['main_sha256']
rng = random.Random(2026091802)
reserved = set()


def panel(repeats):
    jobs = []
    for _ in range(repeats):
        for opponent in opponents:
            for seat in (0, 1):
                seed = rng.randrange(1, 2**31)
                while seed in excluded or seed in reserved:
                    seed = rng.randrange(1, 2**31)
                reserved.add(seed)
                jobs.append([seed, opponent, seat])
    rng.shuffle(jobs)
    assert set(Counter((o, s) for _, o, s in jobs).values()) == {repeats}
    return jobs


development = [panel(4), panel(4)]
final = panel(16)
assert len(reserved) == 720 and not reserved.intersection(excluded)
audit = dict(status='PASS', exclusion_scope='All registered seed/protocol/episode/source metadata found under F:/Kaggriculture/experiments',
             metadata_files=manifest, excluded_positive_integers=sorted(excluded),
             reserved_unique_match_seeds=720, overlapping_seeds=0,
             disjoint_training_development_final=True, repeated_seeds_between_seats=False)
(ROOT / 'SEED_AUDIT.json').write_text(json.dumps(audit, indent=2), encoding='utf8')
code = ROOT / 'frozen_code'
code.mkdir(exist_ok=True)
for name in ('features.py', 'student.py', 'numeric_features.py', 'model_numeric.py', 'model_joint.py', 'model_joint_numeric.py'):
    shutil.copy2(SOURCE / 'code' / name, code / name)
shutil.copy2(SOURCE / 'runs/bc_joint_exact_001/best.pt', ROOT / 'initial.pt')
initial_hash = digest(ROOT / 'initial.pt')
assert initial_hash == '901716e90db0d7bb13e80aa965b89ca5518def192f8fa92c8891508165aeb678'
plan = dict(goal='At least 65% pure wins against all 15 frozen opponents', threshold=.65,
            metric='wins / all games; draws are not wins', final_games=480, required_wins=312,
            opponents=opponents, development_panels=development, final_jobs=final,
            seeds='Every match has its own unique seed, disjoint from all known prior registered seeds',
            seed_audit_sha256=digest(ROOT / 'SEED_AUDIT.json'), initial_sha256=initial_hash,
            pool_sha256=digest(SOURCE / 'frozen/pool/POOL.json'),
            frozen_code={p.name: digest(p) for p in code.glob('*.py')},
            training=dict(steps=1500, batch=64, candidates=[500, 1500], seed=2026091801,
                          learning_rate=1e-5, history_denoising_probability=.20,
                          method='Joint teacher-data fine tuning; predicted previous-action exposure; not on-policy DAgger',
                          trainable='Shared Transformer, unit type/quantity, market skeleton/quantity heads'),
            runtime=dict(backend='CUDA FP32 serial batch-one inference', teacher_fallback=False,
                         competition_timeout_acceptance=False),
            cpu=dict(limit=70, target=55, pause_at=60, resume_below=55,
                     scope='Only owned job and descendants are throttled; unrelated activity can exceed whole-machine ceiling'))
(ROOT / 'PLAN.json').write_text(json.dumps(plan, indent=2), encoding='utf8')
print(json.dumps(dict(status='PASS', metadata_files=len(manifest), excluded_integers=len(excluded),
                      unique_seeds=len(reserved), development=[len(p) for p in development], final=len(final))), flush=True)
