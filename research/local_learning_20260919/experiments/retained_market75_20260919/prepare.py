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
PREVIOUS = Path('F:/Kaggriculture/experiments/market_quantity75_20260918')
MODEL_SOURCE = PREVIOUS / 'frozen_code'
REFERENCE = Path('F:/Kaggriculture/experiments/learned_expand75_20260918/initial.pt')


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
             '*PLAN*.json', '*SEEDS*.json', '*seeds*.json', 'episodes.json', 'sources.jsonl', '*SPLIT*.json', '*REPAIR*.json'):
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
rng = random.Random(2026091901)
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


training_jobs = panel(2)
development = [panel(8)]
final = panel(32)
assert len(reserved) == 1260 and not reserved.intersection(excluded)
audit = dict(status='PASS', exclusion_scope='All registered seed/protocol/episode/source metadata found under F:/Kaggriculture/experiments',
             metadata_files=manifest, excluded_positive_integers=sorted(excluded),
             reserved_unique_match_seeds=1260, overlapping_seeds=0,
             disjoint_training_development_final=True, repeated_seeds_between_seats=False)
(ROOT / 'SEED_AUDIT.json').write_text(json.dumps(audit, indent=2), encoding='utf8')
code = ROOT / 'frozen_code'
code.mkdir(exist_ok=True)
for name in ('features.py', 'student.py', 'numeric_features.py', 'model_numeric.py', 'model_joint.py', 'model_joint_numeric.py'):
    shutil.copy2(MODEL_SOURCE / name, code / name)
shutil.copy2(PREVIOUS / 'run/best.pt', ROOT / 'initial.pt')
initial_hash = digest(ROOT / 'initial.pt')
assert initial_hash == 'a642906d6b8737869efc55dbb52512c633285ef8a2138cafadc248c865bc7120'
reference_hash = digest(REFERENCE)
assert reference_hash == 'cff21b96c0c41b4dadedd23cafbba91d9b5780d4280d644dee3b107f24d57914'
previous_plan = json.loads((PREVIOUS / 'PLAN.json').read_text(encoding='utf8'))
supplemental = previous_plan['supplemental_folders'] + [str(PREVIOUS / 'data/fresh_packed')]
assert len(supplemental) == len({str(Path(x).resolve()) for x in supplemental}) == 4
ready = json.loads((SOURCE / 'data/packed/READY.json').read_text(encoding='utf8'))
plan = dict(goal='Strictly greater than 75% pure wins against all 15 frozen opponents, learned improvements only, expandable market vocabulary', threshold=.75,
            metric='wins / all games strictly > .75; draws are not wins', final_games=960, required_wins=721,
            opponents=opponents, training_jobs=training_jobs, development_panels=development, final_jobs=final,
            seeds='Every match has its own unique seed, disjoint from all known prior registered seeds',
            seed_audit_sha256=digest(ROOT / 'SEED_AUDIT.json'), initial_sha256=initial_hash,
            pool_sha256=digest(SOURCE / 'frozen/pool/POOL.json'),
            frozen_code={p.name: digest(p) for p in code.glob('*.py')},
            teacher_sha256=digest(SOURCE / 'frozen/teacher/main.py'), quantities=ready['quantities'],
            previous_experiment=str(PREVIOUS),
            initialization='Warm start from trained expanded candidate; this is not a promotion of that candidate',
            architecture='Retain the independent market quantity MLP and all existing command classes',
            rare_pattern_reference=str(REFERENCE), rare_pattern_reference_sha256=reference_hash,
            supplemental_folders=supplemental,
            training=dict(steps=6000, batch=64, validate_every=500, patience=4, minimum_improvement=.0001, seed=2026091902,
                          learning_rate=1e-5, embedding_learning_rate=1e-4, quantity_learning_rate=1e-5, history_denoising_probability=.20,
                          group_probabilities=[.80, .025, .025, .025, .025, .10], rare_pattern_probability=.05,
                          rare_sampling='Uniform training examples across every class introduced since the original 1349-class reference, including retained and newly added classes',
                          selection='Lowest validation loss among trained checkpoints; reject before games if full action accuracy drops more than 0.5 percentage points',
                          maximum_action_accuracy_drop=.005,
                          method='Teacher BC with accumulated rare-class rehearsal, fresh balanced rollouts, automatic vocabulary union and independently trainable market quantities',
                          trainable='Market pattern head, pattern embedding and market quantity MLP; shared encoder, unit head and unit quantity MLP frozen'),
            constraints=dict(no_replay_specific_rules=True, no_evaluation_data_in_training=True,
                             vocabulary_source='Union of initialization and training labels only; expandable each round',
                             inference_uses_teacher=False),
            runtime=dict(backend='CUDA FP32 serial batch-one inference', teacher_fallback=False,
                         competition_timeout_acceptance=False),
            cpu=dict(limit=70, target=55, pause_at=60, resume_below=55,
                     scope='Only owned job and descendants are throttled; unrelated activity can exceed whole-machine ceiling'))
(ROOT / 'PLAN.json').write_text(json.dumps(plan, indent=2), encoding='utf8')
print(json.dumps(dict(status='PASS', metadata_files=len(manifest), excluded_integers=len(excluded),
                      unique_seeds=len(reserved), development=[len(p) for p in development], final=len(final))), flush=True)
