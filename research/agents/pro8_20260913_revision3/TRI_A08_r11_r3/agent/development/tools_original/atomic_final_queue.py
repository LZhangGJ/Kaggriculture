import subprocess,json
from pathlib import Path
b=Path("/mnt/data/r3_work")
rows=[]
for job in [['atomic_checks_gcc', '240', 'python', '/mnt/data/r3_work/candidate/tests/run_checks.py', '--sanitize', '--referee', '/mnt/data/r3_work/feedback/referee', '--out-dir', '/mnt/data/r3_work/logs/atomic_checks_gcc'], ['atomic_checks_clang', '240', 'python', '/mnt/data/r3_work/candidate/tests/run_checks.py', '--cxx', 'clang++', '--library', '/mnt/data/r3_work/clang.so', '--sanitize', '--referee', '/mnt/data/r3_work/feedback/referee', '--out-dir', '/mnt/data/r3_work/logs/atomic_checks_clang'], ['atomic_branches', '120', 'python', '/mnt/data/r3_work/tests/own_branch_checks.py'], ['atomic_branches_no_clearance', '120', 'python', '/mnt/data/r3_work/tests/own_branch_checks.py', '--no-clearance'], ['atomic_branches_stress', '120', 'python', '/mnt/data/r3_work/tests/own_branch_checks.py', '--supply', '3']]:
 r=subprocess.run(["python",str(b/"tests/timed_run.py"),*job]);rows.append({"job":job,"returncode":r.returncode});(b/"logs/atomic_final_queue.json").write_text(json.dumps(rows,indent=2))
 if r.returncode:raise SystemExit(r.returncode)
