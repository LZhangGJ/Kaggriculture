import subprocess,json
from pathlib import Path
b=Path("/mnt/data/r3_work")
rows=[]
for job in [['portable_branches', '120', 'python', '-B', '/mnt/data/TRI_A08_r11_r3_release/tests/own_branch_checks.py', '--out-dir', '/mnt/data/r3_work/logs/portable_branches'], ['portable_branches_negative', '120', 'python', '-B', '/mnt/data/TRI_A08_r11_r3_release/tests/own_branch_checks.py', '--out-dir', '/mnt/data/r3_work/logs/portable_branches_negative', '--no-clearance'], ['portable_branches_stress', '120', 'python', '-B', '/mnt/data/TRI_A08_r11_r3_release/tests/own_branch_checks.py', '--out-dir', '/mnt/data/r3_work/logs/portable_branches_stress', '--supply', '3']]:
 r=subprocess.run(["python",str(b/"tests/timed_run.py"),*job]);rows.append({"job":job,"returncode":r.returncode});(b/"logs/portable_queue.json").write_text(json.dumps(rows,indent=2))
 if r.returncode:raise SystemExit(r.returncode)
