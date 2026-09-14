import subprocess,json
from pathlib import Path
b=Path("/mnt/data/r3_work")
rows=[]
for job in [['root_and_cross_precheck', '100', 'python', '/mnt/data/r3_work/tests/final_prechecks.py', 'entry'], ['clean_build_precheck', '120', 'python', '/mnt/data/r3_work/tests/final_prechecks.py', 'clean'], ['atomic_build_negative', '120', 'python', '/mnt/data/r3_work/tests/final_prechecks.py', 'fault']]:
 r=subprocess.run(["python",str(b/"tests/timed_run.py"),*job]);rows.append({"job":job,"returncode":r.returncode});(b/"logs/final_precheck_queue.json").write_text(json.dumps(rows,indent=2))
 if r.returncode:raise SystemExit(r.returncode)
