from pathlib import Path
import subprocess,sys
w=Path(__file__).resolve().parent
for name in ["run_margin_prefixes.py","run_margin_panel.py"]:
 r=subprocess.run([sys.executable,"-B",str(w/name)])
 if r.returncode:raise SystemExit(r.returncode)
