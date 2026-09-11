"""Source-only, reproducible local build (Ubuntu/WSL2, C++20)."""
from pathlib import Path
import subprocess,sys,argparse
R=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--jobs',type=int,default=2);p.add_argument('--policy-only',action='store_true');a=p.parse_args()
def run(args):subprocess.run(args,cwd=R,check=True)
run([sys.executable,'build_policy.py'])
if not a.policy_only:
 run([sys.executable,str(R/'arena/build.py'),'--jobs',str(a.jobs)])
 run([sys.executable,'build_panel.py'])
 run([sys.executable,'build_policy.py','--baseline','j7'])
 run([sys.executable,'build_policy.py','--baseline','auto'])
print('Build complete; no opponent strategy assets are needed by policy/agent.py.')
