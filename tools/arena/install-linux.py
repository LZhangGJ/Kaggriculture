"""Install a CPU-limited user timer for an already configured arena."""
import argparse
import subprocess
from pathlib import Path

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root',required=True,type=Path)
parser.add_argument('--python',required=True,type=Path)
args=parser.parse_args()
repo=Path(__file__).resolve().parents[2]
root=args.root.resolve();python=args.python.resolve()
if any('\n' in str(p) or '%' in str(p) or '"' in str(p) for p in (repo,root,python)):
    raise ValueError('Unsupported unit path')
units=Path.home()/'.config/systemd/user'
units.mkdir(parents=True,exist_ok=True)
(units/'kaggriculture-arena.service').write_text(f'''[Unit]
Description=Kaggriculture CPU arena
[Service]
Type=oneshot
WorkingDirectory="{repo}"
ExecStart="{python}" -m tools.arena.service "{root}"
Environment=OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
Nice=10
CPUQuota=400%
MemoryMax=8G
TasksMax=512
TimeoutStartSec=3600
''')
(units/'kaggriculture-arena.timer').write_text('''[Unit]
Description=Resume Kaggriculture arena after each bounded batch
[Timer]
OnBootSec=120
OnUnitInactiveSec=30
AccuracySec=5
Persistent=true
[Install]
WantedBy=timers.target
''')
subprocess.run(['systemctl','--user','daemon-reload'],check=True)
subprocess.run(['systemctl','--user','enable','--now','kaggriculture-arena.timer'],check=True)
