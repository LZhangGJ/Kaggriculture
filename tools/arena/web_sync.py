"""WRX90-only: refresh authorization and copy checked exports; never copy tokens."""
import json
import shlex
import subprocess
import time
from pathlib import Path
from .check_export import check
from .store import read, write


def sync(root):
    root = Path(root)
    cfg = read(root/'private/web-host.json')
    repo = read(root/'config.json')['github']['repository']
    # A separate 60-second timer prevents training batches from delaying revocations.
    result = subprocess.run(['gh', 'api', '--paginate', '--jq', '.[] | select(.permissions.pull == true) | .id',
                             f'repos/{repo}/collaborators?per_page=100'],
                            capture_output=True, text=True, check=True, timeout=40)
    ids = sorted({int(line) for line in result.stdout.splitlines() if line.strip()})
    if not ids: raise ValueError('Empty membership response')
    membership = root/'private/web-members.json'
    write(membership, dict(repository=repo, checked_at=time.time(), user_ids=ids))
    check(root/'site')
    host, dest = cfg['ssh_host'], cfg['incoming']
    for source, name in [(membership,'members.json'),(root/'site/data.json','data.json')]:
        subprocess.run(['scp','-q',str(source),host+':'+dest+'/'+name],check=True,timeout=30)
    subprocess.run(['ssh',host,'sudo /usr/local/sbin/arena-web-import'],check=True,timeout=30)


if __name__ == '__main__':
    import sys
    sync(sys.argv[1])
