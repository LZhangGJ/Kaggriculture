"""WRX90-only: refresh authorization and copy checked exports; never copy tokens."""
import json
import shlex
import subprocess
import time
import re
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
    from .web_uploads import statuses, MAX_UPLOAD
    status_file=root/'private/web-upload-status.json'
    write(status_file,statuses(root))
    for source, name in [(membership,'members.json'),(root/'site/data.json','data.json'),(status_file,'upload-status.json')]:
        subprocess.run(['scp','-q',str(source),host+':'+dest+'/'+name],check=True,timeout=30)
    subprocess.run(['ssh',host,'sudo /usr/local/sbin/arena-web-import'],check=True,timeout=30)
    spool=cfg.get('upload_dir')
    if not spool:return
    listing=subprocess.run(['ssh',host,'find '+shlex.quote(spool)+" -maxdepth 1 -type f -name '*.json'"],
                           check=True,capture_output=True,text=True,timeout=15)
    inbox=root/'private/web-uploads';inbox.mkdir(exist_ok=True)
    count=0
    for remote in sorted(listing.stdout.splitlines()):
        rid=Path(remote).stem
        if not re.fullmatch('[a-f0-9]{32}',rid) or (inbox/(rid+'.json')).exists():continue
        temp=inbox/(rid+'.pending')
        subprocess.run(['scp','-q',host+':'+spool+'/'+rid+'.json',str(temp)],check=True,timeout=15)
        meta=read(temp)
        if meta.get('user_id') not in ids:continue
        subprocess.run(['scp','-q',host+':'+spool+'/'+rid+'.bin',str(inbox/(rid+'.bin'))],check=True,timeout=30)
        if (inbox/(rid+'.bin')).stat().st_size>MAX_UPLOAD:raise ValueError('Oversized upload')
        temp.replace(inbox/(rid+'.json'))
        count+=1
        if count>=2:break


if __name__ == '__main__':
    import sys
    sync(sys.argv[1])
