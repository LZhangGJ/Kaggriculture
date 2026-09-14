"""Publish only checked summaries to the approved private feature branch."""
import json
import shutil
import subprocess
from pathlib import Path
from .check_export import check
from .store import now, read, write


def publish(root):
    root = Path(root).resolve()
    cfg = read(root/'config.json')
    if not cfg.get('publication_enabled'):
        return
    repo = cfg['github']['repository']
    branch = cfg['publication_branch']
    cwd = Path(__file__).resolve().parents[2]
    def git(*args):
        return subprocess.check_output(['git',*args],cwd=cwd,text=True,timeout=90).strip()
    meta = json.loads(subprocess.check_output(['gh','api','repos/'+repo],text=True,timeout=30))
    if not meta['private'] or git('branch','--show-current') != branch:
        raise RuntimeError('Publication requires the approved private repository and branch')
    check(root/'site')
    target = cwd/'research/arena_live'
    target.mkdir(exist_ok=True)
    names = ['README.md','data.json','index.html']
    for name in names:
        shutil.copyfile(root/'site'/name,target/name)
    paths = ['research/arena_live/'+name for name in names]
    git('add','--',*paths)
    if git('diff','--cached','--name-only','--',*paths):
        # --only never commits other staged work.
        git('commit','--only','-m','Update arena results','--',*paths)
    git('-c','credential.helper=!gh auth git-credential','push','https://github.com/'+repo+'.git','HEAD:refs/heads/'+branch)
    write(root/'published.json',dict(at=now(),commit=git('rev-parse','HEAD')))


if __name__ == '__main__':
    import sys
    publish(sys.argv[1])
