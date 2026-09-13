"""Restore the frozen unchanged-v37 evaluation without starting any games."""
import argparse
import gzip
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile

BASE='86cf19cd528cb071a1fed019e3bc84e45a007e62'
PREFIX='research/evaluation_sets/2026-09-12-v1'
EXT=Path(__file__).resolve().parent
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--out',type=Path,required=True)
p.add_argument('--repo',type=Path,default=EXT)
p.add_argument('--inputs-only',action='store_true')
a=p.parse_args()
out=a.out.resolve()
assert not out.exists(), 'Use a new output directory'
out.mkdir(parents=True)
repo=Path(subprocess.run(['git','rev-parse','--show-toplevel'],cwd=a.repo,capture_output=True,text=True,check=True).stdout.strip())
with tempfile.TemporaryDirectory(dir=out) as temporary:
    archive=Path(temporary)/'baseline.tar'
    subprocess.run(['git','archive',BASE+':'+PREFIX,'--output',str(archive)],cwd=repo,check=True)
    with tarfile.open(archive) as source: source.extractall(out/'bundle',filter='data')
previous=EXT.parent/'pro8_20260913'
shutil.copytree(previous/'runtime/pro8_20260913',out/'bundle/evaluation/runtime/pro8_20260913')
shutil.copyfile(previous/'combined_roster.json',out/'bundle/evaluation/roster.json')
shutil.copytree(EXT/'runtime/public',out/'bundle/evaluation/runtime/public')
shutil.copytree(EXT/'previous_nine',out/'bundle/previous_nine')
for name in ('inputs','tools'): shutil.copytree(EXT/name,out/name)
for name in ('combined_roster.json','condition_thresholds.json','run_host.py','check_cross_host.py',
             'join_and_analyze.py','additional_comparisons.py','render_master_leaderboard.py'):
    shutil.copyfile(EXT/name,out/name)
if not a.inputs_only:
    shutil.copyfile(EXT/'CROSS_HOST_PREFLIGHT.json',out/'CROSS_HOST_PREFLIGHT.json')
    for host in ('local','wrx90'):
        shutil.copyfile(EXT/('HOST_STATUS_'+host+'.json'),out/('HOST_STATUS_'+host+'.json'))
        for phase in ('preflight','benchmark'):
            source=EXT/'runs'/(phase+'-'+host)
            target=out/'runs'/source.name
            target.mkdir(parents=True)
            for file in source.glob('*.json'): shutil.copyfile(file,target/file.name)
            with gzip.open(source/'rows.jsonl.gz','rb') as stream, (target/'rows.jsonl').open('wb') as dest:
                shutil.copyfileobj(stream,dest)
sys.path.insert(0,str(out/'tools'))
from contracts import verify_freeze
verify_freeze(out/'inputs/FREEZE.json',out/'bundle')
print(json.dumps({'status':'PASS','workspace':str(out),'copied_completed_journals':not a.inputs_only}))
