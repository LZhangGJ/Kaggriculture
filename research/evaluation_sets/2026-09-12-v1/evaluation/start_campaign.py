"""Run the authorized development and holdout phases, then validate their results."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
PANELS=HERE.parent/'outputs/seed_sets_v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,d):p.write_bytes((json.dumps(d,indent=2,sort_keys=True)+'\n').encode())
def wsl(p):
    path=Path(p).resolve().as_posix();assert path.startswith('C:/')
    return '/mnt/c/'+path[3:]

roster=json.loads((HERE/'roster.json').read_bytes())
preflight=json.loads((HERE/'runs/preflight-v2/STATUS.json').read_bytes())
assert preflight['complete'] and preflight['invalid']==0
assert preflight['roster_sha256']==sha(HERE/'roster.json')
assert roster['runner_sha256']==sha(HERE/'panel_runner.py')
assert roster['analysis_plan_sha256']==sha(HERE/'ANALYSIS_PLAN.md')
assert roster['condition_thresholds_sha256']==sha(HERE/'condition_thresholds.json')
for name,digest in roster['runtime_files'].items():assert sha(HERE/'runtime'/name)==digest,name
audit=json.loads((HERE/'campaign_audit/report.json').read_bytes());assert audit['status']=='PASS'
holdout=PANELS/'sealed/holdout.json'
# Numerical checksum verification only; no feature or result is examined here.
release=dict(status='released_for_one_frozen_comparison',released_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
             roster_sha256=sha(HERE/'roster.json'),runner_sha256=sha(HERE/'panel_runner.py'),
             analysis_plan_sha256=sha(HERE/'ANALYSIS_PLAN.md'),analysis_code_sha256=sha(HERE/'analyze_panels.py'),
             audit_sha256=sha(HERE/'campaign_audit/report.json'),preflight_rows_sha256=sha(HERE/'runs/preflight-v2/rows.jsonl'),
             holdout_sha256=sha(holdout),policy_changes_after_development_allowed=False,
             purpose='One comparison of all six preselected promising policies; no development-panel selection.',
             retirement='Retire from fresh-confirmation use after this comparison; publish seeds and results.')
release_path=HERE/'HOLDOUT_RELEASE.json'
if release_path.exists():
    old=json.loads(release_path.read_bytes())
    for k in ('roster_sha256','runner_sha256','analysis_code_sha256','holdout_sha256'):assert old[k]==release[k]
else:dump(release_path,release)
campaign=HERE/'CAMPAIGN_STATUS.json'
started=time.time()
for phase in ('development','holdout'):
    out=HERE/'runs'/(phase+'-v1')
    dump(campaign,dict(stage=phase,started_unix=started,updated_unix=time.time(),primary_games_scheduled=122880,
                       roster_sha256=sha(HERE/'roster.json'),pid=__import__('os').getpid()))
    command=['wsl','-d','Ubuntu','--','python3','-B',wsl(HERE/'panel_runner.py'),
             '--roster',wsl(HERE/'roster.json'),'--panels',wsl(PANELS),'--out',wsl(out),
             '--phase',phase,'--workers','16','--audit',wsl(HERE/'campaign_audit/report.json'),
             '--preflight',wsl(HERE/'runs/preflight-v2/STATUS.json')]
    if phase=='holdout':command.extend(['--release',wsl(release_path)])
    print(json.dumps(dict(starting_phase=phase,command=command)),flush=True)
    result=subprocess.run(command)
    if result.returncode:
        dump(campaign,dict(stage='blocked_run_failure',phase=phase,exit_code=result.returncode,updated_unix=time.time()))
        raise SystemExit(result.returncode)
    status=json.loads((out/'STATUS.json').read_bytes());assert status['complete'] and not status['invalid']
    assert sha(HERE/'analyze_panels.py')==release['analysis_code_sha256'],'Analysis code changed after release'
dump(campaign,dict(stage='analysis',updated_unix=time.time(),primary_games_completed=122880))
subprocess.run([sys.executable,str(HERE/'analyze_panels.py'),'--panels',str(PANELS),'--roster',str(HERE/'roster.json'),
                '--development',str(HERE/'runs/development-v1'),'--holdout',str(HERE/'runs/holdout-v1'),
                '--out',str(HERE/'results')],check=True)
dump(campaign,dict(stage='complete_pending_review_and_publication',updated_unix=time.time(),primary_games_completed=122880,
                   results_sha256=sha(HERE/'results/RESULTS.json'),seconds=time.time()-started))
