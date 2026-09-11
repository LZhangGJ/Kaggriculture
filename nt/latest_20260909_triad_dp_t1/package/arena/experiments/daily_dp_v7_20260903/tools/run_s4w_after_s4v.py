"""One-shot local continuation; keep at most 16 simulation threads total."""
from pathlib import Path
import json,subprocess,sys,time
E=Path(__file__).resolve().parents[1]
out=E/'receipts/s4w_execution_v1';out.mkdir(exist_ok=False)
state=dict(status='WAITING_FOR_EXISTING_S4V',steps=[],final_goal_acceptance=False)
while True:
    # Check the actual OS command, not merely a lock/progress file. Never start
    # a duplicate panel if its observation times out or the process disappears.
    ps=subprocess.check_output(['ps','-eo','pid,args'],text=True)
    live=[line for line in ps.splitlines() if str(E/'tools/run_s4v_round.py') in line]
    if not live:break
    state['verified_live']=live;(out/'progress.json').write_text(json.dumps(state,indent=2));print('WAIT verified S4V process',flush=True);time.sleep(30)
assert (E/'receipts/s4v_execution_v1/acceptance.json').exists(),'S4V stopped without acceptance; do not resume blindly'
for label,tool in (('s4v_close','close_s4v_stage.py'),('replacement_probe','probe_service_replacement.py')):
    state['status']='RUNNING';state['active_step']=label;(out/'progress.json').write_text(json.dumps(state,indent=2));print('START '+label,flush=True)
    with (out/(label+'.log')).open('w') as log:r=subprocess.run([sys.executable,str(E/'tools'/tool)],stdout=log,stderr=subprocess.STDOUT)
    state['steps'].append(dict(label=label,returncode=r.returncode))
    if r.returncode:
        state['status']='FAILED_PRESERVED';(out/'failure.json').write_text(json.dumps(state,indent=2));raise SystemExit(r.returncode)
state['status']='COMPLETE_ISOLATED_PROBE_NOT_GOAL';state['active_step']=None;(out/'acceptance.json').write_text(json.dumps(state,indent=2));print(state['status'],flush=True)
