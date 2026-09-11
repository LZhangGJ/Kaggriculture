from pathlib import Path
import hashlib,json,statistics as st
E=Path(__file__).resolve().parents[1];hashes={}
def read(rel):
    p=E/rel;hashes[rel]=hashlib.sha256(p.read_bytes()).hexdigest();return json.loads(p.read_text())
build=read('native/build/build_receipt.json');freeze=read('profiles/s4q/freeze.json');cfg=read('profiles/s4q/configs.json')
assert build['binary_sha256']==freeze['build']['binary_sha256']
assert hashes['profiles/s4q/configs.json']==freeze['configs_sha256']
for rel,h in build['source_hashes'].items():assert hashlib.sha256((E/rel).read_bytes()).hexdigest()==h
prereq=read('receipts/s4q_prerequisites_v2/acceptance.json');assert prereq['status']=='PASS_PREREQUISITES_NOT_STRENGTH'
assert prereq['build']['binary_sha256']==build['binary_sha256']
for rel,h in prereq['verified_receipts'].items():
    r=read(rel);assert hashes[rel]==h and r['status']=='PASS' and r['build']['binary_sha256']==build['binary_sha256']
mechanisms={}
for name in ('day_value','day_scenario','workforce','shared','old','recovery','pickup'):
    r=read(f'receipts/s4q_{name}_mechanisms_v1/acceptance.json');assert r['status']=='PASS'
    mechanisms[name]={k:r[k] for k in ('checks','native_mechanism_checks','animal_cases','windows','transitions') if k in r}
panel=read('receipts/s4q_sixway_N50_v1/results.json');paired=read('receipts/s4q_interaction_N50_v1/summary.json')
audit=read('receipts/s4q_pool_audit_N50_v1/summary.json');cash=read('receipts/s4q_cash_labour_N50_v1/summary.json')
execution=read('receipts/s4q_execution_v1/acceptance.json');runtime=read('receipts/s4q_runtime_probe_v1/acceptance.json')
choices=read('receipts/s4q_value_choices_v1/acceptance.json')
assert panel['status']=='COMPLETE_PANEL_NOT_GOAL_ACCEPTANCE' and len(panel['rows'])==5400
assert audit['status']=='PASS_OFFLINE_AUDIT_NOT_GOAL_ACCEPTANCE' and audit['unchanged_result_games']==3600
assert paired['unchanged_controls']==1800 and len(cash['pairs'])==54
assert execution['status']=='COMPLETED_EXECUTION_NOT_GOAL_ACCEPTANCE'
assert choices['status']=='PASS_READONLY_LIVE_SELECTOR_AUDIT_NOT_COUNTERFACTUAL_LABELS' and len(choices['games'])==64
opps=[k for k in panel['identities'] if k!='pass'];assert len(opps)==8
all90=[label for label,s in panel['summary'].items() if all(s[k]['win_rate']>.9 for k in opps)]
anomalies={k:sum(sum(x['own_production'][k])*x['games'] for x in audit['summary']) for k in ('no_effect','escaped')}
averages={label:st.fmean(s[k]['win_rate'] for k in opps) for label,s in panel['summary'].items()}
bykey={(r['variant'],r['opponent'],r['seed'],r['seat']):r for r in panel['rows']}
pressure=[]
for base in ('all_intraday','all_intraday_insert'):
    changed=flips=0
    for r in panel['rows']:
        if r['variant']!=base+'_value':continue
        b=bykey[base+'_value_public',r['opponent'],r['seed'],r['seat']]
        changed+=any(r[k]!=b[k] for k in ('cash','opponent_cash','overflow'))
        flips+=r['win']!=b['win']
    pressure.append(dict(context=base,changed_terminal_games=changed,changed_wins=flips))
diag=dict(games=64,checks=len(choices['rows']),rejected=sum(r['rejected'] for r in choices['rows']),unknown=sum(r['unknown'] for r in choices['rows']),different_scale_forecasts=sum(r['keep_endpoint']['scale']!=r['proposal_endpoint']['scale'] for r in choices['rows']),scale_forecast_mismatch=sum(any(r['forecast_error']['scale']) for r in choices['rows']),overlapping_checks_not_independent=True)
out=E/'receipts/s4q_stage_acceptance_v1';out.mkdir(exist_ok=False)
scripts=('resume_s4q_prerequisites.py','run_s4q_panel.py','summarize_s4q.py','summarize_s4q_cash_labour.py','build_value_schedule_probe.py','audit_s4q_value_choices.py','write_s4q_acceptance.py')
for name in scripts:hashes['tools/'+name]=hashlib.sha256((E/'tools'/name).read_bytes()).hexdigest()
result=dict(status='STAGE_COMPLETE_NO_PROMOTION_NOT_GOAL_ACCEPTANCE',binary_sha256=build['binary_sha256'],live_games=5400,unchanged_prior_controls=1800,repeated_audit_games=3600,official_sample_games=32,opponent_isolation_checks=6,mechanisms=mechanisms,native_latency_games=runtime['games'],native_max_action_seconds=runtime['hard_max_action_seconds'],actual_own_effect_anomalies=anomalies,mean_eight_opponent_development_win_rates=averages,public_pressure=pressure,diagnostic=diag,development_all_eight_above_90=all90,promoted=False,holdout_used=False,P_used=False,final_goal_complete=False,report='reports/S4Q_DAY_RESIDUAL_VALUE_REVIEW_ZH.md',input_hashes=hashes,caveats=['Not all effects or full task completeness are formally proved.','Official differential and native wrapper latency are finite samples, not final deployment tests.','No new unseen seed used because no candidate merits promotion. N remains development.','Value estimates keep current projects and same-crop renewal, not optimized future investment or true futures.'])
(out/'acceptance.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k not in ('input_hashes','caveats','mechanisms')},indent=2),flush=True)
