"""Describe live paired executions and accounting; no suffix selection or Oracle."""
from pathlib import Path
import argparse
import gzip
import hashlib
import json
import math
import statistics


def interval(values):
    mean = statistics.fmean(values)
    radius = 1.96 * statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.
    return dict(mean=mean, approx95=[mean-radius, mean+radius])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--panel', required=True)
    p.add_argument('--prior-panel', required=True)
    p.add_argument('--audit', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    panel = json.loads(Path(a.panel).read_text())
    prior = json.loads(Path(a.prior_panel).read_text())
    audit = json.loads((Path(a.audit)/'summary.json').read_text())
    assert panel['status'] == 'COMPLETE_PANEL_NOT_GOAL_ACCEPTANCE'
    assert audit['status'] == 'PASS_OFFLINE_AUDIT_NOT_GOAL_ACCEPTANCE'
    records = {(r['variant'], r['opponent'], r['seed'], r['seat']):r for r in panel['rows']}
    controls = 0
    for r in prior['rows']:
        if r['variant'] not in ('old', 'all_guard', 'all_midroute_net'): continue
        n = records[r['variant'], r['opponent'], r['seed'], r['seat']]
        assert all(n[k] == r[k] for k in ('cash', 'opponent_cash', 'margin', 'win', 'overflow'))
        controls += 1
    summaries = {(r['label'],r['opponent']):r for r in audit['summary']}
    pairs = [('old','intraday_stock'), ('old','intraday_funded'),
             ('intraday_stock','intraday_funded'), ('intraday_funded','intraday_funded_exchange'),
             ('all_guard','guard_intraday'), ('all_midroute_net','all_intraday'),
             ('all_intraday','all_intraday_exchange')]
    effects = []
    for baseline, trial in pairs:
        for opp in panel['identities']:
            own = [r for r in panel['rows'] if r['variant']==trial and r['opponent']==opp]
            grouped = {}; changed = improved = harmed = won = lost = 0
            for r in own:
                b = records[baseline,opp,r['seed'],r['seat']]
                delta = (r['cash']-b['cash'], r['margin']-b['margin'], int(r['win'])-int(b['win']))
                grouped.setdefault(r['seed'],[]).append(delta)
                changed += delta[0]!=0 or r['opponent_cash']!=b['opponent_cash']
                improved += delta[0]>0; harmed += delta[0]<0
                won += delta[2]>0; lost += delta[2]<0
            row = dict(baseline=baseline,trial=trial,opponent=opp,changed_games=changed,
                       own_cash_improved_games=improved,own_cash_harmed_games=harmed,
                       loss_to_win=won,win_to_loss=lost,
                       **{name:interval([statistics.fmean(x[i] for x in xs) for xs in grouped.values()])
                          for i,name in enumerate(('cash_delta','margin_delta','win_delta'))})
            if (baseline,opp) in summaries and (trial,opp) in summaries:
                b,t = summaries[baseline,opp],summaries[trial,opp]
                row['actual_cash_delta'] = {k:([y-x for x,y in zip(b['own_cash'][k],v)] if isinstance(v,list)
                                              else v-b['own_cash'][k]) for k,v in t['own_cash'].items()}
                row['actual_quantity_delta'] = {k:([y-x for x,y in zip(b['own_production'][k],v)] if isinstance(v,list)
                                                  else v-b['own_production'][k]) for k,v in t['own_production'].items()}
                row['new_project_means'] = {k:v for k,v in t.items() if k.startswith('planner_mean_intraday') or k.startswith('planner_mean_resource_exchange')}
                row['executed_unit_no_effect'] = sum(t['own_production']['no_effect'])*t['games']
                row['observed_escape'] = sum(t['own_production']['escaped'])*t['games']
            effects.append(row)
    result = dict(status='COMPLETE_ACTUAL_EXECUTION_COMPARISON_NOT_ACCEPTANCE',
                  unchanged_control_games=controls,rows=effects,
                  input_hashes={str(q):hashlib.sha256(q.read_bytes()).hexdigest() for q in (Path(a.panel),Path(a.prior_panel),Path(a.audit)/'summary.json')},
                  caveat='Same development seeds, whole live games, clustered seat pairs. Accounting deltas include downstream effects; they are not isolated causal attribution to individual products.')
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    (out/'summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(status=result['status'],unchanged_control_games=controls,pairs=len(effects))))


if __name__=='__main__': main()
