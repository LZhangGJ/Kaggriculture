"""Pair actual full games and execution ledgers; never ranks future branches."""
from pathlib import Path
import argparse
import gzip
import hashlib
import json
import math
import statistics
from run_production_audit import total as production_total


def ci(x):
    avg = statistics.fmean(x)
    width = 1.96 * statistics.stdev(x) / math.sqrt(len(x)) if len(x) > 1 else 0
    return dict(mean=avg, approx95=[avg - width, avg + width])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--panel', required=True)
    p.add_argument('--prior-panel', required=True)
    p.add_argument('--audit', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    panel = json.loads(Path(a.panel).read_text())
    prior = json.loads(Path(a.prior_panel).read_text())
    audit = Path(a.audit)
    summary = json.loads((audit / 'summary.json').read_text())
    assert panel['status'] == 'COMPLETE_PANEL_NOT_GOAL_ACCEPTANCE'
    assert summary['status'] == 'PASS_OFFLINE_AUDIT_NOT_GOAL_ACCEPTANCE'
    rows = {(r['variant'], r['opponent'], r['seed'], r['seat']): r for r in panel['rows']}
    unchanged = 0
    for r in prior['rows']:
        if r['variant'] not in ('old', 'all_guard', 'all_midroute_net'):
            continue
        n = rows[r['variant'], r['opponent'], r['seed'], r['seat']]
        assert all(n[x] == r[x] for x in ('cash', 'opponent_cash', 'win', 'margin', 'overflow'))
        unchanged += 1
    pairs = [('old', 'resource_exchange'), ('all_guard', 'guard_exchange'), ('all_midroute_net', 'all_exchange')]
    effects = []
    for baseline, trial in pairs:
        for opp in panel['identities']:
            own = [r for k, r in rows.items() if k[:2] == (trial, opp)]
            grouped = {}
            changed = 0
            for r in own:
                b = rows[baseline, opp, r['seed'], r['seat']]
                changed += r['cash'] != b['cash'] or r['opponent_cash'] != b['opponent_cash']
                grouped.setdefault(r['seed'], []).append((r['cash']-b['cash'], r['margin']-b['margin'], int(r['win'])-int(b['win'])))
            effects.append(dict(baseline=baseline, trial=trial, opponent=opp, changed_result_games=changed,
                                **{n: ci([statistics.fmean(x[i] for x in v) for v in grouped.values()])
                                   for i, n in enumerate(('cash_delta', 'margin_delta', 'win_delta'))}))
    actual = []
    protected = ('generated', 'acquired', 'used', 'unit_discard', 'environment_loss', 'drop_loss', 'eod_loss', 'bought', 'sold',
                 'planted', 'fertilized', 'watered', 'placed', 'fed', 'cared', 'escaped')
    for opp in panel['identities']:
        def load(label):
            with gzip.open(audit / f'{label}_{opp}.json.gz', 'rt', encoding='utf-8') as f:
                return json.load(f)['rows']
        old = {(r['seed'], r['seat']): r for r in load('old')}
        trial = load('resource_exchange')
        prod_changed = movement_changed = wages_changed = move_to_pass_games = 0
        pass_delta = []; move_delta = []; calls = []; savings = []; noop = []; escape = []
        for r in trial:
            b = old[r['seed'], r['seat']]; seat = r['seat']
            bt = production_total(b['production'], seat); rt = production_total(r['production'], seat)
            prod_changed += any(bt[k] != rt[k] for k in protected)
            bm, rm = sum(bt['attempts'][1:5]), sum(rt['attempts'][1:5])
            movement_changed += bm != rm
            move_delta.append(rm-bm); pass_delta.append(rt['attempts'][0]-bt['attempts'][0])
            move_to_pass_games += rt['attempts'][0]-bt['attempts'][0] == bm-rm
            wages_changed += any(x[seat]['hired'] != y[seat]['hired'] for x, y in zip(b['cash_ledger'], r['cash_ledger']))
            last = r['planning'][-1]
            calls.append(last['resource_exchange_applied']); savings.append(last['resource_exchange_local_steps_saved'])
            noop.append(sum(rt['no_effect'])); escape.append(sum(rt['escaped']))
        actual.append(dict(opponent=opp, games=len(trial), changed_production_games=prod_changed,
                           changed_movement_games=movement_changed, changed_wage_games=wages_changed,
                           movement_saving_equals_pass_increase_games=move_to_pass_games,
                           mean_move_delta=statistics.fmean(move_delta), mean_pass_delta=statistics.fmean(pass_delta),
                           mean_applied=statistics.fmean(calls), mean_local_saved=statistics.fmean(savings),
                           executed_no_effect=sum(noop), unplanned_escape_observed=sum(escape)))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=False)
    result = dict(status='COMPLETE_ACTUAL_EXECUTION_COMPARISON', prior_config_regression_games=unchanged,
                  panel_sha256=hashlib.sha256(Path(a.panel).read_bytes()).hexdigest(), pairs=effects, actual_single_flag=actual,
                  caveat='Development paired live games, not holdout or Oracle. Equal aggregate production is not proof every trajectory is identical.')
    (out / 'summary.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(dict(regression_games=unchanged, actual_single_flag=actual)))


if __name__ == '__main__':
    main()
