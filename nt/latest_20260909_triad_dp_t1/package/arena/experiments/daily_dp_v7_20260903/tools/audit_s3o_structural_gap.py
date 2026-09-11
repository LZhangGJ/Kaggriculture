"""Summarize existing paired cash/production evidence, without rerunning policies.

Opponent private ledgers are offline attribution only. No field here enters View.
Accounting differences are descriptive, not independent recoverable causal gains.
"""
from pathlib import Path
import argparse
import gzip
import hashlib
import json
import statistics as st


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--audit', required=True)
    p.add_argument('--label', default='expiry_plan')
    p.add_argument('--out', required=True)
    a = p.parse_args()
    src, out = Path(a.audit), Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    root = json.loads((src / 'summary.json').read_text())
    result = []
    for e in root['summary']:
        if e['label'] != a.label:
            continue
        path = src / f"{a.label}_{e['opponent']}.json.gz"
        with gzip.open(path, 'rt') as f:
            rows = json.load(f)['rows']
        days = [d for r in rows for d in r['admissions'] if d['captured']]
        material = [d for d in days if d['resource_dropped_jobs'] > 0]
        poor = [d for d in material if d['cash'] < 100 * d['seed_shortage']]
        stages = []
        for day in (0, 2, 5, 8, 11, 17, 23, 29):
            sides = []
            for side in (0, 1):
                ds = []
                for r in rows:
                    seat = r['seat'] if side == 0 else 1-r['seat']
                    c = [d[seat] for d in r['cash_ledger'][:day+1]]
                    pr = [d[seat] for d in r['production'][:day+1]]
                    ds.append(dict(cash=c[-1]['end'], wages=sum(x['hired'] for x in c),
                        feed_cost=sum(x['products'][0] for x in c),
                        wheat=sum(x['acquired'][0] for x in pr),
                        strawberry=sum(x['acquired'][3] for x in pr),
                        wool=sum(x['acquired'][7] for x in pr),
                        strawberry_planted=sum(x['planted'][3] for x in pr),
                        land=sum(x['lands'] for x in c)))
                sides.append({k: st.fmean(v[k] for v in ds) for k in ds[0]})
            stages.append(dict(day=day, own=sides[0], rival=sides[1]))
        cash = e['own_cash']; rival = e['rival_cash']
        gap = dict(sales=[x-y for x,y in zip(cash['sales'],rival['sales'])],
            product_cost_advantage=sum(rival['products'])-sum(cash['products']),
            seed_cost_advantage=sum(rival['seeds'])-sum(cash['seeds']),
            animal_cost_advantage=sum(rival['animals'])-sum(cash['animals']),
            wage_advantage=rival['hired']-cash['hired'],land_advantage=rival['land']-cash['land'])
        reconstructed = sum(gap['sales']) + sum(v for k,v in gap.items() if k != 'sales')
        assert abs(reconstructed-(cash['cash']-rival['cash'])) < 1e-6
        entry = dict(opponent=e['opponent'],games=len(rows),own_cash=cash['cash'],rival_cash=rival['cash'],
            cash_gap=cash['cash']-rival['cash'],gap_ledger=gap,
            resource_drop_days_per_game=len(material)/len(rows),
            seed_shortage_upper_bound_unfunded_days_per_game=len(poor)/len(rows),
            ready_pack_drop_per_game=e['planner_mean_unassigned_tasks'],
            production_no_effect_per_game=sum(e['own_production']['no_effect']),
            material_gap_conditional_wage_difference=sum(d['conditional_wage_saving'] for d in material)/len(rows),
            source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),stages=stages)
        result.append(entry)
        print(json.dumps({k:v for k,v in entry.items() if k not in ('stages','source_sha256')},ensure_ascii=False),flush=True)
    receipt=dict(status='COMPLETE_DESCRIPTIVE_NOT_CAUSAL_ATTRIBUTION',results=result,
        prior_build=root['build'],source_sha256=hashlib.sha256((src/'summary.json').read_bytes()).hexdigest(),
        caveats=['No new matches; reused development A evidence.',
                 'Resource shortage is not necessarily an economic error.',
                 '100 per missing seed is an upper bound, not its actual price.',
                 'No-effect checks cover recorded operations, not all optimality or legality claims.',
                 'Conditional wage difference is not guaranteed recoverable profit.'])
    (out/'summary.json').write_text(json.dumps(receipt,indent=2),encoding='utf8')


if __name__=='__main__':
    main()
