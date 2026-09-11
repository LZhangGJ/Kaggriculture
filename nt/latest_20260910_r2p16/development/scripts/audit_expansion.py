"""Read-only landed-capacity use audit; no economic causal claim from unused land."""
from pathlib import Path
from collections import Counter
import gzip,json,statistics
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
def load(p):return json.loads(gzip.decompress(p.read_bytes()))
def quadrant(pos):return (pos[0]>=5)+2*(pos[1]>=5)
def main():
    rows=[]
    for p in sorted((HERE/'baseline_audit').glob('*/AUDIT.json.gz')):
        a=load(p);r=a['source_row'];seat=1-r['opponent_seat']
        purchases=[x for x in a['transactions'] if x['seat']==seat and x['op']=='BUY_LAND']
        for ix,tx in enumerate(purchases):
            q=ix+1;t=tx['step'];events=[x for x in a['unit_events'] if x['seat']==seat and x['step']>=t and x['effect'] and x['pos'] and quadrant(x['pos'])==q]
            production=[x for x in events if x['action'][0] in ('PLANT','PLACE')]
            visits=[x for x in events if not x['action'][0].startswith('BUILD_')]
            used_days=Counter(x['step']//24 for x in production)
            rows.append(dict(opponent=r['opponent'],seed=r['seed'],opponent_seat=r['opponent_seat'],win=r['r2_win'],quadrant=q,
                             buy_step=t,cost=-tx['cash_delta'],actions=Counter(x['action'][0] for x in events),
                             first_production_step=production[0]['step'] if production else None,
                             first_visit_step=visits[0]['step'] if visits else None,
                             production_events=len(production),productive_days=len(used_days)))
    summary={}
    for label,group in [('loss',[r for r in rows if not r['win']]),('win_controls',[r for r in rows if r['win']])]:
        delays=[r['first_production_step']-r['buy_step'] for r in group if r['first_production_step'] is not None]
        summary[label]=dict(purchases=len(group),never_produced=sum(r['first_production_step'] is None for r in group),
                            mean_delay=statistics.mean(delays) if delays else None,median_delay=statistics.median(delays) if delays else None,
                            delayed_over_day=sum(d>=24 for d in delays),cost=sum(r['cost'] for r in group))
    out=HERE/'expansion_audit';out.mkdir(exist_ok=True)
    (out/'RESULTS.json').write_text(json.dumps(dict(summary=summary,rows=rows),indent=2))
    print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
