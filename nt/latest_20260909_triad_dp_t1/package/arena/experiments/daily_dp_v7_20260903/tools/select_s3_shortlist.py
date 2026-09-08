"""Preserve a small Pareto-informed shortlist before looking at new seeds."""
import json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
rows=json.loads((root/'receipts/s3_opening_screen/results.json').read_text())['summaries']
ranked=sorted(rows,key=lambda s:(s['g001']['wins'],s['g001']['mean_margin']),reverse=True)
qualified=sorted([s for s in rows if s['pass']['mean_cash']>=180000],key=lambda s:(s['g001']['wins'],s['g001']['mean_margin']),reverse=True)
chosen={s['label']:s['config'] for s in ranked[:4]+qualified[:4]}
chosen['old_opening_control']={'opponent_supply_weight':.75,'competitive_sell_weight':.5,'sell_horizon_days':1}
path=root/'s3_shortlist_configs.json'
if path.exists():raise FileExistsError(path)
path.write_text(json.dumps(chosen,indent=2),encoding='utf8');print(list(chosen))
