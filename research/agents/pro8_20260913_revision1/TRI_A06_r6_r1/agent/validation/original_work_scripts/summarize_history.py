"""Audit all 480 row identities and cash/win denominators (not all replay bytes)."""
from pathlib import Path
import json,hashlib,collections,statistics
R=Path(__file__).resolve().parent;p=R/'input/HISTORICAL_ALL_480_ROWS.json';rows=json.loads(p.read_text());assert len(rows)==480
ids=set();cells=set();opponents=collections.defaultdict(list)
for r in rows:
 assert r['id'] not in ids;ids.add(r['id']);cell=(r['opponent'],r['seed'],r['seat']);assert cell not in cells;cells.add(cell)
 assert r['error'] is None and r['terminal'] and r['steps']==719 and r['status']==['DONE','DONE']
 assert r['candidate_cash']==r['cash'][r['seat']] and r['opponent_cash']==r['cash'][1-r['seat']] and r['cash']==r['rewards']
 assert r['candidate_cash']-r['opponent_cash']==r['margin'];opponents[r['opponent']].append(r)
seeds=sorted({r['seed'] for r in rows});assert len(seeds)==20 and len(opponents)==12
for op,rs in opponents.items():assert {(r['seed'],r['seat']) for r in rs}=={(s,p) for s in seeds for p in (0,1)}
counts={op:{'games':len(rs),'wins':sum(r['margin']>0 for r in rs),'losses':sum(r['margin']<0 for r in rs),'ties':sum(r['margin']==0 for r in rs)} for op,rs in sorted(opponents.items())}
R2='submission_56149565';wins=sum(r['margin']>0 for r in rows);r2=counts[R2];assert wins==374 and r2['wins']==35 and wins-r2['wins']==339
out={'scope':'Historical row-integrity and full-denominator check, not new results; only seven full historical replay files are supplied and separately reproduced.','input_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'rows_checked':len(rows),'unique_seeds':20,'both_seats_per_opponent_seed':True,'by_opponent':counts,'all_wins':wins,'all_games':480,'public_wins':wins-r2['wins'],'public_games':440,'R2_wins':r2['wins'],'R2_games':40,'mean_candidate_cash':statistics.mean(r['candidate_cash'] for r in rows)}
(R/'diagnostics/historical_all480_row_audit.json').write_text(json.dumps(out,indent=2));print(json.dumps(out),flush=True)
