"""Compare exact saved-observation prefixes, never stale counterfactual futures."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--parent',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();cases=[];n=0
keys=('execution_mode','execution_scope','score','roll_own_cash','roll_rival_cash','tail_value','prediction_gate_checks','prediction_gate_activations','prediction_receipt_events','plan_key','first_prediction_investment_step','first_prediction_market')
for f in sorted(a.parent.glob('*.json')):
 if f.name=='SUMMARY.json':continue
 x=json.loads(f.read_text());y=json.loads((a.candidate/f.name).read_text());limit=y['first_divergence'];old={d['step']:d['debug'] for d in x['daily']};count=0;first=None
 for d in y['daily']:
  step=d['step']
  if limit is not None and step>limit:continue
  pa=old[step]['last_search'];ch=d['debug']['last_search']
  if not pa:continue
  byid={z['id']:z for z in ch['candidates']}
  for z in pa['candidates']:
   assert z['id'] in byid,('missing parent candidate',f.name,step,z['id'])
   for k in keys:assert z[k]==byid[z['id']][k],('changed original',f.name,step,z['id'],k)
   count+=1
  if step==limit:
   best=max(ch['candidates'],key=lambda z:z['score']);original=max(pa['candidates'],key=lambda z:z['score'])
   first={'step':step,'new_best_id':best['id'],'new_land_cap':best.get('land_cap'),'conditional_model_score_difference':best['score']-original['score'],'not_realized_cash':True}
 n+=count;cases.append({'id':f.stem,'first_action_divergence':limit,'original_candidate_scores_and_plans_exact_through_divergence':count,'first_divergence_diagnostic':first,'historical_outcome_not_revalidated':limit is not None})
result={'scope':'Original candidates compared only before and at the first different action. No stale future comparison is used to claim preservation or wins.','new_games':0,'original_candidates_bit_exact':n,'cases':cases}
a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
