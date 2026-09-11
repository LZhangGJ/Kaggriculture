"""Summarize finite one-day alternatives without calling them a global oracle."""
from pathlib import Path
from collections import Counter,defaultdict
import argparse,gzip,hashlib,json,math,statistics as st
EXP=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):
    with gzip.open(p,'rt',encoding='utf8') as f:return json.load(f)
def choose_payoff(xs):return max(xs,key=lambda x:(x['cash']>x['opponent_cash'],x['cash']-x['opponent_cash'],x['cash']))
def corr(a,b):
    if len(a)<2:return None
    ma,mb=st.fmean(a),st.fmean(b);va=sum((x-ma)**2 for x in a);vb=sum((x-mb)**2 for x in b)
    return sum((x-ma)*(y-mb) for x,y in zip(a,b))/math.sqrt(va*vb) if va>0 and vb>0 else None
def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    src=EXP/a.input;out=EXP/a.out;out.mkdir(exist_ok=False);meta=json.loads((src/'acceptance.json').read_text())
    files=sorted(src.glob('*.json.gz'));hashes={x.name:sha(x) for x in files};summaries=[];all_rows=[]
    for path in files:
        opponent=path.stem.split('.')[0];rows=load(path);per=[];pair_score=[];pair_margin=[];family=Counter();counts=Counter()
        for row in rows:
            nodes=[]
            for node in row['nodes']:
                choices=node['choices'];keep=choices[0];assert keep['family']=='KEEP' and keep['cash']==row['cash'] and keep['opponent_cash']==row['opponent_cash']
                best_cash=max(choices,key=lambda x:x['cash']);best_payoff=choose_payoff(choices);pred=max(choices,key=lambda x:x['score'])
                unique=len({(x['cash'],x['opponent_cash'],x['suffix_hash_OFFLINE_ONLY']) for x in choices})
                for x in choices[1:]:
                    ds=x['score']-keep['score'];dm=(x['cash']-x['opponent_cash'])-(keep['cash']-keep['opponent_cash'])
                    if dm!=0:pair_score.append(ds);pair_margin.append(dm);counts['direction_correct']+=(ds>0)==(dm>0);counts['direction_total']+=1
                family[best_payoff['family']]+=1;nodes.append(dict(day=node['day'],candidates=len(choices),unique_results=unique,
                    keep_win=keep['cash']>keep['opponent_cash'],predicted_win=pred['cash']>pred['opponent_cash'],
                    cash_best_win=best_cash['cash']>best_cash['opponent_cash'],payoff_best_win=best_payoff['cash']>best_payoff['opponent_cash'],
                    keep_margin=keep['cash']-keep['opponent_cash'],predicted_margin=pred['cash']-pred['opponent_cash'],
                    cash_best_margin=best_cash['cash']-best_cash['opponent_cash'],payoff_best_margin=best_payoff['cash']-best_payoff['opponent_cash'],
                    predicted_family=pred['family'],best_family=best_payoff['family']))
            flat=[x for n in row['nodes'] for x in n['choices']];one=choose_payoff(flat)
            per.append(dict(seed=row['seed'],seat=row['seat'],keep_win=row['cash']>row['opponent_cash'],
                one_change_win=one['cash']>one['opponent_cash'],keep_margin=row['cash']-row['opponent_cash'],
                one_change_margin=one['cash']-one['opponent_cash'],one_change_family=one['family'],nodes=nodes))
        flatnodes=[n for r in per for n in r['nodes']];entry=dict(opponent=opponent,games=len(per),nodes=len(flatnodes),
            keep_wins=sum(r['keep_win'] for r in per),finite_one_change_wins=sum(r['one_change_win'] for r in per),
            predicted_node_win_rate=st.fmean(n['predicted_win'] for n in flatnodes),finite_node_payoff_best_win_rate=st.fmean(n['payoff_best_win'] for n in flatnodes),
            mean_candidates=st.fmean(n['candidates'] for n in flatnodes),mean_behaviorally_distinct=st.fmean(n['unique_results'] for n in flatnodes),
            score_margin_pearson=corr(pair_score,pair_margin),direction_accuracy=counts['direction_correct']/counts['direction_total'] if counts['direction_total'] else None,
            best_family_counts=dict(family),records=per)
        summaries.append(entry);print(json.dumps({k:v for k,v in entry.items() if k!='records'}),flush=True)
    result=dict(status='COMPLETE_FINITE_ONE_DAY_BRANCH_SUMMARY_NOT_DEPLOYABLE_ORACLE',source_status=meta['status'],input_hashes=hashes,
        build=meta['build'],summary=summaries,final_holdout_used=False,full_goal_complete=False,
        caveat='Best result uses true suffix only as an offline label. It changes one day and returns to baseline. Correlated nodes/candidates are not independent; absence is not proof no winning route exists.')
    (out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf8')
if __name__=='__main__':main()
