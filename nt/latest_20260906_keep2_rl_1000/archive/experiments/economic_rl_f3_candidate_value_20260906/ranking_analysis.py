"""Read-only relative ranking audit; separate KEEP mass from non-KEEP skill."""
from pathlib import Path
import json
import numpy as np
P=Path(__file__).resolve().parent

def main():
    states=json.loads((P/'STATE_VALUES.json').read_text());out={}
    for cohort in ('train','new'):
        for tail in (0,2):
            rows=[r for r in states if r['cohort']==cohort and r['tail']==tail]
            pairs=[];hits=[];randomhits=[];topvalues=[];uniformvalues=[]
            for row in rows:
                aa=row['alternatives']
                if len(aa)<2:continue
                values=np.array([a['reward_delta']for a in aa]);probs=np.array([a['model_probability']for a in aa])
                if np.ptp(values)<=1e-6:continue
                pp=[]
                for i in range(len(aa)):
                    for j in range(i+1,len(aa)):
                        diff=values[i]-values[j]
                        if abs(diff)<=1e-6:continue
                        dp=probs[i]-probs[j]
                        pp.append(.5 if abs(dp)<=1e-9 else float(dp*diff>0))
                pairs.append(float(np.mean(pp)))
                best=values.max();top=np.abs(probs-probs.max())<=1e-9
                hits.append(float(np.mean(np.abs(values[top]-best)<=1e-6)))
                randomhits.append(float(np.mean(np.abs(values-best)<=1e-6)))
                topvalues.append(float(values[top].mean()));uniformvalues.append(float(values.mean()))
            out[f'{cohort}_tail{tail}']=dict(informative_states=len(pairs),
                pairwise_order_accuracy=float(np.mean(pairs)),chance_pairwise=.5,
                best_nonkeep_hit=float(np.mean(hits)),random_best_hit=float(np.mean(randomhits)),
                top_nonkeep_mean_reward_gain=float(np.mean(topvalues)),uniform_mean_reward_gain=float(np.mean(uniformvalues)))
    (P/'RANKING_ANALYSIS.json').write_text(json.dumps(out,indent=2))
    print(json.dumps(out,indent=2))

if __name__=='__main__':main()
