from pathlib import Path
import json,gzip,subprocess,statistics
HERE=Path(__file__).resolve().parent;OUT=HERE/'candidate_r2p7';exe=OUT/'audit_market_sequence'
ITEMS=('WHEAT','CARROT','TOMATO','STRAWBERRY','MELON','EGG','MILK','WOOL','FERTILIZER')
def main():
 subprocess.run(['g++-13','-std=c++20','-O2',str(OUT/'audit_market_sequence.cpp'),'-o',str(exe)],check=True)
 proc=subprocess.Popen([str(exe)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,bufsize=1)
 records=[];by=[0]*9
 try:
  for p in sorted((HERE/'forecast_audit').glob('*.json.gz')):
   a=json.loads(gzip.decompress(p.read_bytes()));rows=[]
   for f in a['forecasts']:
    inp=[f['day'],f['supply'],f['competition']]+[f['market']['inventory'][i] for i in ITEMS]
    for key in ['demand','own','rival']:inp.extend(x for r in f[key] for x in r)
    proc.stdin.write(' '.join(map(str,inp))+'\n');proc.stdin.flush()
    vals=proc.stdout.readline().split();assert len(vals)==12,vals
    row=dict(day=f['day'],crosses=int(vals[0]),first_forecast_day=int(vals[1]),conditional_value_delta=float(vals[2]),products=list(map(int,vals[3:])))
    rows.append(row)
    for i,n in enumerate(row['products']):by[i]+=n
   records.append(dict(case=p.stem,actual_win=a['row']['r2_win'],forecasts=rows))
 finally:
  proc.stdin.close();assert proc.wait()==0
 allrows=[r for case in records for r in case['forecasts']]
 summary=dict(cases=len(records),affected_cases=sum(any(r['crosses'] for r in c['forecasts']) for c in records),forecasts=len(allrows),
              affected_forecasts=sum(r['crosses']>0 for r in allrows),crosses=sum(by),by_product=dict(zip(ITEMS,by)),
              median_undiscounted_conditional_score_delta=statistics.median(r['conditional_value_delta'] for r in allrows),
              boundary='These are model-replayed conditional forecasts, not observed sales or causal win improvements. Prefix price-integral correction included in value delta; no fixed costs/labor/discount.')
 (OUT/'FORECAST_FLOOR_AUDIT.json').write_text(json.dumps(dict(summary=summary,records=records),indent=2))
 print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
