from pathlib import Path
import sys,json
from bounded_fix1 import run
w=Path(__file__).resolve().parents[1]
case=sys.argv[1];scenario=sys.argv[2];results=[]
for variant in ('parent','candidate'):
 result=run(case,variant,scenario,672,2609134201,ticks=47,output_group='official_terminal')
 results.append(result);(w/'logs'/f'terminal_{case}_{scenario}.json').write_text(json.dumps(results,indent=2))
a,b=results
print('SUMMARY',case,scenario,'own gain',b['own_cash_end']-a['own_cash_end'],'margin gain',(b['own_cash_end']-b['rival_cash_end'])-(a['own_cash_end']-a['rival_cash_end']),flush=True)
