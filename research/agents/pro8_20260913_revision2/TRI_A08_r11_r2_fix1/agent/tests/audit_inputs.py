"""Recheck exact supplied parent manifest, all result rows and own daily ledgers."""
from __future__ import annotations
import collections,gzip,hashlib,json,pathlib
ROOT=pathlib.Path(__file__).resolve().parents[1];E=ROOT/'input' if (ROOT/'input').exists() else ROOT/'evidence';P=ROOT/'parent_immutable' if (ROOT/'parent_immutable').exists() else ROOT/'reference/parent'
checks=0
def check(b,why):
 global checks
 checks+=1
 if not b:raise AssertionError(why)
m=json.loads((E/'MANIFEST.json').read_text())
for rel,v in m.items():
 p=P/rel.removeprefix('agent/') if rel.startswith('agent/') else E/rel
 check(p.is_file(),rel)
 check(p.stat().st_size==v['bytes'],rel+' size')
 check(hashlib.sha256(p.read_bytes()).hexdigest()==v['sha256'],rel+' sha256')
rows=json.loads((E/'FULL64_ROWS.json').read_text());check(len(rows)==1536,'wrong denominator')
check(len({(x['opponent'],x['seed'],x['seat']) for x in rows})==1536,'duplicate rows')
check(all(x['terminal'] and x['steps']==719 and x['error'] is None for x in rows),'incomplete parent result')
check(sum(x['margin']>0 for x in rows)==1054,'parent wins')
check(sum(x['margin']==0 for x in rows)==0,'parent ties')
check(len({x['seed'] for x in rows})==64,'parent seed count')
sel={x['id']:x for x in json.loads((E/'SELECTED_CASES.json').read_text())};daycount=0
for path in sorted((E/'own_ledgers').glob('*.json')):
 ledger=json.loads(path.read_text());own=ledger['own_player'];flow=collections.Counter();last=None
 check(len(own['days'])==30,'days')
 for d in own['days']:
  daycount+=1
  check(d['cash_start']+sum(d['flow'].values())==d['cash_end'],'cash balance')
  if last is not None:check(last==d['cash_start'],'cash continuity')
  last=d['cash_end']
  for k,v in d['flow'].items():flow[k]+=v
 check(last==own['cash'],'terminal ledger')
 check(dict(flow)==sel[path.stem]['own_totals']['flow'],'sum daily != supplied totals')
 tr=json.loads(gzip.decompress((E/'own_traces'/(path.stem+'.json.gz')).read_bytes()))
 check(len(tr['observations'])==720 and len(tr['own_actions'])==719,'trace length')
 check(tr['observations'][-1]['farms'][ledger['seat']]['money']==own['cash'],'trace terminal cash')
print(json.dumps({'status':'PASS','checks':checks,'manifest_files':len(m),'parent_result_rows':len(rows),'parent_wins':1054,'own_ledger_days':daycount,'new_games':0}))
