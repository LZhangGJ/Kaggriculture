from pathlib import Path
import gzip,json,hashlib
b=Path('/mnt/data/r3_work');fs=[]
for c in json.loads((b/'feedback/SELECTED_CASES.json').read_text()):
 t=json.loads(gzip.decompress((b/'feedback'/c['trace']).read_bytes()));audit=json.loads(gzip.decompress((b/'logs'/('audit_'+c['id']+'.json.gz')).read_bytes()))
 for r in audit:
  if r.get('phase')!=3 or r.get('forecast',0)<99:continue
  s=r['step'];fs.append({'case':c['id'],'step':s,'observation':t['observations'][s],'plans':[u['raw'] for u in r['units']],'parent_forecast':r['forecast']})
p=b/'candidate/tests/fixtures';p.mkdir(exist_ok=True);(p/'delivery_snapshots.json.gz').write_bytes(gzip.compress(json.dumps(fs,separators=(',',':')).encode(),mtime=0));print(len(fs))
