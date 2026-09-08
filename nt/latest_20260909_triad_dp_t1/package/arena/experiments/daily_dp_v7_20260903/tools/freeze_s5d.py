from pathlib import Path
import argparse,hashlib,json,shutil
E=Path(__file__).resolve().parents[1];p=argparse.ArgumentParser();p.add_argument('stage',choices=['before','after']);p.add_argument('--snapshot');a=p.parse_args()
dst=E/'profiles/s5d';dst.mkdir(exist_ok=True);build=json.loads((E/'native/build/build_receipt.json').read_text())
name=a.snapshot or a.stage;assert name.replace('_','').isalnum()
snapshot=dst/name;snapshot.mkdir(exist_ok=False)
for rel,h in build['source_hashes'].items():
 f=E/rel;assert hashlib.sha256(f.read_bytes()).hexdigest()==h
 target=snapshot/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,target)
(dst/(name+'_build.json')).write_text(json.dumps(build,indent=2))
if a.stage=='before':
 old=json.loads((E/'profiles/s5b/configs.json').read_text());cfg={}
 for base in ['all_intraday_insert','full_chain_autonomous','full_chain_autonomous_timing']:
  for tag,values in [('',{}),('_first_value',dict(compile_consequence=True)),('_replant',dict(compile_replant_choices=True)),('_first_value_replant',dict(compile_consequence=True,compile_replant_choices=True))]:cfg[base+tag]={**old[base],**values}
 (dst/'configs.json').write_text(json.dumps(cfg,indent=2))
print('FROZEN '+a.stage)
