from pathlib import Path
import argparse,hashlib,json,shutil
p=argparse.ArgumentParser();p.add_argument('--name',default='s5d_bounded');a=p.parse_args();assert a.name.replace('_','').isalnum()
E=Path(__file__).resolve().parents[1];out=E/'profiles'/a.name;out.mkdir(exist_ok=False)
build=json.loads((E/'native/build/build_receipt.json').read_text())
for rel,h in build['source_hashes'].items():
 src=E/rel;assert hashlib.sha256(src.read_bytes()).hexdigest()==h
 dst=out/'source'/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
(out/'build.json').write_text(json.dumps(build,indent=2))
cfg=json.loads((E/'profiles/s5d/configs.json').read_text())
for label,c in cfg.items():
 if c.get('compile_consequence') or c.get('compile_replant_choices'):c['compile_bounded_rollout']=True
(out/'configs.json').write_text(json.dumps(cfg,indent=2))
print('FROZEN_BOUNDED_NOT_PROMOTED')
