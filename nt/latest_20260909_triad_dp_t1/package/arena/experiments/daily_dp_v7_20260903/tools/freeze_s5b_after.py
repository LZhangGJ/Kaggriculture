from pathlib import Path
import argparse,hashlib,json,shutil
E=Path(__file__).resolve().parents[1];profile=E/'profiles/s5b'
cli=argparse.ArgumentParser();cli.add_argument('--version',default='');args=cli.parse_args();suffix=('_'+args.version) if args.version else ''
old=json.loads((E/'profiles/s4v/configs.json').read_text());cfg={}
for base in ('all_intraday_insert','full_chain_autonomous'):
 cfg[base]=old[base];cfg[base+'_rotation']=dict(old[base],rotate_finite=True)
 cfg[base+'_net']=dict(old[base],portfolio_rotation=True)
 cfg[base+'_timing']=dict(old[base],portfolio_rotation=True,rotation_timing=True)
(profile/'configs.json').write_text(json.dumps(cfg,indent=2))
build=json.loads((E/'native/build/build_receipt.json').read_text())
for rel,h in build['source_hashes'].items():
 assert hashlib.sha256((E/rel).read_bytes()).hexdigest()==h
 target=profile/('after'+suffix)/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(E/rel,target)
(profile/('after_build'+suffix+'.json')).write_text(json.dumps(build,indent=2));print('FROZEN_S5B_AFTER'+suffix)
