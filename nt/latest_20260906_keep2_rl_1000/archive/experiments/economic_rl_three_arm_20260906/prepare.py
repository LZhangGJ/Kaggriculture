"""Copy frozen sources; only generated bridge/config files are new runtime code."""
from pathlib import Path
import hashlib,json,shutil,re
P=Path(__file__).resolve().parent;ROOT=P.parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
sources={
 'c3auto':ROOT/'experiments/daily_dp_c3_autonomous0_20260905/stage',
 'c3j7':ROOT/'experiments/daily_dp_c3_review_20260905/stage',
 'f3':ROOT/'experiments/daily_dp_f5_c2_transfer_20260905/cpp',
 'arena':ROOT/'experiments/daily_dp_v7_20260903/native'}
manifest={}
for name,src in sources.items():
 dst=P/'stage'/name;dst.mkdir(parents=True,exist_ok=True)
 for f in src.rglob('*'):
  rel=f.relative_to(src)
  if not f.is_file() or f.suffix not in ('.hpp','.cpp','.inc','.json','.zlib'):continue
  if any(x in ('build','build_history','__pycache__') or x.endswith('_build') or '_build_' in x for x in rel.parts):continue
  out=dst/rel;out.parent.mkdir(parents=True,exist_ok=True)
  if out.exists():assert sha(out)==sha(f),str(out)
  else:shutil.copy2(f,out)
  manifest[str(f)]=sha(f)
 for f in src.rglob('AGENTS.md'):
  raise RuntimeError(f'Read additional instructions before proceeding: {f}')
for arm in ('c3auto','c3j7'):
 cfg=json.loads((P/'stage'/arm/'config.json').read_text())
 text='#pragma once\ninline competitive::Config frozen_config(){competitive::Config c;\n'
 text+=''.join(f'c.{k}={json.dumps(v)};\n' for k,v in cfg.items())
 text+='return c;}\n'
 (P/f'{arm}_config.hpp').write_text(text)
f=ROOT/'experiments/daily_dp_f5_c2_transfer_20260905/candidates/F3_A0/defaults.json'
manifest[str(f)]=sha(f);shutil.copy2(f,P/'f3_config.json')
(P/'f3_config.hpp').write_text('#pragma once\nconstexpr const char*f3_config=R"CFG('+f.read_text()+')CFG";\n')
reg=ROOT/'experiments/daily_dp_v7_20260903/opponents/registry.json'
data=json.loads(reg.read_text());manifest[str(reg)]=sha(reg)
for n in ('g001','g003','boatlee_v29','kaito_v58','lynn_v5','yhay81_six_day','yhay81_three_day'):
 for k in ('source','asset'):
  if k in data['opponents'][n]:
   f=reg.parent.parent/data['opponents'][n][k];manifest[str(f.resolve())]=sha(f)
(P/'SOURCE_FREEZE.json').write_text(json.dumps(manifest,indent=2))
print('Frozen',len(manifest),'files',flush=True)
