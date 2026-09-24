"""Transplant the existing six-day terminal selector as an optional candidate."""
from pathlib import Path
import difflib
import json
import shutil

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


def main():
    ident='cf_liq_terminal6'
    parent=HERE/'candidates/cf_liq_nointraday';out=HERE/'candidates'/ident
    shutil.copytree(parent,out,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    before={p.relative_to(out).as_posix():p.read_text() for p in (out/'policy').rglob('*') if p.suffix in ('.hpp','.inc','.py')}
    triad=out/'policy/triad.hpp';text=triad.read_text()
    assert text.count(' double r14_fert_floor=0;')==1
    triad.write_text(text.replace(' double r14_fert_floor=0;',' double r14_fert_floor=0;\n double r14_terminal_mode=0;'))
    codec=out/'policy/agent.py';text=codec.read_text()
    codec.write_text(text.replace('_ORDER = tuple(DEFAULTS)','DEFAULTS.update(r14_terminal_mode=0)\n_ORDER = tuple(DEFAULTS)'))
    source=ROOT/'experiments/a06_r12_gptpro_round_robin_20260924/agents/r14_frozen/policy/r14_terminal.inc'
    inc=source.read_text().replace('base.r13_clock>=1?&sale_clock:nullptr','nullptr')
    (out/'policy/r14_terminal.inc').write_text(inc)
    search=out/'policy/search.hpp';text=search.read_text()
    needle=' void choose(const View&o){';assert text.count(needle)==1
    text=text.replace(needle,' #include "r14_terminal.inc"\n'+needle)
    needle='  if(base.scenario!=0&&o.day!=live.core.day)choose(o);';assert text.count(needle)==1
    text=text.replace(needle,'  if(base.r14_terminal_mode>0&&o.day>=24&&o.day!=live.core.day)terminal_choose(o);\n  else if(base.scenario!=0&&o.day!=live.core.day)choose(o);')
    search.write_text(text)
    config=out/'policy/config.json';cfg=json.loads(config.read_text());cfg['r14_terminal_mode']=1
    config.write_text(json.dumps(cfg,indent=2)+'\n')
    diff=[]
    for path,old in before.items():
        new=(out/path).read_text()
        if old!=new:diff.extend(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='parent/'+path,tofile=ident+'/'+path))
    (out/'SOURCE_DIFF.patch').write_text(''.join(diff))
    manifest=HERE/'CANDIDATES.json';rows=json.loads(manifest.read_text())
    rows.append(dict(id=ident,parent='cf_liq_nointraday',ported_module='r14_frozen/policy/r14_terminal.inc',
        changes='Day24+ seven terminal-cash rule alternatives; public-flow assumptions; daily state feedback retained.'))
    manifest.write_text(json.dumps(rows,indent=2)+'\n')
    print(ident,'source prepared; native library must be rebuilt before evaluation')


if __name__=='__main__':main()
