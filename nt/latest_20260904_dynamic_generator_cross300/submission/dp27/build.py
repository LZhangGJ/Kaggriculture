"""Freeze the selected plan and original policy, then build a portable C ABI."""
from pathlib import Path
import hashlib,json,re,shutil,subprocess,time
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
EXP=ROOT/'experiments/daily_dp_v7_20260903'
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def literal(x):
    if isinstance(x,bool):return str(x).lower()
    if isinstance(x,(int,float)):return repr(x)
    if isinstance(x,list):return '{'+','.join(literal(v) for v in x)+'}'
    raise TypeError(x)

def main():
    if (HERE/'build_receipt.json').exists() and read(HERE/'build_receipt.json')['status']!='PASS':
        failed=HERE/'build_failures'/sha(HERE/'build_receipt.json');failed.mkdir(parents=True,exist_ok=True)
        for name in ('build_receipt.json','build.log'):
            if not (failed/name).exists():shutil.copy2(HERE/name,failed/name)
    if (HERE/'agent.so').exists():
        prior=HERE/'build_history'/sha(HERE/'agent.so');prior.mkdir(parents=True,exist_ok=True)
        for name in ('agent.so','build_receipt.json','build.log','validation_smoke.json'):
            path=HERE/name
            if path.exists() and not (prior/name).exists():shutil.copy2(path,prior/name)
    accepted=read(EXP/'search29_cross300/receipts_v1/acceptance.json')
    assert accepted['status']=='COMPLETE_CROSS300_NOT_PROMOTION'
    source=EXP/'search29_cross300/receipts_v1/ranked300.json'
    assert sha(source)==accepted['ranked_sha256']
    selected=next(r for r in read(source) if r['source']=='R03_F001_E104234696_P0' and r['index']==27)
    config=read(EXP/'search29_cross300/receipts_v1/inputs.json')['config']
    build=read(EXP/'native/build/build_receipt.json');frozen=HERE/'source'
    for rel,digest in build['source_hashes'].items():
        original=EXP/rel;assert sha(original)==digest
        dest=frozen/rel;dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists():assert sha(dest)==digest
        else:shutil.copy2(original,dest)
    plan=selected['plan'];assert len(plan)==29
    header=['#pragma once','inline dp7::Params frozen_params(){dp7::Params p;']
    for key,val in config.items():
        assert re.fullmatch('[a-zA-Z_][a-zA-Z_0-9]*',key)
        if key=='opening_animals':val=[{'GOOSE':9,'COW':10,'SHEEP':11}[x] for x in val]
        header.append(f'p.{key}={literal(val)};')
    header+=['return p;}','struct FrozenRecipe{const char*family;int kind,amount;};','inline const FrozenRecipe frozen_plan[29]={']
    header += ['{'+json.dumps(p['family'])+f",{p['kind']},{p['amount']}"+'},' for p in plan]
    header+=['};'];(HERE/'frozen.hpp').write_text('\n'.join(header)+'\n')
    (HERE/'frozen_plan.json').write_text(json.dumps(selected,ensure_ascii=False,indent=2))
    (HERE/'frozen_config.json').write_text(json.dumps(config,indent=2))
    cmd=['g++','-std=c++20','-O3','-DNDEBUG','-march=x86-64','-fPIC','-shared','-static-libstdc++','-static-libgcc','-Wl,-Bsymbolic',
        '-I'+str(frozen/'native'),'-I'+str(HERE),str(HERE/'bridge.cpp'),str(frozen/'native/vendor/simulator.cpp'),'-o',str(HERE/'agent.so')]
    tc=HERE/'toolchain/root/usr'
    cmd[0]=str(tc/'bin/g++-11')
    cmd[1:1]=['-nostdinc++','-isystem',str(tc/'include/c++/11'),'-isystem',str(tc/'include/x86_64-linux-gnu/c++/11'),
        '-L'+str(tc/'lib/gcc/x86_64-linux-gnu/11'),'-B'+str(tc/'lib/gcc/x86_64-linux-gnu/11')+'/']
    start=time.perf_counter();proc=subprocess.run(cmd,text=True,capture_output=True)
    (HERE/'build.log').write_text(proc.stdout+proc.stderr)
    receipt=dict(status='PASS' if proc.returncode==0 else 'FAIL_PRESERVED',returncode=proc.returncode,seconds=time.perf_counter()-start,
        command=cmd,source_hashes=build['source_hashes'],plan_id=selected['plan_id'],plan_sha256=sha(HERE/'frozen_plan.json'),
        bridge_sha256=sha(HERE/'bridge.cpp'),entry_sha256=sha(HERE/'main.py'),binary_sha256=sha(HERE/'agent.so') if proc.returncode==0 else None,
        toolchain_packages={p.name:sha(p) for p in (HERE/'toolchain').glob('*.deb')},
        scope='Fixed daily semantic interventions; original state-dependent execution; no replay playback or opponent-ID input.')
    (HERE/'build_receipt.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps(dict(status=receipt['status'],seconds=receipt['seconds'],plan_id=receipt['plan_id'],error=proc.stderr[-4000:])),flush=True)
    if proc.returncode:raise RuntimeError('Build failed; see build.log')

if __name__=='__main__':main()
