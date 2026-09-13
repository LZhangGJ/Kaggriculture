#!/usr/bin/env python3
"""Verify fail-closed promotion in an isolated copy; no policy modifications.

A test-only copy of the Python checker injects the central feed-bit symptom
into its returned table. This is deliberately NOT a compiler-bug reproduction.
The real production .so and its receipt must remain unchanged after rejection.
"""
from __future__ import annotations
import argparse, hashlib, json, pathlib, shutil, subprocess, sys, time
ROOT=pathlib.Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=pathlib.Path,required=True);ap.add_argument('--cxx',default=None);a=ap.parse_args()
    out=a.out.resolve();out.mkdir(parents=True,exist_ok=False);fixture=out/'fixture';fixture.mkdir()
    receipt=json.loads((ROOT/'policy/tri_a08_r11_r2_fix1.BUILD.json').read_text())
    inputs=list(receipt['sources'])+['policy/tri_a08_r11_r2_fix1.so','policy/tri_a08_r11_r2_fix1.BUILD.json','tests/native_choice_gate.py']
    for rel in inputs:
        dest=fixture/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/rel,dest)
    gate=fixture/'tests/native_choice_gate.py';text=gate.read_text()
    needle='rich=load(9,21,28,px,4.);want='
    if text.count(needle)!=1:raise RuntimeError('Cannot locate unique test-injection point')
    gate.write_text(text.replace(needle,'rich=load(9,21,28,px,4.);rich[((28*2+1)*6)*4+2]=0.;want='))
    (out/'INJECTION.txt').write_text('TEST-ONLY CHECKER COPY: mutate rich feed to 0 before its exact assertion. Does not alter production source or native. Not a GCC13 reproduction.\n')
    native=fixture/'policy/tri_a08_r11_r2_fix1.so';record=native.with_suffix('.BUILD.json');before=(sha(native),sha(record));start=time.perf_counter()
    cmd=[sys.executable,str(fixture/'build.py')]
    if a.cxx:cmd+=['--cxx',a.cxx]
    (out/'command.json').write_text(json.dumps(cmd,indent=2)+'\n')
    with (out/'build.stdout').open('w') as stdout,(out/'build.stderr').open('w') as stderr:
        proc=subprocess.run(['timeout','-k','5s','100s',*cmd],stdout=stdout,stderr=stderr)
    after=(sha(native),sha(record));gates=sorted((fixture/'build_runs').glob('*/native_choice_gate.json'));stages=list((fixture/'policy').glob('*.building-*.so'))
    gate_result=json.loads(gates[-1].read_text()) if gates else {}
    passed=proc.returncode==1 and before==after and gate_result.get('status')=='FAIL' and gate_result.get('error')=='exact positive maintenance assertion' and bool(stages)
    result={'status':'PASS' if passed else 'FAIL','scope':'Isolated expected rejection; not a compiler failure reproduction','build_exit':proc.returncode,'native_and_receipt_preserved':before==after,'before_native_sha256':before[0],'after_native_sha256':after[0],'before_receipt_sha256':before[1],'after_receipt_sha256':after[1],'gate':gate_result,'staging_preserved':[str(p) for p in stages],'production_gate_sha256':sha(ROOT/'tests/native_choice_gate.py'),'test_only_mutated_gate_sha256':sha(gate),'seconds':time.perf_counter()-start,'new_games':0}
    (out/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True);raise SystemExit(not passed)
if __name__=='__main__':main()
