from pathlib import Path
import subprocess,json,hashlib,sys,time
r=Path('/mnt/data/fix1_work');p=r/'guard_fixture';out=p/'policy/tri_a08_r11_r2_fix1.so';sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest();before=sha(out)
t=time.monotonic();cmd=[sys.executable,str(p/'build.py')]
with (r/'logs/transaction_build.stdout').open('w') as o,(r/'logs/transaction_build.stderr').open('w') as e:done=subprocess.run(cmd,stdout=o,stderr=e)
after=sha(out);gates=list((p/'build_runs').glob('*/native_choice_gate.json'));gate=json.loads(gates[-1].read_text());staging=list((p/'policy').glob('*.building-*.so'))
result={'status':'PASS' if done.returncode!=0 and before==after and gate['status']=='FAIL' and staging else 'FAIL','scope':'Expected validation failure injection; not a compiler reproduction','command':cmd,'build_exit':done.returncode,'previous_native_preserved':before==after,'before_sha256':before,'after_sha256':after,'rejected_staging':[str(f) for f in staging],'gate':gate,'seconds':time.monotonic()-t,'new_games':0};(r/'repro/build_transaction.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True);raise SystemExit(result['status']!='PASS')
