from pathlib import Path
import argparse,hashlib,json,subprocess,time,shutil
E=Path(__file__).resolve().parents[1];cli=argparse.ArgumentParser();cli.add_argument('--version',required=True);a=cli.parse_args()
out=E/('receipts/s5b_mechanisms_'+a.version);out.mkdir(exist_ok=False)
sources=['test_rotation_portfolio.cpp','test_rotation_integration.cpp','test_harvest_rotation.cpp']
snapshot=out/'source/native';snapshot.mkdir(parents=True)
for p in [*sorted((E/'native').glob('*.hpp')),*[E/'native'/n for n in sources]]:shutil.copy2(p,snapshot/p.name)
result=dict(status='RUNNING',rows=[]);start=time.perf_counter()
for name in sources:
 src=E/'native'/name;binary=out/src.stem
 command=['g++','-std=c++20','-O2','-I'+str(E/'native'),str(src),str(E/'native/vendor/simulator.cpp'),'-o',str(binary)]
 t=time.perf_counter();r=subprocess.run(command,capture_output=True,text=True);(out/(name+'.compile.log')).write_text(r.stdout+r.stderr)
 row=dict(source=name,sha256=hashlib.sha256(src.read_bytes()).hexdigest(),compile_returncode=r.returncode,compile_seconds=time.perf_counter()-t,command=command)
 if r.returncode==0:
  t=time.perf_counter();test=subprocess.run([str(binary)],capture_output=True,text=True);(out/(name+'.test.log')).write_text(test.stdout+test.stderr)
  row.update(test_returncode=test.returncode,test_seconds=time.perf_counter()-t,stdout=test.stdout,stderr=test.stderr)
 result['rows'].append(row);print(json.dumps({k:v for k,v in row.items() if k!='command'}),flush=True)
 if r.returncode or row.get('test_returncode'):
  result['status']='FAILED_PRESERVED';break
else:result['status']='PASS_MECHANISMS'
result['seconds']=time.perf_counter()-start
result['source_hashes']={str(p.relative_to(E)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((E/'native').glob('*.hpp'))}
(out/'acceptance.json').write_text(json.dumps(result,indent=2));raise SystemExit(0 if result['status']=='PASS_MECHANISMS' else 1)
