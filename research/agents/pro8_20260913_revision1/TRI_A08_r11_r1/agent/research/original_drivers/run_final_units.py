from pathlib import Path
import subprocess,json,datetime,os
w=Path('/mnt/data/TRI_A08_work');out=w/'logs/final_units';out.mkdir(exist_ok=True);bins=w/'test_binaries';bins.mkdir(exist_ok=True)
receipts=[]
for name in ['sale_dp_tests','floor_sale_dp_tests','resume_boundary_tests','floor_observation_guard','floor_margin_regression']:
 for sanitize in [False,True] if name=='floor_sale_dp_tests' else [False]:
  stem=name+('_asan_ubsan' if sanitize else '');target=bins/stem
  flags=['-std=c++20','-O1','-g','-fsanitize=address,undefined','-fno-omit-frame-pointer'] if sanitize else ['-std=c++20','-O2','-march=x86-64','-ffp-contract=off']
  cmd=['/usr/bin/time','-v','-o',str(out/(stem+'.compile.time')),'g++',*flags,str(w/'candidate/tests'/(name+'.cpp')),'-o',str(target)]
  with open(out/(stem+'.compile.stdout'),'w') as o,open(out/(stem+'.compile.stderr'),'w') as e:r=subprocess.run(cmd,stdout=o,stderr=e,timeout=120)
  assert r.returncode==0,(stem,'compile',r.returncode)
  cmd2=['/usr/bin/time','-v','-o',str(out/(stem+'.time')),str(target)]
  with open(out/(stem+'.json'),'w') as o,open(out/(stem+'.stderr'),'w') as e:r=subprocess.run(cmd2,stdout=o,stderr=e,timeout=90,env={**os.environ,'ASAN_OPTIONS':'detect_leaks=1:halt_on_error=1','UBSAN_OPTIONS':'halt_on_error=1:print_stacktrace=1'})
  receipt={'test':stem,'compile_command':cmd,'run_command':cmd2,'returncode':r.returncode}
  (out/(stem+'.receipt.json')).write_text(json.dumps(receipt,indent=2));receipts.append(receipt)
  print(json.dumps(receipt),flush=True);assert r.returncode==0,(stem,'run',r.returncode)
(out/'SUMMARY.json').write_text(json.dumps({'passed':True,'tests':receipts},indent=2))
