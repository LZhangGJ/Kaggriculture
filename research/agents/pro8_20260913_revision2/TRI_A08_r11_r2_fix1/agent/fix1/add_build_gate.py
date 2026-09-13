from pathlib import Path
p=Path('/mnt/data/fix1_work/candidate/build.py');s=p.read_text()
s=s.replace('(parent: exact TRI_A08_r11_r1)', '(parent: exact TRI_A08_r11_r2)')
s=s.replace('import argparse, hashlib, json, os, shutil, subprocess, time','import argparse, datetime, hashlib, json, os, shutil, subprocess, sys, time')
s=s.replace(" if out==ROOT/'reference/P16/policy/startupsupply2.so':raise SystemExit('Refusing to overwrite frozen P16')", " if out.is_relative_to(ROOT/'reference'):raise SystemExit('Refusing to overwrite an immutable ancestor')")
s=s.replace(" cmd=[a.cxx,*flags,str(ROOT/'policy/bridge.cpp'),str(ROOT/'policy/executor/vendor/simulator.cpp'),'-o',str(out)]\n tick=time.perf_counter();subprocess.run(cmd,check=True)", ''' # Compile to a staging file. Never replace a known-good native if compilation
 # OR the exact-native action/value gate fails; rejected artifacts are retained.
 stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
 stage=out.with_name(out.stem+'.building-'+stamp+'.so')
 logdir=ROOT/'build_runs'/stamp;logdir.mkdir(parents=True)
 cmd=[a.cxx,*flags,str(ROOT/'policy/bridge.cpp'),str(ROOT/'policy/executor/vendor/simulator.cpp'),'-o',str(stage)]
 (logdir/'compile.command.json').write_text(json.dumps(cmd,indent=2)+'\\n')
 tick=time.perf_counter()
 with (logdir/'compile.stdout').open('w') as stdout,(logdir/'compile.stderr').open('w') as stderr:
  compiled=subprocess.run(cmd,stdout=stdout,stderr=stderr)
 (logdir/'compile.exit').write_text(str(compiled.returncode)+'\\n')
 if compiled.returncode:raise SystemExit('Compilation failed; prior native preserved; see '+str(logdir))
 check_path=logdir/'native_choice_gate.json'
 check_command=[sys.executable,str(ROOT/'tests/native_choice_gate.py'),'--library',str(stage),'--enabled',str(a.animal_collection_labor),'--out',str(check_path)]
 (logdir/'native_choice_gate.command.json').write_text(json.dumps(check_command,indent=2)+'\\n')
 with (logdir/'native_choice_gate.stdout').open('w') as stdout,(logdir/'native_choice_gate.stderr').open('w') as stderr:
  checked=subprocess.run(check_command,stdout=stdout,stderr=stderr,timeout=45)
 (logdir/'native_choice_gate.exit').write_text(str(checked.returncode)+'\\n')
 if checked.returncode:raise SystemExit('Exact-native choice/value gate FAILED; prior native preserved; rejected staging library and logs retained: '+str(logdir))
 gate=json.loads(check_path.read_text())
 if gate.get('status')!='PASS' or gate.get('native_sha256')!=sha(stage):raise SystemExit('Gate/native identity mismatch; not promoted')
 os.replace(stage,out)
''')
s=s.replace("not x.name.endswith('.BUILD.json')", "not x.name.endswith(('.BUILD.json','.CHECK.json'))")
s=s.replace("'command':cmd,'seconds'", "'command':cmd,'flags':flags,'build_log_directory':str(logdir),'native_choice_gate':gate,'native_choice_gate_command':check_command,'validation_sources':{'tests/native_choice_gate.py':sha(ROOT/'tests/native_choice_gate.py')},'seconds'")
p.write_text(s)
p=Path('/mnt/data/fix1_work/candidate/tests/run_checks.py');s=p.read_text()
s=s.replace("p.add_argument('--saved-limit'", "p.add_argument('--cxx',default=None);p.add_argument('--library',type=pathlib.Path);p.add_argument('--saved-limit'")
s=s.replace("base=[f for f in receipt['command'][1:-5]", "base=[f for f in receipt.get('flags',receipt['command'][1:-5])")
s=s.replace("run(name+'_build',[receipt['command'][0]", "run(name+'_build',[a.cxx or receipt['command'][0]")
s=s.replace(" run('entry_contract'", " run('native_choice_gate',[sys.executable,str(ROOT/'tests/native_choice_gate.py'),'--library',str(a.library or PROD/'policy/tri_a08_r11_r2_fix1.so'),'--out',str(OUT/'native_choice_gate.json')],20)\n run('entry_contract'")
p.write_text(s)
