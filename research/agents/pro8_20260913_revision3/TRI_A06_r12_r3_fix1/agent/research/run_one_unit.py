from pathlib import Path
import json,subprocess,time,hashlib
w=Path(__file__).resolve().parents[1];r=w/'candidate';d=r/'build/tests';name='unit_terminal_supply';cxx='/usr/bin/g++'
flags=[x for x in json.loads((r/'COMPILER_FLAGS.json').read_text()) if x not in ('-O3','-DNDEBUG','-fPIC','-shared','-Wl,-Bsymbolic')]+['-O1']
cmd=[cxx,*flags,str(r/'tests'/f'{name}.cpp'),str(r/'policy/executor/vendor/simulator.cpp'),'-o',str(d/name)];t=time.monotonic()
p=subprocess.run(cmd,capture_output=True,text=True,timeout=45);(d/f'{name}.compile.stdout').write_text(p.stdout);(d/f'{name}.compile.stderr').write_text(p.stderr);p.check_returncode()
p=subprocess.run([str(d/name)],capture_output=True,text=True,timeout=20);(d/f'{name}.stdout').write_text(p.stdout);(d/f'{name}.stderr').write_text(p.stderr)
rec={'name':name,'command':cmd,'compiler':subprocess.getoutput(cxx+' --version').splitlines()[0],'returncode':p.returncode,'elapsed_seconds':time.monotonic()-t,'executable_sha256':hashlib.sha256((d/name).read_bytes()).hexdigest()}
rows=json.loads((d/'receipt.json').read_text());rows=[x for x in rows if x['name']!=name]+[rec];(d/'receipt.json').write_text(json.dumps(rows,indent=2));print(p.stdout,p.stderr,flush=True);p.check_returncode()
