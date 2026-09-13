import pathlib,json,subprocess,argparse,time
p=argparse.ArgumentParser();p.add_argument('--enabled',type=int,choices=[0,1],default=1);p.add_argument('--sanitizer',action='store_true');a=p.parse_args()
root=pathlib.Path(__file__).resolve().parents[1];prod=root/'candidate' if (root/'candidate').exists() else root
receipt=json.loads((prod/'policy/tri_a08_r11_r2_fix1.BUILD.json').read_text());flags=receipt['command'][1:-5]
flags=[x for x in flags if x not in ('-shared','-fPIC','-Wl,-Bsymbolic') and not x.startswith('-DA08_ANIMAL_COLLECTION_LABOR=')]
# Derive all ancestor feature switches from the real production build.
flags += [f'-DA08_ANIMAL_COLLECTION_LABOR={a.enabled}','-I',str(prod)]
name=f'unit_{a.enabled}'+('_ubsan' if a.sanitizer else '')
if a.sanitizer:flags=[x for x in flags if x not in ('-O3','-DNDEBUG')]+['-O1','-g','-fsanitize=undefined','-fno-sanitize-recover=all']
out=root/'probes'/name;out.parent.mkdir(exist_ok=True);cmd=['g++',*flags,str(root/'tests/animal_collection_test.cpp'),str(prod/'policy/executor/vendor/simulator.cpp'),'-o',str(out)]
(root/'logs'/f'{name}.command.json').write_text(json.dumps(cmd,indent=2))
with (root/'logs'/f'{name}.build.stdout').open('w') as o,(root/'logs'/f'{name}.build.stderr').open('w') as e:
 r=subprocess.run(['/usr/bin/time','-v','-o',str(root/'logs'/f'{name}.build.time'),'timeout','-k','5s','40s',*cmd],stdout=o,stderr=e)
print('compile',name,r.returncode);print((root/'logs'/f'{name}.build.stderr').read_text());raise SystemExit(r.returncode)
