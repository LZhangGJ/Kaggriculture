"""Rebuild on Linux x86-64, preferably Ubuntu 22.04 with GCC 11."""
from pathlib import Path
import hashlib,json,os,shutil,subprocess,tarfile
H=Path(__file__).resolve().parent
if __name__=='__main__':
    cxx=os.environ.get('CXX','g++-11')
    target=subprocess.check_output([cxx,'-dumpmachine'],text=True).strip()
    if 'x86_64' not in target:raise SystemExit('Use an x86-64 compiler; do not submit an ARM library')
    m=json.loads((H/'MANIFEST.json').read_text());src=H/'source/policy'
    for name,d in m['source_hashes'].items():assert hashlib.sha256((src/name).read_bytes()).hexdigest()==d,name
    stage=H/'build';runtime=stage/'afs_runtime';runtime.mkdir(parents=True,exist_ok=True)
    shutil.copy2(H/'main.py',stage/'main.py');shutil.copy2(src/'agent.py',runtime/'policy.py');shutil.copy2(src/'config.json',runtime/'config.json');(runtime/'__init__.py').write_text('')
    command=[cxx]+json.loads((H/'BUILD_FLAGS.json').read_text())+[str(src/'bridge.cpp'),str(src/'executor/vendor/simulator.cpp'),'-o',str(runtime/'agent.so')]
    subprocess.run(command,check=True)
    elf=(runtime/'agent.so').read_bytes();assert elf[:6]==b'\x7fELF\x02\x01' and int.from_bytes(elf[18:20],'little')==62
    output=H/'rebuilt_submission.tar.gz'
    with tarfile.open(output,'w:gz') as t:
        for name in m['package_files']:t.add(stage/name,arcname=name)
    (H/'REBUILD.json').write_text(json.dumps(dict(command=command,compiler=subprocess.check_output([cxx,'--version'],text=True).splitlines()[0],sha256=hashlib.sha256(output.read_bytes()).hexdigest()),indent=2))
    print(output)
