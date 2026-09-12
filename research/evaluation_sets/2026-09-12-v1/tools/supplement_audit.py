"""Recover declared CLI ranges and integer seed arrays in NPZ archives."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import shutil
import struct
import zipfile
from audit_seeds import dump,sha,constant


def supplement(repo,out):
    report=[];extra=set()
    # Inspect metadata and source only; large completed row files were fully streamed already.
    for p in sorted(repo.rglob('*')):
        if not p.is_file() or '.git' in p.parts or '__pycache__' in p.parts: continue
        seeds=set();methods=[]
        if p.suffix=='.npz':
            with zipfile.ZipFile(p) as z:
                for name in z.namelist():
                    if 'seed' not in name.lower() or not name.endswith('.npy'): continue
                    raw=z.read(name); assert raw[:6]==b'\x93NUMPY'
                    major=raw[6]; offset=10 if major==1 else 12
                    n=int.from_bytes(raw[8:offset],'little');header=ast.literal_eval(raw[offset:offset+n].decode('latin1'))
                    descriptor=header['descr']; assert descriptor in ('<i8','<u8','<i4','<u4','|i1','|u1'),descriptor
                    fmt={'i8':'q','u8':'Q','i4':'i','u4':'I','i1':'b','u1':'B'}[descriptor[1:]]
                    values=[v[0] for v in struct.iter_unpack('<'+fmt,raw[offset+n:])]
                    assert len(values)==__import__('math').prod(header['shape'])
                    seeds.update(v for v in values if 0<=v<2**64);methods.append('npz_seed_array:'+name)
        elif p.suffix in ('.py','.json','.log','.md','.sh','.ps1') and p.stat().st_size<8*1024*1024:
            text=p.read_text(encoding='utf-8-sig',errors='replace')
            # JSON/Python argv lists and plain shell commands, with flags in either order.
            flags=list(re.finditer(r'''--(seed-start|seeds)["'\s,:=]+(\d+)''',text))
            for i,m in enumerate(flags):
                if m[1]!='seed-start':continue
                neighbours=[f for f in flags[max(0,i-2):i+3] if f[1]=='seeds' and abs(f.start()-m.start())<2048]
                if neighbours:
                    f=min(neighbours,key=lambda f:abs(f.start()-m.start()))
                    start,count=int(m[2]),int(f[2])
                    if 0<count<=100000:seeds.update(range(start,start+count));methods.append('nearby_cli_start_count_conservative')
            if p.suffix=='.py':
                try:
                    tree=ast.parse(text)
                    # Literal argv lists: avoids pairing unrelated lines when such a list is available.
                    for node in ast.walk(tree):
                        if isinstance(node,(ast.List,ast.Tuple)):
                            try:values=constant(node,{})
                            except (KeyError,TypeError,ValueError):continue
                            if '--seed-start' in values and '--seeds' in values:
                                try:
                                    start=int(values[values.index('--seed-start')+1]);count=int(values[values.index('--seeds')+1])
                                    if 0<count<=100000:seeds.update(range(start,start+count));methods.append('constant_argv_range')
                                except (ValueError,IndexError,TypeError):pass
                except SyntaxError:pass
        if seeds:
            extra.update(seeds);report.append(dict(path=p.relative_to(repo).as_posix(),sha256=sha(p),seeds=sorted(seeds),methods=sorted(set(methods))))
    path=out/'exclusions.json';summary=out/'summary.json'
    if (out/'initial_summary.json').exists():raise FileExistsError('Supplement already applied')
    shutil.copyfile(summary,out/'initial_summary.json');shutil.copyfile(path,out/'initial_exclusions.json')
    old=json.loads(path.read_text());before=set(old['seeds']);old['seeds']=sorted(before|extra)
    old['supplement']='NPZ integer seed arrays and declared nearby CLI ranges; ambiguous neighbouring ranges conservatively over-excluded.'
    dump(out/'supplement.json',dict(files=report,additional_unique=len(extra-before),seeds=sorted(extra)))
    dump(path,old);s=json.loads(summary.read_text())
    s.update(unique_exclusions=len(old['seeds']),exclusions_in_sampling_domain=sum(x<2**31 for x in old['seeds']),exclusions_sha256=sha(path),
             supplement_sha256=sha(out/'supplement.json'),supplement_additional_unique=len(extra-before),npz_seed_arrays_recovered=True)
    dump(summary,s)
    print(json.dumps(dict(additional_unique=len(extra-before),total_exclusions=len(old['seeds']),npz_seed_arrays_recovered=True)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();supplement(a.repo.resolve(),a.out.resolve())
