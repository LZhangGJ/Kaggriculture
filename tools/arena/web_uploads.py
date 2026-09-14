"""Treat uploads as data; execution remains in the existing Docker sandbox."""
import json
import re
import zipfile
import tarfile
from pathlib import Path
from . import intake
from .kaggle_export import unpack, pack
from .store import read, write, file_hash

MAX_UPLOAD = 64 * 1024 * 1024


def package(source, meta, destination):
    if source.stat().st_size > MAX_UPLOAD: raise ValueError('File exceeds 64 MiB')
    if file_hash(source) != meta['sha256']: raise ValueError('File hash mismatch')
    suffix = meta['format']
    if suffix == 'py':
        files = {'main.py': source.read_bytes()}
    elif suffix == 'zip':
        intake.inspect_zip(source)
        with zipfile.ZipFile(source) as z:
            files = {i.filename: z.read(i) for i in z.infolist() if not i.is_dir()}
        if 'main.py' not in files and 'arena.json' not in files:
            mains = [n for n in files if n.endswith('/main.py')]
            if len(mains) == 1:
                prefix = mains[0][:-7]
                if all(n.startswith(prefix) for n in files):
                    files = {n[len(prefix):]: v for n,v in files.items()}
    elif suffix == 'tar.gz':
        files = unpack(source.read_bytes())
    else:
        raise ValueError('Use a .py, .zip or .tar.gz file')
    if 'arena.json' in files:
        if len(files['arena.json']) > 16384: raise ValueError('arena.json is too large')
        supplied = json.loads(files.pop('arena.json'))
        m = {k: supplied[k] for k in ('run','build','resources') if k in supplied}
        intake.manifest_check(dict(name='check',author='check',version='check',**m))
    else:
        if 'main.py' not in files: raise ValueError('Package needs main.py, or arena.json for a custom command')
        m = {'run':['python','_arena_bridge.py' if meta['interface']=='kaggle' else 'main.py']}
    if m['run'] == ['python','_arena_bridge.py']:
        pack(files, destination)
    else:
        with zipfile.ZipFile(destination,'w',compression=zipfile.ZIP_DEFLATED) as z:
            for name,value in sorted(files.items()):
                info=zipfile.ZipInfo(name, date_time=(2026,1,1,0,0,0));info.external_attr=0o100644<<16
                z.writestr(info,value)
    m.update(name=meta['name'],version=meta['version'],author='github-'+str(meta['user_id']),origin={'kind':'team'})
    return m


def import_pending(root):
    root=Path(root)
    for path in sorted((root/'private/web-uploads').glob('*.json')):
        meta=read(path);rid=path.stem
        if not re.fullmatch('[a-f0-9]{32}',rid): continue
        result=path.with_suffix('.receipt')
        if result.exists(): continue
        try:
            archive=path.with_suffix('.zip')
            manifest=package(path.with_suffix('.bin'),meta,archive)
            receipt=intake.submit(root,archive,manifest,receipt='web-'+rid)
            write(result,dict(agent=receipt['agent'],status='registered'))
        except (ValueError,OSError,KeyError,TypeError,EOFError,RuntimeError,zipfile.BadZipFile,tarfile.TarError) as e:
            write(result,dict(status='rejected',message=str(e)[:240]))


def statuses(root):
    root=Path(root);out=[]
    for p in sorted((root/'private/web-uploads').glob('*.json')):
        m=read(p);r=read(p.with_suffix('.receipt'),{})
        a=read(root/'agents'/(r.get('agent','none')+'.json'),{})
        status='validation failed' if a.get('validation_failure') else a.get('status',r.get('status','queued'))
        row=dict(id=p.stem,user_id=m['user_id'],name=m['name'],version=m['version'],status=status)
        if r.get('message'):row['message']=r['message']
        if a.get('validation_failure'):row['message']='Sandbox validation failed. Check the agent interface and bundled dependencies.'
        if a:row['agent']=a['id']
        out.append(row)
    return out
