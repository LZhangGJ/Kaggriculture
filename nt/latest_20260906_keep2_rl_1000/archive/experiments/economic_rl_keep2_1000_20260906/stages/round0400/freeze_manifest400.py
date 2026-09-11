from common400 import *
assert read(P/'FINAL_AUDIT.json')['status']=='PASS';check_hashes()
entries={};total=0
for f in sorted(P.rglob('*')):
    if not f.is_file()or '__pycache__'in f.parts or f.name in ('MANIFEST.json','MANIFEST.json.writing'):continue
    h=hashlib.sha256()
    with f.open('rb')as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    size=f.stat().st_size;total+=size
    entries[str(f.relative_to(P))]=dict(sha256=h.hexdigest(),bytes=size)
save(P/'MANIFEST.json',dict(files=entries,count=len(entries),bytes=total,excludes=['__pycache__','MANIFEST.json']))
print('MANIFEST400',len(entries),total,flush=True)
