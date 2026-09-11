"""Verify only reading-package bytes and embedded metadata; never execute Agents."""
import argparse,hashlib,json,re,zipfile
from pathlib import Path,PurePosixPath
ROOT=Path(__file__).resolve().parent
NAME="KAGGRICULTURE_READ_PACK_ZH_EN_20260911.zip"
p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group();g.add_argument('--zip',type=Path);g.add_argument('--directory',type=Path);a=p.parse_args()
archive=a.zip or (ROOT/NAME if not a.directory and (ROOT/NAME).is_file() else None)
folder=a.directory or ROOT
if archive:
 z=zipfile.ZipFile(archive);assert z.testzip() is None,'ZIP CRC failed';get=z.read
 manifest=json.loads(get('PACKAGE_MANIFEST.json'))
 assert set(z.namelist())=={x['path'] for x in manifest['files']}|{'PACKAGE_MANIFEST.json'},'Unexpected ZIP entries'
else:
 get=lambda name:(folder/name).read_bytes()
 manifest=json.loads(get('PACKAGE_MANIFEST.json'))
for x in manifest['files']:
 name=PurePosixPath(x['path']);assert not name.is_absolute() and '..' not in name.parts
 b=get(x['path']);assert len(b)==x['bytes'] and hashlib.sha256(b).hexdigest()==x['sha256'],x['path']
for filename in ['KAGGRICULTURE_STRATEGY_SYNTHESIS.html','KAGGRICULTURE_STRATEGY_SYNTHESIS_EN.html']:
 h=get(filename).decode('utf-8');d=json.loads(re.search(r'<script id="research-data" type="application/json">(.*?)</script>',h,re.S).group(1))
 assert len(d['rows'])==516 and sum(bool(x.get('review')) for x in d['rows'])==514
 assert len(d['ideas'])==16 and len(d['catalog'])==27 and len(d['triage'])==405
 assert all(x['status']=='PROPOSED_NOT_RUN' for x in d['ideas'])
 if filename.endswith('_EN.html'):assert not re.search(r'[\u3400-\u9fff]',h)
if archive:z.close()
print(json.dumps({'status':'PASS','payload_files':len(manifest['files']),'languages':['zh-CN','en'],'reviewed_per_language':514,'browser_validation':'NOT_PERFORMED','agent_execution':False}))
