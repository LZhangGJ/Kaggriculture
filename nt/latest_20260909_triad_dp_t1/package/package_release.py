from pathlib import Path
import zipfile,hashlib,json,shutil
R=Path(__file__).resolve().parent
assert (R/'REPORT_ZH.md').exists() and (R/'ACCEPTANCE.json').exists()
rootfiles=['README_ZH.md','REPORT_ZH.md','DESIGN_AND_SCOPE_ZH.md','TRAINING_BOUNDARY_ZH.md','SOURCE_PROVENANCE_TRIAD.json','SELECTION_BEFORE_HOLDOUT.json','RELEASE_FREEZE.json','ENVIRONMENT.json','ACCEPTANCE.json','QUALITY_ACCEPTANCE.json','holdout_games.csv','split.json','requirements.txt','requirements-training.txt','build_policy.py','build_panel.py','build_all.py','run.py','experiment.py','collect_labels.py','train_value.py','train_ranker.py','make_report.py','final_validate.py','package_release.py','analyze_losses.py','POSTHOLDOUT_LOSS_DIAGNOSIS.json','export_schema.cpp','baseline_auto.so','baseline_j7.so']
folders=['policy','src','tests','arena','agent','runs','labels','models','snapshots']
files=[R/x for x in rootfiles if (R/x).is_file()]
for folder in folders:
 for p in (R/folder).rglob('*'):
  if not p.is_file():continue
  rel=p.relative_to(R)
  if any(part in {'__pycache__','.git','.venv','build'} for part in rel.parts):continue
  if p.suffix in {'.o','.pyc'} or '.tmp' in p.name:continue
  # Native extensions are rebuilt; only primary portable-ABI policy binaries
  # and hashed historical snapshots are retained.
  if p.suffix=='.so' and folder not in {'policy','snapshots'}:continue
  if folder=='tests' and p.suffix=='' and p.name not in {'README'}:continue
  files.append(p)
# Preserve successful and failed/aborted experiment status, without build caches.
index=[]
for d in sorted((R/'runs').iterdir()):
 if not d.is_dir():continue
 summary=d/'summary.json';status='COMPLETED' if summary.exists() else 'NO_COMPLETED_RESULT'
 row=dict(tag=d.name,status=status)
 if summary.exists():row.update(json.loads(summary.read_text())['overall'])
 for f in ['EXPERIMENT_STATUS.json','ABORTED.json']:
  if (d/f).exists():row['note']=json.loads((d/f).read_text())
 index.append(row)
(R/'EXPERIMENT_INDEX.json').write_text(json.dumps(index,indent=2));files.append(R/'EXPERIMENT_INDEX.json')
manifest={str(p.relative_to(R)):dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(set(files))}
(R/'MANIFEST_T1.json').write_text(json.dumps(manifest,indent=2));files.append(R/'MANIFEST_T1.json')
out=Path('/mnt/data/Kaggriculture_Triad_DP_T1_20260908.zip')
with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for p in sorted(set(files)):z.write(p,'Kaggriculture_Triad_DP_T1/'+str(p.relative_to(R)))
with zipfile.ZipFile(out) as z:assert z.testzip() is None
report=Path('/mnt/data/Kaggriculture_Triad_DP_T1_REPORT_ZH.md');shutil.copy2(R/'REPORT_ZH.md',report)
csv_out=Path('/mnt/data/Kaggriculture_Triad_DP_T1_holdout1400.csv');shutil.copy2(R/'holdout_games.csv',csv_out)
print(json.dumps(dict(zip=str(out),bytes=out.stat().st_size,sha256=hashlib.sha256(out.read_bytes()).hexdigest(),files=len(set(files)),report=str(report)),indent=2))
