from pathlib import Path
import shutil,json,subprocess,sys
B=Path('/mnt/data/r3_work');p=B/'checkpoints/harness_atomic_before_fix';p.mkdir()
for src in (B/'candidate/tests/delivery_checks.py',B/'tests/own_branch_checks.py'):
 shutil.copy2(src,p/src.name)
# Keep every old branch and old official validation as superseded evidence.
raw=p/'superseded_raw';raw.mkdir()
for x in (B/'logs').glob('branch_*.json.gz'):shutil.copy2(x,raw/x.name)
for x in (B/'logs').glob('own_branch_summary*.json'):shutil.copy2(x,raw/x.name)
for name in ['seedguard_checks_gcc','seedguard_checks_clang']:
 shutil.copytree(B/'logs'/name,raw/name)
code='''from pathlib import Path
import importlib.util,sys
B=Path('/mnt/data/r3_work')
def load(path,name):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
m=load(B/'candidate/tests/delivery_checks.py','old_units')
e=load(B/'feedback/referee/cpu_runtime.py','rt').load_engine()
o,p=m.synthetic(e,hour=23);o['private']['seeds']={'WHEAT':2};o['farms'][0]['farmer']=[2,2];o['farms'][0]['hands']=[[3,2]]
plans=[[[8,0,1,22]],[[8,0,1,23]]]
r=m.units_only(e,o,plans);planted=sum(t is not None and isinstance(t,dict) and t.get('kind')=='PLANT' for t in [r['farm']['tiles'][2][2],r['farm']['tiles'][2][3]])
print('two funded simultaneous plant requests: expected=2 observed='+str(planted),flush=True)
assert planted==2,'Atomic admission must be frozen before any seed consumption'
'''
(B/'tests/atomic_plant_negative_repro.py').write_text(code)
r=subprocess.run([sys.executable,str(B/'tests/atomic_plant_negative_repro.py')],capture_output=True,text=True)
(p/'negative_repro.stdout').write_text(r.stdout);(p/'negative_repro.stderr').write_text(r.stderr);(p/'negative_repro.result.json').write_text(json.dumps({'returncode':r.returncode,'expected_failure':True},indent=2));assert r.returncode!=0
shutil.copy2(B/'tests/atomic_plant_negative_repro.py',p/'negative_repro.py')
f=B/'candidate/tests/delivery_checks.py';s=f.read_text().replace("demand=collections.Counter(a[1] for a in acts if a[0]=='PLANT')\n  for u", "demand=collections.Counter(a[1] for a in acts if a[0]=='PLANT')\n  blocked={crop for crop,n in demand.items() if n>p['seeds'].get(crop,0)}\n  for u").replace("if a[0]=='PLANT' and demand[a[1]]>p['seeds'].get(a[1],0):a=['PASS']","if a[0]=='PLANT' and a[1] in blocked:a=['PASS']")
marker=" for reflection in range(4):"
insert=""" # The official interpreter freezes all-crop PLANT admission before applying
 # any unit. A sequential re-check after consuming seeds wrongly blocks the
 # second funded planter; this regression is independent of the agent policy.
 for units in (2,3):
  for stock in range(5):
   q,_=synthetic(engine,hour=23);q['farms'][0]['farmer']=[2,2];q['farms'][0]['hands']=[[3+k,2] for k in range(units-1)]
   q['private']['inventories']=[{} for _ in range(units)];q['private']['seeds']={'WHEAT':stock}
   pp=[[[8,0,1,22+u]] for u in range(units)];end=units_only(engine,q,pp);expected=units if stock>=units else 0
   planted=sum(isinstance(end['farm']['tiles'][2][2+u],dict) and end['farm']['tiles'][2][2+u].get('kind')=='PLANT' for u in range(units))
   check(planted==expected,('atomic PLANT admission',units,stock,planted,expected))
   check(end['private']['seeds']['WHEAT']==stock-expected,('atomic seed debit',units,stock))
"""
assert marker in s;s=s.replace(marker,insert+marker).replace("'checks':count,'positive_proposals'","'checks':count,'atomic_plant_unit_phase_cases':10,'positive_proposals'");f.write_text(s)
f=B/'tests/own_branch_checks.py';s=f.read_text().replace("demand=collections.Counter(a[1] for a in acts if a[0]=='PLANT')\n for u", "demand=collections.Counter(a[1] for a in acts if a[0]=='PLANT')\n blocked={crop for crop,n in demand.items() if n>private['seeds'].get(crop,0)}\n for u").replace("if a[0]=='PLANT' and demand[a[1]]>private['seeds'].get(a[1],0):a=['PASS']","if a[0]=='PLANT' and a[1] in blocked:a=['PASS']");f.write_text(s)
r=subprocess.run([sys.executable,str(B/'tests/atomic_plant_negative_repro.py')],capture_output=True,text=True);(B/'logs/atomic_plant_corrected.stdout').write_text(r.stdout);assert r.returncode==0,r.stderr
print('Harness-only atomic PLANT correction; production and native unchanged.')
