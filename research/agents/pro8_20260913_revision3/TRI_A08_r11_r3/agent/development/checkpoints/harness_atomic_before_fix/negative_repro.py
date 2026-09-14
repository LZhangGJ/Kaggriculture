from pathlib import Path
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
