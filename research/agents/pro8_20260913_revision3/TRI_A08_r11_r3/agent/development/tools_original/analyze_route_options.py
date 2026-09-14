import json,gzip,collections
from pathlib import Path
b=Path('/mnt/data/r3_work');items=['WHEAT','CARROT','TOMATO','STRAWBERRY','MELON','EGG','MILK','WOOL','FERTILIZER','GOOSE','COW','SHEEP'];depots=[44,45,54,55]
def dist(x,y):return abs(x%10-y%10)+abs(x//10-y//10)
def nsteps(start,sem):
 n=0
 for op,item,qty,t in sem:
  if op in [5,6]:t=min(depots,key=lambda d:dist(start,d))
  n+=dist(start,t)+1;start=t
 return n
summary={};cases=json.loads((b/'feedback/SELECTED_CASES.json').read_text())
for case in cases:
 rr=json.loads(gzip.decompress((b/'logs'/('audit_'+case['id']+'.json.gz')).read_bytes()));examples=[];counts=collections.Counter();byday={}
 for r in rr:
  t=r.get('step',-1);h=t%24
  if r.get('phase')!=3 or r.get('forecast',0)<=100-r.get('safety',1) or h>=22 or t//24>=29:continue
  for u in r['units']:
   cargo=u['cargo'];q=sum(cargo);sem=[a for a in u['repair'] if a[3]>=0 and a[0] not in [1,2,3,4]]
   if not q or not sem:continue
   sur=[max(0,cargo[i]-u['need'][i]) for i in range(9)];sn=sum(sur)
   reasons=('drop' if u['drop'] else '')+(' mixed' if u['mixed'] else '')
   counts[reasons or 'unblocked']+=1
   for d in depots:
    move=dist(u['pos'],d);full=move+1+nsteps(d,sem);place=move+sum(v>0 for v in sur)+nsteps(d,sem)
    if h+move+1>22:continue
    for mode,l,qty in [('drop',full,q),('selective',place,sn),('one_selective',full,max(sur))]:
     if qty<=0 or (mode=='drop' and u['mixed']):continue
     for last in [22,23]:
      if h+l-1<=last:
       key=mode+'_finish'+str(last);counts['feasible_'+key]+=1
       ex={'step':t,'hour':h,'unit':u['u'],'pos':u['pos'],'depot':d,'forecast':r['forecast'],'qty':qty,'current_cargo':dict(zip(items,cargo)),'needed':dict(zip(items,u['need'])),'remaining_steps':len(u['repair']),'trial_steps':l,'mode':key,'old_reject':reasons,'surplus':dict(zip(items[:9],sur))}
       examples.append(ex);byday.setdefault(str(t//24),[]).append(ex)
 summary[case['id']]={'counts':counts,'examples':examples,'byday':byday}
 print(case['id'],dict(counts))
 for day in (['19','21','27','28'] if 'aurax' in case['id'] else ['15','25','28'] if '1782098909' in case['id'] else []):
  ee=byday.get(day,[]);print('day',day,'feasible',len(ee));print(json.dumps(ee[:3],indent=1))
(b/'logs/route_options.json').write_text(json.dumps(summary,indent=2))
