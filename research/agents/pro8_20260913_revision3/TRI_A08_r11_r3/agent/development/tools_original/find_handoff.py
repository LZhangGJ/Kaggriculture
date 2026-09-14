import json,gzip,collections,itertools,time
from pathlib import Path
b=Path('/mnt/data/r3_work');depots=(44,45,54,55)
def dist(a,b):return abs(a%10-b%10)+abs(a//10-b//10)
def sem(u):return [tuple(a) for a in u['repair'] if a[3]>=0 and a[0] not in [1,2,3,4]]
def route(start,ss,drop=None):
 t=0;r=[]
 if drop is not None:t+=dist(start,drop);r.append((t,5,-1,1,drop));t+=1;start=drop
 for op,i,q,d in ss:
  if op in [5,6]:d=min(depots,key=lambda x:dist(start,x))
  t+=dist(start,d);r.append((t,op,i,q,d));t+=1;start=d
 return t,r

def ordered(units,ss,owner=-1,drop=None):
 res=collections.defaultdict(list)
 for j,(u,s) in enumerate(zip(units,ss)):
  _,r=route(u['pos'],s,drop if j==owner else None)
  for t,op,i,q,d in r:
   if op not in [5,6]:res[d].append((t,j,op,i,q))
 return {p:tuple(tuple(x[2:]) for x in sorted(v)) for p,v in res.items()}

all_results=[]
for case in ['aurax_reactive_v1_1801029924_seat0','submission_56149565_1782098909_seat1','market_smart_v8_290316560_seat0','submission_56149565_447907448_seat0']:
 rr=json.loads(gzip.decompress((b/'logs'/('audit_'+case+'.json.gz')).read_bytes()));found=[]
 for r in rr:
  if r.get('phase')!=3 or r.get('forecast',0)<=100-r.get('safety',1) or r.get('step',9999)%24>=22:continue
  h=r['step']%24;units=r['units'];base=[sem(u) for u in units];origorder=ordered(units,base)
  for j,u in enumerate(units):
   if u['drop'] or u['mixed'] or sum(u['cargo'])<4 or not base[j]:continue
   for d in depots:
    if h+dist(u['pos'],d)+1>22:continue
    oldlen=route(u['pos'],base[j])[0];newlen=route(u['pos'],base[j],d)[0]
    if h+newlen<=23:continue
    groups=[]
    for key,g in itertools.groupby(enumerate(base[j]),key=lambda z:z[1][3]):
     g=list(g)
     if all(a[0] in (9,10,16,17) for k,a in g):groups.append((g[0][0],g[-1][0]+1))
    best=None
    for lo,hi in groups:
     part=base[j][lo:hi];don=base[j][:lo]+base[j][hi:];dl=route(u['pos'],don,d)[0]
     if h+dl>24:continue
     for v,rec in enumerate(units):
      if v==j:continue
      for at in range(len(base[v])+1):
       recv=base[v][:at]+part+base[v][at:];vl=route(rec['pos'],recv)[0]
       if h+vl>24:continue
       trial=base.copy();trial[j]=don;trial[v]=recv
       if ordered(units,trial,j,d)!=origorder:continue
       extra=dl-oldlen+vl-route(rec['pos'],base[v])[0]
       e={'case':case,'step':r['step'],'hour':h,'forecast':r['forecast'],'owner':j,'cargo':u['cargo'],'from_pos':u['pos'],'depot':d,'transfer_to':v,'tasks':part,'insert_at':at,'old_length':oldlen,'new_length':dl,'recipient_new_length':vl,'extra_actions':extra,'old_reject':'remaining-route-completion','task_counts_preserved':True}
       if best is None or (extra,dl,vl)<(best['extra_actions'],best['new_length'],best['recipient_new_length']):best=e
    if best:found.append(best)
 print(case,'feasible_options',len(found),'steps',len({x['step'] for x in found}),flush=True)
 ds=[19,21,27,28] if 'aurax' in case else [15,25,28] if '1782098909' in case else []
 for day in ds:
  ex=[x for x in found if x['step']//24==day];print('day',day,'options',len(ex));print(json.dumps(ex[:2],indent=1))
 all_results+=found
(b/'logs/handoff_options.json').write_text(json.dumps(all_results,indent=2))
