from pathlib import Path
import importlib.util,json,gzip,hashlib,time,resource,sys,ctypes
R=Path(__file__).resolve().parents[1]
p=R/'input/agent/main.py';spec=importlib.util.spec_from_file_location('parentmain',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
f=R/'input/replays/market_smart_v8_4145681829_seat0.json.gz';r=json.loads(gzip.decompress(f.read_bytes()));seat=r['result']['seat'];a=m.create_agent();rows=[];start=time.perf_counter()
for t in range(48):
 o=r['steps'][t][seat]['observation'];beg=time.perf_counter();out=a(o);dur=time.perf_counter()-beg
 assert out==r['actions'][t][seat],(t,out,r['actions'][t][seat]);rows.append({'step':t,'seconds':dur,'match':True,'action':out,'debug':a.debug() if t%24==0 else None})
a.close();summary={'scope':'legal historical-prefix calls, not games','calls':len(rows),'seconds':time.perf_counter()-start,'max_call_seconds':max(x['seconds'] for x in rows),'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'all_equal_saved':True}
(R/'logs/parent_probe.json').write_text(json.dumps({'summary':summary,'rows':rows},indent=2));print(json.dumps(summary,indent=2))
