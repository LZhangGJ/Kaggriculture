#!/usr/bin/env python3
"""Paired invocation on fixed own-visible observations. ZERO new matches.
All old observations after an action divergence remain off-policy probes.
"""
from __future__ import annotations
import argparse,copy,gzip,hashlib,importlib.util,json,pathlib,resource,time

def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def main():
    p=argparse.ArgumentParser();p.add_argument('--left',type=pathlib.Path,required=True);p.add_argument('--right',type=pathlib.Path,required=True);p.add_argument('--right-library',type=pathlib.Path);p.add_argument('--traces',type=pathlib.Path,required=True);p.add_argument('--out',type=pathlib.Path,required=True);p.add_argument('--require-exact',action='store_true');a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False);summaries=[];left=load(a.left.resolve(),'left_entry');right=load(a.right.resolve(),'right_entry');tick=time.perf_counter()
    for trace_path in sorted(a.traces.glob('*.json.gz')):
        trace=json.loads(gzip.decompress(trace_path.read_bytes()));rows=[];left.reset();right.reset()
        alternative=right.create_agent(binary_path=a.right_library.resolve()) if a.right_library else None
        try:
            for i,obs in enumerate(trace['observations'][:719]):
                t=time.perf_counter();x=left.agent(copy.deepcopy(obs),trace['configuration']);lt=time.perf_counter()-t
                t=time.perf_counter();y=alternative(copy.deepcopy(obs),trace['configuration']) if alternative else right.agent(copy.deepcopy(obs),trace['configuration']);rt=time.perf_counter()-t
                rows.append({'step':i,'same':x==y,'left_seconds':lt,'right_seconds':rt,'left_action':x,'right_action':y})
        finally:
            left.reset();right.reset()
            if alternative:alternative.close()
        mismatch=[r['step'] for r in rows if not r['same']]
        raw=json.dumps(rows,separators=(',',':'),allow_nan=False).encode();name=trace_path.name.removesuffix('.json.gz')
        (a.out/(name+'.json.gz')).write_bytes(gzip.compress(raw,mtime=0))
        lh=hashlib.sha256(json.dumps([r['left_action'] for r in rows],sort_keys=True,separators=(',',':')).encode()).hexdigest()
        rh=hashlib.sha256(json.dumps([r['right_action'] for r in rows],sort_keys=True,separators=(',',':')).encode()).hexdigest()
        summary={'case':name,'calls':len(rows),'exact':len(rows)-len(mismatch),'first_difference':mismatch[0] if mismatch else None,'left_action_sha256':lh,'right_action_sha256':rh,'left_seconds':sum(r['left_seconds'] for r in rows),'right_seconds':sum(r['right_seconds'] for r in rows)}
        summaries.append(summary);print(json.dumps(summary),flush=True)
    result={'status':'PASS' if all(x['exact']==x['calls'] for x in summaries) else 'DIFFERENCES','left':str(a.left),'right':str(a.right),'right_library_override':str(a.right_library) if a.right_library else None,'scope':'paired saved-own-observation invocation; not closed-loop game verification','calls_per_variant':sum(x['calls'] for x in summaries),'exact':sum(x['exact'] for x in summaries),'new_games':0,'seconds':time.perf_counter()-tick,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'cases':summaries}
    (a.out/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
    if a.require_exact and result['status']!='PASS':raise SystemExit(1)
if __name__=='__main__':main()
