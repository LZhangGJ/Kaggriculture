"""Timing/intent probe only: activate existing rotation/continuation together."""
from pathlib import Path
import json,gzip,time
import run_panel as panel
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=HERE/'crop_continuation_probe'
def main():
    OUT.mkdir(exist_ok=True);panel.init_worker()
    binary=HERE/'candidate_r2p12/policy/cropchain1.so';assert panel.sha(binary)=='f62c0a303133d28eb93380f52aaead97ff9e7b0ca2613668ac859b638f9adf63'
    config=json.loads((panel.old.R2/'config.json').read_text())
    rows=json.loads((HERE/'crop_chain_screen100/cropchain1/cropchain1/rows.json').read_text());row=next(r for r in rows if r['opponent']=='soil_v219g' and r['seed']==2609110000 and r['opponent_seat']==0)
    trace=json.loads(gzip.decompress((HERE/'crop_chain_screen100/cropchain1/cropchain1'/row['trace']).read_bytes()))
    frames={d['step']:d['observations'][1] for d in trace['days']};result=[]
    for day in (0,3,8,15,24):
        for name,change in [('p12',{}),('rotation_only',{'rotation':1}),('rotation_and_repeat',{'rotation':1,'repeat':1})]:
            agent=panel.R2_MODULE.Agent(config=dict(config,**change),binary_path=str(binary));started=time.perf_counter()
            try:out=agent(frames[day*24]);seconds=time.perf_counter()-started
            finally:agent.close()
            r=dict(day=day,name=name,seconds=seconds,action=out);result.append(r);print(json.dumps(dict(day=day,name=name,seconds=seconds)),flush=True)
            panel.save(OUT/'PARTIAL.json',result)
            if seconds>5:
                panel.save(OUT/'STOPPED.json',dict(reason='single call over5s, not starting arena',row=r));return
    panel.save(OUT/'RESULTS.json',dict(status='TIMING_ONLY_NOT_STRENGTH',rows=result,binary_sha256=panel.sha(binary),boundary='Fresh controllers on five saved legal own observations, not full-game parity or a strength measurement. No opponent identity or future sent to agent.'))
if __name__=='__main__':main()
