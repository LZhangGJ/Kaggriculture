"""400 to 500 continuation, unchanged policy and fixed monitoring seeds."""
from pathlib import Path
import sys,os,time,json,hashlib
P=Path(__file__).resolve().parent
SOURCE=P.parent/'round0400'
ROOT=P.parents[1]
NATIVE=P.parents[2]/'economic_rl_keep2_100_20260906'
sys.path.insert(0,str(SOURCE))
from common400 import read,save,digest,summary,store_result,NAMES,bootstrap,configure_torch,runtime,np,torch
from common400 import Bias2Model,update_aux,forecast_metrics,restore,optimizer_receipt,checkpoint
from common400 import rollout as previous_rollout,chosen_path as previous_chosen_path


def check_hashes():
    for f,h in read(P/'PROTOCOL.json')['hashes'].items():assert digest(f)==h,f


def check_source_manifest():
    counts={}
    for directory in (NATIVE,SOURCE):
        receipt=read(directory/'MANIFEST.json')
        for relative,meta in receipt['files'].items():
            f=directory/relative;assert f.stat().st_size==meta['bytes'],str(f)
            h=hashlib.sha256()
            with f.open('rb')as stream:
                for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
            assert h.hexdigest()==meta['sha256'],str(f)
        counts[directory.name]=receipt['count']
    return counts


def rollout(pool,model='',mode=0,start=70300000,count=100,sample=9921,threads=16,trace=False):
    return previous_rollout(pool,model,mode,start,count,sample,threads,trace)


def evaluate(pool,out,model='',mode=0,start=70300000,count=100,sample=9921):
    r=rollout(pool,model,mode,start,count,sample);store_result(out,r)
    s=summary(r);save(Path(out)/'provenance.json',dict(checkpoint=str(model),sha256=digest(model)if model else None,
       library=str(NATIVE/'build/keep2.so'),library_sha=digest(NATIVE/'build/keep2.so'),seed_start=start,seeds=count,
       sample=sample,mode=mode,keep_bonus=2.))
    print('EVAL',Path(out).name,s['overall'],'nonkeep',s['non_keep'],flush=True)
    return s


def chosen_path(name,step):
    if step<=400:return previous_chosen_path(name,step)
    return P/'training'/name/f'step{step:03}.bin'
