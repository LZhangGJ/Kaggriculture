#!/usr/bin/env python3
"""Independent Bellman/action audit of the EXACT production .so; no games.

Integer diagnostic prices retain the original 1e-8 tolerance. The central
rich assertion remains exact. A deliberate feed-bit mutation proves that
value-only checking is not being mistaken for action correctness.
"""
from __future__ import annotations
import argparse,ctypes,functools,hashlib,json,math,pathlib,time
FIRST=(4,8,6); INTERVAL=(1,2,3); HELD=(4,6,6); PRODUCTS=(5,6,7)

def price_pattern(pattern: int)->list[list[float]]:
    return [[float(1 if pattern==0 else (25 if i==0 else 2 if i==8 else 30) if pattern==1 else (25 if i==0 else 100 if i==8 else 200) if pattern==2 else 1+(d*37+i*71+pattern*13+pattern*d*3)%211) for i in range(9)] for d in range(30)]

def audit(library: pathlib.Path, enabled: int=1)->dict:
    start=time.perf_counter();lib=ctypes.CDLL(str(library.resolve()))
    snapshot=lib.td_service_dp_snapshot
    snapshot.argtypes=[ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.POINTER(ctypes.c_double),ctypes.c_size_t,ctypes.c_double,ctypes.POINTER(ctypes.c_double),ctypes.c_size_t]
    snapshot.restype=ctypes.c_int
    checks=states=solves=0;digest=hashlib.sha256()
    def check(ok,why):
        nonlocal checks
        checks+=1
        if not ok:raise AssertionError(why)
    def load(k,birth,day,px,work):
        nonlocal solves
        inp=(ctypes.c_double*270)(*(v for row in px for v in row));out=(ctypes.c_double*1440)()
        check(snapshot(k,birth,day,inp,270,work,out,1440)==0,'native snapshot rejected valid state');solves+=1
        return list(out)
    def get(table,d,h,b):
        n=((d*2+h)*6+b)*4;return table[n:n+4]
    def transition(k,birth,d,h,b,feed,care,px,work):
        # Action execution is independent of the production loop/lambda. A
        # second unfed night retires the animal before any next-day reward.
        j=k-9;nh=0 if feed else h+1
        if nh>=2:return (0.,None)
        production=d+1-birth>=FIRST[j] and (d+1-birth-FIRST[j])%INTERVAL[j]==0
        quantity=(1+(b if feed else 0)) if production else 0
        next_b=min(HELD[j]-1,(0 if production else b)+care)
        reward=-feed*px[d][0]-(feed+care)*work+quantity*px[d+1][PRODUCTS[j]]
        if enabled:reward+=px[d+1][8]-work*(1+int(quantity>0))
        else:reward+=max(0.,px[d+1][8]-work)
        return reward,(d+1,nh,next_b)
    def validate(k,birth,day,px,work,table):
        nonlocal states
        j=k-9
        @functools.lru_cache(None)
        def oracle(d,h,b):
            if d==29:return 0.,0,0
            production=d+1-birth>=FIRST[j] and (d+1-birth-FIRST[j])%INTERVAL[j]==0
            best=(-1e100,0,0)
            for feed,care in ((0,0),(1,0),(1,1)):
                if care and b>=HELD[j]-1 and not production:continue
                reward,nxt=transition(k,birth,d,h,b,feed,care,px,work)
                candidate=reward+(oracle(*nxt)[0] if nxt else 0.)
                if candidate>best[0]+1e-9:best=candidate,feed,care
            return best
        for d in range(day,30):
            for h in (0,1):
                for b in range(HELD[j]):
                    states+=1;v,qv,feed,care=get(table,d,h,b);truth,tf,tc=oracle(d,h,b)
                    label=f'kind={k} placed={birth} day={d} hunger={h} bonus={b} work={work}'
                    check(all(math.isfinite(x) for x in (v,qv,feed,care)),'nonfinite '+label)
                    check(abs(v-qv)<1e-8,'value != Choice.value '+label)
                    check((feed,care)==(tf,tc),'action != independently selected argmax '+label)
                    check(abs(v-truth)<1e-8,'value != independent optimum '+label)
                    if d<29:
                        reward,nxt=transition(k,birth,d,h,b,int(feed),int(care),px,work)
                        from_action=reward+(get(table,*nxt)[0] if nxt else 0.)
                        check(abs(v-from_action)<1e-8,'stored action does not realize value '+label)
                    # Independently roll out the whole stored policy, not just
                    # one Bellman edge or its chosen score.
                    total=0.;state=(d,h,b)
                    while state is not None and state[0]<29:
                        sd,sh,sb=state;_,_,f,c=get(table,sd,sh,sb)
                        reward,state=transition(k,birth,sd,sh,sb,int(f),int(c),px,work)
                        total+=reward
                    check(abs(v-total)<1e-8,'policy rollout does not realize value '+label)
        digest.update(json.dumps(table,separators=(',',':'),allow_nan=False).encode())
    for k in (9,10,11):
        for horizon in range(1,6):
            day=29-horizon
            for phase in range(4):
                birth=max(0,day-FIRST[k-9]+phase)
                for pattern in range(8):
                    px=price_pattern(pattern);table=load(k,birth,day,px,4.);validate(k,birth,day,px,4.,table)
        for day in (0,7,14,29):
            for pattern in (0,1,2,7):
                px=price_pattern(pattern);table=load(k,0,day,px,1.2);validate(k,0,day,px,1.2,table)
    # Preserve the exact central negative and positive controls.
    px=price_pattern(1);poor=load(9,21,28,px,4.)
    if enabled:check(get(poor,28,1,0)==[0.,0.,0.,0.],'exact negative maintenance assertion')
    for row in px:row[5]=100.
    rich=load(9,21,28,px,4.);rich[((28*2+1)*6)*4+2]=0.;want=65. if enabled else 71.
    check(get(rich,28,1,0)==[want,want,1.,0.],'exact positive maintenance assertion')
    validate(9,21,28,px,4.,rich)
    validated_states=states;validated_checks=checks
    mutated=rich.copy();mutated[((28*2+1)*6)*4+2]=0.
    try:validate(9,21,28,px,4.,mutated)
    except AssertionError as error:
        mutation_message=str(error);check('action != independently selected argmax' in mutation_message,'unexpected mutation result')
    else:raise AssertionError('central feed-bit mutation escaped gate')
    return {'status':'PASS','library':str(library.resolve()),'native_sha256':hashlib.sha256(library.read_bytes()).hexdigest(),'animal_collection_labor':enabled,'table_solves':solves,'states_validated':validated_states,'checks_passed_before_mutation':validated_checks,'mutation_prefix_states':states-validated_states,'checks_including_expected_mutation_failure':checks,'table_digest':digest.hexdigest(),'rich':get(rich,28,1,0),'negative':get(poor,28,1,0),'mutation_rejected':True,'mutation_message':mutation_message,'seconds':time.perf_counter()-start,'new_games':0}

def main():
    p=argparse.ArgumentParser();p.add_argument('--library',type=pathlib.Path,required=True);p.add_argument('--enabled',type=int,choices=(0,1),default=1);p.add_argument('--out',type=pathlib.Path,required=True);args=p.parse_args()
    try:result=audit(args.library,args.enabled)
    except Exception as error:result={'status':'FAIL','error':str(error),'library':str(args.library),'new_games':0}
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
    raise SystemExit(result['status']!='PASS')
if __name__=='__main__':main()
