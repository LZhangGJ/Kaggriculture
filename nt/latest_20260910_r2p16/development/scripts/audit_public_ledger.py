"""Validate observable trade inference and stock bounds against referee labels.

Native agent receives ONLY its normal seat observation. Opponent private stock
and true settled sales are read afterwards in Python for offline audit labels;
neither is ever passed into the inference API or used to choose an action.
"""
from pathlib import Path
import argparse,concurrent.futures as futures,ctypes,gzip,json,multiprocessing,time
from collections import Counter
import run_panel as panel
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
OUT=HERE/'public_ledger_audit_v2';PROBE=HERE/'candidate_r2p5/policy/ledger_probe_v2.so'
ITEMS=('WHEAT','CARROT','TOMATO','STRAWBERRY','MELON','EGG','MILK','WOOL','FERTILIZER')
def load(path):return json.loads(gzip.decompress(path.read_bytes()))
def stock(private):
    out=Counter(private['shed'])
    for inventory in private['inventories']:out.update(inventory)
    return out
def one(row):
    key=f"{row['opponent']}_{row['seed']}_{row['opponent_seat']}";target=OUT/(key+'.json.gz')
    if target.exists():
        data=load(target);assert data['status']=='PASS' and data['source_action_hash']==row['joint_action_sha256']
        return data['summary']
    trace=load(ROOT/row['trace']);audit=load(HERE/'baseline_audit'/key/'AUDIT.json.gz')
    own=1-row['opponent_seat'];rival=1-own;truth={}
    for tx in audit['transactions']:
        if tx['seat']==rival and tx['item'] in ITEMS:
            t=(tx['step'],tx['item']);truth[t]=truth.get(t,0)+(1 if tx['op']=='SELL' else -1 if tx['op']=='BUY_PRODUCT' else 0)*tx['quantity']
    agent=panel.R2_MODULE.Agent(binary_path=str(PROBE))
    agent.lib.td_public_ledger_json.argtypes=[ctypes.c_void_p];agent.lib.td_public_ledger_json.restype=ctypes.c_char_p
    env=panel.old.LocalGame(row['seed'],panel.ENGINE);frames=[];mismatches=[];sale_checks=bound_checks=0;looseness=Counter()
    saved={d['step']:d['observations'] for d in trace['days']}
    try:
        for step,joint in enumerate(trace['actions']):
            obs=env.observation(own)
            if step in saved:assert obs==saved[step][own]
            action=agent(obs);assert action==joint[own],dict(key=key,step=step,original=joint[own],probe=action)
            frame=json.loads(agent.lib.td_public_ledger_json(agent.handle).decode())
            # Offline labels below this boundary, never agent/ledger inputs.
            actual_stock=stock(env.observation(rival)['private'])
            actual_net=[]
            for i,item in enumerate(ITEMS):
                net=truth.get((step-1,item),0);actual_net.append(net)
                if frame['valid'][i]:
                    sale_checks+=1
                    if frame['rival_net'][i]!=net:mismatches.append(dict(type='sale',step=step,item=item,inferred=frame['rival_net'][i],actual=net))
                if 1<=i<=7:
                    bound_checks+=1
                    if actual_stock[item]>frame['upper'][i]+1e-8:mismatches.append(dict(type='upper',step=step,item=item,upper=frame['upper'][i],actual=actual_stock[item]))
                    looseness[item]+=frame['upper'][i]-actual_stock[item]
            frame.update(actual_stock=[actual_stock[i] for i in ITEMS],actual_net=actual_net);frames.append(frame)
            # Advance after recording labels for the observation used above.
            env.advance(joint)
        assert env.done and env.t==719 and env.observation(own)==saved[719][own]
        if frames[-1].get('invalid_market',0):mismatches.append(dict(type='invalid_market',count=frames[-1]['invalid_market']))
        status='PASS' if not mismatches else 'FAIL'
        summary=dict(key=key,status=status,sale_checks=sale_checks,bound_checks=bound_checks,mismatches=len(mismatches),looseness=dict(looseness),
                     skipped_floor=frames[-1]['skipped_floor'],skipped_boundary=frames[-1]['skipped_boundary'])
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(gzip.compress(json.dumps(dict(status=status,summary=summary,source_action_hash=row['joint_action_sha256'],
                                 mismatches=mismatches,frames=frames),separators=(',',':')).encode(),compresslevel=1))
        if mismatches:print(json.dumps(dict(key=key,first_mismatches=mismatches[:6])),flush=True)
        return summary
    finally:agent.close()
def main():
    p=argparse.ArgumentParser();p.add_argument('--limit',type=int,default=0);p.add_argument('--workers',type=int,default=16);args=p.parse_args()
    selection=json.loads((HERE/'baseline_audit/SELECTION.json').read_text())['rows']
    if args.limit:selection=[selection[round(i*(len(selection)-1)/max(1,args.limit-1))] for i in range(args.limit)]
    OUT.mkdir(exist_ok=True)
    protocol=dict(probe_sha256=panel.sha(PROBE),release_sha256=panel.sha(panel.old.R2/'agent.so'),engine_sha256=panel.sha(panel.old.REF/'official/kaggriculture.py'),
                  boundary='Native sees own seat observation and own previous unit actions only. Private labels read only after action.')
    path=OUT/'PROTOCOL.json'
    if path.exists():assert json.loads(path.read_text())==protocol
    else:panel.save(path,protocol)
    start=time.perf_counter();rows=[]
    with futures.ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn'),initializer=panel.init_worker) as executor:
        for f in futures.as_completed([executor.submit(one,row) for row in selection]):
            rows.append(f.result())
            if len(rows)%40==0 or len(rows)==len(selection):print(json.dumps(dict(done=len(rows),total=len(selection),seconds=time.perf_counter()-start)),flush=True)
    status='PASS' if all(r['status']=='PASS' for r in rows) else 'FAIL'
    result=dict(status=status,cases=len(rows),actions=len(rows)*719,seconds=time.perf_counter()-start,
                sale_checks=sum(r['sale_checks'] for r in rows),bound_checks=sum(r['bound_checks'] for r in rows),mismatches=sum(r['mismatches'] for r in rows),rows=rows)
    panel.save(OUT/('PILOT.json' if args.limit else 'RESULTS.json'),result)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)
    if status!='PASS':raise SystemExit(1)
if __name__=='__main__':main()
