"""Audit the unchanged R2's existing short public-world predictions.

The native handle receives normal seat observations only. Recorded opponent
actions and next-day truth are used by the Python referee/labels, never policy.
"""
from pathlib import Path
import argparse,ctypes,gzip,json,concurrent.futures as futures,multiprocessing,time
import run_panel as panel
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
OUT=HERE/'cash_projection_audit';BINARY=HERE/'cash_probe/policy/cash_probe.so'
ITEMS=('WHEAT','CARROT','TOMATO','STRAWBERRY','MELON','EGG','MILK','WOOL','FERTILIZER','GOOSE','COW','SHEEP')

def load(path):return json.loads(gzip.decompress(path.read_bytes()))

def decode(encoded):
    def atom(row):
        op,item,q=row;name=panel.R2_MODULE._OPS[op];a=[name]
        if item>=0:
            a.append(ITEMS[item])
            if name in ('PLACE','PICKUP','BUY_SEED','BUY_ANIMAL','BUY_PRODUCT','SELL'):a.append(q)
        elif q!=1:a.append(q)
        return a
    units=[atom(x) for x in encoded[0]]
    return dict(farmer=units[0] if units else ['PASS'],hands=units[1:],market=[atom(x) for x in encoded[1]])

def private_total(obs):
    p=obs['private'];return [sum(b.get(i,0) for b in [p['shed']]+p['inventories']) for i in ITEMS]

def one(row):
    key=f"{row['opponent']}_{row['seed']}_{row['opponent_seat']}";target=OUT/(key+'.json.gz')
    if target.exists():
        data=load(target);assert data['status']=='PASS' and data['source_action_hash']==row['joint_action_sha256']
        return data['summary']
    trace=load(ROOT/row['trace']);saved={d['step']:d['observations'] for d in trace['days']};own=1-row['opponent_seat']
    agent=panel.R2_MODULE.Agent(binary_path=str(BINARY))
    agent.lib.td_cash_projection_json.argtypes=[ctypes.c_void_p];agent.lib.td_cash_projection_json.restype=ctypes.c_char_p
    env=panel.old.LocalGame(row['seed'],panel.ENGINE);daily=[];pending=None
    try:
        for step,joint in enumerate(trace['actions']):
            obs=env.observation(own)
            if step in saved:assert obs==saved[step][own]
            action=agent(obs)
            assert action==joint[own],dict(key=key,step=step,probe=action,original=joint[own])
            if step%24==0:
                assert pending is None
                pending=json.loads(agent.lib.td_cash_projection_json(agent.handle));pending.update(day=step//24,step=step,initial_own_cash=obs['farms'][own]['money'],initial_rival_cash=obs['farms'][1-own]['money'],initial_shops=obs['town']['unlocked_shops'])
                selected=pending['candidates'][pending['winner']];assert selected['step']==step and selected['end']==min(719,step+24)
                assert len(selected['actions'])==selected['end']-step
                predicted=[decode(a) for a in selected['actions']]
                actual=[a[own] for a in trace['actions'][step:selected['end']]]
                pending['action_comparison']={}
                for name in ('all','units','market'):
                    def project(a):return a if name=='all' else [a['farmer'],a['hands']] if name=='units' else a['market']
                    diffs=[step+i for i,(a,b) in enumerate(zip(predicted,actual)) if project(a)!=project(b)]
                    pending['action_comparison'][name]=dict(differences=len(diffs),first=diffs[0] if diffs else None)
            env.advance(joint)
            if env.t==pending['candidates'][pending['winner']]['end']:
                end=env.observation(own);assert end==saved[env.t][own]
                pending['actual']=dict(own_cash=end['farms'][own]['money'],rival_cash=end['farms'][1-own]['money'],holdings=private_total(end),inventory=[end['market']['inventory'][i] for i in ITEMS[:9]],prices=[end['market']['prices'][i] for i in ITEMS[:9]],shops=end['town']['unlocked_shops'])
                daily.append(pending);pending=None
        assert pending is None and env.done and env.t==719 and len(daily)==30
        summary=dict(key=key,status='PASS',actions=719,days=30)
        target.parent.mkdir(exist_ok=True);target.write_bytes(gzip.compress(json.dumps(dict(status='PASS',summary=summary,row=row,source_action_hash=row['joint_action_sha256'],daily=daily),separators=(',',':')).encode(),compresslevel=1))
        return summary
    finally:agent.close()

def main():
    p=argparse.ArgumentParser();p.add_argument('--pilot',action='store_true');p.add_argument('--workers',type=int,default=16);args=p.parse_args()
    if args.pilot:
        allrows=json.loads((HERE/'baseline_rows_all11.json').read_text());rows=[]
        for op in json.loads((HERE/'pool_all11.json').read_text()):
            for seat in (0,1):rows.append(next(r for r in allrows if r['opponent']==op['id'] and r['opponent_seat']==seat))
    else:rows=json.loads((HERE/'baseline_audit/SELECTION.json').read_text())['rows']
    OUT.mkdir(exist_ok=True)
    protocol=dict(probe_sha256=panel.sha(BINARY),original_sha256=panel.sha(panel.old.R2/'agent.so'),config_sha256=panel.sha(panel.old.R2/'config.json'),official_sha256=panel.sha(panel.old.REF/'official/kaggriculture.py'),boundary='Telemetry-only; all original actions and referee frames must match. No live opponent private or future data passed to native handle.')
    path=OUT/'PROTOCOL.json'
    if path.exists():assert json.loads(path.read_text())==protocol
    else:panel.save(path,protocol)
    started=time.perf_counter();results=[]
    with futures.ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn'),initializer=panel.init_worker) as pool:
        for result in pool.map(one,rows):
            results.append(result)
            if len(results)%40==0 or len(results)==len(rows):
                progress=dict(done=len(results),total=len(rows),seconds=time.perf_counter()-started)
                panel.save(OUT/'PROGRESS.json',progress);print(json.dumps(progress),flush=True)
    panel.save(OUT/('PILOT.json' if args.pilot else 'RESULTS.json'),dict(status='PASS',cases=len(rows),actions=719*len(rows),seconds=time.perf_counter()-started,rows=results))
if __name__=='__main__':main()
