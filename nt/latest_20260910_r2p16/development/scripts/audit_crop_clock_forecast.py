"""Unchanged original actions + native public-only crop-clock telemetry."""
from pathlib import Path
from collections import defaultdict,deque,Counter
import argparse,ctypes,gzip,json,concurrent.futures as futures,multiprocessing,time,statistics
import run_panel as panel
from audit_public_crop_removals import infer
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=HERE/'crop_clock_forecast_audit'
BINARY=HERE/'candidate_r2p11/policy/cropclock2_sale0.so'
ITEMS=('WHEAT','CARROT','TOMATO','STRAWBERRY','MELON','EGG','MILK','WOOL','FERTILIZER')

def one(row):
    key=f"{row['opponent']}_{row['seed']}_{row['opponent_seat']}";target=OUT/(key+'.json.gz')
    if target.exists():
        d=json.loads(gzip.decompress(target.read_bytes()));assert d['binary_sha256']==panel.sha(BINARY);return d['summary']
    trace=json.loads(gzip.decompress((ROOT/row['trace']).read_bytes()));saved={x['step']:x['observations'] for x in trace['days']};seat=row['opponent_seat'];own=1-seat
    agent=panel.R2_MODULE.Agent(binary_path=str(BINARY));agent.lib.td_crop_clock_json.argtypes=[ctypes.c_void_p];agent.lib.td_crop_clock_json.restype=ctypes.c_char_p
    game=panel.old.LocalGame(row['seed'],panel.ENGINE);original=panel.ENGINE._commit_unit;trades=defaultdict(float)
    history={k:deque(maxlen=32) for k in (0,1,4)};records=[];removed=0
    def commit(op,item,price,farm,private,market,shed_capacity=100):
        ok=original(op,item,price,farm,private,market,shed_capacity)
        if ok and farm is game.state[0].observation.farms[seat]:
            if op=='SELL':trades[game.t//24,item]+=1
            elif op=='BUY_PRODUCT':trades[game.t//24,item]-=1
        return ok
    panel.ENGINE._commit_unit=commit
    try:
        for step,joint in enumerate(trace['actions']):
            obs=game.observation(own)
            if step in saved:assert obs==saved[step][own]
            out=agent(obs);assert out==joint[own],dict(key=key,step=step,probe=out,original=joint[own])
            q=json.loads(agent.lib.td_crop_clock_json(agent.handle));assert q['step']==step
            for k,values in history.items():
                assert q['counts'][k]==len(values),(key,step,k,q['counts'],list(values))
                expected=[0]*30
                if len(values)>=3:
                    expected[(4,3,8,10,10)[k]]=1
                    for age in values:expected[age]+=1
                assert q['weights'][k]==expected,(key,step,k)
            if step%24==0:
                records.append(dict(day=step//24,counts=q['counts'],weights=q['weights'],old=q['live_forecast'][step//24],new=q['observed_forecast'][step//24]))
            old=obs['farms'][seat];game.advance(joint);new=game.observation(own)['farms'][seat]
            events=infer(old,new,step);removed+=len(events)
            for e in events:history[ITEMS.index(e['crop'])].append(e['age'])
        assert game.done and game.observation(own)==saved[719][own]
    finally:panel.ENGINE._commit_unit=original;agent.close()
    for d in records:d['actual_net']=[trades[d['day'],i] for i in ITEMS]
    summary=dict(key=key,status='PASS',actions=719,days=len(records),public_removals=removed)
    target.write_bytes(gzip.compress(json.dumps(dict(binary_sha256=panel.sha(BINARY),source_action_hash=row['joint_action_sha256'],row=row,summary=summary,daily=records),separators=(',',':')).encode(),compresslevel=1))
    return summary

def main():
    p=argparse.ArgumentParser();p.add_argument('--pilot',action='store_true');args=p.parse_args()
    rows=json.loads((HERE/'proposal_equivalence_audit/SELECTION.json').read_text());rows=rows[:2] if args.pilot else rows
    build=json.loads((HERE/'candidate_r2p11/cropclock2_sale0.BUILD.json').read_text());assert panel.sha(BINARY)==build['binary_sha256']
    for path,hash in build['sources'].items():assert panel.sha(HERE/'candidate_r2p11/policy'/path)==hash
    OUT.mkdir(exist_ok=True);started=time.perf_counter()
    with futures.ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn'),initializer=panel.init_worker) as pool:
        results=list(pool.map(one,rows))
    buckets=defaultdict(list)
    for r in results:
        data=json.loads(gzip.decompress((OUT/(r['key']+'.json.gz')).read_bytes()))
        for d in data['daily']:
            for k,item in enumerate(ITEMS):
                a=d['actual_net'][k];x=d['old'][k]*.85;y=d['new'][k]*.85
                buckets[item].append((x-a,y-a,x,y,a))
    metrics={k:dict(samples=len(v),old_mae=statistics.mean(abs(x[0]) for x in v),new_mae=statistics.mean(abs(x[1]) for x in v),old_bias=statistics.mean(x[0] for x in v),new_bias=statistics.mean(x[1] for x in v),changed=sum(abs(x[2]-x[3])>1e-8 for x in v),actual_net_mean=statistics.mean(x[4] for x in v)) for k,v in buckets.items()}
    result=dict(status='PASS',cases=len(rows),actions=719*len(rows),rows=results,metrics=metrics,seconds=time.perf_counter()-started,binary_sha256=panel.sha(BINARY),boundary='Original actions unchanged; telemetry predicts conditional net sale only using past public crop removals. Forecast error reduction is not a win-rate test. Private/action labels never enter predictor.')
    panel.save(OUT/('PILOT.json' if args.pilot else 'RESULTS.json'),result)
    if not args.pilot:
        lines=['# 公开作物节奏：原动作不变的预测验证','',f"{len(rows)}场、{719*len(rows)}个动作与原版相同；原始官方帧相同。Python/C++每步窗口、计数和分布一致。",'',
               '|商品|原净卖出预测MAE|新MAE|原偏差|新偏差|改变日数|','|---|---:|---:|---:|---:|---:|']
        for k,v in metrics.items():lines.append(f"|{k}|{v['old_mae']:.2f}|{v['new_mae']:.2f}|{v['old_bias']:+.2f}|{v['new_bias']:+.2f}|{v['changed']}|")
        lines+=['','这不是已加分。模型仍不知道对手私有库存、未来投资与确切售价；只改对当前有限作物收获时点的条件判断。分布按本局最近32次腾地估计，至少3个样本，附1个原默认年龄先验。成熟DIG在新策略中仍可能与HARVEST混淆，不能称万能意图识别。','']
        (OUT/'REPORT_ZH.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)
if __name__=='__main__':main()
