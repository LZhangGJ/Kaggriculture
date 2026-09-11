"""Does today's intent key imply equal intraday policy behavior? Read only."""
from pathlib import Path
import argparse,ctypes,gzip,json,concurrent.futures as futures,multiprocessing,time,statistics
import run_panel as panel
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
OUT=HERE/'proposal_equivalence_audit';BINARY=HERE/'proposal_equivalence_probe/policy/equivalence_probe.so'

def one(row):
    key=f"{row['opponent']}_{row['seed']}_{row['opponent_seat']}";path=OUT/(key+'.json.gz')
    if path.exists():
        data=json.loads(gzip.decompress(path.read_bytes()));assert data['hash']==row['joint_action_sha256']
        return data['summary']
    trace=json.loads(gzip.decompress((ROOT/row['trace']).read_bytes()));saved={x['step']:x['observations'] for x in trace['days']};seat=1-row['opponent_seat']
    agent=panel.R2_MODULE.Agent(binary_path=str(BINARY));agent.lib.td_equivalence_json.argtypes=[ctypes.c_void_p];agent.lib.td_equivalence_json.restype=ctypes.c_char_p
    env=panel.old.LocalGame(row['seed'],panel.ENGINE);records=[];started=time.perf_counter()
    try:
        for step,joint in enumerate(trace['actions']):
            obs=env.observation(seat)
            if step in saved:assert obs==saved[step][seat]
            a=agent(obs);assert a==joint[seat],dict(key=key,step=step,original=joint[seat],probe=a)
            if step%24==0:records.append(json.loads(agent.lib.td_equivalence_json(agent.handle)))
            env.advance(joint)
        assert env.done and env.t==719 and env.observation(seat)==saved[719][seat]
        discarded=[p for d in records for p in d['discarded']]
        summary=dict(key=key,status='PASS',actions=719,days=len(records),raw=sum(d['raw'] for d in records),kept=sum(d['kept'] for d in records),discarded=len(discarded),different_actions=sum(x['different_actions']>0 for x in discarded),different_account=sum(not x['same_end_account'] for x in discarded),different_score=sum(abs(x['score_delta'])>1e-6 for x in discarded),positive_score=sum(x['score_delta']>1e-6 for x in discarded),seconds=time.perf_counter()-started)
        path.write_bytes(gzip.compress(json.dumps(dict(status='PASS',hash=row['joint_action_sha256'],summary=summary,row=row,days=records),separators=(',',':')).encode(),compresslevel=1))
        return summary
    finally:agent.close()

def main():
    p=argparse.ArgumentParser();p.add_argument('--pilot',action='store_true');p.add_argument('--workers',type=int,default=2);args=p.parse_args()
    assert 1<=args.workers<=16
    allrows=json.loads((HERE/'baseline_rows_all11.json').read_text());opponents=json.loads((HERE/'pool_all11.json').read_text());rows=[]
    for op in opponents:
        for won in (False,True):
            available=[r for r in allrows if r['opponent']==op['id'] and r['opponent_seat']==0 and r['r2_win']==won]
            for frac in (.25,.75):
                if available:rows.append(available[int(frac*(len(available)-1))])
    rows=list({(r['opponent'],r['seed'],r['opponent_seat']):r for r in rows}.values())
    if args.pilot:rows=rows[:2]
    OUT.mkdir(exist_ok=True)
    protocol=dict(binary_sha256=panel.sha(BINARY),official_sha256=panel.sha(panel.old.REF/'official/kaggriculture.py'),original_binary_sha256=panel.sha(panel.old.R2/'agent.so'),config_sha256=panel.sha(panel.old.R2/'config.json'),boundary='Only policy-local conditional worlds; discarded scores never choose live actions; all actual actions verified against original frozen traces.')
    file=OUT/'PROTOCOL.json'
    if file.exists():assert json.loads(file.read_text())==protocol
    else:panel.save(file,protocol)
    panel.save(OUT/('PILOT_SELECTION.json' if args.pilot else 'SELECTION.json'),rows)
    start=time.perf_counter();result=[]
    with futures.ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn'),initializer=panel.init_worker) as pool:
        for r in pool.map(one,rows):
            result.append(r);print(json.dumps(r),flush=True)
            panel.save(OUT/'PROGRESS.json',dict(done=len(result),total=len(rows),seconds=time.perf_counter()-start))
    totals={k:sum(r[k] for r in result) for k in ('actions','days','raw','kept','discarded','different_actions','different_account','different_score','positive_score')}
    summary=dict(status='PASS',cases=len(result),totals=totals,seconds=time.perf_counter()-start,rows=result)
    panel.save(OUT/('PILOT.json' if args.pilot else 'RESULTS.json'),summary)
    if not args.pilot:
        text=['# 候选等价性审计','',f"{len(result)}场原R2官方原轨迹、{totals['actions']}步，探针动作全部一致。没有改原策略。",'',f"条件日模拟：原候选{totals['raw']}，初始意图去重保留{totals['kept']}；丢弃{totals['discarded']}。",f"其中{totals['different_actions']}个被丢候选后续动作不同；{totals['different_account']}个期末现金/库存不同；{totals['different_score']}个条件分数不同（{totals['positive_score']}个比其代表高）。",'', '这说明的是初始去重键是否忠实代表24步行为，不是事后最优胜率，也不是去重修复一定能加分。若有不等价，再做实时原程序多seed对照。']
        (OUT/'REPORT_ZH.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k!='rows'}),flush=True)
if __name__=='__main__':main()
