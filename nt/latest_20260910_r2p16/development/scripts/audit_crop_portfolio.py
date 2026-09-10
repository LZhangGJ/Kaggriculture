"""Read-only finite-crop portfolio audit. Both probes must reproduce frozen P12."""
from pathlib import Path
from collections import Counter
import json,gzip,ctypes,time
import run_panel as panel
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=HERE/'crop_portfolio_audit'
SOURCE=HERE/'crop_chain_screen100/cropchain1/cropchain1'
def main(kind='crop'):
    global OUT
    assert kind in ('crop','fert')
    number=13 if kind=='crop' else 14;prefix='cropportfolio' if kind=='crop' else 'fertportfolio'
    api='td_crop_portfolio_json' if kind=='crop' else 'td_fert_portfolio_json'
    OUT=HERE/f'{kind}_portfolio_audit'
    OUT.mkdir(exist_ok=True);panel.init_worker()
    rows={(r['opponent'],r['seed'],r['opponent_seat']):r for r in json.loads((SOURCE/'rows.json').read_text())}
    selected=json.loads((HERE/'proposal_equivalence_audit/SELECTION.json').read_text())
    binaries={m:HERE/f'candidate_r2p{number}/policy/{prefix}{m}.so' for m in (0,2)}
    for mode,path in binaries.items():
        receipt=json.loads((HERE/f'candidate_r2p{number}/{prefix}{mode}.BUILD.json').read_text());assert panel.sha(path)==receipt['binary_sha256']
        for p,h in receipt['sources'].items():assert panel.sha(HERE/f'candidate_r2p{number}/policy'/p)==h
    started=time.perf_counter();results=[]
    for ref in selected:
        row=rows[ref['opponent'],ref['seed'],ref['opponent_seat']];key=f"{row['opponent']}_{row['seed']}_{row['opponent_seat']}";target=OUT/(key+'.json')
        if target.exists():
            result=json.loads(target.read_text());assert result['binary_sha256']=={str(m):panel.sha(p) for m,p in binaries.items()};results.append(result);continue
        trace=json.loads(gzip.decompress((SOURCE/row['trace']).read_bytes()));saved={d['step']:d['observations'] for d in trace['days']};own=1-row['opponent_seat']
        agents={m:panel.R2_MODULE.Agent(binary_path=str(p)) for m,p in binaries.items()}
        for agent in agents.values():getattr(agent.lib,api).argtypes=[ctypes.c_void_p];getattr(agent.lib,api).restype=ctypes.c_char_p
        game=panel.old.LocalGame(row['seed'],panel.ENGINE);records=[]
        try:
            for step,joint in enumerate(trace['actions']):
                obs=game.observation(own)
                if step in saved:assert obs==saved[step][own]
                for mode,agent in agents.items():
                    action=agent(obs);assert action==joint[own],dict(key=key,step=step,mode=mode,actual=action,expected=joint[own])
                if step%24==0:records.append(json.loads(getattr(agents[2].lib,api)(agents[2].handle)))
                game.advance(joint)
            assert game.done and game.observation(own)==saved[719][own]
        finally:
            for agent in agents.values():agent.close()
        result=dict(status='PASS',key=key,actions=719,source_action_sha256=row['joint_action_sha256'],binary_sha256={str(m):panel.sha(p) for m,p in binaries.items()},daily=records)
        panel.save(target,result);results.append(result);print(json.dumps(dict(key=key,status='PASS',done=len(results))),flush=True)
    counter=Counter();changes=[];trials=0
    for result in results:
        for d in result['daily']:
            trials+=d['trials']
            for c in d['changes']:
                category=('early' if c['new_finish']<c['old_finish'] else 'later' if c['new_finish']>c['old_finish'] else 'same_day_service') if kind=='crop' else 'skip_today_fertilizer_same_finish'
                counter[str(c['kind'])+':'+category]+=1;changes.append(dict(key=result['key'],day=d['day'],**c))
    summary=dict(status='PASS',cases=len(results),actions_per_mode=719*len(results),official_frames_match=True,active_actions_unchanged=True,trials=trials,changes=len(changes),by_kind_direction=dict(counter),seconds=time.perf_counter()-started,binary_sha256={str(m):panel.sha(p) for m,p in binaries.items()},boundary='Model conditional value differences only, not realized outcome gains. Source games from development P12, no new32 confirmation or final100 labels used.')
    panel.save(OUT/'CHANGES.json',changes);panel.save(OUT/'RESULTS.json',summary)
    lines=[f'# 有限作物{kind}全场比较：只读审计','',f"{len(results)}局；OFF和只读模式分别{719*len(results)}动作与冻结P12相同；官方保存帧全部一致。",'',f"候选比较{trials}次，模型认为更好{len(changes)}次；这些是模型条件价值变化，不是实测现金或胜率变化。",'', '|作物ID/方向|次数|','|---|---:|']
    for k,v in sorted(counter.items()):lines.append(f'|{k}|{v}|')
    lines+=['','ID0小麦、1胡萝卜、4甜瓜。接替已承诺的轮作跳过。需要实际对战才能判断全场比较是否改善竞争，没有使用新32seed或最终留出结果。']
    (OUT/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
