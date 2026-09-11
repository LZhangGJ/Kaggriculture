"""Small rule differential: native candidate vs original Python G001.

Not a Python translation test of the new native-only policy experiments.
Bulk search remains inside native.batch; Python is used only as the referee here.
"""
from pathlib import Path
import argparse, copy, gzip, hashlib, json, sys, time, zlib
EXP=Path(__file__).resolve().parents[1];ROOT=EXP.parents[1]
sys.path.insert(0,str(EXP/'native/build'))
sys.path.insert(0,str(ROOT/'gpt_review/codex/G001_CPU_FOR_GPT_20260903'))
import _dp7_native as native
from cpu_runtime import LocalGame, load_agent, pass_agent
from check_native import normalized, canon

def main():
    p=argparse.ArgumentParser();p.add_argument('--configs',required=True);p.add_argument('--label',required=True)
    p.add_argument('--seed',type=int,default=20260904);p.add_argument('--seats',default='0,1');p.add_argument('--count',type=int,default=1)
    p.add_argument('--opponent',default='g001',choices=['g001','g003','pass']);p.add_argument('--out',required=True);p.add_argument('--save-replays',action='store_true');a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False);params=json.loads(Path(a.configs).read_text())[a.label]
    build=json.loads((EXP/'native/build/build_receipt.json').read_text())
    for rel,h in build['source_hashes'].items():assert hashlib.sha256((EXP/rel).read_bytes()).hexdigest()==h
    opponent_source = (EXP/'opponents/g003/source/main.py') if a.opponent=='g003' else (ROOT/'research/team_mate/Kaggriculture_main_512631c/agents/route_clustering_switch_agent/main.py')
    asset=EXP/f'native/{a.opponent}_frozen.json.zlib'
    data=json.loads(zlib.decompress(asset.read_bytes())) if a.opponent!='pass' else None
    if data: assert hashlib.sha256(opponent_source.read_bytes()).hexdigest()==data['source_sha256']
    # G001 is the legacy binding name of the shared searched-route executor.
    # G003 supplies its own complete library, trees and opening, not G001 data.
    g=native.G001(data) if data else None
    rows=[];started=time.perf_counter()
    for seed in range(a.seed,a.seed+a.count):
        for seat in map(int,a.seats.split(',')):
            env=native.Env(seed);ctl=native.Controller(params);st=native.G001State();official=LocalGame(seed)
            rival=load_agent(opponent_source) if g else pass_agent
            trace=[];max_latency=0
            for step in range(719):
                tic=time.perf_counter();own=ctl.act(env,seat);max_latency=max(max_latency,time.perf_counter()-tic)
                other=g.act(env,1-seat,st) if g else pass_agent(official.observation(1-seat),official.configuration)
                ref=rival(official.observation(1-seat),copy.deepcopy(official.configuration))
                if normalized(other)!=normalized(ref):
                    controller = getattr(rival, '__globals__', {}).get('_S_CONTROLLER')
                    failure = dict(seed=seed, seat=seat, step=step, kind='opponent_action',
                                   actual=other, expected=ref,
                                   native_route=data['families'][st.current] if data else None,
                                   reference_route=getattr(controller,'current',None),
                                   observation=canon(env.observation(1-seat)))
                    (out/'failure.json').write_text(json.dumps(failure,indent=2),encoding='utf8')
                    raise AssertionError((seed,seat,step,'opponent_action',failure['native_route'],failure['reference_route']))
                actions=[None,None];actions[seat]=own;actions[1-seat]=other
                if a.save_replays:trace.append(dict(step=step,observation=canon(env.observation(seat)),actions=actions))
                env.step(actions);official.advance(actions)
                for side in (0,1):
                    actual=canon(env.observation(side));expected=canon(official.observation(side))
                    for field in ('farms','private','market','town'):
                        if actual[field]!=expected[field]:
                            (out/'failure.json').write_text(json.dumps(dict(seed=seed,seat=seat,step=step,field=field,actual=actual[field],expected=expected[field])),encoding='utf8')
                            raise AssertionError((seed,seat,step,field))
            assert env.done
            farm=env.observation(seat)['farms'];row=dict(seed=seed,seat=seat,cash=farm[seat]['money'],opponent_cash=farm[1-seat]['money'],max_native_action_seconds=max_latency,opponent_switched=st.switched)
            rows.append(row);print(json.dumps(row),flush=True)
            if a.save_replays:
                (out/f'{seed}_seat{seat}.json.gz').write_bytes(gzip.compress(json.dumps(dict(trace=trace,final=canon(env.observation(seat)))).encode()))
    (out/'acceptance.json').write_text(json.dumps(dict(status='PASS',check='official_rule_state_and_original_opponent_action_parity',candidate_python_translation_checked=False,opponent=a.opponent,opponent_source_sha256=data['source_sha256'] if data else None,opponent_asset_sha256=hashlib.sha256(asset.read_bytes()).hexdigest() if data else None,params=params,build=build,rows=rows,seconds=time.perf_counter()-started),indent=2),encoding='utf8')

if __name__=='__main__':main()
