"""Full public EcoBot v7 state/action parity against frozen official 1.32.7."""
from pathlib import Path
import argparse
import copy
from dataclasses import asdict
import gzip
import hashlib
import importlib.util
import json
import sys
import time

EXP=Path(__file__).resolve().parents[1];ROOT=EXP.parents[1]
sys.path.insert(0,str(EXP/'native/build'));sys.path.insert(0,str(EXP/'native/eco7_probe_build'))
sys.path.insert(0,str(ROOT/'gpt_review/codex/G001_CPU_FOR_GPT_20260903'))
import _dp7_native as arena
import _eco7_probe as port
from cpu_runtime import LocalGame
from check_native import canon,normalized

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def reference(source,index):
    spec=importlib.util.spec_from_file_location(f'eco7_original_{index}',source);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
    original=m.evaluate_turn
    def capture(obs,state):
        d=original(obs,state);m.last_decision=d;return d
    m.evaluate_turn=capture
    return m

def ref_debug(ref,seat):
    mem=ref._MEMORIES[seat];drift=asdict(mem.eval_state.drift);drift['observed']={k:v for k,v in drift['observed'].items() if v!=0}
    cull=asdict(mem.eval_state.cull);cull['downsized']=sorted(cull['downsized']);cull['negative_days']={sp:cull['negative_days'].get(sp,0) for sp in ('COW','SHEEP','GOOSE')}
    return canon(dict(drift=drift,cull=cull,plan_day=mem.day_plan.day if mem.day_plan else -1,
        queues=mem.day_plan.routes if mem.day_plan else {},decision=asdict(ref.last_decision)))

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--seeds',default='20261401,20261404');p.add_argument('--seats',default='0,1');p.add_argument('--save-replays',action='store_true');p.add_argument('--arena-adapter',action='store_true')
    p.add_argument('--configs',default=str(EXP/'profiles/candidates.json'));p.add_argument('--label',default='S3C03');a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False);source=EXP/'opponents/ecobot_v7/output/main.py';source_hash=sha(source)
    assert source_hash=='0dc02e03c94ef60c06b5093efc2e2fd0530aa6eea20df507a90b90d6651bd067'
    build=json.loads((EXP/'native/build/build_receipt.json').read_text());pb=json.loads((EXP/'native/eco7_probe_build/build_receipt.json').read_text())
    for module,b in ((arena,build),(port,pb)):
        assert sha(Path(module.__file__))==b['binary_sha256']
        for rel,h in b['source_hashes'].items():assert sha(EXP/rel)==h,rel
    params=json.loads(Path(a.configs).read_text())[a.label];receipt=dict(status='RUNNING',build=build,probe_build=pb,args=vars(a),params=params,source_sha256=source_hash,rows=[])
    (out/'plan.json').write_text(json.dumps(receipt,indent=2));started=time.perf_counter()
    def fail(kind,**kw):
        (out/'failure.json').write_text(json.dumps(dict(kind=kind,**kw),indent=2));raise AssertionError((kind,kw.get('seed'),kw.get('seat'),kw.get('step')))
    for seed in map(int,a.seeds.split(',')):
        for seat in map(int,a.seats.split(',')):
            env=arena.Env(seed);own=arena.Controller(params);rival=port.EcoBot();official=LocalGame(seed);ref=reference(source,len(receipt['rows']));trace=[]
            native_op=arena.EcoBotV7() if a.arena_adapter else None
            max_cpp=max_py=0;orders=0;hints=set();cull_days=[]
            for step in range(719):
                obs=canon(official.observation(1-seat));actual_input=canon(env.observation(1-seat))
                for field in ('farms','private','market','town','day','hour','step'):
                    if obs[field]!=actual_input[field]:fail('projection',seed=seed,seat=seat,step=step,field=field)
                start=time.perf_counter();expected=ref.agent(copy.deepcopy(obs));max_py=max(max_py,time.perf_counter()-start)
                start=time.perf_counter();actual=rival.act(actual_input);max_cpp=max(max_cpp,time.perf_counter()-start)
                if normalized(actual)!=normalized(expected):fail('action',seed=seed,seat=seat,step=step,expected=expected,actual=actual,observation=obs,expected_state=ref_debug(ref,1-seat),actual_state=rival.debug())
                rs=ref_debug(ref,1-seat);cs=canon(rival.debug());cs['cull']['downsized'].sort()
                if rs!=cs:fail('memory',seed=seed,seat=seat,step=step,expected=rs,actual=cs,observation=obs)
                if native_op:
                    direct=native_op.act(env,1-seat)
                    if normalized(direct)!=normalized(expected):fail('arena_adapter_action',seed=seed,seat=seat,step=step,expected=expected,actual=direct,observation=obs)
                    actual=direct
                if step%24==0:
                    hints.add(json.dumps(rs['decision']['hints'],sort_keys=True))
                    if rs['cull']['downsized']:cull_days.append(dict(day=obs['day'],species=rs['cull']['downsized']))
                actions=[None,None];actions[seat]=own.act(env,seat);actions[1-seat]=actual;original=copy.deepcopy(actions);original[1-seat]=expected;orders+=len(actual['market'])
                if a.save_replays:trace.append(dict(step=step,observation=canon(env.observation(seat)),actions=actions,original_opponent_action=expected,opponent_memory=cs))
                env.step(actions);official.advance(original)
                for side in (0,1):
                    no=canon(env.observation(side));po=canon(official.observation(side))
                    for field in ('farms','private','market','town','day','hour','step'):
                        if no[field]!=po[field]:fail('official_state',seed=seed,seat=seat,step=step+1,side=side,field=field,actual=no[field],expected=po[field])
            assert env.done and env.step_count==719;farms=env.observation(seat)['farms']
            row=dict(seed=seed,seat=seat,steps=719,cash=farms[seat]['money'],opponent_cash=farms[1-seat]['money'],
                action_mismatches=0,memory_mismatches=0,official_state_mismatches=0,distinct_daily_hints=len(hints),cull_days=cull_days,market_orders=orders,max_cpp_action_seconds=max_cpp,max_python_action_seconds=max_py)
            receipt['rows'].append(row);(out/'progress.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(row),flush=True)
            if a.save_replays:(out/f'{seed}_seat{seat}.json.gz').write_bytes(gzip.compress(json.dumps(dict(trace=trace,final=canon(env.observation(seat)))).encode()))
    receipt.update(status='PASS',seconds=time.perf_counter()-started,coverage_caveat='Finite full-game differential checks, not exhaustive state proof or final Kaggle sandbox validation.')
    (out/'acceptance.json').write_text(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
