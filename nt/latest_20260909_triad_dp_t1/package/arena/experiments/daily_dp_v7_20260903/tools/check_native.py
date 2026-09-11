"""Exact per-step differential checks; Python is a referee, not bulk rollout."""
from pathlib import Path
import argparse, copy, gzip, hashlib, importlib.util, json, sys, time, zlib
EXP=Path(__file__).resolve().parents[1]; ROOT=EXP.parents[1]
sys.path.insert(0,str(EXP/'native/build'))
sys.path.insert(0,str(ROOT/'gpt_review/codex/G001_CPU_FOR_GPT_20260903'))
import _dp7_native as native
from cpu_runtime import LocalGame, load_agent, pass_agent

def load_module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def normalized(a):
    def atom(x):
        # Official 1.32.7 defaults omitted PICKUP/PLACE quantities to one.
        # Preserve explicit quantities: PICKUP(item, 2) must remain different.
        if x[0] in ('PICKUP','PLACE'):return x[:2]+[x[2] if len(x)>2 else 1]
        if x[0] in ('BUY_PRODUCT','BUY_ANIMAL','BUY_SEED','SELL'): return x[:3]
        if x[0]=='PLANT':return x[:2]
        return x[:1]
    return dict(farmer=atom(a.get('farmer',['PASS'])), hands=[atom(x) for x in a.get('hands',[])], market=[atom(x) for x in a.get('market',[])])

def canon(x):
    return json.loads(json.dumps(x))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,default=20260903);ap.add_argument('--count',type=int,default=1)
    ap.add_argument('--opponent',choices=['pass','g001'],default='pass');ap.add_argument('--seats',default='0');ap.add_argument('--variant',choices=['none','core'],default='core');ap.add_argument('--official',action='store_true');ap.add_argument('--out',required=True)
    ap.add_argument('--prior-replay');ap.add_argument('--params',default='{}');args=ap.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    prior=json.loads(gzip.decompress(Path(args.prior_replay).read_bytes()))['steps'] if args.prior_replay else None
    mod=load_module(EXP/'agents/agent_v7.py','dp7_reference')
    config={f:args.variant=='core' for f in ('fix_resources','fix_expiry','fix_values','fix_calendar')}
    config.update(json.loads(args.params))
    data=json.loads(zlib.decompress((EXP/'native/g001_frozen.json.zlib').read_bytes()));op=native.G001(data) if args.opponent=='g001' else None
    records=[];started=time.perf_counter()
    for seed in range(args.seed,args.seed+args.count):
      for seat in map(int,args.seats.split(',')):
        env=native.Env(seed);c=native.Controller(config);ref=mod.DailyDPController(mod.Params(**config));st=native.G001State()
        rival=load_agent(ROOT/'research/team_mate/Kaggriculture_main_512631c/agents/route_clustering_switch_agent/main.py') if op else pass_agent
        game=LocalGame(seed) if args.official else None
        for step in range(719):
          if prior:
            for pp in (0,1):
              no=canon(env.observation(pp));po=canon(prior[step][pp]['observation'])
              for kk in ('farms','private','market','town'):
                if no[kk]!=po[kk]:
                  report=dict(seed=seed,seat=seat,step=step,error='historical_state',key=kk,native=no[kk],prior=po[kk])
                  (out/'failure.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:report[k] for k in ('seed','seat','step','error','key')}),flush=True);raise RuntimeError('historical_state')
          obs=env.observation(seat);pyact=ref.act(game.observation(seat) if game else obs);ccact=c.act(env,seat)
          expected=rival(game.observation(1-seat) if game else env.observation(1-seat),copy.deepcopy(game.configuration) if game else None);actual=op.act(env,1-seat,st) if op else expected
          why=None
          if normalized(pyact)!=normalized(ccact):why='candidate_action'
          elif normalized(expected)!=normalized(actual):why='g001_action'
          if prior and (normalized(prior[step+1][seat]['action'])!=normalized(ccact) or normalized(prior[step+1][1-seat]['action'])!=normalized(actual)):
            why='historical_divergence'
            print(json.dumps(dict(prior_own=prior[step+1][seat]['action'],prior_opponent=prior[step+1][1-seat]['action'])),flush=True)
          if why:
            report=dict(seed=seed,seat=seat,step=step,error=why,python_action=pyact,cpp_action=ccact,python_rival=expected,cpp_rival=actual,cpp_debug=c.debug(),python_debug=ref.debug,python_target=list(ref.target.items()),observation=obs)
            (out/'failure.json').write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps({k:report[k] for k in ('seed','seat','step','error','python_action','cpp_action','python_rival','cpp_rival')}),flush=True);raise RuntimeError(why)
          acts=[None,None];acts[seat]=ccact;acts[1-seat]=actual;env.step(acts)
          if game:
            game.advance(acts)
            for p in (0,1):
              no=canon(env.observation(p));po=canon(game.observation(p))
              for key in ('farms','private','market','town','day','hour'):
                if no[key]!=po[key]:
                  (out/'failure.json').write_text(json.dumps(dict(seed=seed,seat=seat,step=step+1,error='official_state',key=key,native=no[key],official=po[key]),indent=2));raise RuntimeError(('official_state',seed,seat,step+1,key))
        assert env.done
        record=dict(seed=seed,seat=seat,steps=719,action_mismatches=0,official_state_mismatches=0 if game else None,cash=env.observation(seat)['farms'][seat]['money'],opponent_cash=env.observation(seat)['farms'][1-seat]['money'],g001_switched=bool(st.switched))
        records.append(record);print(json.dumps(record),flush=True)
    receipt=dict(args=vars(args),seconds=time.perf_counter()-started,games=records,build=json.loads((EXP/'native/build/build_receipt.json').read_text()),python_sha256=hashlib.sha256((EXP/'agents/agent_v7.py').read_bytes()).hexdigest())
    (out/'acceptance.json').write_text(json.dumps(receipt,indent=2),encoding='utf8')

if __name__=='__main__':main()
