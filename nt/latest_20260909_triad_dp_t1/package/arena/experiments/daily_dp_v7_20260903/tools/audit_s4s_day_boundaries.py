"""Paired unchanged live policies: does earlier work change the day-end state?"""
from pathlib import Path
import hashlib,json,sys,time,zlib,collections
E=Path(__file__).resolve().parents[1];out=E/'receipts/s4s_day_boundaries_v1';out.mkdir(exist_ok=False)
b=json.loads((E/'native/build/build_receipt.json').read_text())
for rel,h in b['source_hashes'].items():assert hashlib.sha256((E/rel).read_bytes()).hexdigest()==h
sys.path.insert(0,str(E/'native/build'));import _dp7_native as native
p=json.loads((E/'receipts/s4s_fourway_N50_v1/results.json').read_text());ref={(r['variant'],r['opponent'],r['seed'],r['seat']):r for r in p['rows']}
classes={'g001':(native.G001,native.G001State),'g003':(native.G001,native.G001State),'boatlee_v29':(native.BoatleeV29,native.BoatleeState),'kaito_v58':(native.KaitoV58,native.KaitoState),'lynn_v5':(native.LynnV5,native.LynnState)}
direct={'yhay81_six_day':native.Fieldbook,'yhay81_three_day':native.ThreeDay,'ecobot_v7':native.EcoBotV7}
games=[];days=[];tic=time.perf_counter()
for base in ('all_intraday','all_intraday_insert'):
    for opp,entry in p['identities'].items():
        if opp=='pass':continue
        agent=None
        if opp in classes:
            f=E/entry['asset'];assert hashlib.sha256(f.read_bytes()).hexdigest()==entry['asset_sha256'];agent=classes[opp][0](json.loads(zlib.decompress(f.read_bytes())))
        for seed in (20262701,20262702):
            for seat in (0,1):
                envs=[native.Env(seed),native.Env(seed)];ctls=[native.Controller(p['configurations'][l]) for l in (base,base+'_handoff')];rs=[classes[opp][1]() if agent else direct[opp]() for _ in range(2)]
                last_handoffs=0;local=[];different_actions=0;hand_days=0
                for step in range(719):
                    acts=[]
                    for env,c,r in zip(envs,ctls,rs):
                        own=c.act(env,seat);other=agent.act(env,1-seat,r) if agent else r.act(env,1-seat);acts.append(own);env.step([own,other] if seat==0 else [other,own])
                    different_actions+=acts[0]!=acts[1]
                    if (step+1)%24==0 or step==718:
                        a,z=(env.observation(seat) for env in envs);now=native.idle_handoff_stats(ctls[1])['applied'];n=now-last_handoffs;last_handoffs=now
                        own_a=a['farms'][seat];own_z=z['farms'][seat]
                        components={k:a[k]==z[k] for k in ('farms','private','market','town')}
                        row=dict(base=base,opponent=opp,seed=seed,seat=seat,day=step//24,step=step+1,handoffs=n,physical_equal=all(components.values()),components_equal=components,cash_delta=own_z['money']-own_a['money'],opponent_cash_delta=z['farms'][1-seat]['money']-a['farms'][1-seat]['money'],tiles_equal=own_a['tiles']==own_z['tiles'])
                        local.append(row)
                for i,label in enumerate((base,base+'_handoff')):
                    actual=envs[i].observation(seat)['farms'];expected=ref[label,opp,seed,seat];assert actual[seat]['money']==expected['cash'] and actual[1-seat]['money']==expected['opponent_cash']
                stats=native.idle_handoff_stats(ctls[1]);games.append(dict(base=base,opponent=opp,seed=seed,seat=seat,stats=stats,action_different_steps=different_actions,any_state_difference=any(not d['physical_equal'] for d in local),final_cash_difference=local[-1]['cash_delta'],final_win_changed=ref[base,opp,seed,seat]['win']!=ref[base+'_handoff',opp,seed,seat]['win']));days+=local
        print(json.dumps(dict(base=base,opponent=opp,pairs=len(games))),flush=True)
summary=dict(pairs=len(games),full_games=2*len(games),pair_days=len(days),handoffs=sum(g['stats']['applied'] for g in games),pairs_with_handoff=sum(g['stats']['applied']>0 for g in games),handoff_days=sum(d['handoffs']>0 for d in days),handoff_days_identical_endpoint=sum(d['handoffs']>0 and d['physical_equal'] for d in days),handoff_days_same_own_tiles=sum(d['handoffs']>0 and d['tiles_equal'] for d in days),pairs_changed_cash=sum(g['final_cash_difference']!=0 for g in games),pairs_changed_win=sum(g['final_win_changed'] for g in games),action_different_steps=sum(g['action_different_steps'] for g in games))
(out/'acceptance.json').write_text(json.dumps(dict(status='PASS_PAIRED_LIVE_DAY_BOUNDARIES',build=b,panel_sha256=hashlib.sha256((E/'receipts/s4s_fourway_N50_v1/results.json').read_bytes()).hexdigest(),summary=summary,games=games,days=days,seconds=time.perf_counter()-tic,caveat='Original live policies independently respond each turn. No future supplied to policy or alternate future selection. 64 matched pairs/128 repeated games; not extra independent strength samples. Same day-end physical state is not proof that every opponent has identical hidden policy memory.'),indent=2));print(json.dumps(summary),flush=True)
