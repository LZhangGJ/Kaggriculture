"""Read-only live trajectory audit; do not convert realized futures to policy."""
from pathlib import Path
import argparse,gzip,hashlib,json,shlex,subprocess,sys,sysconfig,time,zlib
E=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    cli=argparse.ArgumentParser();cli.add_argument('--build-only',action='store_true');a=cli.parse_args()
    build=json.loads((E/'native/build/build_receipt.json').read_text())
    for rel,h in build['source_hashes'].items():assert sha(E/rel)==h
    dst=E/'native/market_execution_probe_build';dst.mkdir(exist_ok=True)
    binary=dst/('_dp7_market_probe'+sysconfig.get_config_var('EXT_SUFFIX'))
    src=E/'native/market_execution_probe.cpp';stamp=dst/'build_receipt.json'
    fresh=stamp.exists() and binary.exists()
    if fresh:
        r=json.loads(stamp.read_text());fresh=r['build']['binary_sha256']==build['binary_sha256'] and r['probe_source_sha256']==sha(src) and r['binary_sha256']==sha(binary)
    if not fresh:
        assert not binary.exists(),'Preserve old probe before rebuilding'
        include=shlex.split(subprocess.check_output([sys.executable,'-m','pybind11','--includes'],text=True))
        cmd=['g++','-std=c++20','-O2','-ffp-contract=off','-fPIC','-shared',*include,'-I'+str(E/'native'),str(src),'-o',str(binary)]
        subprocess.run(cmd,check=True);stamp.write_text(json.dumps(dict(build=build,probe_source_sha256=sha(src),binary_sha256=sha(binary),command=cmd),indent=2))
    if a.build_only:print('READ_ONLY_PROBE_BUILT',flush=True);return
    # This module uses the currently validated production Controller ABI.
    sys.path.insert(0,str(E/'native/build'));sys.path.insert(0,str(dst))
    import _dp7_native as native
    import _dp7_market_probe as probe
    panel_path=E/'receipts/s4u_fiveway_N50_v1/results.json';panel=json.loads(panel_path.read_text())
    assert panel['status']=='COMPLETE_PANEL_NOT_GOAL_ACCEPTANCE'
    expected={(r['variant'],r['opponent'],r['seed'],r['seat']):r for r in panel['rows']}
    out=E/'receipts/s4v_market_execution_audit_v1';out.mkdir(exist_ok=False)
    classes={'g001':(native.G001,native.G001State),'g003':(native.G001,native.G001State),'boatlee_v29':(native.BoatleeV29,native.BoatleeState),'kaito_v58':(native.KaitoV58,native.KaitoState),'lynn_v5':(native.LynnV5,native.LynnState)}
    direct={'yhay81_six_day':native.Fieldbook,'yhay81_three_day':native.ThreeDay,'ecobot_v7':native.EcoBotV7}
    items=['WHEAT','CARROT','TOMATO','STRAWBERRY','MELON','EGG','MILK','WOOL','FERTILIZER']
    games=[];tic=time.perf_counter()
    for label in ('all_intraday','all_intraday_insert','full_chain_autonomous'):
        for opp,entry in panel['identities'].items():
            if opp=='pass':continue
            asset=None
            if opp in classes:
                asset=E/entry['asset'];assert sha(asset)==entry['asset_sha256'];agent=classes[opp][0](json.loads(zlib.decompress(asset.read_bytes())))
            for seed in (20262701,20262702):
                for seat in (0,1):
                    env=native.Env(seed);c=native.Controller(panel['configurations'][label]);state=classes[opp][1]() if asset else direct[opp]()
                    samples=[];offered=0;omitted=0;active_steps=0;miss_steps=0
                    for step in range(719):
                        obs=env.observation(seat);own=c.act(env,seat);debug=c.debug();before=native.schedule_cache_stats(c)
                        info=probe.inspect(c,env,seat)
                        assert c.debug()==debug and native.schedule_cache_stats(c)==before,'probe mutated policy'
                        sales={i:sum(int(x[2]) for x in own['market'] if x[0]=='SELL' and x[1]==i) for i in items}
                        picks={i:sum(int(x[2]) for x in [own['farmer'],*own['hands']] if x[0]=='PICKUP' and x[1]==i) for i in items}
                        missing={i:max(0,info['tactical'][j]-sales[i]-picks[i]) for j,i in enumerate(items)}
                        quantity=sum(info['tactical']);gap=sum(missing.values());offered+=quantity;omitted+=gap;active_steps+=quantity>0;miss_steps+=gap>0
                        other=agent.act(env,1-seat,state) if asset else state.act(env,1-seat)
                        if gap>0:
                            samples.append(dict(step=step,day=obs['day'],hour=obs['hour'],phase=info['phase'],missing=missing,
                                available=obs['private']['shed'],inventory=obs['market']['inventory'],prices=obs['market']['prices'],
                                tactical=info['tactical'],own_action=own,opponent_action=other,
                                caveat='Opponent action recorded after decision, offline only. Repeated inventory across steps is not unique lost product.'))
                        env.step([own,other] if seat==0 else [other,own])
                    exp=expected[label,opp,seed,seat];farm=env.observation(seat)['farms']
                    assert farm[seat]['money']==exp['cash'] and farm[1-seat]['money']==exp['opponent_cash']
                    row=dict(label=label,opponent=opp,seed=seed,seat=seat,steps=719,offered_quantity_steps=offered,omitted_quantity_steps=omitted,tactical_steps=active_steps,omitted_steps=miss_steps,cash=exp['cash'],opponent_cash=exp['opponent_cash'])
                    name=f'{label}_{opp}_{seed}_{seat}.json.gz';(out/name).write_bytes(gzip.compress(json.dumps(dict(summary=row,samples=samples)).encode()))
                    games.append(row);(out/'progress.json').write_text(json.dumps(games,indent=2));print(json.dumps(row),flush=True)
    for rel,h in build['source_hashes'].items():assert sha(E/rel)==h
    (out/'acceptance.json').write_text(json.dumps(dict(status='PASS_READ_ONLY_LIVE_MARKET_AUDIT',build=build,rows=games,full_games=len(games),seconds=time.perf_counter()-tic,
        panel_sha256=sha(panel_path),script_sha256=sha(Path(__file__)),probe_build_sha256=sha(stamp),
        caveat='Same 96 development matches repeated, not new strength evidence. An omitted heuristic is not proven lost profit; no hindsight action inserted.'),indent=2))
if __name__=='__main__':main()
