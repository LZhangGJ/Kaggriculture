"""Existing policy's harvest/sale gates and price forecast, evaluated post hoc.

No policy choice is made using future prices; these are historical predictions
versus subsequently observed prices, NOT counterfactual trading profits.
"""
from pathlib import Path
import argparse,gzip,hashlib,json,shlex,subprocess,sys,sysconfig,time,zlib
E=Path(__file__).resolve().parents[1]
cli=argparse.ArgumentParser();cli.add_argument('--build-only',action='store_true');cli.add_argument('--build',default='native/cash_window_probe_build_v2');a=cli.parse_args()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
build=json.loads((E/'native/build/build_receipt.json').read_text())
for rel,h in build['source_hashes'].items():assert sha(E/rel)==h
dst=E/a.build;src=E/'native/cash_window_probe.cpp';stamp=dst/'build_receipt.json'
if a.build_only:
    dst.mkdir(exist_ok=False);binary=dst/('_dp7_cash_window_probe'+sysconfig.get_config_var('EXT_SUFFIX'))
    inc=shlex.split(subprocess.check_output([sys.executable,'-m','pybind11','--includes'],text=True));cmd=['g++','-std=c++20','-O2','-ffp-contract=off','-fPIC','-shared',*inc,'-I'+str(E/'native'),str(src),'-o',str(binary)]
    r=subprocess.run(cmd,capture_output=True,text=True);(dst/'compile.log').write_text(r.stdout+r.stderr)
    if r.returncode:print(r.stderr[-3500:]);raise SystemExit(r.returncode)
    stamp.write_text(json.dumps(dict(build=build,source_sha256=sha(src),binary_sha256=sha(binary),command=cmd),indent=2));print('COMPILE_ONLY_PASS');raise SystemExit(0)
frozen=json.loads(stamp.read_text());assert frozen['build']==build and frozen['source_sha256']==sha(src)
sys.path.insert(0,str(E/'native/build'));sys.path.insert(0,str(dst));import _dp7_native as n;import _dp7_cash_window_probe as probe
panel_path=E/'receipts/s4v_twelveway_N50_v1/results.json';panel=json.loads(panel_path.read_text());refs={(r['variant'],r['opponent'],r['seed'],r['seat']):r for r in panel['rows']}
classes={'g001':(n.G001,n.G001State),'g003':(n.G001,n.G001State),'boatlee_v29':(n.BoatleeV29,n.BoatleeState),'kaito_v58':(n.KaitoV58,n.KaitoState),'lynn_v5':(n.LynnV5,n.LynnState)}
direct={'yhay81_six_day':n.Fieldbook,'yhay81_three_day':n.ThreeDay,'ecobot_v7':n.EcoBotV7}
names=['WHEAT','CARROT','TOMATO','STRAWBERRY','MELON','EGG','MILK','WOOL','FERTILIZER']
out=E/'receipts/s4x_cash_window_audit_v1';out.mkdir(exist_ok=False);games=[];tic=time.perf_counter()
for label in ('all_intraday_insert','full_chain_autonomous'):
 for opp,entry in panel['identities'].items():
  if opp=='pass':continue
  rival=None
  if opp in classes:
   asset=E/entry['asset'];assert sha(asset)==entry['asset_sha256'];rival=classes[opp][0](json.loads(zlib.decompress(asset.read_bytes())))
  for seed in (20262701,20262702):
   for seat in (0,1):
    env=n.Env(seed);c=n.Controller(panel['configurations'][label]);state=classes[opp][1]() if rival else direct[opp]();rows=[];market=[];net=[0]*9;net_history=[]
    for step in range(719):
     own=c.act(env,seat);before=c.debug();r=probe.inspect(c,env,seat);assert before==c.debug()
     obs=env.observation(seat);issued=set()
     for u,action in enumerate([own['farmer'],*own['hands']]):
      if action[0]=='HARVEST':
       pos=obs['farms'][seat]['farmer'] if u==0 else obs['farms'][seat]['hands'][u-1]
       issued.add(pos[1]*10+pos[0])
     latent=[x for x in r['mature'] if not x['generated'] and not x['scheduled'] and x['pos'] not in issued]
     market.append(r['inventory']);net_history.append(list(net))
     if r['same_day_controller'] and step%24<23 and (latent or any(r['tactical'])):
      rows.append(dict(step=step,phase=r['phase'],latent=latent,tactical=r['tactical'],inventory=r['inventory'],forecast_inventory=r['forecast_inventory'],forecast_end=r['forecast_end'],own_action=own))
     other=rival.act(env,1-seat,state) if rival else state.act(env,1-seat)
     env.step([own,other] if seat==0 else [other,own]);fills=probe.fills(env,seat)
     for order,q in zip(own['market'],fills):
      if order[0] in ('SELL','BUY_PRODUCT') and order[1] in names:net[names.index(order[1])]+=q if order[0]=='SELL' else -q
    ref=refs[label,opp,seed,seat];farms=env.observation(seat)['farms'];assert farms[seat]['money']==ref['cash'] and farms[1-seat]['money']==ref['opponent_cash']
    # Comparing SAME unmodified trajectory only. Includes feedback of actual
    # own orders; adjusted stock removes own filled orders but NOT feedback.
    for r in rows:
     h=r['forecast_end'];s=r['step']
     if h>=719:continue
     r['later_observed_inventory']=market[h]
     r['own_net_orders_before_horizon']=[net_history[h][i]-net_history[s][i] for i in range(9)]
     r['later_inventory_excluding_direct_own_orders']=[market[h][i]-r['own_net_orders_before_horizon'][i] for i in range(9)]
    summary=dict(label=label,opponent=opp,seed=seed,seat=seat,cash=ref['cash'],opponent_cash=ref['opponent_cash'],latent_steps=sum(bool(r['latent']) for r in rows),tactical_steps=sum(any(r['tactical']) for r in rows),control_equal=True)
    (out/f'{label}_{opp}_{seed}_{seat}.json.gz').write_bytes(gzip.compress(json.dumps(dict(summary=summary,rows=rows)).encode()));games.append(summary);(out/'progress.json').write_text(json.dumps(games,indent=2));print(json.dumps(summary),flush=True)
(out/'acceptance.json').write_text(json.dumps(dict(status='PASS_READ_ONLY_HARVEST_MARKET_AUDIT',build=frozen,rows=games,seconds=time.perf_counter()-tic,panel_sha256=sha(panel_path),script_sha256=sha(Path(__file__)),final_goal_acceptance=False,
 caveat='Repeated stock is quantity-steps, not distinct opportunities. Future observations only evaluate historical predictions, never select policy actions. No counterfactual profit claim.'),indent=2));print('PASS_READ_ONLY_HARVEST_MARKET_AUDIT',flush=True)
