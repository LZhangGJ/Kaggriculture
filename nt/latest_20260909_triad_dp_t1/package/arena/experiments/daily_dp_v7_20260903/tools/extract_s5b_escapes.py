from pathlib import Path
import gzip,hashlib,json
E=Path(__file__).resolve().parents[1];out=E/'receipts/s5b_escape_triage_v1';out.mkdir(exist_ok=False)
panel=json.loads((E/'receipts/s5b_eightway_N10_v1/results.json').read_text());found=[]
for label,opponent in [('full_chain_autonomous_net','kaito_v58'),('full_chain_autonomous_net','yhay81_three_day'),('full_chain_autonomous_timing','yhay81_three_day')]:
 path=E/f'receipts/s5b_pool_audit_N10_v1/{label}_{opponent}.json.gz';raw=json.load(gzip.open(path,'rt'))
 base=json.load(gzip.open(E/f'receipts/s4u_pool_audit_N50_v1/full_chain_autonomous_{opponent}.json.gz','rt'))['rows'];base={(r['seed'],r['seat']):r for r in base}
 for r in raw['rows']:
  seat=r['seat'];days=[i for i,d in enumerate(r['production']) if sum(d[seat]['escaped'])]
  if not days:continue
  previous=base[r['seed'],seat];old=sum(sum(d[seat]['escaped']) for d in previous['production'])
  event=dict(label=label,opponent=opponent,seed=r['seed'],seat=seat,cash=r['money'][seat],baseline_cash=previous['money'][seat],baseline_escapes=old,escape_days=days,escaped=sum(sum(d[seat]['escaped']) for d in r['production']))
  scope=sorted({i for day in days for i in range(max(0,day-2),day+1)})
  detail=dict(summary=event,window=[dict(day=i,production=r['production'][i][seat],cash=r['cash_ledger'][i][seat],planning=r['planning'][i]) for i in scope])
  (out/f'{label}_{opponent}_{r["seed"]}_{seat}.json').write_text(json.dumps(detail,indent=2));found.append(event)
receipt=dict(status='COMPLETE_REAL_FAILURE_TRIAGE',rows=found,failed_games=len(found),total_escaped=sum(r['escaped'] for r in found),build=panel['build'],reason='No planned animal release exists in these configs. Treat escapes as hard regression, not profitable deliberate exit.')
(out/'acceptance.json').write_text(json.dumps(receipt,indent=2));print(json.dumps({k:v for k,v in receipt.items() if k!='build'},indent=2))
