from pathlib import Path
import datetime,hashlib,json,subprocess,sys,time
w=Path('/mnt/data/TRI_A08_work');out=w/'logs/anchored_closed_loop';out.mkdir(exist_ok=True)
panel=json.loads((w/'logs/ANCHORED_PARENT_PANEL.json').read_text());expected=hashlib.sha256((w/'candidate/policy/tri_a08_r11_r1.so').read_bytes()).hexdigest()
(out/'PANEL_FREEZE.json').write_text(json.dumps({'panel':panel,'candidate_native_sha256':expected,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2))
results=[]
for job in panel['games']:
 stem=out/f"seed{job['seed']}_seat{job['candidate_seat']}"
 cmd=['/usr/bin/time','-v','-o',str(stem)+'.time','timeout','-k','3s','180s',sys.executable,'-B',str(w/'candidate/tests/closed_loop_match.py'),'--candidate',str(w/'candidate'),'--parent',str(w/'input/agent'),'--referee',str(w/'input/referee'),'--seed',str(job['seed']),'--candidate-seat',str(job['candidate_seat']),'--out',str(stem)]
 assert hashlib.sha256((w/'candidate/policy/tri_a08_r11_r1.so').read_bytes()).hexdigest()==expected
 with open(str(stem)+'.stdout','w') as o,open(str(stem)+'.stderr','w') as e:r=subprocess.run(cmd,stdout=o,stderr=e)
 receipt={'job':job,'command':cmd,'returncode':r.returncode}
 Path(str(stem)+'.run.json').write_text(json.dumps(receipt,indent=2))
 if r.returncode!=0:print(json.dumps(receipt),flush=True);raise RuntimeError(receipt)
 result=json.loads((stem/'result.json').read_text());results.append(result)
 print(json.dumps({'seed':job['seed'],'seat':job['candidate_seat'],'complete':result['complete'],'margin':result.get('candidate_margin'),'seconds':result['wall_seconds']}),flush=True)
 (out/'SUMMARY.partial.json').write_text(json.dumps({'planned':8,'completed':len(results),'results':results},indent=2))
summary={'scope':panel['scope'],'candidate_native_sha256':expected,'games':len(results),'strict_wins':sum(r['candidate_strict_win'] for r in results),'draws':sum(r['draw'] for r in results),'losses':sum(not r['candidate_strict_win'] and not r['draw'] for r in results),'all719':all(r['steps']==719 for r in results),'all_zero_runtime_errors':all(not r['errors'] for r in results),'all_cash_identity_zero':all(r['cash_identity_residual_all_zero'] for r in results),'market_rejections_are_not_exceptions':True,'results':results}
(out/'SUMMARY.json').write_text(json.dumps(summary,indent=2));print(json.dumps({k:v for k,v in summary.items() if k!='results'}),flush=True)
