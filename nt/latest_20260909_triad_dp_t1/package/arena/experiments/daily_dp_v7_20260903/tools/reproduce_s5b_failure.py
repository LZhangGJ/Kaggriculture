from pathlib import Path
import json,sys,zlib
E=Path(__file__).resolve().parents[1];sys.path.insert(0,str(E/'native/build'));import _dp7_native as n
out=E/'receipts/s5b_runtime_failure_v1';out.mkdir(exist_ok=False)
label='full_chain_autonomous_timing';cfg=json.loads((E/'profiles/s5b/configs.json').read_text())[label]
env=n.Env(20262701);c=n.Controller(cfg);r=n.G001(json.loads(zlib.decompress((E/'native/g003_frozen.json.zlib').read_bytes())));st=n.G001State()
while not env.done:
 step=env.step_count
 try:own=c.act(env,0)
 except Exception as ex:
  data=dict(status='REPRODUCED_FAILURE',label=label,seed=20262701,seat=0,step=step,error=str(ex),debug=c.debug(),rotation=n.rotation_stats(c),observation=env.observation(0),build=json.loads((E/'native/build/build_receipt.json').read_text()))
  (out/'failure.json').write_text(json.dumps(data,indent=2));print(json.dumps({k:data[k] for k in ('status','step','error','debug','rotation')}),flush=True);break
 other=r.act(env,1,st);env.step([own,other])
else:raise RuntimeError('original failure not reproduced')
