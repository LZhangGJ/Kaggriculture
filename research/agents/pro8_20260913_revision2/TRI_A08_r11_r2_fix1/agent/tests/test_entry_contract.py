"""Actual deployed root, fixed config, library receipt and legal-input contract."""
from __future__ import annotations
import copy,gzip,hashlib,importlib.util,json,pathlib,sys
ROOT=pathlib.Path(__file__).resolve().parents[1]
PROD=ROOT/'candidate' if (ROOT/'candidate').exists() else ROOT
EVIDENCE=ROOT/'input' if (ROOT/'input').exists() else ROOT/'evidence'
checks=0
def check(b,why):
 global checks
 checks+=1
 if not b:raise AssertionError(why)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
receipt=json.loads((PROD/'policy/tri_a08_r11_r2_fix1.BUILD.json').read_text())
check(sha(PROD/'policy/tri_a08_r11_r2_fix1.so')==receipt['binary_sha256'],'native receipt mismatch')
for name,value in receipt['sources'].items():check(sha(PROD/name)==value,'source receipt mismatch: '+name)
parent=ROOT/'parent_immutable' if (ROOT/'parent_immutable').exists() else ROOT/'reference/parent'
check((PROD/'policy/config.json').read_bytes()==(parent/'policy/config.json').read_bytes(),'fixed config changed')
spec=importlib.util.spec_from_file_location('contract_candidate',PROD/'main.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
traces=[json.loads(gzip.decompress((EVIDENCE/'own_traces'/f'{name}.json.gz').read_bytes())) for name in ('flexon_v5_113649547_seat0','submission_56149565_521596133_seat1')]
try:
 independent=[]
 for tr in traces:
  a=mod.create_agent();expected=[]
  check(pathlib.Path(a.lib._name).resolve()==(PROD/'policy/tri_a08_r11_r2_fix1.so').resolve(),'wrong native loaded')
  check(a.lib.td_settings_count()==len(mod.codec._ORDER),'ABI settings mismatch')
  for obs in tr['observations'][:48]:expected.append(a(copy.deepcopy(obs)))
  independent.append(expected);a.close()
 # Interleaving two real own-visible traces must equal two isolated contexts.
 for i in range(48):
  for seat,tr in enumerate(traces):
   check(mod.agent(copy.deepcopy(tr['observations'][i]),tr['configuration'])==independent[seat][i],'seat state contamination')
 # Fresh episode reset must be deterministic.
 for seat,tr in enumerate(traces):check(mod.agent(copy.deepcopy(tr['observations'][0]))==independent[seat][0],'step-zero reset mismatch')
 mod.reset()
 # Noncontract metadata is not an input to the packed observation or policy.
 for seat,tr in enumerate(traces):
  a=mod.create_agent()
  for i in range(8):
   obs=copy.deepcopy(tr['observations'][i]);plain=bytes(mod.codec._pack(obs))
   obs.update(seed=987654321,opponent_entry_label='ignored_metadata',future_state={'fabricated':True})
   for farm in obs['farms']:farm['private']={'this_is_not_a_contract_field':True}
   check(bytes(mod.codec._pack(obs))==plain,'ignored metadata leaked into codec')
   check(a(obs,{'seed':987654321})==independent[seat][i],'metadata-based policy branch')
  a.close()
 # Fail loudly on missing library, unknown config and malformed observations.
 try:mod.create_agent('/does/not/exist/missing.so')
 except OSError:check(True,'')
 else:check(False,'missing library silently accepted')
 try:mod.codec.Agent(config={'not_a_setting':1},binary_path=PROD/'policy/tri_a08_r11_r2_fix1.so')
 except ValueError:check(True,'')
 else:check(False,'unknown setting accepted')
 a=mod.create_agent()
 try:a({'step':0,'player':0,'farms':[]})
 except ValueError:check(True,'')
 else:check(False,'malformed observation silently accepted')
 a.close()
finally:mod.reset()
print(json.dumps({'status':'PASS','checks':checks,'new_games':0,'native_sha256':receipt['binary_sha256'],'scope':'Root dispatch, receipt, ABI, context isolation and input contract; not a sandbox timing guarantee'}))
