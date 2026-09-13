"""Root callable, native ABI and unknown-field noninterference on legal observations."""
import pathlib,sys,json,importlib.util,copy,hashlib
R=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(R/'input/referee'));from cpu_runtime import LocalGame,load_engine
s=importlib.util.spec_from_file_location('runtime_privacy_root',R/'candidate/main.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
eng=load_engine();panel=json.loads((R/'diagnostics/closed_loop_panel.json').read_text());rows=[]
for seed in panel['seeds'][:2]:
 g=LocalGame(seed,eng)
 for seat in (0,1):
  obs=g.observation(seat);other=copy.deepcopy(obs)
  # Fabricated canaries only: do not retrieve or pass real opponent private state.
  other['seed']=987654321;other['opponent_name']='CANARY_NOT_AN_OPPONENT_ID'
  other['future_states']=[{'unexpected':'must_not_be_encoded'}]
  other['farms'][1-seat]['private']={'shed':{'WHEAT':99999999}}
  a=bytes(m.codec._pack(obs));b=bytes(m.codec._pack(other));assert a==b
  m.reset();normal=m.agent(obs,g.configuration);m.reset();modified=m.agent(other,{'seed':1234,'hidden_script':'canary'});assert normal==modified
  count=m._seats[seat].lib.td_settings_count();assert count==len(m.codec._ORDER)
  rows.append({'environment_seed_not_passed_to_policy':seed,'seat':seat,'packed_observation_sha256':hashlib.sha256(a).hexdigest(),'canary_invariance':True,'root_action_sha256':hashlib.sha256(json.dumps(normal,sort_keys=True).encode()).hexdigest(),'abi_settings_count':count});m.reset();assert not m._seats
out={'scope':'Unknown-field noninterference and actual main.py:agent/reset entry tests; fabricated canaries are not real private data. No games or win-rate estimate.','cases':rows,'all_passed':True}
(R/'diagnostics/runtime_privacy.json').write_text(json.dumps(out,indent=2));print(json.dumps(out),flush=True)
