"""Read-only structural audit of generated training schedules, no simulation."""
from pathlib import Path
import ast,json,hashlib
ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT.parent/'economic_rl_keep2_300_20260906'

def tree(path):return ast.parse(path.read_text(encoding='utf8'))
def calls(t,name):
    return [ast.dump(n,include_attributes=False)for n in ast.walk(t)if isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id==name]
old=tree(OLD/'train300.py');receipt={}
for end in range(400,1001,100):
    start=end-100;p=ROOT/'stages'/f'round{end:04}'
    training=tree(p/f'train{end}.py')
    for name in ('update_aux','rollout','evaluate','checkpoint'):
        assert calls(training,name)==calls(old,name),(end,name)
    loops=[n for n in ast.walk(training)if isinstance(n,ast.For)and isinstance(n.target,ast.Name)and n.target.id=='step']
    assert len(loops)==1 and isinstance(loops[0].iter,ast.Call)
    assert ast.literal_eval(loops[0].iter.args[1])==end+1
    assert f"source/'step{start}.pt'"in (p/f'train{end}.py').read_text()
    assert f"final_optimizer['steps']==[{end*56}]"in (p/f'train{end}.py').read_text()
    audit=(p/f'audit{end}.py').read_text()
    assert f'list(range({start+1},{end+1}))'in audit
    assert f'selection[:{start//20+1}]'in audit
    source_text=(p/f'common{end}.py').read_text()
    assert f'if step<={start}:'in source_text
    run=(p/f'run_all{end}.py').read_text()
    assert f'start_round={start},end_round={end}'in run
    assert f'new_training_seed_starts=[{69600000+start*16},{69700000+start*16}]'in run
    assert 'final_seed_start=70300000,final_seeds=100'in run
    pre=(p/f'preflight{end}.py').read_text()
    assert f"before['steps']==[{start*56}]"in pre
    assert f"after['steps']==[{start*56+4}]"in pre
    for file in p.glob('*.py'):tree(file)
    receipt[str(end)]=dict(start=start,end=end,core_calls_unchanged=True,expected_adam_updates=end*56,
        cumulative_games_per_model=end*224,training_seed_intervals=[[b+start*16,b+end*16-1]for b in (69600000,69700000)])
result=dict(status='PASS',stages=receipt,note='Static schedule and call-equivalence checks; runtime and performance require separate stage audits.')
(ROOT/'diagnostics'/'SCHEDULE_ACCEPTANCE.json').write_text(json.dumps(result,indent=2),encoding='utf8')
print(json.dumps(result),flush=True)
