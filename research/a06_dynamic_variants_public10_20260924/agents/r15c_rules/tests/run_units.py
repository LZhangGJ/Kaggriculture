"""Native ABI, rule-only configuration and context-lifetime checks (not a game test)."""
from pathlib import Path
import argparse,importlib.util,json
p=argparse.ArgumentParser();p.add_argument('--cxx');p.parse_args()
r=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('runtime_test',r/'main.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
a=m.create_agent();b=m.create_agent()
try:
    assert a.handle and b.handle and a.handle!=b.handle
    assert a.lib.td_settings_count()==len(m.codec._ORDER)
    assert a.debug()['learned_value_available'] is False
    assert a.debug()['learned_value_active'] is False
    assert a.config['scenario']>=0 and a.config['rule_opening']==0
    bad=dict(a.config);bad['scenario']=-1
    try:
        rejected=m.codec.Agent(config=bad,binary_path=r/'policy/a06.so')
    except ValueError:pass
    else:
        rejected.close();raise AssertionError('Learned mode must be rejected')
    print(json.dumps({'passed':True,'native_settings_count':a.lib.td_settings_count(),
        'no_learned_weights':True,'negative_scenario_rejected':True,
        'independent_contexts':True,'scope':'ABI/configuration only, not gameplay parity'}))
finally:
    a.close();b.close()
