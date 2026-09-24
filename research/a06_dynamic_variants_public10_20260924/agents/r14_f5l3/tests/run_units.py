"""Standard-library packaging/API sanity checks, not performance certification."""
from pathlib import Path
import argparse,ast,copy,importlib.util,json,unittest
P=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser();parser.add_argument('--cxx');args,rest=parser.parse_known_args()
spec=importlib.util.spec_from_file_location('tested_entry',P/'main.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def observation():
    farm={'money':3000,'farmer':[4,4],'hands':[],'unlocked_quadrants':['NW'],'hires_today':0,'tiles':[[None if x<5 and y<5 else 'LOCKED' for x in range(10)] for y in range(10)]}
    items=m.codec._ITEMS
    return {'step':0,'day':0,'hour':0,'player':0,'farms':[copy.deepcopy(farm),copy.deepcopy(farm)],'private':{'shed':{k:0 for k in items},'seeds':{k:0 for k in items[:5]},'inventories':[{}]},'market':{'inventory':{k:10000 for k in items[:9]},'prices':dict(zip(items[:9],[25,35,60,120,250,50,160,200,100]))},'town':{'unlocked_shops':[]}}
class Checks(unittest.TestCase):
    def test_explicit_last_entry(self):
        names=[n.name for n in ast.parse((P/'main.py').read_text()).body if isinstance(n,ast.FunctionDef)]
        self.assertEqual(names[-1],'agent')
    def test_no_trained_ranker_file(self):
        self.assertFalse((P/'policy/learned_value.hpp').exists())
        text=(P/'policy/search.hpp').read_text()
        self.assertNotIn('#include "learned_value.hpp"',text)
    def test_seed_and_other_private_not_packed(self):
        a=observation();b=copy.deepcopy(a);b['seed']=987654321;b['opponent_private']={'money':10**9}
        self.assertEqual(list(m.codec._pack(a)),list(m.codec._pack(b)))
    def test_native_config_and_no_ml(self):
        a=m.create_agent()
        try:
            self.assertEqual(a.lib.td_settings_count(),len(m.codec._ORDER))
            self.assertFalse(a.debug()['learned_value_active'])
            self.assertFalse(a.debug()['learned_value_available'])
        finally:a.close()
    def test_reject_ml_selector(self):
        cfg=json.loads((P/'policy/config.json').read_text());cfg['scenario']=-1
        with self.assertRaises(ValueError):m.codec.Agent(cfg,P/'policy/a06.so')
    def test_reject_unknown_config(self):
        with self.assertRaises(ValueError):m.codec.Agent({'not_a_setting':1},P/'policy/a06.so')
    def test_isolated_determinism_and_wrapper(self):
        a=m.create_agent();b=m.create_agent();o=observation();saved=copy.deepcopy(o)
        try:
            expected=a(o,{});self.assertEqual(expected,b(o,{'seed':9988}))
            m.reset();self.assertEqual(expected,m.agent(o,{}));self.assertEqual(o,saved)
            self.assertIsInstance(expected['market'],list)
            self.assertLessEqual(len(expected['market']),10)
        finally:a.close();b.close();m.reset()
if __name__=='__main__':unittest.main(argv=['run_units.py']+rest,verbosity=2)
