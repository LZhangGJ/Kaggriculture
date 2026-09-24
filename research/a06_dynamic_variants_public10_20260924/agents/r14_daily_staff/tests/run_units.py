"""Native settings/context contract tests, not simulator parity tests."""
import argparse, importlib.util, json, sys, unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("candidate_main",ROOT/"main.py")
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class Contract(unittest.TestCase):
    def test_independent_contexts(self):
        a=m.create_agent();b=m.create_agent()
        try:
            self.assertNotEqual(a.handle,b.handle)
            self.assertFalse(a.debug()["learned_value_active"])
            self.assertFalse(b.debug()["learned_value_available"])
            self.assertEqual(a.lib.td_settings_count(),len(m.codec._ORDER))
        finally:a.close();b.close()
    def test_no_learned_model(self):
        c=json.loads((ROOT/"policy/config.json").read_text());c["scenario"]=-1
        with self.assertRaises(ValueError):m.codec.Agent(c,ROOT/"policy/a06.so")
    def test_unknown_setting_rejected(self):
        with self.assertRaises(ValueError):m.codec.Agent({"unknown_setting":1},ROOT/"policy/a06.so")
    def test_release_and_reset(self):
        a=m.create_agent();a.close();self.assertIsNone(a.handle);a.reset()
        self.assertIsNotNone(a.handle);a.close()
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--cxx");p.parse_args()
    unittest.main(argv=[sys.argv[0]])
