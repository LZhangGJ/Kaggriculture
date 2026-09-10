"""Mock-only CLI budget/resume tests. These are not farming-game results."""
import contextlib,io,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import evaluate
class FakePool:
 calls=0
 def __init__(self,*a,**k):pass
 def __enter__(self):return self
 def __exit__(self,*a):pass
 def map(self,fn,jobs):
  for arm,job in jobs:
   FakePool.calls+=1;op,seed,seat,*_=job
   yield {'arm':arm,'opponent':op,'seed':seed,'opponent_seat':seat,'runtime_error':None,'win':True,'tie':False}
class ResumeTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.r=Path(self.tmp.name)
  for d in ['policy','baselines','referee/official']: (self.r/d).mkdir(parents=True)
  for n in ['policy/joint.so','baselines/merged.so','policy/config.json','referee/official/kaggriculture.py']:(self.r/n).write_text(n)
  (self.r/'POOL.json').write_text(json.dumps([{'id':str(i)}for i in range(11)]));FakePool.calls=0
 def tearDown(self):self.tmp.cleanup()
 def invoke(self,*extra):
  with patch.object(evaluate,'R',self.r),patch.object(evaluate.cf,'ProcessPoolExecutor',FakePool),patch.object(sys,'argv',['evaluate.py','--out',str(self.r/'out'),*extra]),contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):evaluate.main()
 def test_three_then_eight_then_repeat(self):
  self.invoke('--seeds','3');self.assertEqual(FakePool.calls,132)
  self.invoke('--seeds','8');self.assertEqual(FakePool.calls,352)
  self.invoke('--seeds','8');self.assertEqual(FakePool.calls,352)
  x=json.loads((self.r/'out/RESULTS.json').read_text());self.assertEqual(x['overall']['joint']['games'],176)
 def test_seed_cap(self):
  with self.assertRaises(SystemExit):self.invoke('--seeds','9')
 def test_worker_cap(self):
  with self.assertRaises(SystemExit):self.invoke('--workers','5')
 def test_binary_hash_changed(self):
  self.invoke('--seeds','3');(self.r/'policy/joint.so').write_text('changed')
  with self.assertRaises(SystemExit):self.invoke('--seeds','8')
 def test_seed_range_changed(self):
  self.invoke('--seeds','3')
  with self.assertRaises(SystemExit):self.invoke('--seed-start','100','--seeds','8')
 def test_duplicate_log(self):
  self.invoke('--seeds','3');p=self.r/'out/rows.jsonl';p.write_text(p.read_text()+p.read_text().splitlines()[0]+'\n')
  with self.assertRaises(SystemExit):self.invoke('--seeds','8')
if __name__=='__main__':unittest.main()
