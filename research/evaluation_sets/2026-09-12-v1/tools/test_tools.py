"""Focused contamination and release-gate tests, using synthetic files only."""
import contextlib
import gzip
import io
import json
from pathlib import Path
import tempfile
import unittest
from audit_seeds import audit,dump,sha
from supplement_audit import supplement
from seed_sets import sample,ROOT
from plan_jobs import plan
from release_holdout import release


class ToolTests(unittest.TestCase):
    def test_seed_recovery_from_manifest_rows_gzip_cli_and_code(self):
        with tempfile.TemporaryDirectory(dir=ROOT.parents[1]/'work') as td:
            base=Path(td);repo=base/'repo';repo.mkdir()
            dump(repo/'manifest.json',{'seeds':[9001,9002],'training_seeds':[17],'future_seed':8123})
            (repo/'run.jsonl').write_text('{"seed": 7711}\n{"future_seed": 8711}\n')
            (repo/'trace_seed901.json.gz').write_bytes(gzip.compress(b'{"seed": 1011}'))
            (repo/'driver.py').write_text("train_seeds=list(range(8800,8803))\ncommand=['--seed-start','7700','--seeds','4']\n")
            with contextlib.redirect_stdout(io.StringIO()):audit(repo,base/'audit');supplement(repo,base/'audit')
            found=set(json.loads((base/'audit/exclusions.json').read_text())['seeds'])
            self.assertTrue({9001,9002,17,8123,7711,8711,901,1011,8800,8801,8802,7700,7701,7702,7703}<=found)

    def test_sampling_rejects_exclusions_without_redrawing(self):
        key='01'*32;first,_=sample(key,'fixture',40,set())
        second,_=sample(key,'fixture',30,set(first[:10]))
        self.assertEqual(second,first[10:]);self.assertEqual(len(set(second)),30)

    def test_holdout_job_plan_refuses_missing_release_before_write(self):
        with tempfile.TemporaryDirectory(dir=ROOT.parents[1]/'work') as td:
            out=Path(td)/'jobs'
            with self.assertRaisesRegex(ValueError,'release receipt'):plan(ROOT,'holdout','connected-day3-v1',out)
            self.assertFalse(out.exists())

    def test_synthetic_release_checks_frozen_bytes_and_contamination(self):
        with tempfile.TemporaryDirectory(dir=ROOT.parents[1]/'work') as td:
            base=Path(td);root=base/'bundle';repo=base/'source';repo.mkdir();root.mkdir()
            candidates=[];frozen={}
            for name in ('connected-day3-v1','expected-selector-v1','original-afs-r2'):
                folder=repo/name;folder.mkdir();path=folder/'main.py';path.write_text('# fixture '+name)
                rel=f'{name}/main.py' if name=='original-afs-r2' else 'main.py'
                candidates.append(dict(id=name,path=name,runtime_files={rel:sha(path)}));frozen[name]={str(path):sha(path)}
            evaluator=repo/'runner.py';evaluator.write_text('# fixture runner')
            analysis=root/'analysis.md';analysis.write_text('Frozen synthetic analysis')
            dump(root/'candidates.json',dict(candidates=candidates));dump(root/'opponents.json',dict(opponents=[]))
            dump(root/'provenance.json',dict(source_root=str(repo)))
            dump(root/'sealed/holdout.json',dict(seeds=list(range(10000,10256))))
            dump(root/'holdout_receipt.json',dict(sha256=sha(root/'sealed/holdout.json')))
            freeze=base/'freeze.json';dump(freeze,dict(finalists_frozen=True,candidates=frozen,evaluator_files={str(evaluator):sha(evaluator)},
                analysis_files={str(analysis):sha(analysis)},opponents_sha256=sha(root/'opponents.json'),frozen_utc='2026-01-01T00:00:00Z'))
            aud=base/'audit';dump(aud/'exclusions.json',dict(seeds=[1,2,3]))
            summary=dict(source_root=str(repo),inventory_started_utc='2026-01-02T00:00:00Z',changed_files=[],new_files_after_inventory=[],
                         npz_seed_arrays_recovered=True,exclusions_sha256=sha(aud/'exclusions.json'),gaps=[])
            dump(aud/'summary.json',summary)
            with contextlib.redirect_stdout(io.StringIO()):release(root,freeze,aud,base/'released')
            self.assertTrue((base/'released/release_receipt.json').is_file())
            evaluator.write_text('# changed')
            with self.assertRaisesRegex(ValueError,'Frozen file mismatch'):release(root,freeze,aud,base/'changed')
            evaluator.write_text('# fixture runner');dump(aud/'exclusions.json',dict(seeds=[10000]));summary['exclusions_sha256']=sha(aud/'exclusions.json');dump(aud/'summary.json',summary)
            with self.assertRaisesRegex(ValueError,'contamination'):release(root,freeze,aud,base/'contaminated')
            self.assertFalse((base/'contaminated').exists())


if __name__=='__main__':unittest.main()
