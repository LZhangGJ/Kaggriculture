"""Regression tests use synthetic files and results; no policy is imported."""
import contextlib
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import contracts as c
import panel_runner as runner
import holdout
import analyze_panels as analysis
import numpy as np


def game(key, winner=False, draw=False):
    cid, panel, seed, opponent, seat = key
    shops = ['YARN_STORE'] * 8
    economy = dict(samples=180, unlock_events=[dict(step=s, shop='YARN_STORE') for s in range(72, 577, 72)],
                   market={item: dict(floor_samples=0, inventory_sum=1800) for item in analysis.PRODUCTS})
    own, rival = (11, 10) if winner else (10, 10) if draw else (9, 10)
    return dict(candidate_id=cid, panel=panel, seed=seed, opponent=opponent, opponent_seat=seat,
                candidate_seat=1-seat, valid=True, steps=719, runtime_error=None, own_cash=own,
                opponent_cash=rival, margin=own-rival, win=winner, tie=draw, action_hash='synthetic',
                terminal_shops=shops, economy=economy, latency_max=.001, seconds=.01,
                observations_checked=1440)


class Fixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.bundle = self.root / 'bundle'
        self.inputs = self.root / 'inputs'
        runtime = self.bundle / 'evaluation/runtime'
        required = [
            'research/robust90/stress.py',
            'nt/latest_20260911_afs_workflow_repair_r1_r2/r2/policy/agent.so',
            'nt/latest_20260911_afs_workflow_repair_r1_r2/r2/policy/config.json',
            'nt/latest_20260911_p16_jointafs_r1/agent/policy/agent.py',
        ]
        for name in required:
            path = runtime / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'SYNTHETIC TEST DATA - NEVER IMPORT')
        files = {name: c.sha(runtime / name) for name in required}
        self.roster = dict(candidates=[dict(id=cid, name=cid, binary=required[1],
                           binary_sha256=files[required[1]], config={}) for cid in ('alpha', 'beta')],
                           runtime_files=files)
        c.write(self.bundle / 'evaluation/roster.json', self.roster)
        c.write(self.bundle / 'opponents.json', {'opponents': [dict(id=f'opp{i}',runtime_files=files) for i in range(16)]})
        for name, values in [('representative', [101, 202]), ('stress', [303, 404]), ('stress_pool', [303, 404, 505])]:
            c.write(self.bundle / f'manifests/{name}.json', {'seeds': values})
        for name, values in [('representative', [101, 202]), ('stress', [303, 404])]:
            path = self.bundle / f'features/{name}.jsonl'
            path.parent.mkdir(parents=True, exist_ok=True)
            rows = [dict(seed=s, reference_first_shop='YARN_STORE', features=dict(
                ref_milk_minus_wool=i*2-1, ref_crop_minus_animal=-1, ref_shop_variety=1,
                ref_max_duplicate=8)) for i,s in enumerate(values)]
            path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        c.write(self.bundle / 'evaluation/condition_thresholds.json',
                {'thresholds': {'ref_milk_minus_wool': {'q25': -.5, 'q75': .5}}})
        c.write(self.bundle / 'campaign_audit/report.json', {'status': 'PASS'})
        c.write(self.bundle / 'campaign_audit/exclusions.json', {'seeds': [1, 2, 3]})
        self.retired = self.root / 'retired.json'
        c.write(self.retired, {'seeds': list(range(1000,1256))})
        c.write(self.bundle / 'holdout_receipt.json', {'count': 256, 'sha256': c.sha(self.retired)})
        c.write(self.bundle / 'evaluation/HOLDOUT_RELEASE.json',
                {'holdout_sha256': c.sha(self.retired), 'roster_sha256': c.sha(self.bundle / 'evaluation/roster.json')})
        for name in ('evaluation/panel_runner.py', 'evaluation/ANALYSIS_PLAN.md', 'tools/economy_features.py'):
            path = self.bundle / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('SYNTHETIC TEST DATA')
        (self.bundle/'tools/economy_features.py').write_text(
            "def characterize(seed):\n    return {'seed': seed, 'source': 'frozen generator'}\n")
        c.create_freeze(self.bundle, self.inputs)
        self.freeze = self.inputs / 'FREEZE.json'
        _, self.plan, self.roster, self.opponents = c.verify_freeze(self.freeze, self.bundle)

    def make_preflight(self):
        output = self.root / 'preflight'
        manifest, expected, _ = runner.build_manifest(self.freeze, self.bundle, 'preflight', 1)
        c.write(output / 'MANIFEST.json', manifest)
        rows = []
        for key in expected:
            result = game(key)
            rows.append(dict(candidate_id=key[0], panel=key[1], seed=key[2], opponent=key[3],
                opponent_seat=key[4], valid=True, comparisons={'parity_to_direct': {}},
                parity=result, direct=result, package=None))
        (output / 'rows.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        c.write(output / 'STATUS.json', dict(complete=True, invalid=0, missing=0,
            scheduled=len(expected), completed=len(expected), phase='preflight', freeze_sha256=c.sha(self.freeze)))
        return output

    def new_release_inputs(self):
        new = self.root / 'new.json'
        c.write(new, {'seeds': list(range(2000,2256))})
        audit = self.root / 'fresh/report.json'
        now = datetime.now(timezone.utc).isoformat()
        c.write(audit, dict(status='PASS', errors=[], created_utc=now, completed_utc=now,
                           npz_files_checked=1, remote_registry_hashes_rechecked=1))
        c.write(audit.parent / 'exclusions.json', {'seeds': [1,2,3]})
        completed = self.root / 'original_status.json'
        count = 256 * 16 * 2 * len(self.roster['candidates'])
        c.write(completed, dict(complete=True, invalid=0, missing=0, completed=count, scheduled=count,
            phase='holdout', roster_sha256=c.sha(self.bundle / 'evaluation/roster.json')))
        return new, audit, completed


class FreezeTests(Fixture):
    def test_four_distinct_seeds_and_separate_diagnostics(self):
        selected = self.plan['preflight_seeds']
        self.assertEqual(selected, {'representative':[101,202], 'stress':[303,404]})
        self.assertEqual(len(self.plan['primary_opponents']),16)
        self.assertEqual(len(self.plan['diagnostic_opponents']),5)
        self.assertEqual(len(c.jobs(self.plan,self.roster,self.opponents,'preflight')),336)

    def test_changed_unplayed_seed_rejected(self):
        c.write(self.bundle / 'manifests/representative.json', {'seeds':[101,999]})
        with self.assertRaisesRegex(ValueError,'Frozen input changed'):
            runner.build_manifest(self.freeze,self.bundle,'preflight',1)

    def test_opponent_analysis_audit_and_feature_changes_rejected(self):
        for name in ('opponents.json','evaluation/ANALYSIS_PLAN.md','campaign_audit/report.json',
                     'features/stress.jsonl','evaluation/condition_thresholds.json'):
            with self.subTest(name=name):
                path=self.bundle/name;original=path.read_bytes()
                path.write_bytes(original+b' ')
                with self.assertRaisesRegex(ValueError,'Frozen input changed'):
                    c.verify_freeze(self.freeze,self.bundle)
                path.write_bytes(original)

    def test_candidate_roster_expansion_rejected(self):
        roster=c.read(self.inputs/'roster.json');roster['candidates'].append(dict(roster['candidates'][0],id='third'))
        c.write(self.inputs/'roster.json',roster)
        with self.assertRaisesRegex(ValueError,'Frozen input changed'):
            c.verify_freeze(self.freeze,self.bundle)

    def test_missing_frozen_input_and_path_escape_rejected(self):
        with self.assertRaisesRegex(ValueError,'leaves frozen root'):
            c.inside(self.bundle,'../outside')
        (self.bundle/'features/stress.jsonl').unlink()
        with self.assertRaisesRegex(ValueError,'Frozen input changed'):
            c.verify_freeze(self.freeze,self.bundle)

    def test_runtime_mutation_rejected(self):
        name=next(iter(self.roster['runtime_files']))
        (self.bundle/'evaluation/runtime'/name).write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'Frozen input changed'):
            c.verify_freeze(self.freeze,self.bundle)

    def test_added_runtime_code_is_rejected(self):
        (self.bundle/'evaluation/runtime/new_helper.py').write_text('new code')
        with self.assertRaisesRegex(ValueError,'inventory changed'):
            c.verify_freeze(self.freeze,self.bundle)

    def test_analysis_code_change_is_rejected(self):
        tools=self.root/'copied_tools';tools.mkdir()
        for name in c.read(self.freeze)['files']['tools']:
            (tools/name).write_bytes((c.HERE/name).read_bytes())
        (tools/'analyze_panels.py').write_bytes(b'changed analysis')
        with patch.object(c,'HERE',tools),self.assertRaisesRegex(ValueError,'Frozen input changed: tools'):
            c.verify_freeze(self.freeze,self.bundle)

    def test_resume_requires_identical_manifest(self):
        path=self.root/'manifest.json';current={'freeze':'abc','jobs':'first','workers':4}
        c.resume_manifest(path,current);c.resume_manifest(path,current)
        with self.assertRaisesRegex(ValueError,'Resume inputs'):
            c.resume_manifest(path,dict(current,jobs='second'))

    def test_duplicate_missing_unexpected_invalid_rows_block_completion(self):
        expected=[('alpha','representative',101,'opp0',0)]
        path=self.root/'rows.jsonl';row=game(expected[0])
        for rows,pattern in [([], 'Missing'),([row,row],'Duplicate'),
                             ([dict(row,seed=999)],'Unexpected'),([dict(row,valid=False)],'Invalid')]:
            with self.subTest(pattern=pattern):
                path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
                with self.assertRaisesRegex(ValueError,pattern):
                    c.journal_rows(path,expected,complete=True)

    def test_preflight_contents_and_receipts_are_bound_to_resume(self):
        pre=self.make_preflight()
        first,_,_=runner.build_manifest(self.freeze,self.bundle,'development',1,pre)
        manifest=self.root/'run/MANIFEST.json';c.resume_manifest(manifest,first)
        path=pre/'rows.jsonl';path.write_text(path.read_text().replace('"seconds": 0.01','"seconds": 0.02'))
        changed,_,_=runner.build_manifest(self.freeze,self.bundle,'development',1,pre)
        with self.assertRaisesRegex(ValueError,'Resume inputs'):
            c.resume_manifest(manifest,changed)

    def test_wrong_or_incomplete_preflight_rejected(self):
        pre=self.make_preflight()
        status=c.read(pre/'STATUS.json');status['missing']=1;c.write(pre/'STATUS.json',status)
        with self.assertRaisesRegex(ValueError,'incomplete'):
            runner.build_manifest(self.freeze,self.bundle,'development',1,pre)

    def test_preflight_observation_coverage_rejected(self):
        pre=self.make_preflight();path=pre/'rows.jsonl'
        path.write_text(path.read_text().replace('"observations_checked": 1440','"observations_checked": 2',1))
        with self.assertRaisesRegex(ValueError,'all observations'):
            runner.build_manifest(self.freeze,self.bundle,'development',1,pre)

    def test_concurrent_run_lock_rejected(self):
        output=self.root/'locked';output.mkdir()
        with runner.run_lock(output):
            with self.assertRaises(FileExistsError):
                with runner.run_lock(output):pass
        self.assertFalse((output/'RUNNING.json').exists())


class HoldoutTests(Fixture):
    def test_old_release_cannot_authorize_new_comparison(self):
        with self.assertRaisesRegex(ValueError,'original release'):
            holdout.verify_release(self.bundle/'evaluation/HOLDOUT_RELEASE.json',self.freeze,self.bundle)

    def test_running_original_comparison_blocks_new_release(self):
        new,audit,status=self.new_release_inputs()
        value=c.read(status);value['complete']=False;c.write(status,value)
        with self.assertRaisesRegex(ValueError,'incomplete'):
            holdout.release_new(self.freeze,self.bundle,new,audit,self.retired,status,self.root/'release')

    def test_retired_seeds_rejected_even_when_file_encoding_changes(self):
        new,audit,status=self.new_release_inputs()
        new.write_text(json.dumps(c.read(self.retired),separators=(',',':')))
        self.assertNotEqual(c.sha(new),c.sha(self.retired))
        with self.assertRaisesRegex(ValueError,'overlaps'):
            holdout.release_new(self.freeze,self.bundle,new,audit,self.retired,status,self.root/'release')

    def test_stress_pool_and_audited_history_are_reserved(self):
        new,audit,status=self.new_release_inputs()
        for forbidden in (505,2):
            with self.subTest(seed=forbidden):
                c.write(new,{'seeds':[forbidden]+list(range(2001,2256))})
                with self.assertRaisesRegex(ValueError,'overlaps'):
                    holdout.release_new(self.freeze,self.bundle,new,audit,self.retired,status,self.root/'release')

    def test_stale_audit_rejected(self):
        new,audit,status=self.new_release_inputs();value=c.read(audit)
        value['created_utc']='2020-01-01T00:00:00Z';c.write(audit,value)
        with self.assertRaisesRegex(ValueError,'after freezing'):
            holdout.release_new(self.freeze,self.bundle,new,audit,self.retired,status,self.root/'release')

    def test_valid_release_resume_and_second_run_rejection(self):
        new,audit,status=self.new_release_inputs();output=self.root/'release'
        holdout.release_new(self.freeze,self.bundle,new,audit,self.retired,status,output)
        release=output/'RELEASE.json'
        self.assertEqual(holdout.verify_release(release,self.freeze,self.bundle),list(range(2000,2256)))
        manifest=dict(freeze_sha256=c.sha(self.freeze),jobs_sha256='fixed-jobs')
        holdout.claim_release(release,self.root/'first',manifest)
        holdout.claim_release(release,self.root/'first',manifest)
        with self.assertRaisesRegex(ValueError,'already used'):
            holdout.claim_release(release,self.root/'second',manifest)
        (output/'holdout.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'Released file changed'):
            holdout.verify_release(release,self.freeze,self.bundle)


class AnalysisTests(Fixture):
    def test_added_holdout_feature_file_is_ignored(self):
        (self.bundle/'features/holdout.jsonl').write_text(json.dumps({'seed':2000,'source':'stale'})+'\n')
        rows=analysis.load_feature_rows(self.bundle,'holdout',[2000])
        self.assertEqual(rows,[{'seed':2000,'source':'frozen generator'}])

    def test_paired_bootstrap_cancels_common_seed_variation(self):
        a=np.array([.2,.8,.4,.6]);b=a+.1
        rng=np.random.default_rng(13);indices=rng.integers(0,4,size=(4000,4))
        boot=np.stack([a[indices].mean(axis=1),b[indices].mean(axis=1)])
        result=analysis.simultaneous_pairs(np.array([a.mean(),b.mean()]),boot,['a','b'])
        pair=result['pairs'][0]
        self.assertAlmostEqual(pair['difference'],.1)
        self.assertAlmostEqual(pair['ci95'][0],.1)
        self.assertAlmostEqual(pair['ci95'][1],.1)

    def test_end_to_end_diagnostics_cannot_inflate_primary(self):
        pre=self.make_preflight();run=self.root/'development'
        manifest,expected,_=runner.build_manifest(self.freeze,self.bundle,'development',1,pre)
        c.write(run/'MANIFEST.json',manifest)
        rows=[game(key,winner=key[3].startswith('stress:'),draw=not key[3].startswith('stress:')) for key in expected]
        (run/'rows.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
        c.write(run/'STATUS.json',dict(complete=True,invalid=0,missing=0,completed=len(expected),scheduled=len(expected),
                                     freeze_sha256=c.sha(self.freeze)))
        out=self.root/'results'
        argv=['analyze','--bundle',str(self.bundle),'--freeze',str(self.freeze),'--preflight',str(pre),
              '--development',str(run),'--out',str(out)]
        with patch.object(sys,'argv',argv),contextlib.redirect_stdout(io.StringIO()):
            analysis.main()
        result=c.read(out/'RESULTS.json')
        for panel in ('representative','stress'):
            primary=result['panels'][panel]['overall']['candidates']['alpha']
            diagnostic=result['panels'][panel]['diagnostic_overall']['candidates']['alpha']
            self.assertEqual(primary['games'],64)
            self.assertEqual(primary['win_rate'],0.)
            self.assertEqual(primary['match_score'],.5)
            self.assertEqual(diagnostic['games'],20)
            self.assertEqual(diagnostic['win_rate'],1.)
            self.assertEqual(primary['paired_differences']['alpha']['ci95'],[0.,0.])
        self.assertTrue((out/'REALIZED_MARKETS.json').exists())
        # Altering the preflight receipt blocks a later analysis of this run.
        path=pre/'STATUS.json';value=c.read(path);value['new_field']='changed';c.write(path,value)
        argv[-1]=str(self.root/'new_results')
        with patch.object(sys,'argv',argv),self.assertRaisesRegex(ValueError,'manifest'):
            analysis.main()


if __name__=='__main__':unittest.main()
